# src/training/parallel_trainer_base.py
"""
REINFORCE trainer with optional dynamic complexity.
Optimized rollout collection for unchanged VectorizedMazeEnv.
Uses minimal scalar extraction to avoid expensive .cpu() calls.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

from __future__ import annotations

import time
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .parallel_trainer_base import ParallelTrainerBase
from src.training.losses import PolicyLoss, AuxiliaryLoss


class ReinforceTrainer(ParallelTrainerBase):
    """
    REINFORCE with baseline (normalized returns) and entropy bonus.
    Supports:
        - per-episode updates
        - per-epoch updates
        - optional auxiliary losses
        - dynamic curriculum complexity
    """

    def __init__(self, config):
        # REINFORCE does not use a value head
        config["model"]["use_value_head"] = False

        super().__init__(config)

        train_cfg = self.config["training"]

        self.policy_loss_fn = PolicyLoss(
            gamma=train_cfg["gamma"],
            entropy_coef=train_cfg["entropy_coef"],
            normalize_advantages=True,
        )

        if self.agent.use_auxiliary:
            self.aux_loss_fn = AuxiliaryLoss(
                energy_coef=0.1,
                obs_coef=0.05,
            )
        else:
            self.aux_loss_fn = None

    @staticmethod
    def _obs_to_numpy(obs_array) -> np.ndarray:
        """
        Fast conversion from pybind nested lists.
        """
        return np.asarray(obs_array, dtype=np.int64)

    @staticmethod
    def _scalar(x):
        """
        Convert a scalar tensor to a Python float using .item().
        This is cheaper here than .detach().cpu().
        """
        if isinstance(x, torch.Tensor):
            return x.item()
        return float(x)

    def _collect_experiences_parallel(self, full_reset: bool = True) -> dict:
        """
        Collect one episode from each parallel environment.
        """
        max_steps = self.vector_env[0].max_steps

        # Reset recurrent hidden state
        self.agent.reset()

        # Reset environments
        if full_reset:
            obs_array, _ = self.vector_env.reset()
        else:
            obs_array, _ = self.vector_env.soft_reset_all()

        obs_np = self._obs_to_numpy(obs_array)

        if obs_np.ndim == 1:
            obs_np = obs_np[:, None]

        batch_size = obs_np.shape[0]
        obs_dim = obs_np.shape[1]

        # Initial observations
        observations = (
            torch.from_numpy(obs_np)
            .to(self.device, non_blocking=True)
            .unsqueeze(1)
        )  # [B, 1, K]

        # Preallocate rollout buffers
        obs_buf = torch.empty(
            (batch_size, max_steps, obs_dim),
            dtype=torch.long,
            device=self.device,
        )

        next_obs_buf = torch.empty(
            (batch_size, max_steps, obs_dim),
            dtype=torch.long,
            device=self.device,
        )

        act_buf = torch.empty(
            (batch_size, max_steps),
            dtype=torch.long,
            device=self.device,
        )

        rew_buf = torch.empty(
            (batch_size, max_steps),
            dtype=torch.float32,
            device=self.device,
        )

        energy_buf = torch.empty(
            (batch_size, max_steps),
            dtype=torch.float32,
            device=self.device,
        )

        current_energies = np.fromiter(
            (self.vector_env[i].energy for i in range(batch_size)),
            dtype=np.float32,
            count=batch_size,
        )

        t = 0

        with torch.inference_mode():
            for step in range(max_steps):
                # Store observations
                obs_buf[:, step, :] = observations.squeeze(1)

                # Store energies
                energy_buf[:, step] = torch.from_numpy(current_energies).to(
                    self.device,
                    non_blocking=True,
                )

                # Forward pass
                logits = self.agent.network(observations).squeeze(1)
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

                # pybind expects a python integer sequence
                actions_np = actions.cpu().tolist()

                (
                    obs_array,
                    r_list,
                    terminated_list,
                    truncated_list,
                    infos,
                ) = self.vector_env.step(actions_np)

                # Next observations
                next_obs_np = self._obs_to_numpy(obs_array)
                if next_obs_np.ndim == 1:
                    next_obs_np = next_obs_np[:, None]

                next_obs_tensor = (
                    torch.from_numpy(next_obs_np)
                    .to(self.device, non_blocking=True)
                    .unsqueeze(1)
                )

                # Rewards / dones
                rewards_np = np.asarray(r_list, dtype=np.float32)
                terminated_np = np.asarray(terminated_list, dtype=bool)
                truncated_np = np.asarray(truncated_list, dtype=bool)
                dones = terminated_np | truncated_np

                # Store rollout
                rew_buf[:, step] = torch.from_numpy(rewards_np).to(
                    self.device,
                    non_blocking=True,
                )
                act_buf[:, step] = actions
                next_obs_buf[:, step, :] = next_obs_tensor.squeeze(1)

                # Update energies
                current_energies = np.fromiter(
                    (info.energy for info in infos),
                    dtype=np.float32,
                    count=batch_size,
                )

                observations = next_obs_tensor
                t += 1

                if dones.all():
                    break

        return {
            "observations": obs_buf[:, :t],
            "actions": act_buf[:, :t],
            "rewards": rew_buf[:, :t],
            "mask": torch.ones(
                (batch_size, t),
                dtype=torch.float32,
                device=self.device,
            ),
            "energy_targets": energy_buf[:, :t],
            "next_obs_targets": next_obs_buf[:, :t],
        }

    def _compute_loss(self, experiences):
        """
        Compute policy and optional auxiliary losses.
        Returns a loss tensor plus metrics dictionary.
        """
        obs = experiences["observations"]
        actions = experiences["actions"]
        rewards = experiences["rewards"]
        mask = experiences.get("mask")

        # Reset recurrent state for training batch
        self.agent.reset()

        reward_scalar = rewards.sum(dim=1).mean()

        if self.aux_loss_fn and self.agent.use_auxiliary:
            out = self.agent.network(obs, return_auxiliary=True)

            if not isinstance(out, tuple) or len(out) != 3:
                raise ValueError("Network did not return auxiliary outputs")

            logits, energy_pred, obs_pred_logits = out   # <-- renamed from obs_pred

            # Compute policy loss WITHOUT entropy for logging
            # Use unnormalized advantages for logging to avoid zero values
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Compute returns (discounted)
            returns = self.policy_loss_fn._compute_returns(rewards)
            
            # Use raw returns (not normalized) for policy loss logging
            # This ensures non-zero values even when all returns are similar
            policy_loss_only = -(action_log_probs * returns.detach()).mean()

            # Full policy loss with entropy (used for training) - uses normalized advantages
            policy_loss_with_entropy, entropy = self.policy_loss_fn(
                logits,
                actions,
                rewards,
            )

            energy_target = experiences["energy_targets"]
            obs_target = experiences["next_obs_targets"]   # integer tokens, not float

            # pass obs_pred_logits (logits) and obs_target (int)
            aux_loss = self.aux_loss_fn(energy_pred, energy_target,
                                        obs_pred_logits, obs_target, mask)

            # Extract component losses for logging
            with torch.no_grad():
                energy_loss = F.mse_loss(energy_pred.squeeze(-1), energy_target)
                # Cross-entropy loss is already logged as aux_loss's obs part,
                # but we can still compute it separately for logging if desired.
                # Here we compute it again for consistency.
                B, T, obs_size, vocab_size = obs_pred_logits.shape
                obs_ce_loss = F.cross_entropy(
                    obs_pred_logits.view(-1, vocab_size),
                    obs_target.view(-1).long()
                )
                if mask is not None:
                    valid_ratio = mask.sum() / (mask.numel() + 1e-8)
                    energy_loss = energy_loss * valid_ratio
                    obs_ce_loss = obs_ce_loss * valid_ratio

            total_loss = policy_loss_with_entropy + aux_loss

            metrics = {
                "loss": self._scalar(total_loss.detach()),
                "reward": self._scalar(reward_scalar.detach()),
                "policy_loss": self._scalar(policy_loss_only.detach()),
                "aux_loss": self._scalar(aux_loss.detach()),
                "energy_loss": self._scalar(energy_loss.detach()),
                "obs_loss": self._scalar(obs_ce_loss.detach()),   # now cross-entropy value
            }

        else:
            logits = self.agent.network(obs)

            # Compute policy loss WITHOUT entropy for logging
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            
            # Compute returns (discounted)
            returns = self.policy_loss_fn._compute_returns(rewards)
            
            # Use raw returns (not normalized) for policy loss logging
            policy_loss_only = -(action_log_probs * returns.detach()).mean()

            # Full policy loss with entropy (used for training) - uses normalized advantages
            policy_loss_with_entropy, entropy = self.policy_loss_fn(
                logits,
                actions,
                rewards,
            )

            total_loss = policy_loss_with_entropy

            metrics = {
                "loss": self._scalar(total_loss.detach()),
                "reward": self._scalar(reward_scalar.detach()),
                "policy_loss": self._scalar(policy_loss_only.detach()),
            }

        return total_loss, metrics

    def _train_step(self, experiences):
        """
        Perform one optimization step.
        """
        self.agent.network.train()

        loss, metrics = self._compute_loss(experiences)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.gradient_clipper.clip(self.agent.network.parameters())
        self.optimizer.step()

        if hasattr(self.agent.network, "flush_cache_buffer"):
            self.agent.network.flush_cache_buffer()

        return metrics

    def train(self):
        """
        Main REINFORCE training loop.
        """
        self._run_initial_test()
        dummy = self._setup_visualization()

        pbar = tqdm(
            range(self.start_epoch, self.epochs),
            desc="Epochs",
        )

        self._start_training_timer()
        start_time = time.time()

        try:
            for epoch in pbar:
                # Generate first grid
                current_config = self._generate_grid_config()
                self._grid_pool = [current_config]
                self._apply_grid_config(
                    current_config,
                    reset_hidden=True,
                )

                batched_experiences = None
                epoch_rewards = []

                if self._should_test(epoch):
                    self._run_test(epoch)

                if self._should_save(epoch):
                    self._save_checkpoint(epoch)

                # Collect episodes
                for ep_idx in range(self.reinforce_intra_epochs):
                    if ep_idx == 0:
                        full_reset = True
                    else:
                        if np.random.random() < self.grid_change_prob:
                            new_config = self._generate_grid_config()
                            self._grid_pool.append(new_config)
                            current_config = new_config
                            full_reset = True
                        else:
                            chosen_config = random.choice(self._grid_pool)
                            if chosen_config == current_config:
                                full_reset = False
                            else:
                                current_config = chosen_config
                                full_reset = True

                        if full_reset:
                            self._apply_grid_config(
                                current_config,
                                reset_hidden=False,
                            )

                    experiences = self._collect_experiences_parallel(
                        full_reset=full_reset
                    )

                    if self.update_per_episode:
                        metrics = self._train_step(experiences)
                        reward = metrics["reward"]

                        self.metrics["train_rewards"].append(reward)
                        self.metrics["train_losses"].append(metrics["loss"])
                        self.metrics.setdefault("policy_losses", []).append(metrics.get("policy_loss", 0))
                        if self.agent.use_auxiliary:
                            self.metrics.setdefault("aux_losses", []).append(metrics.get("aux_loss", 0.0))
                            self.metrics.setdefault("energy_losses", []).append(metrics.get("energy_loss", 0.0))
                            self.metrics.setdefault("obs_losses", []).append(metrics.get("obs_loss", 0.0))
                        epoch_rewards.append(reward)
                        self.lr_scheduler.step()

                    else:
                        if batched_experiences is None:
                            batched_experiences = experiences
                        else:
                            for k in batched_experiences:
                                batched_experiences[k] = torch.cat(
                                    [batched_experiences[k], experiences[k]],
                                    dim=1,
                                )

                # Batched update
                if not self.update_per_episode:
                    metrics = self._train_step(batched_experiences)
                    reward = metrics["reward"]

                    self.metrics["train_rewards"].append(reward)
                    self.metrics["train_losses"].append(metrics["loss"])
                    self.metrics.setdefault("policy_losses", []).append(metrics.get("policy_loss", 0))
                    if self.agent.use_auxiliary:
                        self.metrics.setdefault("aux_losses", []).append(metrics.get("aux_loss", 0.0))
                        self.metrics.setdefault("energy_losses", []).append(metrics.get("energy_loss", 0.0))
                        self.metrics.setdefault("obs_losses", []).append(metrics.get("obs_loss", 0.0))
                    epoch_rewards = [reward]
                    self.lr_scheduler.step()

                avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
                self.metrics.setdefault("epoch_rewards", []).append(avg_epoch_reward)

                if self._post_epoch_hook(epoch, dummy) is True:
                    break

                self._update_progress_bar(pbar, avg_epoch_reward)

        except StopIteration:
            print("\nStopped by post_epoch_hook (stagnation).")

        finally:
            cv2.destroyAllWindows()

        self._finalise_total_training_time()
        self._finalize_training()
        self._print_training_summary(start_time)

    def _print_training_summary(self, start_time):
        total_time = time.time() - start_time
        print(f"\n{'=' * 80}\nREINFORCE TRAINING SUMMARY\n{'=' * 80}")
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        print(f"Best reward: {self.metrics['best_reward']:.2f}")
        if self.dynamic:
            print(f"Final stage: {self.complexity_manager.get_current_task_class()}")
            print(f"Final complexity: {self.complexity_manager.get_current_complexity():.2f}")
            print(f"Total adjustments: {self.complexity_manager.adjustments_made}")
        print(f"Model saved in: {self.experiment_dir}")
        print(f"{'=' * 80}")