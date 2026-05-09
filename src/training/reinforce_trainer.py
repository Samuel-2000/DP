"""
REINFORCE trainer with optional dynamic complexity.
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import cv2
from .parallel_trainer_base import ParallelTrainerBase
from src.training.losses import PolicyLoss, AuxiliaryLoss
import random

class ReinforceTrainer(ParallelTrainerBase):
    """
    REINFORCE with baseline (normalized returns) and entropy bonus.
    Updates the policy either per episode or per epoch (batch of episodes).
    """

    def __init__(self, config):
        # Ensure no value head (REINFORCE doesn't use it)
        config['model']['use_value_head'] = False
        super().__init__(config)
        train_cfg = self.config['training']
        self.policy_loss_fn = PolicyLoss(
            gamma=train_cfg['gamma'],
            entropy_coef=train_cfg['entropy_coef'],
            normalize_advantages=True
        )
        if self.agent.use_auxiliary:
            self.aux_loss_fn = AuxiliaryLoss(energy_coef=0.1, obs_coef=0.05)
        else:
            self.aux_loss_fn = None

    def _collect_experiences_parallel(self, full_reset: bool = True) -> dict:
        """
        Collect one episode from each parallel environment.
        If full_reset is True, generate new random grids and reset agent memory.
        Otherwise, soft reset (keep grid layout, only reset agent position/food).
        Returns a dictionary with batched tensors.
        """
        max_steps = self.vector_env[0].max_steps
        self.agent.reset()

        if full_reset:
            obs_array, _ = self.vector_env.reset()
        else:
            obs_array, _ = self.vector_env.soft_reset_all()

        observations = torch.as_tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)
        all_obs = []
        all_actions = []
        all_rewards = []
        all_energies = []
        all_next_obs = []
        # Get initial energies directly from the environment (fast property)
        current_energies = [self.vector_env[i].energy for i in range(self.batch_size)]

        for step in range(max_steps):
            all_obs.append(observations.clone())
            all_energies.append(torch.tensor(current_energies, dtype=torch.float32, device=self.device))

            with torch.no_grad():
                logits = self.agent.network(observations).squeeze(1)
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)

            self.action_buffer[:] = actions.cpu().numpy()
            actions_np = self.action_buffer

            # Step the environment – returns lists from C++
            obs_array, r_list, terminated_list, truncated_list, infos = self.vector_env.step(actions_np)
            
            # Convert to numpy arrays for vector operations
            r = np.array(r_list, dtype=np.float32)
            terminated = np.array(terminated_list, dtype=bool)
            truncated = np.array(truncated_list, dtype=bool)
            dones = terminated | truncated
            
            next_obs_tensor = torch.as_tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)

            all_actions.append(actions)
            all_rewards.append(torch.tensor(r, dtype=torch.float32, device=self.device))
            all_next_obs.append(next_obs_tensor.clone())
            
            # infos are StepInfo objects – direct attribute access (fast, no dict)
            current_energies = [info.energy for info in infos]
            observations = next_obs_tensor

            if dones.all():
                break

        return {
            'observations': torch.cat(all_obs, dim=1),          # [B, T, K]
            'actions': torch.stack(all_actions, dim=1),         # [B, T]
            'rewards': torch.stack(all_rewards, dim=1),         # [B, T]
            'mask': torch.ones_like(torch.stack(all_rewards, dim=1)),  # all valid
            'energy_targets': torch.stack(all_energies, dim=1), # [B, T]
            'next_obs_targets': torch.cat(all_next_obs, dim=1), # [B, T, K]
        }

    def _compute_loss(self, experiences):
        """
        Compute the policy gradient loss (and optional auxiliary loss).
        """
        obs = experiences['observations']
        actions = experiences['actions']
        rewards = experiences['rewards']
        mask = experiences.get('mask')

        self.agent.reset()   # Reset LSTM/transformer state for this batch

        if self.aux_loss_fn and self.agent.use_auxiliary:
            out = self.agent.network(obs, return_auxiliary=True)
            if isinstance(out, tuple):
                logits, energy_pred, obs_pred = out
            else:
                raise ValueError("Network did not return auxiliary outputs")
            policy_loss, entropy = self.policy_loss_fn(logits, actions, rewards)
            energy_target = experiences['energy_targets']
            obs_target = experiences['next_obs_targets']
            aux_loss = self.aux_loss_fn(energy_pred, energy_target, obs_pred, obs_target.float(), mask)
            total_loss = policy_loss + aux_loss
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'aux_loss': aux_loss.item(),
                'energy_loss': (energy_pred - energy_target.unsqueeze(-1)).pow(2).mean().item(),
                'obs_loss': (obs_pred - obs_target.float()).pow(2).mean().item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item(),
            }
        else:
            logits = self.agent.network(obs)
            policy_loss, entropy = self.policy_loss_fn(logits, actions, rewards)
            total_loss = policy_loss
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item(),
            }
        return total_loss, metrics

    def _train_step(self, experiences):
        """Perform one gradient update using the collected experiences."""
        self.agent.network.train()
        loss, metrics = self._compute_loss(experiences)
        self.optimizer.zero_grad()
        loss.backward()
        self.gradient_clipper.clip(self.agent.network.parameters())
        self.optimizer.step()
        if hasattr(self.agent.network, 'flush_cache_buffer'):
            self.agent.network.flush_cache_buffer()
        return metrics

    def train(self):
        """
        Main training loop for REINFORCE.
        Supports both per-episode updates and per-epoch updates (batch of episodes).
        Also handles dynamic complexity and visualisation.
        """
        self._run_initial_test()
        dummy = self._setup_visualization()

        pbar = tqdm(range(self.start_epoch, self.epochs), desc="Epochs")
        self._start_training_timer()
        start_time = time.time()

        try:
            for epoch in pbar:
                # Generate first grid for this epoch
                current_config = self._generate_grid_config()
                self._grid_pool = [current_config]
                self._apply_grid_config(current_config, reset_hidden=True)

                batched_experiences = None
                epoch_rewards = []
                
                if self._should_test(epoch):
                    self._run_test(epoch)

                if self._should_save(epoch):
                    self._save_checkpoint(epoch)

                # Collect episodes (consecutive episodes on same or different grids)
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
                            self._apply_grid_config(current_config, reset_hidden=False)

                    experiences = self._collect_experiences_parallel(full_reset=full_reset)

                    if self.update_per_episode:
                        metrics = self._train_step(experiences)
                        reward = metrics['reward']
                        self.metrics['train_rewards'].append(reward)
                        self.metrics['train_losses'].append(metrics['loss'])
                        epoch_rewards.append(reward)
                        self.lr_scheduler.step()
                    else:
                        if batched_experiences is None:
                            batched_experiences = experiences
                        else:
                            for k in batched_experiences:
                                batched_experiences[k] = torch.cat([batched_experiences[k], experiences[k]], dim=1)

                if not self.update_per_episode:
                    metrics = self._train_step(batched_experiences)
                    reward = metrics['reward']
                    self.metrics['train_rewards'].append(reward)
                    self.metrics['train_losses'].append(metrics['loss'])
                    epoch_rewards = [reward]
                    self.lr_scheduler.step()

                avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                self.metrics.setdefault('epoch_rewards', []).append(avg_epoch_reward)

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
        print(f"\n{'='*80}\nREINFORCE TRAINING SUMMARY\n{'='*80}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best reward: {self.metrics['best_reward']:.2f}")
        if self.dynamic:
            print(f"Final stage: {self.complexity_manager.get_current_task_class()}")
            print(f"Final complexity: {self.complexity_manager.get_current_complexity():.2f}")
            print(f"Total adjustments: {self.complexity_manager.adjustments_made}")
        print(f"Model saved in: {self.experiment_dir}")
        print(f"{'='*80}")