"""
PPO trainer with rollout buffer, GAE, and clipped surrogate objective.
Optimized for speed: vectorized GAE, pre-allocated buffers, no_grad for rollout only.
Supports auxiliary tasks.
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import cv2
from .parallel_trainer_base import ParallelTrainerBase
from src.training.losses import PPOLoss, AuxiliaryLoss

from src.core.constants import OBSERVATION_SIZE


class PPOTrainer(ParallelTrainerBase):
    def __init__(self, config):
        config['model']['use_value_head'] = True
        super().__init__(config)

        train_cfg = self.config['training']
        self.ppo_intra_epochs = train_cfg['ppo_intra_epochs']
        self.mini_batch_size = train_cfg['mini_batch_size']
        self.clip_epsilon = train_cfg['clip_epsilon']
        self.value_coef = train_cfg['value_coef']
        self.entropy_coef = train_cfg['entropy_coef']
        self.gamma = train_cfg['gamma']
        self.gae_lambda = train_cfg['gae_lambda']

        # Disable AMP – huge overhead due to .item() calls inside GradScaler
        self.use_amp = False
        self.scaler = None

        # ---------- Pre-allocated tensors (avoids repeated GPU allocations) ----------
        self.obs_tensor = torch.empty(
            (self.batch_size, 1, OBSERVATION_SIZE),
            dtype=torch.long, device=self.device
        )
        self.reward_tensor = torch.empty(self.batch_size, dtype=torch.float32, device=self.device)
        self.done_tensor = torch.empty(self.batch_size, dtype=torch.float32, device=self.device)
        self.action_buffer = np.zeros(self.batch_size, dtype=np.int64)   # for CPU transfer
        # ----------------------------------------------------------------------------

        self.ppo_loss_fn = PPOLoss(
            clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        if self.agent.use_auxiliary:
            self.aux_loss_fn = AuxiliaryLoss(energy_coef=0.1, obs_coef=0.05)
        else:
            self.aux_loss_fn = None

    def _collect_rollout(self) -> dict:
        max_steps = self.vector_env[0].max_steps
        B = self.batch_size
        device = self.device

        obs_array, _ = self.vector_env.reset()
        obs_array = np.array(obs_array, dtype=np.int64)
        self.agent.reset()

        self.obs_tensor.copy_(
            torch.from_numpy(obs_array).unsqueeze(1).to(device, non_blocking=True)
        )

        storage = {
            'obs': [None] * max_steps,
            'actions': [None] * max_steps,
            'rewards': [None] * max_steps,
            'dones': [None] * max_steps,
            'values': [None] * max_steps,
            'logits': [None] * max_steps,
        }
        if self.agent.use_auxiliary:
            storage['energies'] = [None] * max_steps
            storage['next_obs'] = [None] * max_steps

        with torch.no_grad():
            for step in range(max_steps):
                outputs = self.agent.network(
                    self.obs_tensor,
                    return_auxiliary=self.agent.use_auxiliary,
                    return_value=True
                )
                logits = outputs[0].squeeze(1)      # [B, A]
                value = outputs[-1].squeeze(1)      # [B]

                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)

                storage['obs'][step] = self.obs_tensor.squeeze(1).clone()
                storage['actions'][step] = actions
                storage['logits'][step] = logits
                storage['values'][step] = value

                actions_cpu = actions.cpu().numpy()
                self.action_buffer[:] = actions_cpu
                actions_np = self.action_buffer

                obs_array, rewards_list, terminated_list, truncated_list, infos = self.vector_env.step(actions_np)

                obs_array = np.array(obs_array, dtype=np.int64)
                rewards = np.array(rewards_list, dtype=np.float32)
                terminated = np.array(terminated_list, dtype=bool)
                truncated = np.array(truncated_list, dtype=bool)
                dones = terminated | truncated

                self.reward_tensor.copy_(torch.from_numpy(rewards).to(device, non_blocking=True))
                storage['rewards'][step] = self.reward_tensor.clone()

                self.done_tensor.copy_(torch.from_numpy(dones).to(device, non_blocking=True))
                storage['dones'][step] = self.done_tensor.clone()

                if self.agent.use_auxiliary:
                    # Store current energy and next observation as targets
                    energies = [info.energy for info in infos]
                    if not hasattr(self, 'energy_tensor'):
                        self.energy_tensor = torch.empty(B, dtype=torch.float32, device=device)
                    self.energy_tensor.copy_(torch.from_numpy(np.array(energies)).to(device, non_blocking=True))
                    storage['energies'][step] = self.energy_tensor.clone()

                    # Next observation target (what we will predict from current state)
                    next_obs_tensor = torch.from_numpy(obs_array).unsqueeze(1).to(device, non_blocking=True)
                    storage['next_obs'][step] = next_obs_tensor.squeeze(1).clone()

                # Prepare next observation
                self.obs_tensor.copy_(torch.from_numpy(obs_array).unsqueeze(1).to(device, non_blocking=True))

                if dones.all():
                    for k in storage:
                        storage[k] = storage[k][:step+1]
                    break

        for k in storage:
            storage[k] = torch.stack(storage[k], dim=1)

        storage['values'] = storage['values'].squeeze(-1)

        mask = torch.ones_like(storage['rewards'])
        advantages, returns = self.ppo_loss_fn.compute_gae(
            storage['rewards'],
            storage['values'],
            storage['dones'],
            mask
        )

        experiences = {
            'observations': storage['obs'],
            'actions': storage['actions'],
            'rewards': storage['rewards'],
            'advantages': advantages,
            'returns': returns,
            'old_logits': storage['logits'],
            'old_values': storage['values'],
            'mask': mask,
        }
        if self.agent.use_auxiliary:
            experiences['energy_targets'] = storage['energies']
            experiences['obs_targets'] = storage['next_obs']

        return experiences

    def _train_step(self, experiences: dict) -> dict:
        """Perform multiple PPO epochs over the rollout buffer using mini-batches."""
        B, T = experiences['observations'].shape[:2]
        total_envs = B
        indices = torch.randperm(total_envs, device=self.device)

        # Detach tensors that should not have gradients
        for key in ['old_logits', 'old_values', 'advantages', 'returns', 'mask']:
            experiences[key] = experiences[key].detach()
        if self.agent.use_auxiliary:
            experiences['energy_targets'] = experiences['energy_targets'].detach()
            experiences['obs_targets'] = experiences['obs_targets'].detach()

        metrics_sum = {}
        network = self.agent.network
        optimizer = self.optimizer
        scaler = self.scaler
        use_amp = self.use_amp

        network.train()

        for _ in range(self.ppo_intra_epochs):
            for start in range(0, total_envs, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_envs)
                env_idx = indices[start:end]
                batch = {k: v[env_idx] for k, v in experiences.items()}

                self.agent.reset()   # resets LSTM/transformer memory to zeros

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = network(
                            batch['observations'],
                            return_auxiliary=self.agent.use_auxiliary,
                            return_value=True
                        )
                        logits = outputs[0]
                        values = outputs[-1].squeeze(-1)
                        loss, metrics = self.ppo_loss_fn(
                            logits, batch['old_logits'], batch['actions'],
                            batch['advantages'], batch['returns'], values, batch['mask']
                        )
                        if self.agent.use_auxiliary:
                            energy_pred = outputs[1]
                            obs_pred = outputs[2]
                            aux_loss = self.aux_loss_fn(
                                energy_pred, batch['energy_targets'],
                                obs_pred, batch['obs_targets'].float(),
                                batch['mask']
                            )
                            loss = loss + aux_loss
                            metrics['aux_loss'] = aux_loss.item()
                            # Compute component losses for logging
                            with torch.no_grad():
                                energy_loss = torch.nn.functional.mse_loss(energy_pred.squeeze(-1), batch['energy_targets'])
                                obs_loss = torch.nn.functional.mse_loss(obs_pred, batch['obs_targets'].float())
                                if batch['mask'] is not None:
                                    valid_ratio = batch['mask'].sum() / (batch['mask'].numel() + 1e-8)
                                    energy_loss = energy_loss * valid_ratio
                                    obs_loss = obs_loss * valid_ratio
                                metrics['energy_loss'] = energy_loss.item()
                                metrics['obs_loss'] = obs_loss.item()
                else:
                    outputs = network(
                        batch['observations'],
                        return_auxiliary=self.agent.use_auxiliary,
                        return_value=True
                    )
                    logits = outputs[0]
                    values = outputs[-1].squeeze(-1)
                    loss, metrics = self.ppo_loss_fn(
                        logits, batch['old_logits'], batch['actions'],
                        batch['advantages'], batch['returns'], values, batch['mask']
                    )
                    if self.agent.use_auxiliary:
                        energy_pred = outputs[1]
                        obs_pred = outputs[2]
                        aux_loss = self.aux_loss_fn(
                            energy_pred, batch['energy_targets'],
                            obs_pred, batch['obs_targets'].float(),
                            batch['mask']
                        )
                        loss = loss + aux_loss
                        metrics['aux_loss'] = aux_loss.item()
                        with torch.no_grad():
                            energy_loss = torch.nn.functional.mse_loss(energy_pred.squeeze(-1), batch['energy_targets'])
                            obs_loss = torch.nn.functional.mse_loss(obs_pred, batch['obs_targets'].float())
                            if batch['mask'] is not None:
                                valid_ratio = batch['mask'].sum() / (batch['mask'].numel() + 1e-8)
                                energy_loss = energy_loss * valid_ratio
                                obs_loss = obs_loss * valid_ratio
                            metrics['energy_loss'] = energy_loss.item()
                            metrics['obs_loss'] = obs_loss.item()

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                self.gradient_clipper.clip(network.parameters())

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Detach hidden state to prevent backprop through time across updates
                if hasattr(network, 'hidden_state') and network.hidden_state is not None:
                    h, c = network.hidden_state
                    network.hidden_state = (h.detach(), c.detach())

                # Aggregate metrics
                for k, v in metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v / self.ppo_intra_epochs / (total_envs / self.mini_batch_size)

        return metrics_sum

    def train(self):
        """Main training loop."""
        self._run_initial_test()
        dummy = self._setup_visualization()

        pbar = tqdm(range(self.start_epoch, self.epochs), desc="PPO Epochs")
        self._start_training_timer()
        start_time = time.time()

        for epoch in pbar:
            if self._should_test(epoch):
                self._run_test(epoch)

            if self._should_save(epoch):
                self._save_checkpoint(epoch)

            experiences = self._collect_rollout()
            train_metrics = self._train_step(experiences)

            avg_reward = experiences['rewards'].sum(dim=1).mean().item()
            self.metrics['train_rewards'].append(avg_reward)
            self.metrics['train_losses'].append(train_metrics.get('loss', 0))
            self.metrics.setdefault('policy_losses', []).append(train_metrics.get('policy_loss', 0))
            if 'value_loss' in train_metrics:
                self.metrics.setdefault('value_losses', []).append(train_metrics['value_loss'])
            if 'entropy' in train_metrics:
                self.metrics.setdefault('entropies', []).append(train_metrics['entropy'])
            if self.agent.use_auxiliary:
                self.metrics.setdefault('aux_losses', []).append(train_metrics.get('aux_loss', 0.0))
                self.metrics.setdefault('energy_losses', []).append(train_metrics.get('energy_loss', 0.0))
                self.metrics.setdefault('obs_losses', []).append(train_metrics.get('obs_loss', 0.0))

            self.lr_scheduler.step()

            # Hook for dynamic complexity or early stop
            if self._post_epoch_hook(epoch, dummy) is True:
                break

            self._update_progress_bar(pbar, avg_reward)

        self._finalise_total_training_time()
        self._finalize_training()
        self._print_training_summary(start_time)

    def _print_training_summary(self, start_time):
        total_time = time.time() - start_time
        print(f"\n{'='*80}\nPPO TRAINING SUMMARY\n{'='*80}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best reward: {self.metrics['best_reward']:.2f}")
        if getattr(self, 'dynamic', False):
            print(f"Final stage: {self.complexity_manager.get_current_task_class()}")
            print(f"Final complexity: {self.complexity_manager.get_current_complexity():.2f}")
            print(f"Total adjustments: {self.complexity_manager.adjustments_made}")
        print(f"Model saved in: {self.experiment_dir}")
        print(f"{'='*80}")