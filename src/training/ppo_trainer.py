"""
PPO trainer with rollout buffer, GAE, and clipped surrogate objective.
Optimized for speed: vectorized GAE, pre-allocated buffers, no_grad for rollout only.
"""

import torch
import numpy as np
from tqdm import tqdm
import time
from .parallel_trainer_base import ParallelTrainerBase
from src.training.losses import PPOLoss


class PPOTrainer(ParallelTrainerBase):
    def __init__(self, config):
        config['model']['use_value_head'] = True
        super().__init__(config)

        train_cfg = self.config['training']
        self.ppo_epochs = train_cfg['ppo_epochs']
        self.mini_batch_size = train_cfg['mini_batch_size']
        self.clip_epsilon = train_cfg['clip_epsilon']
        self.value_coef = train_cfg['value_coef']
        self.entropy_coef = train_cfg['entropy_coef']
        self.gamma = train_cfg['gamma']
        self.gae_lambda = train_cfg['gae_lambda']

        # Mixed precision (GPU only)
        self.use_amp = self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp) if self.use_amp else None

        self.ppo_loss_fn = PPOLoss(
            clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

    def _collect_rollout(self) -> dict:
        """Collect one full episode from each parallel environment. Optimized."""
        max_steps = self.vector_env.envs[0].max_steps
        B = self.batch_size
        device = self.device

        obs_array, _ = self.vector_env.reset()
        self.agent.reset()

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

        obs = torch.tensor(obs_array, dtype=torch.long, device=device).unsqueeze(1)

        # Inference only – no gradients needed
        with torch.no_grad():
            for step in range(max_steps):
                outputs = self.agent.network(obs, return_auxiliary=self.agent.use_auxiliary, return_value=True)
                logits = outputs[0].squeeze(1)
                value = outputs[-1].squeeze(1)

                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()

                storage['obs'][step] = obs.squeeze(1)
                storage['actions'][step] = actions
                storage['logits'][step] = logits
                storage['values'][step] = value

                actions_np = actions.cpu().numpy()
                obs_array, rewards, terminated, truncated, infos = self.vector_env.step(actions_np)
                dones = terminated | truncated

                storage['rewards'][step] = torch.tensor(rewards, dtype=torch.float32, device=device)
                storage['dones'][step] = torch.tensor(dones, dtype=torch.float32, device=device)

                if self.agent.use_auxiliary:
                    energies = [info.get('energy', 0.0) for info in infos]
                    storage['energies'][step] = torch.tensor(energies, dtype=torch.float32, device=device)

                obs = torch.tensor(obs_array, dtype=torch.long, device=device).unsqueeze(1)

                if dones.all():
                    for k in storage:
                        storage[k] = storage[k][:step+1]
                    break

        for k in storage:
            storage[k] = torch.stack(storage[k], dim=1)

        storage['values'] = storage['values'].squeeze(-1)   # [B, T]

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

        return experiences

    def _train_step(self, experiences: dict) -> dict:
        """Perform multiple PPO epochs over the rollout buffer using mini-batches."""
        B, T = experiences['observations'].shape[:2]
        total_envs = B
        indices = torch.randperm(total_envs, device=self.device)

        # Detach tensors that should not have gradients
        for key in ['old_logits', 'old_values', 'advantages', 'returns', 'mask']:
            experiences[key] = experiences[key].detach()

        metrics_sum = {}
        network = self.agent.network
        optimizer = self.optimizer
        scaler = self.scaler
        use_amp = self.use_amp

        # Ensure network is in training mode for backward pass
        network.train()

        for _ in range(self.ppo_epochs):
            for start in range(0, total_envs, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_envs)
                env_idx = indices[start:end]
                batch = {k: v[env_idx] for k, v in experiences.items()}

                self.agent.reset()   # resets LSTM/transformer memory to zeros

                # Forward pass – need gradients
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = network(batch['observations'], return_auxiliary=False, return_value=True)
                        logits = outputs[0]
                        values = outputs[-1].squeeze(-1)
                        loss, metrics = self.ppo_loss_fn(
                            logits, batch['old_logits'], batch['actions'],
                            batch['advantages'], batch['returns'], values, batch['mask']
                        )
                else:
                    outputs = network(batch['observations'], return_auxiliary=False, return_value=True)
                    logits = outputs[0]
                    values = outputs[-1].squeeze(-1)
                    loss, metrics = self.ppo_loss_fn(
                        logits, batch['old_logits'], batch['actions'],
                        batch['advantages'], batch['returns'], values, batch['mask']
                    )

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
                    metrics_sum[k] = metrics_sum.get(k, 0) + v / self.ppo_epochs / (total_envs / self.mini_batch_size)

        # Optionally set back to eval mode after training (but not necessary)
        # network.eval()

        return metrics_sum

    def train(self):
        """Main training loop."""
        training_cfg = self.config['training']
        epochs = training_cfg['epochs']
        test_interval = training_cfg['test_interval']
        save_interval = training_cfg['save_interval']
        start_epoch = len(self.metrics['train_rewards'])

        if not self.metrics['test_epochs']:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(0)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

        pbar = tqdm(range(start_epoch, epochs), desc="PPO Epochs")
        start_time = time.time()

        for epoch in pbar:
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

            self.lr_scheduler.step()
            self._post_epoch_hook(epoch)

            if epoch % test_interval == 0 and epoch > 0:
                test_metrics = self._test_valid(epochs=4)
                self.metrics['test_epochs'].append(epoch)
                self.metrics['test_rewards'].append(test_metrics['reward'])
                if test_metrics['reward'] > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_metrics['reward']
                    self._save_model('best')

            if epoch % save_interval == 0 and epoch > 0:
                self._save_model(f'epoch_{epoch:06d}')

            pbar.set_postfix({
                'reward': f"{avg_reward:.2f}",
                'loss': f"{train_metrics.get('loss', 0):.4f}",
                'best': f"{self.metrics['best_reward']:.2f}"
            })

        self._save_model('final')
        self._save_metrics()
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