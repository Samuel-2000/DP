"""
PPO trainer with rollout buffer, GAE, and clipped surrogate objective.
"""

import torch
import numpy as np
from tqdm import tqdm
import time
from .parallel_trainer_base import ParallelTrainerBase
from src.training.losses import PPOLoss


class PPOTrainer(ParallelTrainerBase):
    """
    Proximal Policy Optimization with:
      - Value head for state value estimation
      - GAE (Generalized Advantage Estimation)
      - Multiple PPO epochs over a fixed-size rollout buffer
      - Mini-batch updates
    """

    def __init__(self, config):
        # Force value head to be used (required for PPO)
        config['model']['use_value_head'] = True
        super().__init__(config)

        train_cfg = self.config['training']
        self.rollout_steps = train_cfg['rollout_steps']
        self.ppo_epochs = train_cfg['ppo_epochs']
        self.mini_batch_size = train_cfg['mini_batch_size']
        self.clip_epsilon = train_cfg['clip_epsilon']
        self.value_coef = train_cfg['value_coef']
        self.entropy_coef = train_cfg['entropy_coef']
        self.gamma = train_cfg['gamma']
        self.gae_lambda = train_cfg['gae_lambda']

        self.ppo_loss_fn = PPOLoss(
            clip_epsilon=self.clip_epsilon,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

    def _collect_rollout(self) -> dict:
        """
        Collect exactly `rollout_steps` steps from the parallel environments.
        Returns a dictionary with tensors of shape [total_steps, ...].
        """
        num_envs = self.batch_size
        steps_needed = self.rollout_steps
        total_steps = 0

        # Reset all environments and agent memory
        obs_array, _ = self.vector_env.reset()
        self.agent.reset()
        obs = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)

        storage = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'logits': [],
        }
        if self.agent.use_auxiliary:
            storage['energies'] = []

        while total_steps < steps_needed:
            with torch.no_grad():
                outputs = self.agent.network(obs, return_auxiliary=self.agent.use_auxiliary, return_value=True)

                logits = outputs[0]
                value = outputs[-1]
            
            logits = logits.squeeze(1)   # [B, A]
            value = value.squeeze(1)     # [B]

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()

            storage['obs'].append(obs.squeeze(1))
            storage['actions'].append(actions)
            storage['logits'].append(logits)
            storage['values'].append(value)

            actions_np = actions.cpu().numpy()
            obs_array, rewards, terminated, truncated, infos = self.vector_env.step(actions_np)
            dones = terminated | truncated
            storage['rewards'].append(torch.tensor(rewards, dtype=torch.float32, device=self.device))
            storage['dones'].append(torch.tensor(dones, dtype=torch.float32, device=self.device))

            if self.agent.use_auxiliary:
                energies = [info.get('energy', 0.0) for info in infos]
                storage['energies'].append(torch.tensor(energies, dtype=torch.float32, device=self.device))

            obs = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)
            total_steps += num_envs

        # Concatenate all stored tensors
        for k in storage:
            storage[k] = torch.cat(storage[k], dim=0)

        # Compute advantages and returns using GAE
        advantages, returns = self.ppo_loss_fn.compute_gae(
            storage['rewards'].unsqueeze(0),
            storage['values'].unsqueeze(0),
            storage['dones'].unsqueeze(0),
            mask=torch.ones_like(storage['rewards']).unsqueeze(0)
        )
        advantages = advantages.squeeze(0)
        returns = returns.squeeze(0)

        experiences = {
            'observations': storage['obs'],
            'actions': storage['actions'],
            'advantages': advantages,
            'returns': returns,
            'old_logits': storage['logits'],
            'old_values': storage['values'],
            'mask': torch.ones_like(storage['rewards']),
        }
        if self.agent.use_auxiliary:
            experiences['energy_targets'] = storage['energies']
            # Next observation targets are optional and omitted for brevity
        return experiences

    def _train_step(self, experiences: dict) -> dict:
        """
        Perform multiple PPO epochs over the collected rollout buffer,
        using mini-batch gradient descent.
        """
        total_steps = experiences['observations'].shape[0]
        indices = torch.randperm(total_steps)
        metrics_sum = {}

        for epoch in range(self.ppo_epochs):
            for start in range(0, total_steps, self.mini_batch_size):
                end = min(start + self.mini_batch_size, total_steps)
                idx = indices[start:end]
                batch = {k: v[idx] for k, v in experiences.items()}

                obs = batch['observations'].unsqueeze(1)  # add seq dim

                outputs = self.agent.network(obs, return_auxiliary=False, return_value=True)
                logits = outputs[0]
                values = outputs[-1]

                logits = logits.squeeze(1)
                values = values.squeeze(1)

                loss, metrics = self.ppo_loss_fn(
                    logits, batch['old_logits'], batch['actions'],
                    batch['advantages'], batch['returns'], values, batch['mask']
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.gradient_clipper.clip(self.agent.network.parameters())
                self.optimizer.step()

                for k, v in metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v / self.ppo_epochs / (total_steps / self.mini_batch_size)

        return metrics_sum

    def train(self):
        """
        Main training loop for PPO.
        Each iteration:
          - Collects a rollout of `rollout_steps`
          - Updates policy and value network using several PPO epochs
          - Tests and saves periodically
        """
        training_cfg = self.config['training']
        epochs = training_cfg['epochs']
        test_interval = training_cfg['test_interval']
        save_interval = training_cfg['save_interval']

        start_epoch = len(self.metrics['train_rewards'])

        # Initial test
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
            # Collect a fixed-size rollout
            experiences = self._collect_rollout()
            # Perform multiple PPO updates on this rollout
            train_metrics = self._train_step(experiences)

            avg_reward = experiences['rewards'].mean().item()
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
        if self.dynamic:
            print(f"Final stage: {self.complexity_manager.get_current_task_class()}")
            print(f"Final complexity: {self.complexity_manager.get_current_complexity():.2f}")
            print(f"Total adjustments: {self.complexity_manager.adjustments_made}")
        print(f"Model saved in: {self.experiment_dir}")
        print(f"{'='*80}")