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
        max_steps = self.vector_env.envs[0].max_steps
        self.agent.reset()

        if full_reset:
            obs_array, _ = self.vector_env.reset()
        else:
            obs_array, _ = self.vector_env.soft_reset_all()

        observations = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)
        all_obs = []
        all_actions = []
        all_rewards = []
        all_energies = []
        all_next_obs = []
        current_energies = [env.energy for env in self.vector_env.envs]

        for step in range(max_steps):
            all_obs.append(observations.clone())
            all_energies.append(torch.tensor(current_energies, dtype=torch.float32, device=self.device))

            with torch.no_grad():
                logits = self.agent.network(observations).squeeze(1)
                probs = torch.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, 1).squeeze(-1)

            actions_np = actions.cpu().numpy()
            obs_array, rewards, terminated, truncated, infos = self.vector_env.step(actions_np)
            next_obs_tensor = torch.tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)

            all_actions.append(actions)
            all_rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self.device))
            all_next_obs.append(next_obs_tensor.clone())
            current_energies = [info.get('energy', 0.0) for info in infos]
            observations = next_obs_tensor

            if (terminated | truncated).all():
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
            policy_loss, entropy = self.policy_loss_fn(logits, actions, rewards, mask)
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
            policy_loss, entropy = self.policy_loss_fn(logits, actions, rewards, mask)
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
        training_cfg = self.config['training']
        epochs = training_cfg['epochs']
        test_interval = training_cfg['test_interval']
        save_interval = training_cfg['save_interval']

        start_epoch = len(self.metrics['train_rewards'])
        episode_counter = len(self.metrics['train_rewards'])

        # Initial test
        if not self.metrics['test_epochs']:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(0)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

        # Visualisation control window
        cv2.namedWindow('Training Controls', cv2.WINDOW_NORMAL)
        dummy = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(dummy, "Press 'v' to visualise", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dummy, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Training Controls', dummy)
        cv2.waitKey(1)

        pbar = tqdm(range(start_epoch, epochs), desc="Epochs")
        start_time = time.time()

        try:
            for epoch in pbar:
                # Generate first grid for this epoch
                current_config = self._generate_grid_config()
                self._grid_pool = [current_config]
                self._apply_grid_config(current_config, reset_hidden=True)

                batched_experiences = None
                epoch_rewards = []

                # Collect episodes (consecutive episodes on same or different grids)
                for ep_idx in range(self.consecutive_episodes):
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
                        episode_counter += 1
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
                    episode_counter += 1
                    self.lr_scheduler.step()

                avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                self.metrics.setdefault('epoch_rewards', []).append(avg_epoch_reward)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('v'):
                    self._visualize_current_environments(epoch)
                    cv2.imshow('Training Controls', dummy)
                elif key == ord('q'):
                    print("\nEarly stop requested.")
                    self._save_model('interrupted')
                    cv2.destroyAllWindows()
                    break

                # Dynamic complexity hook
                self._post_epoch_hook(epoch)

                # Periodic test and save
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
                    'reward': f"{avg_epoch_reward:.2f}",
                    'best': f"{self.metrics['best_reward']:.2f}",
                    'pool': len(self._grid_pool)
                })

        except StopIteration:
            print("\nStopped by post_epoch_hook (stagnation).")
        finally:
            cv2.destroyAllWindows()

        # Final test and save
        if epochs > 0 and epochs - 1 not in self.metrics['test_epochs']:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(epochs - 1)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

        self._save_model('final')
        self._save_metrics()
        self._print_training_summary(start_time)

    def _visualize_current_environments(self, epoch):
        """Display a montage of the current training environments (first few envs)."""
        print(f"\nVisualizing at epoch {epoch}")
        num_to_show = min(4, len(self.vector_env.envs))
        cell_size = 256
        padding = 10
        cols = 2
        rows = (num_to_show + cols - 1) // cols
        total_w = cols * (cell_size + padding) + padding
        total_h = rows * (cell_size + padding) + padding
        combined = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        for i in range(num_to_show):
            env = self.vector_env.envs[i]
            original = env.render_size
            env.render_size = cell_size
            if hasattr(env, '_render_buffer'):
                env._render_buffer = None
            frame = env.render()
            env.render_size = original
            if frame is None:
                frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
            if frame.shape[:2] != (cell_size, cell_size):
                frame = cv2.resize(frame, (cell_size, cell_size))
            row, col = i // cols, i % cols
            y = padding + row * (cell_size + padding)
            x = padding + col * (cell_size + padding)
            combined[y:y+cell_size, x:x+cell_size] = frame
        info = f"Stage: {self.get_environment_config()['task_class']}, Complexity: {self.get_environment_config()['complexity_level']:.2f}"
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('Training Visualization', combined)
        cv2.waitKey(0)

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