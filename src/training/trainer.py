# src/training/trainer.py - OPTIMIZED PARALLEL VERSION
"""
Training module with optimizations and parallel execution
"""

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
from datetime import datetime

from src.core.environment import GridMazeWorld
from src.core.vector_env import VectorizedMazeEnv  # New import
from src.core.agent import Agent
from src.core.utils import setup_logging, seed_everything
from .losses import PolicyLoss, AuxiliaryLoss
from .optimizers import GradientClipper, LearningRateScheduler


class ParallelTrainer:
    """Main trainer class with parallel execution"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 use_wandb: bool = False):
        
        self.config = config
        self.experiment_name = f"{config["model"]["type"]}_" \
                                f"{config["training"]["batch_size"]}b_" \
                                f"{config["training"]["learning_rate"]}lr_" \
                                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
        if config["training"].get("auxiliary_tasks", None):
            self.experiment_name += "_aux"
        
        
        self.use_wandb = use_wandb
        
        # Setup
        self.logger = setup_logging(self.experiment_name)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Set seed
        seed_everything(config.get('seed', 42))
        
        # Get batch size for parallel environments
        training_config = self.config['training']
        self.batch_size = training_config['batch_size']
        
        # Create vectorized environment
        self.vector_env = self._create_vectorized_env()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        
        # Create loss functions
        self.policy_loss_fn = PolicyLoss(
            gamma=training_config.get('gamma', 0.97),
            entropy_coef=training_config.get('entropy_coef', 0.01),
            normalize_advantages=True
        )
        
        # Only create auxiliary loss if configured
        if self.agent.use_auxiliary:
            self.aux_loss_fn = AuxiliaryLoss(
                energy_coef=0.1,
                obs_coef=0.05,
                obs_prediction_type='classification'
            )
        else:
            self.aux_loss_fn = None
            
        self.gradient_clipper = GradientClipper(
            max_norm=training_config.get('max_grad_norm', 1.0)
        )
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            mode='cosine',
            lr_start=training_config.get('learning_rate', 0.0005),
            lr_min=1e-6
        )
        
        # Metrics
        self.metrics = {
            'train_rewards': [],
            'train_losses': [],
            'test_rewards': [],
            'best_reward': -np.inf,
            'timing': {
                'collection': [],
                'training': [],
                'total': []
            }
        }
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="maze-rl",
                name=self.experiment_name,
                config=config
            )
    
    def _create_vectorized_env(self) -> VectorizedMazeEnv:
        """Create vectorized training environment"""
        env_config = self.config['environment']
        return VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config
        )
    
    def _create_agent(self) -> Agent:
        """Create agent with specified network"""
        model_config = self.config['model']
        
        agent = Agent(
            network_type=model_config['type'],
            observation_size=10,  # Fixed observation size
            action_size=self.vector_env.action_space.n,
            hidden_size=model_config.get('hidden_size', 512),
            use_auxiliary=model_config.get('use_auxiliary', False),
            device=self.device
        )
        
        # Load pretrained if specified
        if 'pretrained_path' in model_config:
            agent.load(model_config['pretrained_path'])
        
        return agent
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        training_config = self.config['training']
        
        optimizer_type = training_config.get('optimizer', 'adam')
        lr = training_config.get('learning_rate', 0.0005)
        weight_decay = training_config.get('weight_decay', 0.0)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.agent.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.agent.network.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train(self):
        """Main training loop with parallel execution"""
        training_config = self.config['training']
        epochs = training_config.get('epochs', 10000)
        save_interval = training_config.get('save_interval', 1000)
        test_interval = training_config.get('test_interval', 500)
        
        # Create progress bar
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        
        start_time = time.time()
        
        for epoch in pbar:
            epoch_start = time.time()
            
            # Training phase with timing
            coll_start = time.time()
            experiences = self._collect_experiences_parallel()
            coll_time = time.time() - coll_start
            
            train_start = time.time()
            train_metrics = self._train_step(experiences)
            train_time = time.time() - train_start
            
            # Update metrics
            self.metrics['train_rewards'].append(train_metrics['reward'])
            self.metrics['train_losses'].append(train_metrics['loss'])
            self.metrics['timing']['collection'].append(coll_time)
            self.metrics['timing']['training'].append(train_time)
            self.metrics['timing']['total'].append(time.time() - epoch_start)
            
            # Test phase
            if epoch % test_interval == 0 and epoch != 0:
                test_metrics = self._test_epoch(episodes=10)
                test_reward = test_metrics['reward']
                self.metrics['test_rewards'].append(test_reward)
                
                # Update best model
                if test_reward > self.metrics['best_reward']:
                    self.metrics['best_reward'] = test_reward
                    self._save_model('best')
                    self.logger.info(f"New best model with reward: {test_reward:.2f}")
            
            # Save checkpoint
            if epoch % save_interval == 0 and epoch != 0:
                self._save_model(f'epoch_{epoch:06d}')
            
            # Update progress bar
            avg_coll_time = np.mean(self.metrics['timing']['collection'][-10:]) if len(self.metrics['timing']['collection']) > 10 else coll_time
            avg_train_time = np.mean(self.metrics['timing']['training'][-10:]) if len(self.metrics['timing']['training']) > 10 else train_time
            
            pbar.set_postfix({
                'reward': f"{train_metrics['reward']:.2f}",
                'loss': f"{train_metrics['loss']:.4f}",
                'best': f"{self.metrics['best_reward']:.2f}",
                'coll': f"{avg_coll_time:.2f}s",
                'train': f"{avg_train_time:.2f}s",
                'eps/s': f"{self.batch_size/(avg_coll_time+avg_train_time):.1f}",
            })
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'train/reward': train_metrics['reward'],
                    'train/loss': train_metrics['loss'],
                    'train/entropy': train_metrics.get('entropy', 0.0),
                    'lr': self.lr_scheduler.get_lr(),
                    'timing/collection': coll_time,
                    'timing/training': train_time,
                    'timing/env_steps_per_sec': self.batch_size / coll_time,
                })
                
                if epoch % test_interval == 0 and epoch != 0:
                    wandb.log({
                        'test/reward': test_metrics['reward'],
                        'test/success_rate': test_metrics['success_rate'],
                    })
            
            # Update learning rate
            self.lr_scheduler.step()
        
        # Save final model
        self._save_model('final')
        
        # Save training metrics
        self._save_metrics()
        
        # Print timing summary
        total_time = time.time() - start_time
        avg_collection = np.mean(self.metrics['timing']['collection'])
        avg_training = np.mean(self.metrics['timing']['training'])
        avg_total = np.mean(self.metrics['timing']['total'])
        
        self.logger.info(f"Training completed in {total_time:.1f}s")
        self.logger.info(f"Average timing: Collection={avg_collection:.3f}s, "
                        f"Training={avg_training:.3f}s, Total={avg_total:.3f}s")
        self.logger.info(f"Average environment steps per second: {self.batch_size/avg_collection:.1f}")
        
        # Close wandb
        if self.use_wandb:
            wandb.finish()
    
    def _collect_experiences_parallel(self) -> Dict[str, torch.Tensor]:
        """
        Collect experiences in parallel across all environments
        """
        max_steps = self.vector_env.envs[0].max_steps
        self.agent.reset()  # Reset network state once
        
        # Reset all environments
        obs_array, _ = self.vector_env.reset()
        
        # Convert to tensor
        observations = torch.tensor(obs_array, dtype=torch.long).to(self.device)
        observations = observations.unsqueeze(1)  # [B, 1, K]
        
        # Storage
        all_observations = []
        all_actions = []
        all_rewards = []
        
        # Run for max_steps or until all environments are done
        active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        
        for step in range(max_steps):
            # Store current observations
            all_observations.append(observations.clone())
            
            # Get actions from network
            with torch.no_grad():
                # Ensure network is in eval mode for inference
                was_training = self.agent.network.training
                self.agent.network.eval()
                
                logits = self.agent.network(observations)  # [B, 1, A]
                logits = logits.squeeze(1)  # [B, A]
                
                if was_training:
                    self.agent.network.train()
                
                # Sample actions during training
                if self.agent.network.training:
                    probs = torch.softmax(logits, dim=-1)
                    actions = torch.multinomial(probs, 1).squeeze(-1)  # [B]
                else:
                    actions = logits.argmax(dim=-1)  # [B]
            
            # Convert to numpy for environment step
            actions_np = actions.cpu().numpy()
            
            # Step all environments in parallel
            obs_array, rewards, terminated, truncated, _ = self.vector_env.step(actions_np)
            
            # Convert to tensors
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            terminated_tensor = torch.tensor(terminated, dtype=torch.bool, device=self.device)
            truncated_tensor = torch.tensor(truncated, dtype=torch.bool, device=self.device)
            
            # Store actions and rewards
            all_actions.append(actions)
            all_rewards.append(rewards_tensor)
            
            # Update active mask
            done_mask = terminated_tensor | truncated_tensor
            active_mask = active_mask & ~done_mask
            
            # Break if all environments are done
            if not active_mask.any():
                break
            
            # Prepare next observations
            observations = torch.tensor(obs_array, dtype=torch.long, device=self.device)
            observations = observations.unsqueeze(1)  # [B, 1, K]
        
        # Stack all collected data
        T = len(all_observations)
        
        observations_tensor = torch.cat(all_observations, dim=1)  # [B, T, K]
        actions_tensor = torch.stack(all_actions, dim=1)  # [B, T]
        rewards_tensor = torch.stack(all_rewards, dim=1)  # [B, T]
        
        # Create mask for valid steps
        mask = torch.ones_like(rewards_tensor, dtype=torch.float32)
        
        return {
            'observations': observations_tensor,
            'actions': actions_tensor,
            'rewards': rewards_tensor,
            'mask': mask
        }
    
    def _train_step(self, 
                   experiences: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step"""
        self.agent.network.train()
        
        # Compute loss
        loss, metrics = self._compute_loss(experiences)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.gradient_clipper.clip(self.agent.network.parameters())
        self.optimizer.step()
        
        # Flush cache if using multi-memory
        if hasattr(self.agent.network, 'flush_cache_buffer'):
            self.agent.network.flush_cache_buffer()
        
        return metrics
    
    def _compute_loss(self, 
                     experiences: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Compute policy loss with auxiliary losses"""
        obs = experiences['observations']
        actions = experiences['actions']
        rewards = experiences['rewards']
        mask = experiences.get('mask', None)
        
        # Reset network for batch processing
        self.agent.reset()
        
        # Forward pass
        if self.aux_loss_fn and self.agent.use_auxiliary:
            # Get outputs from network with auxiliary predictions
            outputs = self.agent.network(obs, return_auxiliary=True)
            
            if isinstance(outputs, tuple):
                if len(outputs) == 4:
                    logits, energy_pred, obs_pred, _ = outputs
                else:
                    logits, energy_pred, obs_pred = outputs
            else:
                # Network doesn't return auxiliary outputs
                logits = outputs
                energy_pred = None
                obs_pred = None
            
            # Policy loss
            policy_loss, entropy = self.policy_loss_fn(
                logits, actions, rewards, mask
            )
            
            total_loss = policy_loss
            
            metrics = {
                'loss': policy_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item()
            }
            
            # Add auxiliary loss if available
            if energy_pred is not None and obs_pred is not None:
                # Note: We don't have energy targets in basic implementation
                # You'd need to collect these during experience collection
                pass
            
        else:
            # Standard forward pass without auxiliary tasks
            logits = self.agent.network(obs)
            policy_loss, entropy = self.policy_loss_fn(
                logits, actions, rewards, mask
            )
            
            total_loss = policy_loss
            
            metrics = {
                'loss': total_loss.item(),
                'policy_loss': policy_loss.item(),
                'entropy': entropy.item(),
                'reward': rewards.sum(dim=1).mean().item()
            }
        
        return total_loss, metrics
    
    def _test_epoch(self, episodes: int = 10) -> Dict[str, float]:
        """Test agent performance"""
        self.agent.network.eval()
        
        total_reward = 0.0
        success_count = 0
        episode_lengths = []
        
        with torch.no_grad():
            for _ in range(episodes):
                # Use single environment for testing
                test_env = GridMazeWorld(**self.config['environment'])
                obs, info = test_env.reset()
                self.agent.reset()
                
                episode_reward = 0.0
                steps = 0
                terminated = truncated = False
                
                while not (terminated or truncated) and steps < test_env.max_steps:
                    action = self.agent.act(obs, training=False)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                
                total_reward += episode_reward
                episode_lengths.append(steps)
                
                # Consider episode successful if agent survives to end
                if steps == test_env.max_steps:
                    success_count += 1
        
        avg_reward = total_reward / episodes
        success_rate = success_count / episodes * 100
        avg_length = np.mean(episode_lengths)
        
        return {
            'reward': avg_reward,
            'success_rate': success_rate,
            'avg_length': avg_length
        }
    
    def _save_model(self, name: str):
        """Save model checkpoint without duplication"""
        save_dir = Path(self.config.get('save_dir', 'models'))
        save_dir.mkdir(exist_ok=True)
        
        # For 'best' and 'final', save the agent file (lightweight, for deployment)
        if name in ['best', 'final']:
            agent_path = save_dir / f"{self.experiment_name}_{name}.pt"
            self.agent.save(str(agent_path))
            self.logger.info(f"Saved agent to {agent_path}")
        
        # Save checkpoint (for resuming training)
        checkpoint_path = save_dir / f"{self.experiment_name}_{name}_checkpoint.pt"
        
        # Save checkpoint WITHOUT agent_state (already saved separately)
        torch.save({
            'epoch': len(self.metrics['train_rewards']),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }, str(checkpoint_path))
        
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _save_metrics(self):
        """Save training metrics"""
        metrics_dir = Path('logs/metrics')
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_path = metrics_dir / f"{self.experiment_name}_metrics.npz"
        np.savez(str(metrics_path), **self.metrics)
        
        # Plot metrics
        self._plot_metrics()
    
    def _plot_metrics(self):
        """Plot training metrics"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Raw and smoothed rewards
        ax = axes[0, 0]
        rewards = self.metrics['train_rewards']
        ax.plot(rewards, alpha=0.3, color='gray', linewidth=0.5, label='Raw')
        
        # Add smoothed line
        if len(rewards) >= 100:
            window = 100
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), smoothed, 
                   'r-', linewidth=2, label=f'Smoothed (window={window})')
        
        ax.set_title('Training Rewards')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Training losses
        ax = axes[0, 1]
        ax.plot(self.metrics['train_losses'], alpha=0.7, linewidth=1)
        ax.set_title('Training Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # 3. Test rewards
        if self.metrics['test_rewards']:
            ax = axes[1, 0]
            ax.plot(self.metrics['test_rewards'], 'o-', linewidth=2, markersize=6)
            ax.set_title('Test Rewards')
            ax.set_xlabel('Test Interval')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
        
        # 4. Progress comparison: Early vs Late training
        ax = axes[1, 1]
        
        if len(rewards) >= 200:
            # Compare first and last 100 epochs (or halves)
            split_point = len(rewards) // 2
            
            early_rewards = rewards[:split_point]
            late_rewards = rewards[split_point:]
            
            # Smooth both halves
            if len(early_rewards) >= 50 and len(late_rewards) >= 50:
                window = min(50, len(early_rewards)//2, len(late_rewards)//2)
                
                early_smoothed = np.convolve(early_rewards, np.ones(window)/window, mode='valid')
                late_smoothed = np.convolve(late_rewards, np.ones(window)/window, mode='valid')
                
                ax.plot(range(window-1, len(early_rewards)), early_smoothed, 
                       'b-', linewidth=2, label=f'First {split_point} epochs')
                ax.plot(range(split_point + window-1, len(rewards)), late_smoothed, 
                       'r-', linewidth=2, label=f'Last {len(rewards)-split_point} epochs')
                
                ax.set_title('Training Progress: Early vs Late')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Smoothed Reward (window=50)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Not enough data for comparison', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Training Progress')
        else:
            ax.text(0.5, 0.5, 'Need at least 200 epochs for comparison', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Progress')
        
        plt.tight_layout()
        plot_path = Path('results/plots') / f"{self.experiment_name}_metrics.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also create a dedicated smoothed rewards plot
        self._create_smoothed_rewards_plot()
        
        self.logger.info(f"Saved metrics plot to {plot_path}")
    
    def _create_smoothed_rewards_plot(self):
        """Create a detailed smoothed rewards plot"""
        import matplotlib.pyplot as plt
        
        rewards = self.metrics['train_rewards']
        
        if len(rewards) < 20:
            return  # Not enough data
        
        fig = plt.figure(figsize=(12, 8))
        
        # Create 3 subplots
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, (3, 4))
        
        # 1. Raw rewards with smoothing overlay
        ax1.plot(rewards, alpha=0.2, color='gray', linewidth=0.5, label='Raw')
        
        window_sizes = [10, 50, 100]
        colors = ['red', 'green', 'blue']
        
        for i, window in enumerate(window_sizes):
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), smoothed, 
                        color=colors[i], linewidth=1.5, 
                        label=f'Window={window}')
        
        ax1.set_title('Rewards with Different Smoothing')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # 2. Rolling statistics
        if len(rewards) >= 100:
            window = 100
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            rolling_std = np.array([
                np.std(rewards[max(0, i-window//2):min(len(rewards), i+window//2)])
                for i in range(window-1, len(rewards))
            ])
            
            ax2.plot(range(window-1, len(rewards)), rolling_mean, 
                    'b-', linewidth=2, label='Rolling mean')
            ax2.fill_between(range(window-1, len(rewards)),
                            rolling_mean - rolling_std,
                            rolling_mean + rolling_std,
                            alpha=0.2, color='blue', label='±1 std')
            
            ax2.set_title(f'Rolling Statistics (window={window})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Reward')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. Cumulative rewards and learning curve
        cumulative = np.cumsum(rewards)
        ax3.plot(cumulative, 'k-', linewidth=1.5, label='Cumulative reward')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Cumulative Reward', color='k')
        ax3.tick_params(axis='y', labelcolor='k')
        ax3.grid(True, alpha=0.3)
        
        # Add smoothed rewards on secondary y-axis
        if len(rewards) >= 50:
            ax3b = ax3.twinx()
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            ax3b.plot(range(49, len(rewards)), smoothed, 
                     'r-', linewidth=1.5, label='Smoothed reward (window=50)')
            ax3b.set_ylabel('Smoothed Reward', color='r')
            ax3b.tick_params(axis='y', labelcolor='r')
            
            # Combine legends
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax3.set_title('Cumulative and Smoothed Rewards')
        
        plt.suptitle(f'Training Progress: {self.experiment_name}', fontsize=14)
        plt.tight_layout()
        
        smooth_path = Path('results/plots') / f"{self.experiment_name}_smoothed_rewards.png"
        plt.savefig(str(smooth_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved smoothed rewards plot to {smooth_path}")


# Backward compatibility - keep original Trainer for now
Trainer = ParallelTrainer