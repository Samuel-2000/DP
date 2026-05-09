"""
Base parallel trainer with vectorized environments, metrics, saving, dynamic complexity,
and training plots.
"""

import time
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from collections import deque
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any

from core.obsolete.env_factory_vector import VectorizedMazeEnv
from src.core.agent import Agent
from src.core.utils import setup_logging, seed_everything
from src.training.optimizers import GradientClipper, LearningRateScheduler, OptimizerFactory
from src.core.constants import OBSERVATION_SIZE, ACTION_SIZE


def generate_plots_from_metrics(metrics: Dict[str, Any], plots_dir: Path, increase_threshold: float, decrease_threshold: float):
    """Generate all training plots from a metrics dictionary."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(fig, name):
        path = plots_dir / f"{name}.png"
        fig.savefig(str(path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ---- 1. Training Rewards (raw) + Test Rewards (exact epochs) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    rewards = np.array(metrics['train_rewards'])
    epochs = np.arange(len(rewards))
    ax.plot(epochs, rewards, 'b-', alpha=0.7, linewidth=1, label='Train Reward (raw)')
    if 'test_rewards' in metrics and len(metrics['test_rewards']) > 0:
        test_rewards = metrics['test_rewards']
        if 'test_epochs' in metrics and len(metrics['test_epochs']) == len(test_rewards):
            test_epochs = metrics['test_epochs']
        else:
            test_interval = max(1, len(rewards) // len(test_rewards))
            test_epochs = np.arange(0, len(test_rewards) * test_interval, test_interval)
        ax.plot(test_epochs, test_rewards, 'g-o', linewidth=1.5, markersize=6, label='Test Reward')
    ax.set_title('Training & Test Rewards (raw)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add total training time annotation
    total_seconds = metrics.get('total_training_time', 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    time_str = f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}"
    ax.text(0.98, 0.98, time_str, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    save_plot(fig, 'rewards')

    # ---- 2. Training Losses (total + policy) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics['train_losses'], 'r-', alpha=0.7, label='Total Loss')
    if 'policy_losses' in metrics and len(metrics['policy_losses']) > 0:
        ax.plot(metrics['policy_losses'], 'r--', alpha=0.5, label='Policy Loss')
    ax.set_title('Training Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add total training time annotation
    total_seconds = metrics.get('total_training_time', 0.0)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    time_str = f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}"
    ax.text(0.98, 0.98, time_str, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    save_plot(fig, 'losses')

    # ---- 3. Auxiliary Losses ----
    if 'aux_losses' in metrics and len(metrics['aux_losses']) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metrics['aux_losses'], label='Total Aux Loss', color='purple')
        if 'energy_losses' in metrics and len(metrics['energy_losses']) > 0:
            ax.plot(metrics['energy_losses'], label='Energy MSE', color='orange')
        if 'obs_losses' in metrics and len(metrics['obs_losses']) > 0:
            ax.plot(metrics['obs_losses'], label='Obs MSE', color='green')
        ax.set_title('Auxiliary Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_plot(fig, 'aux_losses')

    # ---- 4. Complexity & Task Class Progression ----
    if 'complexity_history' in metrics and len(metrics['complexity_history']) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metrics['complexity_history'], 'b-', linewidth=1, label='Complexity')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Complexity Level', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        if 'task_class_history' in metrics and len(metrics['task_class_history']) > 0:
            ax2 = ax.twinx()
            stage_map = {'basic': 0.0, 'doors': 0.33, 'buttons': 0.66, 'complex': 1.0}
            task_numeric = []
            for t in metrics['task_class_history']:
                if isinstance(t, str):
                    task_numeric.append(stage_map.get(t, 0.0))
                else:
                    task_numeric.append(float(t))
            ax2.plot(task_numeric, 'g--', linewidth=1.5, label='Task Class', alpha=0.9)
            ax2.set_ylabel('Task Class', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.set_yticks([0.0, 0.33, 0.66, 1.0])
            ax2.set_yticklabels(['Basic', 'Doors', 'Buttons', 'Complex'])
            ax2.set_ylim(-0.1, 1.1)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.legend()
        ax.set_title('Complexity Progression (raw)')
        save_plot(fig, 'complexity')

    # ---- 5. Performance Scores (colored by config change) ----
    if 'performance_scores' in metrics and len(metrics['performance_scores']) > 0 and 'complexity_history' in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        scores = np.array(metrics['performance_scores'])
        window = max(1, len(metrics['train_rewards']) // len(scores))
        perf_epochs = np.arange(len(scores)) * window
        complexities = np.array(metrics['complexity_history'])
        valid_mask = perf_epochs < len(complexities)
        perf_epochs = perf_epochs[valid_mask]
        scores = scores[:len(perf_epochs)]
        if len(scores) > 0:
            tasks = metrics.get('task_class_history', ['basic'] * len(complexities))
            if isinstance(tasks, list):
                tasks = np.array(tasks)
            complexities_at_perf = complexities[perf_epochs]
            tasks_at_perf = tasks[perf_epochs]
            change_indices = []
            for i in range(1, len(complexities_at_perf)):
                if abs(complexities_at_perf[i] - complexities_at_perf[i-1]) > 1e-6 or tasks_at_perf[i] != tasks_at_perf[i-1]:
                    change_indices.append(i)
            colors = ['orange', 'blue']
            start = 0
            for idx, split in enumerate(change_indices):
                seg_x = perf_epochs[start:split]
                seg_y = scores[start:split]
                if len(seg_x) > 0:
                    ax.plot(seg_x, seg_y, color=colors[idx % 2], linewidth=1.5)
                    ax.axvline(x=perf_epochs[split], color='gray', linestyle=':', alpha=0.5)
                start = split
            if start < len(perf_epochs):
                ax.plot(perf_epochs[start:], scores[start:], color=colors[len(change_indices) % 2], linewidth=1.5)
            ax.axhline(increase_threshold, color='green', linestyle='--', alpha=0.7, label=f'Increase ({increase_threshold})')
            ax.axhline(decrease_threshold, color='red', linestyle='--', alpha=0.7, label=f'Decrease ({decrease_threshold})')
            ax.legend()
            ax.set_title('Performance Scores (colored by config change)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            save_plot(fig, 'performance_scores')

    # ---- 6. Complexity vs Reward (raw) ----
    if 'complexity_history' in metrics and len(metrics['train_rewards']) > 10:
        fig, ax = plt.subplots(figsize=(8, 5))
        complexities = np.array(metrics['complexity_history'])
        rewards_raw = np.array(metrics['train_rewards'])
        min_len = min(len(complexities), len(rewards_raw))
        complexities = complexities[:min_len]
        rewards_raw = rewards_raw[:min_len]
        sc = ax.scatter(complexities, rewards_raw, c=range(min_len), cmap='viridis', alpha=0.7, s=10)
        plt.colorbar(sc, ax=ax, label='Epoch')
        if min_len > 1:
            corr = np.corrcoef(complexities, rewards_raw)[0, 1]
            ax.set_title(f'Complexity vs Reward (raw, corr: {corr:.3f})')
        else:
            ax.set_title('Complexity vs Reward (raw)')
        ax.set_xlabel('Complexity Level')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        save_plot(fig, 'complexity_vs_reward')

    # ---- 7. Reward vs Complexity per stage ----
    if 'task_class_history' in metrics and len(metrics['task_class_history']) > 0:
        stage_order = ['basic', 'doors', 'buttons', 'complex']
        unique_stages = [s for s in stage_order if s in metrics['task_class_history']]
        rewards_raw = np.array(metrics['train_rewards'])
        complexities_raw = np.array(metrics['complexity_history'])
        stages_raw = np.array(metrics['task_class_history'])
        min_len = min(len(rewards_raw), len(complexities_raw), len(stages_raw))
        rewards_raw = rewards_raw[:min_len]
        complexities_raw = complexities_raw[:min_len]
        stages_raw = stages_raw[:min_len]
        for stage in unique_stages:
            fig, ax = plt.subplots(figsize=(8, 5))
            mask = (stages_raw == stage)
            ax.scatter(complexities_raw, rewards_raw, c='gray', alpha=0.2, s=10, label='All epochs')
            epochs_of_stage = np.arange(min_len)[mask]
            if len(epochs_of_stage) > 0:
                sc = ax.scatter(complexities_raw[mask], rewards_raw[mask], c=epochs_of_stage, cmap='viridis', alpha=0.8, s=30, label=f'{stage.capitalize()} active')
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label('Epoch')
            if np.sum(mask) > 1:
                x_vals = complexities_raw[mask]
                y_vals = rewards_raw[mask]
                valid = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isinf(x_vals) | np.isinf(y_vals))
                x_vals = x_vals[valid]
                y_vals = y_vals[valid]
                if len(x_vals) >= 2 and np.std(x_vals) > 1e-6:
                    try:
                        z = np.polyfit(x_vals, y_vals, 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend: {z[0]:.2f}*x + {z[1]:.2f}')
                    except np.linalg.LinAlgError:
                        pass
            ax.set_title(f'Reward vs Complexity – {stage.capitalize()} stage (raw data)')
            ax.set_xlabel('Complexity Level')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            ax.legend()
            save_plot(fig, f'reward_vs_complexity_stage_{stage}')


class ParallelTrainerBase:
    """
    Base class for parallel training with vectorized environments.
    Handles:
      - Vectorized environment creation and resetting
      - Agent, optimizer, learning rate scheduler
      - Metrics logging, checkpoint saving/loading
      - Dynamic complexity (optional)
      - Test evaluation
      - Plot generation
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_seed = config['experiment']['seed']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        seed_everything(self.base_seed)

        training_cfg = config['training']
        self.batch_size = training_cfg['batch_size']
        self.reinforce_intra_epochs = training_cfg['reinforce_intra_epochs']
        self.grid_change_prob = training_cfg['grid_change_prob']
        self.update_per_episode = training_cfg['update_per_episode']
        self.epochs = training_cfg['epochs']
        self.test_interval = training_cfg['test_interval']
        self.save_interval = training_cfg['save_interval']
        self.test_task_class = training_cfg['test_task_class']
        self.test_complexity_level = training_cfg['test_complexity_level']

        self.action_buffer = np.zeros(self.batch_size, dtype=np.int64)
        
        # Dynamic complexity
        self.dynamic = training_cfg['dynamic_complexity']
        if self.dynamic:
            from .dynamic_complexity import ComplexityManager
            self.complexity_manager = ComplexityManager(config)
        else:
            self.complexity_manager = None

        self.model_name = self._build_model_name()
        self._setup_experiment_dirs()

        self.agent = self._create_agent()
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            mode='cosine',
            lr_start=training_cfg['learning_rate'],
            lr_min=1e-6
        )
        self.gradient_clipper = GradientClipper(max_norm=training_cfg['max_grad_norm'])

        # Metrics dictionary (full set)
        self.metrics = {
            'train_rewards': [],
            'train_losses': [],
            'policy_losses': [],
            'test_epochs': [],
            'test_rewards': [],
            'best_reward': -np.inf,
            'total_training_time': 0.0,   # cumulative training time across runs
        }
        self.training_start_time = None   # for tracking current session

        if self.dynamic:
            self.metrics['complexity_history'] = []
            self.metrics['task_class_history'] = []
            self.metrics['performance_scores'] = []
        if self.agent.use_auxiliary:
            self.metrics['aux_losses'] = []
            self.metrics['energy_losses'] = []
            self.metrics['obs_losses'] = []

        resume_path = config['experiment'].get('resume')
        if resume_path and Path(resume_path).exists():
            self._load_checkpoint(resume_path)

        self.start_epoch = len(self.metrics['train_rewards'])

        self.vector_env = self._create_vectorized_env()
        self._grid_pool = []

    def _build_model_name(self) -> str:
        train_cfg = self.config['training']
        batch_size = train_cfg['batch_size']
        lr = train_cfg['learning_rate']
        grid_size = self.config['environment']['grid_size']
        base = f"{batch_size}b_{lr}lr_gs{grid_size}"
        if train_cfg['algorithm'] == 'ppo':
            ppo_intra_epochs = train_cfg['ppo_intra_epochs']
            mini_batch_size = train_cfg['mini_batch_size']
            base += f"_pie{ppo_intra_epochs}_mb{mini_batch_size}"
        return base

    def _setup_experiment_dirs(self):
        exp_cfg = self.config['experiment']
        network_type = self.config['model']['type']
        use_aux = self.config['model']['use_auxiliary']
        aux_str = 'aux' if use_aux else 'no_aux'
        algorithm = self.config['training']['algorithm']

        prefix = exp_cfg.get('prefix')
        if prefix:
            base_dir = Path(exp_cfg['save_dir']) / prefix / network_type / algorithm / aux_str / self.model_name
        else:
            base_dir = Path(exp_cfg['save_dir']) / network_type / algorithm / aux_str / self.model_name

        date_subfolder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.experiment_dir = base_dir / date_subfolder
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.weights_dir = self.experiment_dir / 'weights'
        self.metrics_dir = self.experiment_dir / 'metrics'
        self.plots_dir = self.experiment_dir / 'plots'

        self.weights_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        self.metrics_path = self.metrics_dir / 'metrics.npz'

        self.logger = setup_logging(f"{self.model_name}_{date_subfolder}")

    def _create_agent(self) -> Agent:
        model_cfg = self.config['model']
        agent = Agent(
            network_type=model_cfg['type'],
            observation_size=OBSERVATION_SIZE,
            action_size=ACTION_SIZE,
            hidden_size=model_cfg['hidden_size'],
            use_auxiliary=model_cfg['use_auxiliary'],
            use_value_head=model_cfg.get('use_value_head', False),
            device=self.device
        )
        return agent

    def _create_optimizer(self) -> optim.Optimizer:
        train_cfg = self.config['training']
        return OptimizerFactory.create(
            optimizer_name=train_cfg['optimizer'],
            parameters=self.agent.network.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay'],
        )

    #def _create_vectorized_env(self) -> VectorizedMazeEnv:
    #    env_config = self.get_environment_config()
    #    return VectorizedMazeEnv(
    #        num_envs=self.batch_size,
    #        env_config=env_config,
    #        base_seed=self.base_seed
    #    )
    
    def _create_vectorized_env(self) -> "VectorizedMazeEnv":
        env_config = self.get_environment_config()
        import maze_core
        return maze_core.VectorizedMazeEnv(
            num_envs=self.batch_size,
            grid_size=env_config['grid_size'],
            max_steps=env_config['max_steps'],
            n_food_sources=env_config['n_food_sources'],
            food_energy=env_config['food_energy'],
            initial_energy=env_config['initial_energy'],
            energy_decay=env_config['energy_decay'],
            energy_per_step=env_config['energy_per_step'],
            task_class=env_config['task_class'],
            complexity_level=env_config['complexity_level'],
            n_doors=env_config.get('n_doors', 0),
            door_open_duration=env_config['door_open_duration'],
            door_close_duration=env_config['door_close_duration'],
            n_buttons_per_door=env_config.get('n_buttons_per_door', 0),
            button_break_probability=env_config.get('button_break_probability', 0.0),
            base_seed=self.base_seed
        )

    def get_environment_config(self) -> dict:
        if self.dynamic:
            base = self.config['environment'].copy()
            base['task_class'] = self.complexity_manager.get_current_task_class()
            base['complexity_level'] = self.complexity_manager.get_current_complexity()
            stage = base['task_class']
            if stage == 'basic':
                base['n_doors'] = 0
                base['n_buttons_per_door'] = 0
                base['button_break_probability'] = 0.0
            elif stage == 'doors':
                base['n_doors'] = None
                base['n_buttons_per_door'] = 0
                base['button_break_probability'] = 0.0
            elif stage == 'buttons':
                base['n_doors'] = None
                base['n_buttons_per_door'] = None
                base['button_break_probability'] = None
            elif stage == 'complex':
                base['n_doors'] = None
                base['n_buttons_per_door'] = None
                base['button_break_probability'] = None
            return base
        else:
            return self.config['environment'].copy()

    def _generate_grid_config(self) -> tuple:
        env_config = self.get_environment_config()
        import random
        seed = random.randint(0, 2**31 - 1)
        return (seed,
                env_config['task_class'],
                env_config['complexity_level'],
                env_config.get('n_doors'),
                env_config.get('n_buttons_per_door'),
                env_config.get('button_break_probability'))

    def _apply_grid_config(self, config: tuple, reset_hidden: bool = False):
        seed, task_class, complexity, n_doors, n_buttons, break_prob = config
        env_config = self.get_environment_config()
        env_config.update({
            'task_class': task_class,
            'complexity_level': complexity,
            'n_doors': n_doors,
            'n_buttons_per_door': n_buttons,
            'button_break_probability': break_prob
        })
        if hasattr(self, 'vector_env'):
            self.vector_env.close()
        self.vector_env = VectorizedMazeEnv(
            num_envs=self.batch_size,
            env_config=env_config,
            base_seed=seed
        )
        if reset_hidden:
            self.agent.reset()

    def _post_epoch_hook(self, epoch: int, dummy):
        """
        Called after each epoch.
        If dynamic complexity is enabled, it updates the performance history,
        checks for complexity adjustments or stage switches, and records metrics.
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            self._visualize_current_environments(epoch)
            cv2.imshow('Training Controls', dummy)
        elif key == ord('q'):
            print("\nEarly stop requested.")
            self._save_model('interrupted')
            cv2.destroyAllWindows()
            return True

        if self.dynamic and self.complexity_manager is not None:
            if 'epoch_rewards' in self.metrics and self.metrics['epoch_rewards']:
                avg_reward = self.metrics['epoch_rewards'][-1]
            elif self.metrics['train_rewards']:
                avg_reward = self.metrics['train_rewards'][-1]
            else:
                return
            self.complexity_manager.add_performance(avg_reward)
            adjustment = self.complexity_manager.adjust_complexity(epoch)
            if adjustment:
                action = adjustment['action']
                if 'new_stage' in adjustment:
                    self.logger.info(f"Epoch {epoch}: {action} - "
                                    f"{adjustment['old_stage']} -> {adjustment['new_stage']}, "
                                    f"complexity {adjustment['old_complexity']:.2f} -> {adjustment['new_complexity']:.2f}")
                else:
                    self.logger.info(f"Epoch {epoch}: {action} on {adjustment['old_stage']} - "
                                    f"complexity {adjustment['old_complexity']:.2f} -> {adjustment['new_complexity']:.2f}")
                self.vector_env.close()
                self.vector_env = self._create_vectorized_env()
                self._grid_pool = []
            self.metrics['complexity_history'].append(self.complexity_manager.get_current_complexity())
            self.metrics['task_class_history'].append(self.complexity_manager.get_current_task_class())
            self.metrics['performance_scores'].append(self.complexity_manager.calculate_performance_score())
        
        return False

    def _start_training_timer(self):
        """Start or resume the cumulative training timer."""
        if self.training_start_time is None:
            self.training_start_time = time.time()

    def _finalise_total_training_time(self):
        """Compute final cumulative training time and store in metrics."""
        if self.training_start_time is not None:
            elapsed = time.time() - self.training_start_time
            self.metrics['total_training_time'] += elapsed
            self.training_start_time = None

    def _test_valid(self, epochs: int = 10) -> dict:
        """Run test epochs using the configured test environment (task class + complexity)."""
        self.agent.network.eval()

        test_env_config = self.config['environment'].copy()
        test_env_config['task_class'] = self.test_task_class
        test_env_config['complexity_level'] = self.test_complexity_level
        # Let door/button parameters be auto‑configured based on task class
        test_env_config['n_doors'] = None
        test_env_config['n_buttons_per_door'] = None
        test_env_config['button_break_probability'] = None
        test_env_config['render_size'] = 0

        test_seed = self.base_seed + 12345   # fixed offset for reproducibility

        total_episodes = epochs * self.batch_size
        all_rewards = []
        all_lengths = []

        for _ in range(epochs):
            test_env = VectorizedMazeEnv(
                num_envs=self.batch_size,
                env_config=test_env_config,
                base_seed=test_seed
            )
            max_steps = test_env.envs[0].max_steps
            obs_array, _ = test_env.reset()
            obs_t = torch.as_tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)

            rewards = np.zeros(self.batch_size)
            lengths = np.zeros(self.batch_size, dtype=int)

            with torch.no_grad():
                for step in range(max_steps):
                    logits = self.agent.network(obs_t).squeeze(1)
                    actions = logits.argmax(dim=-1).cpu().numpy()
                    obs_array, r, terminated, truncated, _ = test_env.step(actions)
                    obs_t = torch.as_tensor(obs_array, dtype=torch.long, device=self.device).unsqueeze(1)
                    rewards += r
                    lengths += 1
                    if (terminated | truncated).all():
                        break
            test_env.close()
            all_rewards.extend(rewards)
            all_lengths.extend(lengths)

        avg_reward = np.mean(all_rewards)
        avg_length = np.mean(all_lengths)
        success_rate = np.sum(np.array(all_lengths) == max_steps) / total_episodes * 100

        return {
            'reward': avg_reward,
            'success_rate': success_rate,
            'avg_length': avg_length
        }

    def _save_model(self, name: str):
        agent_path = self.weights_dir / f"{name}.pt"
        self.agent.save(str(agent_path))
        self.logger.info(f"Saved agent to {agent_path}")

        checkpoint_path = self.weights_dir / f"{name}_checkpoint.pt"
        checkpoint = {
            'epoch': len(self.metrics['train_rewards']),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'metrics': self.metrics,
            'model_state_dict': self.agent.network.state_dict(),
            'model_config': {
                'network_type': self.agent.network_type,
                'use_auxiliary': self.agent.use_auxiliary,
                'hidden_size': self.agent.network.hidden_size,
                'observation_size': OBSERVATION_SIZE,
                'action_size': ACTION_SIZE
            },
            'config': self.config.copy()
        }
        if self.dynamic and self.complexity_manager:
            checkpoint['complexity_manager_state'] = {
                'current_stage_idx': self.complexity_manager.current_stage_idx,
                'performance_history': list(self.complexity_manager.performance_history),
                'adjustments_made': self.complexity_manager.adjustments_made,
                'stage_complexities': self.complexity_manager.stage_complexities,
                'max_rewards_by_stage': self.complexity_manager.max_rewards_by_stage,
                'epochs_without_progress': self.complexity_manager.epochs_without_progress,
                'last_complexity_increase_epoch': self.complexity_manager.last_complexity_increase_epoch,
                'last_max_reward': self.complexity_manager.last_max_reward,
            }
        torch.save(checkpoint, str(checkpoint_path))
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.metrics = checkpoint['metrics']
        if 'total_training_time' not in self.metrics:
            self.metrics['total_training_time'] = 0.0
        self.training_start_time = None

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.agent.network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if self.dynamic and self.complexity_manager and 'complexity_manager_state' in checkpoint:
            cm_state = checkpoint['complexity_manager_state']
            self.complexity_manager.current_stage_idx = cm_state['current_stage_idx']
            self.complexity_manager.performance_history = deque(cm_state['performance_history'],
                                                                maxlen=self.complexity_manager.performance_window)
            self.complexity_manager.adjustments_made = cm_state['adjustments_made']
        self.logger.info(f"Resumed from {path} at epoch {len(self.metrics['train_rewards'])}")

    def _save_metrics(self):
        save_dict = {
            'train_rewards': self.metrics['train_rewards'],
            'train_losses': self.metrics['train_losses'],
            'test_epochs': self.metrics['test_epochs'],
            'test_rewards': self.metrics['test_rewards'],
            'total_training_time': self.metrics['total_training_time'],
        }
        if self.dynamic:
            stage_map = {'basic': 0.0, 'doors': 0.33, 'buttons': 0.66, 'complex': 1.0}
            task_numeric = [stage_map.get(s, 0.0) for s in self.metrics['task_class_history']]
            save_dict['complexity_history'] = self.metrics['complexity_history']
            save_dict['task_class_history'] = task_numeric
            save_dict['performance_scores'] = self.metrics['performance_scores']
        if 'aux_losses' in self.metrics:
            save_dict['aux_losses'] = self.metrics['aux_losses']
            save_dict['energy_losses'] = self.metrics['energy_losses']
            save_dict['obs_losses'] = self.metrics['obs_losses']
        np.savez(self.metrics_path, **save_dict)

        increase_threshold = self.config['training'].get('complexity_increase_threshold', 0.65)
        decrease_threshold = self.config['training'].get('complexity_decrease_threshold', 0.4)
        generate_plots_from_metrics(self.metrics, self.plots_dir, increase_threshold, decrease_threshold)

    def _visualize_current_environments(self, epoch: int):
        print(f"\n📸 Visualizing environments at epoch {epoch}")

        num_to_show = min(4, len(self.vector_env.envs))
        cell_size = 256
        padding = 10
        cols = 2
        rows = (num_to_show + cols - 1) // cols
        total_width = cols * cell_size + (cols + 1) * padding
        total_height = rows * cell_size + (rows + 1) * padding
        combined = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        """
        for i in range(num_to_show):
            env = self.vector_env.envs[i]
            original_size = env.render_size
            env.render_size = cell_size
            if hasattr(env, '_render_buffer'):
                env._render_buffer = None
            frame = super(type(env), env).render()
            env.render_size = original_size
            if frame is None:
                frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                cv2.putText(frame, f"Env {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            if frame.shape[:2] != (cell_size, cell_size):
                frame = cv2.resize(frame, (cell_size, cell_size))
            col = i % cols
            row = i // cols
            x = padding + col * (cell_size + padding)
            y = padding + row * (cell_size + padding)
            combined[y:y+cell_size, x:x+cell_size] = frame

        if self.dynamic:
            status = self.complexity_manager.get_status()
            print(f"  Stage: {status['current_stage']}, Complexity: {status['current_complexity']:.2f}")

        """
            
        cv2.imshow('Training Visualization', combined)
        cv2.waitKey(0)

    def _setup_visualization(self):
        print("\n🎮 Visualisation Controls:")
        print("  Press 'v' to visualise current environments")
        print("  Press 'q' to stop training early")
        print("=" * 50)
        cv2.namedWindow('Training Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Training Controls', 400, 100)
        dummy = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(dummy, "Press 'v' to visualise", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(dummy, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow('Training Controls', dummy)
        cv2.waitKey(1)
        return dummy

    def _run_initial_test(self):
        if not self.metrics['test_epochs']:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(0)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

    def _should_test(self, epoch: int) -> bool:
        return (epoch > 0 and epoch % self.test_interval == 0 and
                epoch not in self.metrics['test_epochs'])

    def _run_test(self, epoch: int):
        test_metrics = self._test_valid(epochs=4)
        self.metrics['test_epochs'].append(epoch)
        self.metrics['test_rewards'].append(test_metrics['reward'])
        if test_metrics['reward'] > self.metrics['best_reward']:
            self.metrics['best_reward'] = test_metrics['reward']
            self._save_model('best')

    def _should_save(self, epoch: int) -> bool:
        return epoch > 0 and epoch % self.save_interval == 0

    def _save_checkpoint(self, epoch: int):
        self._save_model(f'epoch_{epoch:06d}')

    def _finalize_training(self):
        if self.epochs not in self.metrics['test_epochs']:
            test_metrics = self._test_valid(epochs=4)
            self.metrics['test_epochs'].append(self.epochs)
            self.metrics['test_rewards'].append(test_metrics['reward'])
            if test_metrics['reward'] > self.metrics['best_reward']:
                self.metrics['best_reward'] = test_metrics['reward']
                self._save_model('best')

        self._save_model('final')
        self._save_metrics()

    def _update_progress_bar(self, pbar, reward: float):
        pbar.set_postfix({
            'reward': f"{reward:.2f}",
            'best': f"{self.metrics['best_reward']:.2f}"
        })

    def train(self):
        raise NotImplementedError("Subclasses must implement train()")