# run.py
#!/usr/bin/env python3
import sys
from pathlib import Path
from parser import parse_args
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.parallel_trainer_base import generate_plots_from_metrics

from src.core.agent import Agent
from src.core.env_factory import EnvironmentFactory
from src.core.agent_human import HumanAgent
from src.core.utils import get_model_name_from_path
from src.training.trainer import Trainer
from src.core.constants import (
    DEFAULT_GRID_SIZE, DEFAULT_MAX_STEPS, DEFAULT_OBSTACLE_FRACTION,
    DEFAULT_FOOD_SOURCES, DEFAULT_FOOD_ENERGY, DEFAULT_INITIAL_ENERGY,
    DEFAULT_ENERGY_DECAY, DEFAULT_ENERGY_PER_STEP, DEFAULT_RENDER_SIZE,
    DEFAULT_DOOR_OPEN_DURATION, DEFAULT_DOOR_CLOSE_DURATION,
    DEFAULT_HIDDEN_SIZE, DEFAULT_GAMMA, DEFAULT_ENTROPY_COEF,
    DEFAULT_MAX_GRAD_NORM, DEFAULT_SAVE_INTERVAL, DEFAULT_TEST_INTERVAL,
)


def main():
    args = parse_args()
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)


    if args.command == "plot":
        path = Path(args.metrics_path)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Resolve metrics.npz
        if path.is_file():
            if path.suffix == '.npz':
                metrics_path = path
        else:  # directory
            metrics_path = path / 'metrics' / 'metrics.npz'

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

        data = np.load(metrics_path, allow_pickle=True)
        metrics = {key: data[key] for key in data.files}

        # Convert numeric task class back to strings if needed
        if 'task_class_history' in metrics and metrics['task_class_history'].dtype.kind in 'iuf':
            stage_map = {0.0: 'basic', 0.33: 'doors', 0.66: 'buttons', 1.0: 'complex'}
            metrics['task_class_history'] = [stage_map[v] for v in metrics['task_class_history']]

        increase_threshold = metrics.get('increase_threshold', 0.65)
        decrease_threshold = metrics.get('decrease_threshold', 0.4)

        # Plots go into the plots subdirectory inside the timestamp folder
        plots_dir = metrics_path.parent.parent / 'plots'
        generate_plots_from_metrics(metrics, plots_dir, increase_threshold, decrease_threshold)
        print(f"Plots saved to {plots_dir}")
        return

    env_config = {
        "grid_size": DEFAULT_GRID_SIZE,
        "max_steps": DEFAULT_MAX_STEPS,
        "obstacle_fraction": DEFAULT_OBSTACLE_FRACTION,
        "n_food_sources": DEFAULT_FOOD_SOURCES,
        "food_energy": DEFAULT_FOOD_ENERGY,
        "initial_energy": DEFAULT_INITIAL_ENERGY,
        "energy_decay": DEFAULT_ENERGY_DECAY,
        "energy_per_step": DEFAULT_ENERGY_PER_STEP,
        "render_size": DEFAULT_RENDER_SIZE,
        "task_class": args.task_class,
        "complexity_level": args.complexity_level,
        "n_doors": args.n_doors,
        "door_open_duration": DEFAULT_DOOR_OPEN_DURATION,
        "door_close_duration": DEFAULT_DOOR_CLOSE_DURATION,
        "n_buttons_per_door": args.n_buttons_per_door,
        "button_break_probability": args.button_break_probability
    }

    if args.command == "train":
        # Build configuration dictionary directly from arguments
        config = {
            "experiment": {
                "name": args.experiment_name or f"{args.network_type}_{args.batch_size}b_{args.lr}lr",
                "prefix": args.experiment_name,   # None if not provided
                "save_dir": args.save_dir,
                "seed": args.seed,
                "resume": args.resume
            },
            "environment": env_config,
            "model": {
                "type": args.network_type,
                "hidden_size": DEFAULT_HIDDEN_SIZE,
                "use_auxiliary": args.auxiliary_tasks,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "gamma": DEFAULT_GAMMA,
                "entropy_coef": DEFAULT_ENTROPY_COEF,
                "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
                "save_interval": DEFAULT_SAVE_INTERVAL,
                "test_interval": DEFAULT_TEST_INTERVAL,
                "dynamic_complexity": args.dynamic_complexity,
                "performance_window": args.performance_window,
                "complexity_increase_threshold": args.complexity_increase_threshold,
                "complexity_decrease_threshold": args.complexity_decrease_threshold,
                "complexity_step": args.complexity_step,
                "min_complexity": args.min_complexity,
                "max_complexity": args.max_complexity,
                "adjustment_interval": args.adjustment_interval,
                "stagnation_switch_interval": args.stagnation_switch_interval,
                "stagnation_termination": args.stagnation_termination,
                "min_basic_complexity": args.min_basic_complexity,
                "curriculum_stages": args.curriculum_stages,
                "auxiliary_tasks": args.auxiliary_tasks,
                "reinforce_intra_epochs": args.reinforce_intra_epochs,
                "grid_change_prob": args.grid_change_prob,
                "update_per_episode": args.update_per_episode
            },
        }

        config['training']['algorithm'] = args.algorithm
        if args.algorithm == 'ppo':
            config['training']['ppo_intra_epochs'] = args.ppo_intra_epochs
            config['training']['mini_batch_size'] = args.mini_batch_size
            config['training']['clip_epsilon'] = args.clip_epsilon
            config['training']['value_coef'] = args.value_coef
            config['training']['gae_lambda'] = args.gae_lambda
        else:
            # These keys will still be present (set to None) to avoid KeyError
            config['training']['ppo_intra_epochs'] = None
            config['training']['mini_batch_size'] = None
            config['training']['clip_epsilon'] = None
            config['training']['value_coef'] = None
            config['training']['gae_lambda'] = None
            
        trainer = Trainer(config)
        trainer.train()

    elif args.command == "test":
        if args.play:
            # Human play mode
            agent = HumanAgent()
        else:
            agent = Agent.load(args.model)
            base_name = get_model_name_from_path(args.model)


        # Determine test configurations
        if args.dynamic_complexity:
            test_configs = []
            for stage in args.stages:
                for comp in args.complexities:
                    test_configs.append(env_config.copy())
                    test_configs[-1]["task_class"] = stage
                    test_configs[-1]["complexity_level"] = comp
        else:
            test_configs = [env_config]


        for idx, cfg in enumerate(test_configs):
            print(f"\nTest {idx+1}/{len(test_configs)}: {cfg['task_class']} comp={cfg['complexity_level']:.2f}")
            env = EnvironmentFactory.create_from_config(cfg, test_mode=True)
            if args.play:
                results = agent.test(env, args)
            else:
                model_name = f"{base_name}_{cfg['task_class']}_comp_{cfg['complexity_level']:.2f}".replace('.','_')
                results = agent.test(env, args, model_name, args.seed)

            print(f"  Reward: {results['avg_reward']:.2f}, Success: {results['success_rate']:.1f}%, Avg Steps: {results['avg_steps']:.1f}")

    else:
        print("Please specify a command. Use --help for usage.")


if __name__ == "__main__":
    main()