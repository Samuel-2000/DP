#!/usr/bin/env python3
"""
profile_run.py – Run run.py with full profiling (cProfile and PyTorch profiler).
Example:
    python profile_run.py train --network-type lstm --epochs 200 --batch-size 64 \
        --lr 0.0005 --dynamic-complexity --algorithm reinforce --reinforce-intra-epochs 1
"""

import sys
import cProfile
import pstats
import io
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the original argument parser and trainer factory
from parser import parse_args
from src.training.trainer import Trainer
from src.core.utils import seed_everything


def run_training_profiled(args, enable_pytorch_profiler: bool = False):
    """Run the training with profiling."""
    # Parse args is already done in main; we receive the parsed namespace.
    # Ensure seed for reproducibility
    seed_everything(args.seed)

    # Build config from parsed args (same as run.py does)
    from run import env_config, config as build_config  # we'll replicate the logic
    # Instead of importing run.py (which would execute), we re-implement the config building.
    # But that's messy. Simpler: call the main training part directly.
    # Let's construct the trainer from config.

    # We'll reuse the config building logic from run.py's train branch.
    # For clarity, we copy the relevant parts.
    from src.core.constants import (
        DEFAULT_GRID_SIZE, DEFAULT_MAX_STEPS,
        DEFAULT_FOOD_SOURCES, DEFAULT_FOOD_ENERGY, DEFAULT_INITIAL_ENERGY,
        DEFAULT_ENERGY_DECAY, DEFAULT_ENERGY_PER_STEP, DEFAULT_RENDER_SIZE,
        DEFAULT_DOOR_OPEN_DURATION, DEFAULT_DOOR_CLOSE_DURATION,
        DEFAULT_HIDDEN_SIZE, DEFAULT_GAMMA, DEFAULT_ENTROPY_COEF,
        DEFAULT_MAX_GRAD_NORM, DEFAULT_SAVE_INTERVAL, DEFAULT_TEST_INTERVAL,
    )

    env_config = {
        "grid_size": DEFAULT_GRID_SIZE,
        "max_steps": DEFAULT_MAX_STEPS,
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

    config = {
        "experiment": {
            "name": args.experiment_name or f"{args.network_type}_{args.batch_size}b_{args.lr}lr",
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
            "update_per_episode": args.update_per_episode,
            "algorithm": args.algorithm,
        },
    }

    # Add PPO specific args if needed
    if args.algorithm == 'ppo':
        config['training']['ppo_intra_epochs'] = args.ppo_intra_epochs
        config['training']['mini_batch_size'] = args.mini_batch_size
        config['training']['clip_epsilon'] = args.clip_epsilon
        config['training']['value_coef'] = args.value_coef
        config['training']['gae_lambda'] = args.gae_lambda
    else:
        config['training']['ppo_intra_epochs'] = None
        config['training']['mini_batch_size'] = None
        config['training']['clip_epsilon'] = None
        config['training']['value_coef'] = None
        config['training']['gae_lambda'] = None

    # Create trainer
    trainer = Trainer(config)

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Optional PyTorch profiler
    pytorch_prof = None
    if enable_pytorch_profiler and torch.cuda.is_available():
        import torch
        pytorch_prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_traces/train_run'),
            record_shapes=True,
            profile_memory=True
        )
        pytorch_prof.__enter__()

    start_time = time.time()
    try:
        trainer.train()
    except Exception as e:
        print(f"Training interrupted: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        profiler.disable()
        if pytorch_prof:
            pytorch_prof.__exit__(None, None, None)

        # Save cProfile results
        results_dir = Path("profiler_results")
        results_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"train_{args.network_type}_{args.algorithm}_{ts}"

        # Save binary profile
        prof_file = results_dir / f"{prefix}.prof"
        profiler.dump_stats(str(prof_file))

        # Generate human-readable stats sorted by cumulative time
        s_cum = io.StringIO()
        ps_cum = pstats.Stats(profiler, stream=s_cum).sort_stats(pstats.SortKey.CUMULATIVE)
        ps_cum.print_stats(100)
        with open(results_dir / f"{prefix}_cumulative.txt", 'w') as f:
            f.write(s_cum.getvalue())

        # Sorted by internal time
        s_int = io.StringIO()
        ps_int = pstats.Stats(profiler, stream=s_int).sort_stats(pstats.SortKey.TIME)
        ps_int.print_stats(100)
        with open(results_dir / f"{prefix}_internal.txt", 'w') as f:
            f.write(s_int.getvalue())

        # Top 10 functions by tottime
        stats_dict = {}
        for func, (cc, nc, tt, ct, callers) in ps_int.stats.items():
            stats_dict[func[2]] = tt
        sorted_funcs = sorted(stats_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        print("\n" + "="*80)
        print("PROFILING RESULTS")
        print("="*80)
        print(f"Total training time: {total_time:.1f}s")
        print(f"cProfile saved to: {prof_file}")
        print(f"Human-readable reports saved in {results_dir}/")
        if pytorch_prof:
            print("PyTorch profiler traces saved to ./profiler_traces/")
        print("\nTop 10 functions by total time (tottime):")
        for name, tt in sorted_funcs:
            print(f"  {name:50s} : {tt:.3f} s")
        print("="*80)


def main():
    # Parse arguments using the same parser as run.py
    # But we need to add a flag for enabling PyTorch profiler.
    original_parser = argparse.ArgumentParser(add_help=False)
    # Add our own flag before parsing the rest
    original_parser.add_argument('--pytorch-profiler', action='store_true', help='Enable PyTorch profiler (requires CUDA)')
    # Parse known args so we can get the flag, then parse the rest with the real parser
    preliminary_args, remaining_argv = original_parser.parse_known_args()

    # Now parse the full arguments using run.py's parser (which expects sys.argv without the new flag)
    sys.argv = [sys.argv[0]] + remaining_argv
    args = parse_args()  # this is the original parser

    # Add the pytorch_profiler attribute to args
    args.pytorch_profiler = preliminary_args.pytorch_profiler

    # Now run training with profiling
    run_training_profiled(args, enable_pytorch_profiler=args.pytorch_profiler)


if __name__ == "__main__":
    # Need torch for PyTorch profiler
    import torch
    main()