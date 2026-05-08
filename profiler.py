#!/usr/bin/env python3
"""
profiler.py - Minimal yet detailed profiling.
Prints benchmark summary (component timings) and top 3 slowest functions.
Full cProfile and PyTorch profiler results are saved to files.
"""

import torch
import numpy as np
import cProfile
import pstats
import io
from pathlib import Path
import sys
import time
import argparse
from pstats import SortKey

sys.path.insert(0, str(Path(__file__).parent))
from src.training.trainer import Trainer
from src.core.utils import seed_everything


class MinimalProfiler:
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = self._create_config()
        print("\n" + "=" * 80)
        print("PROFILER – Benchmark Summary")
        print("=" * 80)
        print(f"Algorithm: {args.algorithm} | Network: {args.network_type} | Batch: {args.batch_size}")
        print(f"Task: {args.task_class} | Complexity: {args.complexity_level} | Doors: {args.n_doors}")
        print(f"Device: {self.device}")
        print("=" * 80)

    def _create_config(self):
        config = {
            'experiment': {'name': f'profile_{self.args.task_class}', 'seed': 42, 'save_dir': 'models', 'resume': None},
            'environment': {
                'grid_size': 11, 'max_steps': 100, 'obstacle_fraction': 0.25,
                'n_food_sources': 4, 'food_energy': 10.0, 'initial_energy': 30.0,
                'energy_decay': 0.98, 'energy_per_step': 0.1, 'render_size': 0,
                'task_class': self.args.task_class, 'complexity_level': self.args.complexity_level,
                'n_doors': self.args.n_doors, 'door_open_duration': 10, 'door_close_duration': 20,
                'n_buttons_per_door': self.args.n_buttons_per_door, 'button_break_probability': 0.0,
            },
            'model': {
                'type': self.args.network_type, 'hidden_size': 512, 'use_auxiliary': False,
                'use_value_head': (self.args.algorithm == 'ppo')
            },
            'training': {
                'epochs': 10, 'batch_size': self.args.batch_size, 'learning_rate': 0.0005,
                'gamma': 0.97, 'entropy_coef': 0.01, 'max_grad_norm': 1.0,
                'save_interval': 1000, 'test_interval': 500, 'optimizer': 'adam',
                'weight_decay': 0.0, 'dynamic_complexity': False, 'reinforce_intra_epochs': 1,
                'grid_change_prob': 0.0, 'update_per_episode': True, 'algorithm': self.args.algorithm,
                'ppo_intra_epochs': 4, 'mini_batch_size': self.args.batch_size, 'clip_epsilon': 0.2,
                'value_coef': 0.5, 'gae_lambda': 0.95,
            }
        }
        return config

    def _get_collect_method(self, trainer):
        if hasattr(trainer, '_collect_rollout'):
            return trainer._collect_rollout
        elif hasattr(trainer, '_collect_experiences_parallel'):
            return trainer._collect_experiences_parallel
        else:
            raise AttributeError(f"No collection method found for {type(trainer)}")

    def _get_train_method(self, trainer):
        return trainer._train_step

    def _run_training_steps(self, trainer, num_steps=5):
        collect = self._get_collect_method(trainer)
        train_step = self._get_train_method(trainer)
        for _ in range(num_steps):
            trainer.vector_env.reset()
            trainer.agent.reset()
            exp = collect()
            train_step(exp)
            if hasattr(trainer.agent.network, 'flush_cache_buffer'):
                trainer.agent.network.flush_cache_buffer()

    def run_benchmark(self, trainer):
        collect = self._get_collect_method(trainer)
        train_step = self._get_train_method(trainer)

        # Environment full reset
        reset_times = []
        for _ in range(5):
            start = time.perf_counter()
            trainer.vector_env.reset()
            reset_times.append(time.perf_counter() - start)
        reset_ms = np.mean(reset_times) * 1000
        reset_std = np.std(reset_times) * 1000

        # Environment soft reset (on a single environment, after a full reset)
        # Get a single environment instance from the vector env
        single_env = trainer.vector_env.envs[0]
        # Perform a full reset first to ensure clean state
        single_env.reset(seed=42)
        soft_reset_times = []
        # Measure many soft resets (1000) for stable microsecond timing
        for _ in range(1000):
            start = time.perf_counter()
            single_env.soft_reset()
            soft_reset_times.append(time.perf_counter() - start)
        soft_reset_us = np.mean(soft_reset_times) * 1e6
        soft_reset_std = np.std(soft_reset_times) * 1e6

        # Experience collection
        coll_times = []
        for i in range(5):
            start = time.perf_counter()
            exp = collect()
            coll_times.append(time.perf_counter() - start)
            if i == 0:
                B, T, _ = exp['observations'].shape
                total_steps = B * T
        coll_mean = np.mean(coll_times)
        coll_std = np.std(coll_times)

        # Training step
        train_times = []
        for _ in range(5):
            exp = collect()
            start = time.perf_counter()
            train_step(exp)
            train_times.append(time.perf_counter() - start)
        train_mean = np.mean(train_times)
        train_std = np.std(train_times)

        # Network forward – skip if half precision to avoid dtype mismatch
        original_dtype = next(trainer.agent.network.parameters()).dtype
        if original_dtype == torch.float16:
            print("  Warning: Skipping forward pass timing (half precision).")
            fwd_ms = 0.0
            fwd_std = 0.0
        else:
            dummy = torch.randint(0, 19, (self.args.batch_size, 1, 10), device=self.device).long()
            trainer.agent.network.eval()
            fwd_times = []
            with torch.no_grad():
                for _ in range(10):
                    start = time.perf_counter()
                    _ = trainer.agent.network(dummy)
                    fwd_times.append(time.perf_counter() - start)
            fwd_ms = np.mean(fwd_times) * 1000
            fwd_std = np.std(fwd_times) * 1000

        # Door updates
        env = trainer.vector_env.envs[0]
        door_us = None
        if env.doors:
            door_us_list = []
            for _ in range(100):
                start = time.perf_counter()
                env._update_door_states()
                door_us_list.append(time.perf_counter() - start)
            door_us = np.mean(door_us_list) * 1e6
            door_std = np.std(door_us_list) * 1e6

        # Template matching
        tm_us = None
        if hasattr(env, 'template_matcher'):
            tm_list = []
            for _ in range(100):
                y = np.random.randint(1, env.grid_size-1)
                x = np.random.randint(1, env.grid_size-1)
                start = time.perf_counter()
                env.template_matcher.matches(env.grid, y, x)
                tm_list.append(time.perf_counter() - start)
            tm_us = np.mean(tm_list) * 1e6
            tm_std = np.std(tm_list) * 1e6

        print("\nComponent Timings (average ± std):")
        print(f"  Environment full reset: {reset_ms:.1f} ± {reset_std:.1f} ms")
        print(f"  Environment soft reset: {soft_reset_us:.1f} ± {soft_reset_std:.1f} µs")
        print(f"  Experience collection : {coll_mean:.3f} ± {coll_std:.3f} s  ({total_steps} steps, {total_steps/coll_mean:.0f} steps/s)")
        print(f"  Training step         : {train_mean:.3f} ± {train_std:.3f} s")
        if original_dtype != torch.float16:
            print(f"  Network forward       : {fwd_ms:.2f} ± {fwd_std:.2f} ms")
        else:
            print(f"  Network forward       : skipped (half precision)")
        if door_us is not None:
            print(f"  Door updates          : {door_us:.1f} ± {door_std:.1f} µs")
        if tm_us is not None:
            print(f"  Template matching     : {tm_us:.1f} ± {tm_std:.1f} µs")

    def run_cprofile_and_show_top3(self, trainer):
        print("\n" + "=" * 80)
        print("cProfile – Top 3 Slowest Functions (by tottime)")
        print("=" * 80)
        profiler = cProfile.Profile()
        profiler.enable()
        self._run_training_steps(trainer, num_steps=5)
        profiler.disable()

        results_dir = Path("profiler_results")
        results_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.args.task_class}_{self.args.algorithm}_{ts}"

        # Cumulative report
        s_cum = io.StringIO()
        ps_cum = pstats.Stats(profiler, stream=s_cum).sort_stats(SortKey.CUMULATIVE)
        ps_cum.print_stats(100)
        with open(results_dir / f"{prefix}_cumulative.txt", 'w') as f:
            f.write(s_cum.getvalue())

        # Internal time report
        s_int = io.StringIO()
        ps_int = pstats.Stats(profiler, stream=s_int).sort_stats(SortKey.TIME)
        ps_int.print_stats(100)
        with open(results_dir / f"{prefix}_internal.txt", 'w') as f:
            f.write(s_int.getvalue())

        # Top 3 by tottime
        stats_dict = {}
        for func, (cc, nc, tt, ct, callers) in ps_int.stats.items():
            stats_dict[func[2]] = tt
        sorted_funcs = sorted(stats_dict.items(), key=lambda x: x[1], reverse=True)[:3]

        print("\nFunction (tottime):")
        for name, tt in sorted_funcs:
            print(f"  {name:40s} : {tt:.3f} s")

    def run(self):
        try:
            seed_everything(42)
            print("\nCreating trainer...")
            trainer = Trainer(self.config)

            self.run_benchmark(trainer)
            self.run_cprofile_and_show_top3(trainer)

            if torch.cuda.is_available():
                print("\nRunning PyTorch profiler (GPU). Traces saved to ./profiler_traces/")
                collect = self._get_collect_method(trainer)
                train_step = self._get_train_method(trainer)
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profiler_traces/{self.args.task_class}'),
                    record_shapes=True, profile_memory=True
                ) as prof:
                    for _ in range(5):
                        trainer.vector_env.reset()
                        trainer.agent.reset()
                        with torch.profiler.record_function("experience_collection"):
                            exp = collect()
                        with torch.profiler.record_function("training_step"):
                            train_step(exp)
                        prof.step()
                print("  PyTorch profiler finished. Use tensorboard --logdir=./profiler_traces")
            else:
                print("\nSkipping PyTorch profiler (CUDA not available).")

            print("\n" + "=" * 80)
            print("PROFILING COMPLETE. Detailed reports saved in 'profiler_results/'")
            print("=" * 80)
            return True
        except Exception as e:
            print(f"\n❌ Profiling failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='reinforce', choices=['reinforce','ppo'])
    parser.add_argument('--network-type', default='lstm', choices=['lstm','transformer','multimemory'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--task-class', default='doors', choices=['basic','doors','buttons','complex'])
    parser.add_argument('--complexity-level', type=float, default=1.0)
    parser.add_argument('--n-doors', type=int, default=5)
    parser.add_argument('--n-buttons-per-door', type=int, default=4, choices=[0,1,2,3,4])
    args = parser.parse_args()

    print(f"\nSystem: PyTorch {torch.__version__}, CUDA {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    profiler = MinimalProfiler(args)
    return 0 if profiler.run() else 1


if __name__ == "__main__":
    sys.exit(main())