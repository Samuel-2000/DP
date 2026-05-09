#!/usr/bin/env python3
"""
benchmark_cpp_env_fixed.py – Performance benchmark for the C++ GridMazeWorld.
Also benchmarks the original Python+Numba environment from src/core/obsolete.
"""

import sys
import time
import numpy as np
import argparse
from pathlib import Path

# --- Add project root to sys.path (same as run.py) ---
sys.path.insert(0, str(Path(__file__).parent / "src"))

# --- Ensure C++ module and OpenCV DLLs are set up ---
from src.core.cpp_build import ensure_cpp_module
ensure_cpp_module()   # This sets up OpenCV path and loads maze_core

# Now import the C++ environment factory
from src.core.env_factory_cpp import EnvironmentFactoryCPP as EnvironmentFactory

# --- Try to import original Python environment from obsolete folder ---
PYTHON_AVAILABLE = False
PythonGridMazeWorld = None
try:
    # The obsolete environment lives in src/core/obsolete/environment.py
    from src.core.obsolete.environment import GridMazeWorld as PythonGridMazeWorld
    PYTHON_AVAILABLE = True
    print("✓ Original Python environment (obsolete) loaded successfully")
except ImportError as e:
    print(f"⚠️ Original Python environment not found: {e}")


def measure_reset(env, num_runs):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        env.reset(seed=None)
        times.append(time.perf_counter_ns() - start)
    times_ns = np.array(times)
    mean_us = np.mean(times_ns) / 1000.0
    std_us = np.std(times_ns) / 1000.0
    total_s = np.sum(times_ns) / 1e9
    return mean_us, std_us, total_s


def measure_soft_reset(env, num_runs):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        env.soft_reset()
        times.append(time.perf_counter_ns() - start)
    times_ns = np.array(times)
    mean_us = np.mean(times_ns) / 1000.0
    std_us = np.std(times_ns) / 1000.0
    total_s = np.sum(times_ns) / 1e9
    return mean_us, std_us, total_s


def measure_step(env, num_steps, random_actions=True):
    env.reset(seed=42)
    actions = [0, 1, 2, 3, 4, 5]  # LEFT, RIGHT, UP, DOWN, STAY, BUTTON
    times = []
    for i in range(num_steps):
        action = np.random.choice(actions) if random_actions else (i % len(actions))
        start = time.perf_counter_ns()
        obs, reward, terminated, truncated, info = env.step(action)
        times.append(time.perf_counter_ns() - start)
        if terminated or truncated:
            env.soft_reset()
    times_ns = np.array(times)
    mean_us = np.mean(times_ns) / 1000.0
    std_us = np.std(times_ns) / 1000.0
    total_s = np.sum(times_ns) / 1e9
    return mean_us, std_us, total_s


def benchmark(env, name):
    NUM_RESET = 1000
    NUM_SOFT = 100_000
    NUM_STEPS = 100_000

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")
    print(f"Running {NUM_RESET} resets, {NUM_SOFT} soft_resets, {NUM_STEPS} steps...")

    reset_mean, reset_std, reset_total = measure_reset(env, NUM_RESET)
    print(f"reset      : {reset_mean:8.2f} ± {reset_std:5.2f} µs  (total {reset_total:.3f} s)  [{NUM_RESET} runs]")

    soft_mean, soft_std, soft_total = measure_soft_reset(env, NUM_SOFT)
    print(f"soft_reset : {soft_mean:8.2f} ± {soft_std:5.2f} µs  (total {soft_total:.3f} s)  [{NUM_SOFT} runs]")

    step_mean, step_std, step_total = measure_step(env, NUM_STEPS)
    print(f"step       : {step_mean:8.2f} ± {step_std:5.2f} µs  (total {step_total:.3f} s)  [{NUM_STEPS} steps]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="doors", choices=["basic","doors","buttons","complex"])
    parser.add_argument("--complexity", type=float, default=1.0)
    parser.add_argument("--n_doors", type=int, default=5)
    parser.add_argument("--n_buttons", type=int, default=4)
    parser.add_argument("--grid_size", type=int, default=11)
    args = parser.parse_args()

    print(f"Grid size: {args.grid_size} | Task: {args.task} | Complexity: {args.complexity}")
    print(f"Doors: {args.n_doors} | Buttons per door: {args.n_buttons}")

    env_config = {
        "grid_size": 39,#args.grid_size,
        "max_steps": 100,
        "n_food_sources": 4,
        "food_energy": 10.0,
        "initial_energy": 30.0,
        "energy_decay": 0.98,
        "energy_per_step": 0.1,
        "render_size": 0,
        "task_class": "buttons",# args.task,
        "complexity_level": args.complexity,
        "n_doors": args.n_doors,
        "door_open_duration": 10,
        "door_close_duration": 20,
        "n_buttons_per_door": args.n_buttons,
        "button_break_probability": 0.0,
    }

    # --- C++ environment (via factory) ---
    cpp_env = EnvironmentFactory.create_from_config(env_config, test_mode=False)
    benchmark(cpp_env, "C++ (maze_core)")

    # --- Original Python environment (obsolete) ---
    if PYTHON_AVAILABLE and PythonGridMazeWorld is not None:
        py_env = PythonGridMazeWorld(**env_config)
        benchmark(py_env, "Python+Numba (original / obsolete)")
    else:
        print("\n⚠️ Skipping Python environment benchmark (not found)")

    print("\n" + "="*60)
    print("Benchmark complete.")
    print("="*60)


if __name__ == "__main__":
    main()