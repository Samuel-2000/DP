#!/usr/bin/env python3
"""
benchmark_vectorized_env.py - Performance of VectorizedMazeEnv with batch size 64.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.cpp_build import ensure_cpp_module
ensure_cpp_module()

# Import the vectorized environment class
try:
    from maze_core import VectorizedMazeEnv
except ImportError:
    print("Error: VectorizedMazeEnv not found in maze_core.")
    sys.exit(1)

# For sequential baseline (single env repeated)
from src.core.env_factory_cpp import EnvironmentFactoryCPP as EnvironmentFactory


def measure_vectorized_reset(env_vec, num_runs):
    """Measure reset() on the vectorized env (all environments at once)."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        env_vec.reset()          # returns obs, infos (we ignore)
        times.append(time.perf_counter_ns() - start)
    times_ns = np.array(times)
    mean_us = np.mean(times_ns) / 1000.0
    std_us = np.std(times_ns) / 1000.0
    total_s = np.sum(times_ns) / 1e9
    return mean_us, std_us, total_s


def measure_vectorized_soft_reset(env_vec, num_runs):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter_ns()
        env_vec.soft_reset()
        times.append(time.perf_counter_ns() - start)
    times_ns = np.array(times)
    mean_us = np.mean(times_ns) / 1000.0
    std_us = np.std(times_ns) / 1000.0
    total_s = np.sum(times_ns) / 1e9
    return mean_us, std_us, total_s


def measure_vectorized_step(env_vec, num_steps, batch_size):
    """Step with random batch actions of length batch_size."""
    env_vec.reset()
    times = []
    for _ in range(num_steps):
        actions = np.random.randint(0, 6, size=batch_size).tolist()
        start = time.perf_counter_ns()
        _, _, _, _, _ = env_vec.step(actions)
        times.append(time.perf_counter_ns() - start)
    times_ns = np.array(times)
    mean_us = np.mean(times_ns) / 1000.0
    std_us = np.std(times_ns) / 1000.0
    total_s = np.sum(times_ns) / 1e9
    return mean_us, std_us, total_s


def measure_sequential_baseline(env_config, num_envs, num_resets, num_soft, num_steps):
    """
    Create num_envs single environments and run them sequentially.
    Returns per-operation times (total time / num_envs) for fair comparison.
    """
    envs = [EnvironmentFactory.create_from_config(env_config) for _ in range(num_envs)]

    # Reset all sequentially
    reset_times = []
    for _ in range(num_resets):
        for env in envs:
            start = time.perf_counter_ns()
            env.reset()
            reset_times.append(time.perf_counter_ns() - start)

    # Soft reset all sequentially
    soft_times = []
    for _ in range(num_soft):
        for env in envs:
            start = time.perf_counter_ns()
            env.soft_reset()
            soft_times.append(time.perf_counter_ns() - start)

    # Step all sequentially (random actions, soft reset on terminal/truncated)
    step_times = []
    for env in envs:
        env.reset()
    step_count = 0
    while step_count < num_steps:
        for env in envs:
            action = np.random.randint(0, 6)
            start = time.perf_counter_ns()
            _, _, term, trunc, _ = env.step(action)
            step_times.append(time.perf_counter_ns() - start)
            step_count += 1
            if step_count >= num_steps:
                break
            if term or trunc:
                env.soft_reset()

    def stats(arr):
        arr_ns = np.array(arr)
        mean_us = np.mean(arr_ns) / 1000.0
        std_us = np.std(arr_ns) / 1000.0
        total_s = np.sum(arr_ns) / 1e9
        return mean_us, std_us, total_s

    return stats(reset_times), stats(soft_times), stats(step_times)


def main():
    # Configuration - same as original benchmark
    env_config = {
        "grid_size": 19,
        "max_steps": 100,
        "n_food_sources": 4,
        "food_energy": 10.0,
        "initial_energy": 30.0,
        "energy_decay": 0.98,
        "energy_per_step": 0.1,
        "render_size": 0,
        "task_class": "complex",
        "complexity_level": 1.0,
        "n_doors": 5,
        "door_open_duration": 10,
        "door_close_duration": 20,
        "n_buttons_per_door": 4,
        "button_break_probability": 0.0,
    }

    BATCH_SIZE = 64
    NUM_RESET = 100      # total number of reset calls on the vectorized env
    NUM_SOFT = 1000
    NUM_STEPS = 10000

    print("=" * 70)
    print(f"Vectorized Environment Benchmark (batch size = {BATCH_SIZE})")
    print("=" * 70)

    # --- Create vectorized environment ---
    # NOTE: adapt constructor arguments to match your VectorizedMazeEnv C++ class
    vec_env = VectorizedMazeEnv(
        num_envs=BATCH_SIZE,
        grid_size=env_config["grid_size"],
        max_steps=env_config["max_steps"],
        n_food_sources=env_config["n_food_sources"],
        food_energy=env_config["food_energy"],
        initial_energy=env_config["initial_energy"],
        energy_decay=env_config["energy_decay"],
        energy_per_step=env_config["energy_per_step"],
        task_class=env_config["task_class"],
        complexity_level=env_config["complexity_level"],
        n_doors=env_config["n_doors"],
        door_open_duration=env_config["door_open_duration"],
        door_close_duration=env_config["door_close_duration"],
        n_buttons_per_door=env_config["n_buttons_per_door"],
        button_break_probability=env_config["button_break_probability"],
        base_seed=42
    )

    # --- Vectorized measurements ---
    print("\n>>> Vectorized (batch) operations <<<")
    reset_mean, reset_std, reset_total = measure_vectorized_reset(vec_env, NUM_RESET)
    print(f"reset      : {reset_mean:8.2f} ± {reset_std:5.2f} µs  (total {reset_total:.3f} s)  [{NUM_RESET} vector resets]")

    soft_mean, soft_std, soft_total = measure_vectorized_soft_reset(vec_env, NUM_SOFT)
    print(f"soft_reset : {soft_mean:8.2f} ± {soft_std:5.2f} µs  (total {soft_total:.3f} s)  [{NUM_SOFT} vector soft resets]")

    step_mean, step_std, step_total = measure_vectorized_step(vec_env, NUM_STEPS, BATCH_SIZE)
    print(f"step       : {step_mean:8.2f} ± {step_std:5.2f} µs  (total {step_total:.3f} s)  [{NUM_STEPS} vector steps]")


    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()