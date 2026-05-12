#!/usr/bin/env python3
"""
Sequential, adaptive hyperparameter tuning for Maze RL.
Follows the test order:
  1. Algorithm (REINFORCE vs PPO)
  2. PPO intra-epochs (if PPO)
  3. PPO mini-batch size
  4. Batch size (keeping total steps constant)
  5. Optimizer (adam, sgd, adamw, rmsprop)
  6. Learning rate

Results are cached in 'experiment_cache/' and skipped if already run.
"""

import subprocess
import sys
import json
import argparse
import re
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# ------------------------------------------------------------
# Base configuration – matches your typical run.py calls
# ------------------------------------------------------------
BASE_CONFIG = {
    "network_type": "lstm",
    "epochs": 5000,                     # will be adjusted later
    "batch_size": 64,
    "lr": 0.0005,
    "task_class": "basic",
    "complexity_level": 0.5,
    "test_task_class": "basic",
    "test_complexity_level": 1.0,
    "algorithm": "ppo",
    "ppo_intra_epochs": 4,
    "mini_batch_size": 64,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "seed": 42,
    # Default grid size is 11 (from constants.py) – do NOT pass --grid-size to run.py
    # "grid_size": 11,   # not used in command line
}

# Constant total steps for batch size stage (5000*64 = 320000)
CONSTANT_TOTAL_STEPS = 320000

# Number of repetitions per configuration (for robustness)
REPEATS_PER_CONFIG = 3

# Maximum parallel subprocesses per stage
MAX_WORKERS = 4

# Directory where experiment results are cached
RESULTS_CACHE_DIR = Path("experiment_cache")
RESULTS_CACHE_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Helper: unique config ID and result caching
# ------------------------------------------------------------
def config_id(config: Dict[str, Any]) -> str:
    """Create a unique identifier for a configuration (ignores seed)."""
    ignore_keys = {"seed"}
    filtered = {k: v for k, v in config.items() if k not in ignore_keys}
    items = sorted(filtered.items())
    hash_str = hashlib.md5(json.dumps(items).encode()).hexdigest()[:12]
    return hash_str

def load_cached_result(config: Dict[str, Any]) -> Optional[float]:
    """If the configuration has been run before, return its final reward."""
    cid = config_id(config)
    cache_file = RESULTS_CACHE_DIR / f"{cid}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data.get("final_reward", None)
    return None

def save_cached_result(config: Dict[str, Any], reward: float):
    """Save the final reward for this configuration."""
    cid = config_id(config)
    cache_file = RESULTS_CACHE_DIR / f"{cid}.json"
    with open(cache_file, "w") as f:
        json.dump({"config": config, "final_reward": reward}, f, indent=2)

# ------------------------------------------------------------
# Running a single experiment (subprocess)
# ------------------------------------------------------------
def build_cmd(config: Dict[str, Any]) -> List[str]:
    """Convert config dict to run.py train command."""
    cmd = [
        sys.executable, "run.py", "train",
        "--network-type", config["network_type"],
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--task-class", config["task_class"],
        "--complexity-level", str(config["complexity_level"]),
        "--test-task-class", config["test_task_class"],
        "--test-complexity-level", str(config["test_complexity_level"]),
        "--algorithm", config["algorithm"],
        "--optimizer", config["optimizer"],
        "--weight-decay", str(config["weight_decay"]),
        "--seed", str(config["seed"]),
    ]
    if config["algorithm"] == "ppo":
        cmd.extend([
            "--ppo-intra-epochs", str(config["ppo_intra_epochs"]),
            "--mini-batch-size", str(config["mini_batch_size"])
        ])
    # Do NOT pass --grid-size or --max-steps – they are not in the original parser
    # (defaults from constants.py are used)
    return cmd

def extract_reward_from_output(output: str) -> Optional[float]:
    """Parse final test reward from run.py stdout/stderr."""
    for line in output.splitlines():
        if "Test reward" in line or "Final validation" in line:
            m = re.search(r"reward[=:]\s*([0-9.]+)", line, re.IGNORECASE)
            if m:
                return float(m.group(1))
        if "Best reward:" in line:
            m = re.search(r"Best reward:\s*([0-9.]+)", line)
            if m:
                return float(m.group(1))
    return None

def run_single(config: Dict[str, Any], timeout: int = 7200) -> float:
    """Execute one training run and return the final test reward."""
    # Check cache
    cached = load_cached_result(config)
    if cached is not None:
        print(f"  ⏩ Skipping {config_id(config)} (cached reward = {cached:.2f})")
        return cached

    cmd = build_cmd(config)
    print(f"  Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        print(f"  ⚠️ Timeout after {timeout}s")
        return 0.0

    reward = extract_reward_from_output(stdout + stderr)
    if reward is None:
        print("  ⚠️ Could not parse reward, using 0.0")
        reward = 0.0
    else:
        print(f"  ✅ Reward = {reward:.2f}")

    save_cached_result(config, reward)
    return reward

# ------------------------------------------------------------
# Stage definition: test a list of values for one parameter
# ------------------------------------------------------------
def run_stage(param_name: str, values: List[Any],
              base: Dict[str, Any], repeats: int = REPEATS_PER_CONFIG) -> Any:
    """
    Test all values for a given parameter (e.g., 'optimizer').
    Returns the best value (the one with highest average reward).
    """
    print(f"\n{'='*70}")
    print(f"STAGE: tuning {param_name} -> values: {values}")
    print(f"{'='*70}")

    # Build all configurations for this stage
    configs = []
    for val in values:
        for rep in range(repeats):
            cfg = base.copy()
            cfg[param_name] = val
            cfg["seed"] = base["seed"] + rep * 1000 + hash(str(val)) % 1000
            # Special handling for batch size: keep total steps constant
            if param_name == "batch_size" and CONSTANT_TOTAL_STEPS:
                cfg["epochs"] = max(1, CONSTANT_TOTAL_STEPS // val)
            configs.append(cfg)

    # Run in parallel (within this stage only)
    results = {}   # value -> list of rewards
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            cfg = futures[future]
            val = cfg[param_name]
            reward = future.result()
            results.setdefault(val, []).append(reward)

    # Compute average reward per value
    import numpy as np
    avg_rewards = {val: np.mean(rewards) for val, rewards in results.items()}
    best_val = max(avg_rewards.items(), key=lambda x: x[1])[0]
    print(f"\n🏆 Best {param_name} = {best_val} (avg reward {avg_rewards[best_val]:.2f})")
    return best_val

# ------------------------------------------------------------
# Main sequential tuning pipeline (exact order from your tests)
# ------------------------------------------------------------
def run_sequential_tuning():
    """Run the adaptive pipeline as described."""
    config = BASE_CONFIG.copy()

    # Stage 1: algorithm (REINFORCE vs PPO)
    best_algo = run_stage("algorithm", ["reinforce", "ppo"], config)
    config["algorithm"] = best_algo

    # Stages 2 & 3: PPO-specific parameters (only if algorithm is PPO)
    if config["algorithm"] == "ppo":
        best_intra = run_stage("ppo_intra_epochs", [1, 2, 4], config)
        config["ppo_intra_epochs"] = best_intra

        best_mini = run_stage("mini_batch_size", [16, 64, 256], config)
        config["mini_batch_size"] = best_mini

    # Stage 4: batch size (with constant total steps)
    best_bs = run_stage("batch_size", [32, 64, 128], config)
    config["batch_size"] = best_bs
    if CONSTANT_TOTAL_STEPS:
        config["epochs"] = CONSTANT_TOTAL_STEPS // best_bs

    # Stage 5: optimizer (adam, sgd, adamw, rmsprop)
    best_opt = run_stage("optimizer", ["adam", "sgd", "adamw", "rmsprop"], config)
    config["optimizer"] = best_opt

    # Stage 6: learning rate
    best_lr = run_stage("lr", [0.001, 0.0005, 0.0001], config)
    config["lr"] = best_lr

    # Final validation: use best hyperparameters, train on harder task (basic/1.0)
    print("\n" + "="*70)
    print("FINAL TRAINING on basic/1.0 (grid size 11, max_steps=100)")
    print("="*70)

    final_config = config.copy()
    final_config.update({
        "complexity_level": 1.0,
        "test_complexity_level": 1.0,
        "epochs": 10000,          # longer final training
    })
    # Run 3 repeats and average
    final_rewards = []
    for rep in range(3):
        rep_cfg = final_config.copy()
        rep_cfg["seed"] = final_config["seed"] + rep * 12345
        reward = run_single(rep_cfg, timeout=7200)
        final_rewards.append(reward)

    import numpy as np
    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    print(f"\n🎉 FINAL RESULT (basic/1.0): {mean_reward:.2f} ± {std_reward:.2f}")

# ------------------------------------------------------------
# Command line interface
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help="Max parallel subprocesses per stage")
    parser.add_argument("--repeats", type=int, default=REPEATS_PER_CONFIG,
                        help="Number of repetitions per configuration")
    args = parser.parse_args()

    MAX_WORKERS = args.max_workers
    REPEATS_PER_CONFIG = args.repeats

    run_sequential_tuning()