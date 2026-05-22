#!/usr/bin/env python3
"""
run_experiments.py - Run hyperparameter experiments sequentially, using the best
found hyperparameters from previous experiments as the base for later ones.
Supports resuming from crashes: completed experiments are recorded and skipped.
Usage: python run_experiments.py [--experiment-name NAME] [--epochs N] [--log-file PATH] [--dry-run]
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import argparse

# Make sure we are in the project root (where run.py is)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PYTHON = sys.executable

# ----------------------------------------------------------------------
# Global best configuration storage
# ----------------------------------------------------------------------
GLOBAL_BEST_FILE = Path("models/global_best_config.json")
COMPLETED_FILE = Path("models/completed_experiments.json")

# Default hyperparameters (used if no global best exists yet)
DEFAULT_HYPERPARAMS = {
    "algorithm": "ppo",
    "batch_size": 64,
    "lr": 0.0005,
    "optimizer": "adam",
    "gamma": 0.97,
    "entropy_coef": 0.01,
    "max_grad_norm": 1.0,
    "ppo_intra_epochs": 2,
    "mini_batch_size": 64,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "gae_lambda": 0.95,
    "network_type": "lstm",
    "hidden_size": 512,
    "use_auxiliary": False,
    "grid_size": 11,
    "max_steps": 100,
    "n_food_sources": 0,
    "food_energy": 10.0,
    "initial_energy": 30.0,
    "energy_decay": 0.98,
    "energy_per_step": 0.1,
    "dynamic_complexity": False,
    "task_class": "basic",
    "complexity_level": 0.5,
    "n_doors": 0,
    "n_buttons_per_door": 0,
    "button_break_probability": 0.0,
    # PPO specific defaults (if algorithm is ppo)
    "ppo_intra_epochs": 2,
    "mini_batch_size": 6400,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "gae_lambda": 0.95,
}

# Mapping from config key to command line argument
ARG_MAP = {
    "algorithm": "--algorithm",
    "batch_size": "--batch-size",
    "lr": "--lr",
    "optimizer": "--optimizer",
    "gamma": "--gamma",
    "entropy_coef": "--entropy-coef",
    "max_grad_norm": "--max-grad-norm",
    "ppo_intra_epochs": "--ppo-intra-epochs",
    "mini_batch_size": "--mini-batch-size",
    "clip_epsilon": "--clip-epsilon",
    "value_coef": "--value-coef",
    "gae_lambda": "--gae-lambda",
    "network_type": "--network-type",
    "hidden_size": "--hidden-size",
    "use_auxiliary": "--auxiliary-tasks",
    "grid_size": "--grid-size",
    "max_steps": "--max-steps",
    "n_food_sources": "--food-sources",
    "food_energy": "--food-energy",
    "initial_energy": "--initial-energy",
    "energy_decay": "--energy-decay",
    "energy_per_step": "--energy-per-step",
    "dynamic_complexity": "--dynamic-complexity",
    "task_class": "--task-class",
    "complexity_level": "--complexity-level",
    "n_doors": "--n-doors",
    "n_buttons_per_door": "--n-buttons-per-door",
    "button_break_probability": "--button-break-probability",
    "curriculum_stages": "--curriculum-stages",
    "performance_window": "--performance-window",
    "complexity_increase_threshold": "--complexity-increase-threshold",
    "complexity_decrease_threshold": "--complexity-decrease-threshold",
    "complexity_step": "--complexity-step",
    "min_complexity": "--min-complexity",
    "max_complexity": "--max-complexity",
    "adjustment_interval": "--adjustment-interval",
    "test_task_class": "--test-task-class",
    "test_complexity_level": "--test-complexity-level",
    "test_complexity_step": "--test-complexity-step",
    "test_complexity_range": "--test-complexity-range",
}


def load_completed_experiments() -> set:
    """Load the set of completed experiment identifiers."""
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_completed_experiment(exp_id: str):
    """Add an experiment ID to the completed set and save."""
    completed = load_completed_experiments()
    completed.add(exp_id)
    with open(COMPLETED_FILE, 'w') as f:
        json.dump({"completed": list(completed)}, f, indent=2)


def load_global_best() -> dict:
    """Load the best hyperparameters found so far. If none, return defaults."""
    if GLOBAL_BEST_FILE.exists():
        with open(GLOBAL_BEST_FILE, 'r') as f:
            return json.load(f)
    else:
        return DEFAULT_HYPERPARAMS.copy()


def save_global_best(hyperparams: dict):
    """Save new global best hyperparameters."""
    with open(GLOBAL_BEST_FILE, 'w') as f:
        json.dump(hyperparams, f, indent=2)


def update_global_best_from_experiment(exp_dir: Path):
    """
    Given an experiment directory (the timestamp folder), read its best_test_reward.txt
    and best_hyperparams.json. If the reward is higher than the current global best,
    update the global best file.
    """
    metrics_dir = exp_dir / "metrics"
    best_reward_file = metrics_dir / "best_test_reward.txt"
    best_hyperparams_file = metrics_dir / "best_hyperparams.json"

    if not best_reward_file.exists() or not best_hyperparams_file.exists():
        print(f"  Warning: No best reward/hyperparams found in {exp_dir}, skipping update.")
        return None

    with open(best_reward_file, 'r') as f:
        reward = float(f.read().strip())
    with open(best_hyperparams_file, 'r') as f:
        hyperparams = json.load(f)

    current_best = load_global_best()
    current_best_reward = current_best.get("_best_test_reward", -float("inf"))

    if reward > current_best_reward:
        hyperparams["_best_test_reward"] = reward
        hyperparams["_best_model_dir"] = str(exp_dir)
        save_global_best(hyperparams)
        print(f"  New global best! Reward = {reward:.3f}")
        return reward
    else:
        print(f"  Reward {reward:.3f} <= current best {current_best_reward:.3f}, no update.")
        return reward


def find_latest_experiment_dir(experiment_name: str, hyperparams: dict) -> Path:
    """
    Locate the most recent timestamp subdirectory for a given experiment run.
    The path structure is:
        models/<experiment_name>/<network_type>/<algorithm>/<optimizer>/<aux_str>/<model_name>/
    where aux_str = 'with_aux' if use_auxiliary else 'no_aux'
    and model_name = f"{batch_size}b_{lr}lr_gs{grid_size}" + (optional ppo suffix)
    """
    aux_str = "with_aux" if hyperparams.get("use_auxiliary", False) else "no_aux"
    model_name = f"{hyperparams['batch_size']}b_{hyperparams['lr']}lr_gs{hyperparams['grid_size']}"
    if hyperparams.get("algorithm") == "ppo":
        pie = hyperparams.get("ppo_intra_epochs", 2)
        mb = hyperparams.get("mini_batch_size", hyperparams["batch_size"])
        model_name += f"_pie{pie}_mb{mb}"
    base_dir = Path("models") / experiment_name / hyperparams["network_type"] / hyperparams["algorithm"] / hyperparams["optimizer"] / aux_str / model_name
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    # Find all timestamp subdirectories (format YYYY-MM-DD_HH-MM-SS)
    timestamps = [d for d in base_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not timestamps:
        raise FileNotFoundError(f"No timestamp directories found in {base_dir}")
    # Return the most recent (by name, since timestamp format is sortable)
    latest = max(timestamps)
    return latest


def build_command(base_args: list, hyperparams: dict, override_key: str = None, override_value=None) -> list:
    """
    Build a command list from base_args (e.g., ['python', 'run.py', 'train'])
    and a dictionary of hyperparameters. Each hyperparameter is converted to
    command line flags using ARG_MAP. Boolean flags are handled specially:
    if value is True, the flag is added without argument; if False, omitted.
    If override_key is given, that specific argument is overridden.
    """
    cmd = base_args.copy()
    for key, value in hyperparams.items():
        if key.startswith("_"):  # skip internal keys
            continue
        if key not in ARG_MAP:
            continue
        flag = ARG_MAP[key]
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(flag)
            cmd.append(str(value))
    # Apply override if specified
    if override_key is not None and override_key in ARG_MAP:
        # Remove existing occurrences of the flag and its value (if any)
        flag = ARG_MAP[override_key]
        i = 0
        while i < len(cmd):
            if cmd[i] == flag:
                # Remove flag
                del cmd[i]
                # If next token is not a flag (i.e., its value), remove it too
                if i < len(cmd) and not cmd[i].startswith("-"):
                    del cmd[i]
            else:
                i += 1
        # Add the new override
        cmd.append(flag)
        if isinstance(override_value, list):
            cmd.extend(str(v) for v in override_value)
        elif not isinstance(override_value, bool):
            cmd.append(str(override_value))
        elif override_value is True:
            # Boolean flag already added, no value needed
            pass
        # Note: if override_value is False, we simply omit the flag entirely
    return cmd


def log_experiment_run(log_path: Path, experiment_name: str, hyperparam_key: str, hyperparam_value,
                       best_reward: float, exp_dir: Path, success: bool = True):
    """Append a log entry for a completed experiment run."""
    timestamp = datetime.now().isoformat()
    entry = {
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "hyperparam_key": hyperparam_key,
        "hyperparam_value": hyperparam_value,
        "best_reward": best_reward if best_reward is not None else None,
        "exp_dir": str(exp_dir) if exp_dir is not None else None,
        "success": success,
    }
    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def run_command(cmd, log_path: Path, experiment_name: str, hyperparam_key: str, hyperparam_value,
                exp_id: str, dry_run: bool = False):
    """Run a shell command and log the result. Returns True if successful, False otherwise."""
    if dry_run:
        print(f"  [dry-run] Would run: {' '.join(cmd)}")
        # Mark as completed even in dry-run? No, because we didn't actually run.
        # But we return True to indicate it would have been ok (so we can skip in future dry runs? 
        # Better not mark completed on dry-run.)
        return True

    print(f"\n>>> Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Experiment failed with error: {e}")
        log_experiment_run(log_path, experiment_name, hyperparam_key, hyperparam_value,
                           best_reward=None, exp_dir=None, success=False)
        return False

    # After successful run, locate the experiment directory and get best reward
    best = load_global_best()  # the hyperparams that were used (before any potential update)
    try:
        exp_dir = find_latest_experiment_dir(experiment_name, best)
        reward = update_global_best_from_experiment(exp_dir)
        log_experiment_run(log_path, experiment_name, hyperparam_key, hyperparam_value,
                           best_reward=reward, exp_dir=exp_dir, success=True)
        # Mark this experiment as completed
        save_completed_experiment(exp_id)
        return True
    except FileNotFoundError as e:
        print(f"  Warning: Could not locate experiment directory: {e}")
        log_experiment_run(log_path, experiment_name, hyperparam_key, hyperparam_value,
                           best_reward=None, exp_dir=None, success=False)
        return False


def main():
    parser = argparse.ArgumentParser(description="Adaptive hyperparameter search with resume support")
    parser.add_argument("--experiment-name", type=str, default="adaptive_search",
                        help="Base name for the experiment folder (default: adaptive_search)")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Number of training epochs per experiment (default: 5000)")
    parser.add_argument("--log-file", type=str, default="models/experiment_log.txt",
                        help="Path to log file (default: models/experiment_log.txt)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print commands, do not execute")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Ignore completed experiments and rerun all (use with caution)")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Adaptive Hyperparameter Search (with resume)")
    print("=" * 50)
    print(f"Experiment name: {args.experiment_name}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Log file: {log_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Force rerun: {args.force_rerun}")
    print("=" * 50)

    # Base command (shared for all experiments)
    base_cmd = [PYTHON, "run.py", "train",
                "--epochs", str(args.epochs),
                "--experiment-name", args.experiment_name,
                "--test-task-class", "basic",
                "--test-complexity-level", "1.0",
                ]

    # Define all experiment configurations as tuples: (key, list_of_values, description)
    experiments = [
        ("batch_size", [32, 128], "Batch size test"),
        ("ppo_intra_epochs", [2, 4], "PPO intra-epochs test"),
        ("mini_batch_size", [1600, 640, 256, 16], "Mini-batch size test"),
        ("optimizer", ["adamw", "sgd", "rmsprop"], "Optimizer comparison"),
        ("lr", [0.001, 0.0001, 0.0002, 0.0003, 0.0004], "Learning rate test"),
    ]

    total_skipped = 0
    total_success = 0
    total_failed = 0

    for exp_key, values, description in experiments:
        print(f"\n>>> {description}")
        for val in values:
            exp_id = f"{exp_key}_{val}"
            if not args.force_rerun and exp_id in load_completed_experiments():
                print(f"  Skipping already completed: {exp_id}")
                total_skipped += 1
                continue

            print(f"\n--- Testing {exp_key} = {val} ---")
            best = load_global_best()
            cmd = build_command(base_cmd, best, override_key=exp_key, override_value=val)
            success = run_command(cmd, log_path, args.experiment_name, exp_key, val, exp_id,
                                  dry_run=args.dry_run)
            if success:
                total_success += 1
            else:
                total_failed += 1
                print(f"  Experiment {exp_id} failed. It will be retried on next run.")

    print("\n" + "=" * 50)
    print("All experiments processed.")
    print(f"  Completed successfully: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Skipped (already completed): {total_skipped}")
    print(f"Final global best configuration saved to {GLOBAL_BEST_FILE}")
    print(f"Full experiment log saved to {log_path}")
    print(f"Completed experiments list saved to {COMPLETED_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()