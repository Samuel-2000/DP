#!/usr/bin/env python3
"""
run_experiments.py - Adaptive hyperparameter search with resume.
Uses recursive scan to locate experiment directories (robust against naming changes).
Logs all commands to commands_log.txt (plain commands, no timestamps).
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PYTHON = sys.executable

GLOBAL_BEST_FILE = Path("experiments/global_best_config.json")
COMPLETED_FILE = Path("experiments/completed_experiments.json")
COMMANDS_LOG_FILE = Path("experiments/commands_log.txt")

# Total number of environment steps for batch size experiments (batch_size * epochs = 320k)
TOTAL_STEPS = 320_000

DEFAULT_HYPERPARAMS = {
    "algorithm": "ppo",
    "batch_size": 64,
    "lr": 0.0005,
    "optimizer": "adam",
    "ppo_intra_epochs": 1,
    "mini_batch_size": 6400,
    "network_type": "lstm",
    "hidden_size": 512,
    "task_class": "basic",
    "complexity_level": 0.5,
    "grid_size": 11,
    "max_steps": 100,
    "dynamic_complexity": False,
    "curriculum_stages": ["basic"],
    "use_auxiliary": False,
}

def load_completed_experiments() -> set:
    if COMPLETED_FILE.exists():
        with open(COMPLETED_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()

def save_completed_experiment(exp_id: str):
    completed = load_completed_experiments()
    completed.add(exp_id)
    with open(COMPLETED_FILE, 'w') as f:
        json.dump({"completed": list(completed)}, f, indent=2)

def load_global_best() -> dict:
    if GLOBAL_BEST_FILE.exists():
        with open(GLOBAL_BEST_FILE, 'r') as f:
            return json.load(f)
    else:
        return DEFAULT_HYPERPARAMS.copy()

def save_global_best(hyperparams: dict):
    with open(GLOBAL_BEST_FILE, 'w') as f:
        json.dump(hyperparams, f, indent=2)

def find_most_recent_experiment_dir(experiment_name: str) -> Path:
    base = Path("experiments") / experiment_name
    if not base.exists():
        raise FileNotFoundError(f"Experiment base directory not found: {base}")
    candidates = []
    for dirpath in base.rglob("*"):
        if dirpath.is_dir() and (dirpath / "metrics" / "best_test_reward.txt").exists():
            mtime = dirpath.stat().st_mtime
            candidates.append((mtime, dirpath))
    if not candidates:
        raise FileNotFoundError(f"No experiment directory with best_test_reward.txt found under {base}")
    return max(candidates, key=lambda x: x[0])[1]

def update_global_best_from_experiment(exp_dir: Path):
    metrics_dir = exp_dir / "metrics"
    reward_file = metrics_dir / "best_test_reward.txt"
    config_file = metrics_dir / "config.json"
    if not reward_file.exists() or not config_file.exists():
        print(f"  Warning: No best reward or config found in {exp_dir}")
        return None
    with open(reward_file, 'r') as f:
        reward = float(f.read().strip())
    with open(config_file, 'r') as f:
        full_config = json.load(f)
    env = full_config.get('environment', {})
    model = full_config.get('model', {})
    training = full_config.get('training', {})
    hyperparams = {
        "algorithm": training.get('algorithm', 'ppo'),
        "batch_size": training.get('batch_size', 64),
        "lr": training.get('learning_rate', 0.0005),
        "optimizer": training.get('optimizer', 'adam'),
        "ppo_intra_epochs": training.get('ppo_intra_epochs', 1),
        "mini_batch_size": training.get('mini_batch_size', 6400),
        "network_type": model.get('type', 'lstm'),
        "hidden_size": model.get('hidden_size', 512),
        "task_class": env.get('task_class', 'basic'),
        "complexity_level": env.get('complexity_level', 0.5),
        "grid_size": env.get('grid_size', 11),
        "max_steps": env.get('max_steps', 100),
        "dynamic_complexity": training.get('dynamic_complexity', False),
        "curriculum_stages": training.get('curriculum_stages', ["basic"]),
        "use_auxiliary": model.get('use_auxiliary', False),
    }
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

def build_command(hyperparams: dict, experiment_name: str, epochs_override: int = None, resume_from: str = None) -> list:
    cmd = [PYTHON, "run.py", "train"]
    cmd.extend(["--network-type", hyperparams["network_type"]])
    cmd.extend(["--hidden-size", str(hyperparams["hidden_size"])])
    cmd.extend(["--batch-size", str(hyperparams["batch_size"])])
    cmd.extend(["--lr", str(hyperparams["lr"])])
    # Only add task_class and complexity_level if NOT dynamic complexity
    if not hyperparams.get("dynamic_complexity", False):
        cmd.extend(["--task-class", hyperparams["task_class"]])
        cmd.extend(["--complexity-level", str(hyperparams["complexity_level"])])
    cmd.extend(["--algorithm", hyperparams["algorithm"]])
    if hyperparams["algorithm"] == "ppo":
        cmd.extend(["--ppo-intra-epochs", str(hyperparams["ppo_intra_epochs"])])
        cmd.extend(["--mini-batch-size", str(hyperparams["mini_batch_size"])])
    if hyperparams["optimizer"] != "adam":
        cmd.extend(["--optimizer", hyperparams["optimizer"]])
    if hyperparams["grid_size"] != 11:
        cmd.extend(["--grid-size", str(hyperparams["grid_size"])])
    if hyperparams["max_steps"] != 100:
        cmd.extend(["--max-steps", str(hyperparams["max_steps"])])
    if hyperparams.get("dynamic_complexity", False):
        cmd.append("--dynamic-complexity")
        cmd.extend(["--curriculum-stages"] + hyperparams["curriculum_stages"])
    if hyperparams.get("use_auxiliary", False):
        cmd.append("--auxiliary-tasks")
    if epochs_override is not None:
        cmd.extend(["--epochs", str(epochs_override)])
    else:
        cmd.extend(["--epochs", "5000"])
    if resume_from is not None:
        cmd.extend(["--resume", resume_from])
    cmd.extend(["--experiment-name", experiment_name])
    return cmd

def log_command(command: list):
    if command and (command[0].endswith("python.exe") or command[0].endswith("python")):
        log_cmd = ["python"] + command[1:]
    else:
        log_cmd = command[:]
    with open(COMMANDS_LOG_FILE, 'a') as f:
        f.write(' '.join(log_cmd) + '\n')

def run_experiment(cmd, exp_id: str, experiment_name: str, dry_run: bool = False) -> bool:
    # Check if already completed
    if exp_id in load_completed_experiments():
        print(f"  Skipping already completed: {exp_id}")
        return True

    if dry_run:
        print(f"  [dry-run] Would run: {' '.join(cmd)}")
        log_command(cmd)
        return True

    print(f"\n>>> Running: {' '.join(cmd)}")
    log_command(cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  Experiment failed with error: {e}")
        return False

    try:
        exp_dir = find_most_recent_experiment_dir(experiment_name)
        reward = update_global_best_from_experiment(exp_dir)
        if reward is not None:
            save_completed_experiment(exp_id)
            return True
        else:
            return False
    except FileNotFoundError as e:
        print(f"  Warning: Could not locate experiment directory: {e}")
        return False

def run_incremental_complexity(base_params: dict, experiment_name: str, dry_run: bool) -> bool:
    """
    Run a sequence of training sessions:
    Start at complexity 0.0 for 500 epochs, then each step increase by 0.05,
    resume from the previous checkpoint, and train another 500 epochs.
    """
    exp_id = "bigger_grid_grid19_inc_comp"
    if exp_id in load_completed_experiments():
        print(f"  Skipping already completed: {exp_id}")
        return True

    start_complexity = 0.0
    end_complexity = 1.0
    step = 0.05
    epochs_per_step = 500
    total_steps = int((end_complexity - start_complexity) / step) + 1  # 21 steps from 0.0 to 1.0 inclusive
    print(f"  Running incremental complexity from {start_complexity} to {end_complexity} in steps of {step} (total {total_steps} runs)")

    prev_checkpoint = None
    for i in range(total_steps):
        complexity = start_complexity + i * step
        params = base_params.copy()
        params["complexity_level"] = complexity
        params["dynamic_complexity"] = False
        # Override grid size and max steps
        params["grid_size"] = 19
        params["max_steps"] = 200
        # Build command with resume if not first run
        cmd = build_command(params, experiment_name, epochs_override=epochs_per_step, resume_from=prev_checkpoint)
        print(f"  Step {i+1}/{total_steps}: complexity={complexity:.2f}, epochs={epochs_per_step}")
        if not dry_run:
            # Run the command
            cmd_log = [PYTHON, "run.py", "train"] + cmd[cmd.index("run.py")+2:]  # Remove the initial PYTHON for logging? Actually build_command returns full list.
            # But we have the full cmd already. We'll just run it.
            try:
                subprocess.run(cmd, check=True)
                # Find the checkpoint from this run to resume next time
                exp_dir = find_most_recent_experiment_dir(experiment_name)
                # The checkpoint is usually in weights/ subdirectory with a name like "final_checkpoint.pt" or "best_checkpoint.pt"
                # We need the most recent checkpoint file. Let's use "best_checkpoint.pt" as it is saved at each best.
                checkpoint_path = exp_dir / "weights" / "best_checkpoint.pt"
                if checkpoint_path.exists():
                    prev_checkpoint = str(checkpoint_path)
                else:
                    # Fallback: use final_checkpoint.pt
                    checkpoint_path = exp_dir / "weights" / "final_checkpoint.pt"
                    if checkpoint_path.exists():
                        prev_checkpoint = str(checkpoint_path)
                    else:
                        print(f"  Warning: No checkpoint found after step {i+1}")
                        return False
            except subprocess.CalledProcessError as e:
                print(f"  Step {i+1} failed: {e}")
                return False
        else:
            # Dry run: just log the command
            log_command(cmd)

    # After all steps, mark as completed
    if not dry_run:
        save_completed_experiment(exp_id)
    return True

def run_experiment_1(experiment_name: str, dry_run: bool) -> bool:
    print("\n>>> Experiment 1: REINFORCE vs PPO")
    params = DEFAULT_HYPERPARAMS.copy()
    params["algorithm"] = "reinforce"
    cmd = build_command(params, experiment_name)
    run_experiment(cmd, "exp1_reinforce", experiment_name, dry_run)

    params = DEFAULT_HYPERPARAMS.copy()
    params["algorithm"] = "ppo"
    cmd = build_command(params, experiment_name)
    run_experiment(cmd, "exp1_ppo", experiment_name, dry_run)

    if not dry_run:
        best = load_global_best()
        best_algorithm = best.get("algorithm", "unknown")
        best_reward = best.get("_best_test_reward", -float("inf"))
        print(f"\nExperiment 1 completed. Best algorithm: {best_algorithm} (reward {best_reward:.3f})")
        return best_algorithm == "ppo"
    else:
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    COMMANDS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Adaptive Hyperparameter Search")
    print("=" * 50)
    print(f"Dry run: {args.dry_run}")
    print(f"Force rerun: {args.force_rerun}")
    print("=" * 50)

    print("Completed experiments:", load_completed_experiments())

    ppo_wins = run_experiment_1("rf_vs_ppo_experiment", args.dry_run)

    if not ppo_wins:
        print("\nREINFORCE won. Skipping all remaining experiments (PPO-specific).")
        return

    print("\nPPO won. Proceeding with all experiments...")

    simple_experiments = [
        ("ppo_intra_epochs", [2, 4], "PPO intra-epochs test", None),
        ("mini_batch_size", [1600, 256, 64], "Mini-batch size test", None),
        ("batch_size", [(32, 10000), (128, 2500)], "Batch size test", "per_value"),
        ("optimizer", ["adamw", "sgd", "rmsprop"], "Optimizer comparison", "adaptive_epochs"),
        ("lr", [0.001, 0.0001, 0.0002, 0.0003, 0.0004, 0.0006, 0.0007, 0.0008, 0.0009], "Learning rate test", "adaptive_epochs"),
    ]

    for exp_key, values, description, epochs_info in simple_experiments:
        print(f"\n>>> {description}")
        base_hyperparams = load_global_best()
        for val in values:
            if exp_key == "batch_size":
                batch_val, epochs_val = val
                exp_id = f"{exp_key}_{batch_val}"
                if not args.force_rerun and exp_id in load_completed_experiments():
                    print(f"  Skipping already completed: {exp_id}")
                    continue
                params = base_hyperparams.copy()
                params[exp_key] = batch_val
                cmd = build_command(params, f"{exp_key}_experiment", epochs_override=epochs_val)
                run_experiment(cmd, exp_id, f"{exp_key}_experiment", args.dry_run)
            else:
                exp_id = f"{exp_key}_{val}"
                if not args.force_rerun and exp_id in load_completed_experiments():
                    print(f"  Skipping already completed: {exp_id}")
                    continue
                params = base_hyperparams.copy()
                params[exp_key] = val
                if epochs_info == "adaptive_epochs":
                    batch_size = params["batch_size"]
                    epochs_override = TOTAL_STEPS // batch_size
                else:
                    epochs_override = epochs_info if epochs_info != "per_value" else None
                cmd = build_command(params, f"{exp_key}_experiment", epochs_override=epochs_override)
                run_experiment(cmd, exp_id, f"{exp_key}_experiment", args.dry_run)

    print("\n>>> Bigger grid size experiment")
    base = load_global_best()
    base["grid_size"] = 19
    base["max_steps"] = 200

    # Standard variants (non‑incremental)
    variants = [
        {"dynamic": False, "complexity": 1.0, "epochs": 10000, "aux": False, "desc": "grid19_cl10"},
        {"dynamic": True,  "complexity": 1.0, "epochs": 10000, "aux": False, "desc": "grid19_dyn"},
        {"dynamic": True,  "complexity": 1.0, "epochs": 10000, "aux": True, "desc": "grid19_dyn_aux"},
    ]
    for v in variants:
        exp_id = f"bigger_grid_{v['desc']}"
        if not args.force_rerun and exp_id in load_completed_experiments():
            print(f"  Skipping already completed: {exp_id}")
            continue
        params = base.copy()
        params["dynamic_complexity"] = v["dynamic"]
        if not v["dynamic"]:
            params["complexity_level"] = v["complexity"]
        else:
            params["curriculum_stages"] = ["basic"]
        params["use_auxiliary"] = v["aux"]
        cmd = build_command(params, "bigger_grid_size_experiment", epochs_override=v["epochs"])
        run_experiment(cmd, exp_id, "bigger_grid_size_experiment", args.dry_run)

    # Incremental complexity variant (manual resume)
    print("\n>>> Running incremental complexity variant (0.0 → 1.0 in 0.05 steps every 500 epochs)")
    run_incremental_complexity(base, "bigger_grid_size_experiment", args.dry_run)

    print("\n>>> Networks experiment")
    base = load_global_best()
    base["grid_size"] = 19
    base["max_steps"] = 200
    base["dynamic_complexity"] = True
    base["curriculum_stages"] = ["basic"]
    base["use_auxiliary"] = True
    exp_id = "network_transformer"
    if not args.force_rerun and exp_id in load_completed_experiments():
        print(f"  Skipping already completed: {exp_id}")
    else:
        params = base.copy()
        params["network_type"] = "transformer"
        params["hidden_size"] = 512
        cmd = build_command(params, "networks_experiment", epochs_override=20000)
        run_experiment(cmd, exp_id, "networks_experiment", args.dry_run)

    exp_id = "network_lstm_hs768"
    if not args.force_rerun and exp_id in load_completed_experiments():
        print(f"  Skipping already completed: {exp_id}")
    else:
        params = base.copy()
        params["network_type"] = "lstm"
        params["hidden_size"] = 768
        cmd = build_command(params, "networks_experiment", epochs_override=20000)
        run_experiment(cmd, exp_id, "networks_experiment", args.dry_run)

    print("\n" + "=" * 50)
    print("All experiments processed.")
    print(f"Commands log: {COMMANDS_LOG_FILE}")
    print(f"Global best: {GLOBAL_BEST_FILE}")
    print(f"Completed list: {COMPLETED_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()