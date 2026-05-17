#!/usr/bin/env python3
"""
run_experiments.py - Run hyperparameter experiments sequentially.
Usage: python run_experiments.py
"""

import subprocess
import sys
import os

# Make sure we are in the project root (where run.py is)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PYTHON = sys.executable

def run_command(cmd):
    """Run a shell command and exit if it fails."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("=" * 50)
    print("Starting hyperparameter experiments")
    print("=" * 50)

    # ------------------------------------------------------------
    # Experiment 1: REINFORCE vs PPO
    # ------------------------------------------------------------
    print("\n>>> Experiment 1: REINFORCE vs PPO")
    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "5000", "--batch-size", "64", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "reinforce",
        "--experiment-name", "rf_vs_ppo_experiment"
    ])

    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "5000", "--batch-size", "64", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "1", "--mini-batch-size", "64",
        "--experiment-name", "rf_vs_ppo_experiment"
    ])

    # ------------------------------------------------------------
    # Experiment 2: PPO intra-epochs test (2, 4) - 1 already done above
    # ------------------------------------------------------------
    print("\n>>> Experiment 2: PPO intra-epochs test (2, 4)")
    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "5000", "--batch-size", "64", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "64",
        "--experiment-name", "ppo_intra_epochs_experiment"
    ])

    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "5000", "--batch-size", "64", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "4", "--mini-batch-size", "64",
        "--experiment-name", "ppo_intra_epochs_experiment"
    ])

    # ------------------------------------------------------------
    # Experiment 3: Mini-batch size test (256, 16) – baseline 64 already covered
    # ------------------------------------------------------------
    print("\n>>> Experiment 3: Mini-batch size test (256, 16)")
    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "5000", "--batch-size", "64", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "256",
        "--experiment-name", "mini_batch_size_experiment"
    ])

    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "5000", "--batch-size", "64", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "16",
        "--experiment-name", "mini_batch_size_experiment"
    ])

    # ------------------------------------------------------------
    # Experiment 4: Batch size test (32, 128)
    # ------------------------------------------------------------
    print("\n>>> Experiment 4: Batch size test (32, 128)")
    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "10000", "--batch-size", "32", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "64",
        "--experiment-name", "batch_size_experiment"
    ])

    run_command([
        PYTHON, "run.py", "train",
        "--network-type", "lstm", "--epochs", "2500", "--batch-size", "128", "--lr", "0.0005",
        "--task-class", "basic", "--complexity-level", "0.5",
        "--test-task-class", "basic", "--test-complexity-level", "1.0",
        "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "64",
        "--experiment-name", "batch_size_experiment"
    ])

    # ------------------------------------------------------------
    # Experiment 5: Optimizer comparison (adamw, sgd, rmsprop)
    # ------------------------------------------------------------
    print("\n>>> Experiment 5: Optimizer comparison")
    for opt in ["adamw", "sgd", "rmsprop"]:
        run_command([
            PYTHON, "run.py", "train",
            "--network-type", "lstm", "--epochs", "2500", "--batch-size", "128", "--lr", "0.0005",
            "--task-class", "basic", "--complexity-level", "0.5",
            "--test-task-class", "basic", "--test-complexity-level", "1.0",
            "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "64",
            "--optimizer", opt,
            "--experiment-name", "optimizer_experiment"
        ])

    # ------------------------------------------------------------
    # Experiment 6: Learning rate test (various values)
    # ------------------------------------------------------------
    print("\n>>> Experiment 6: Learning rate test")
    for lr in [0.001, 0.0001, 0.0002, 0.0003, 0.0004]:
        run_command([
            PYTHON, "run.py", "train",
            "--network-type", "lstm", "--epochs", "2500", "--batch-size", "128", "--lr", str(lr),
            "--task-class", "basic", "--complexity-level", "0.5",
            "--test-task-class", "basic", "--test-complexity-level", "1.0",
            "--algorithm", "ppo", "--ppo-intra-epochs", "2", "--mini-batch-size", "64",
            "--experiment-name", "learning_rate_experiment"
        ])

    print("\n" + "=" * 50)
    print("All experiments completed successfully.")
    print("=" * 50)

if __name__ == "__main__":
    main()