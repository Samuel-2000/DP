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
    print("Starting hidden size experiments")
    print("=" * 50)

    # Experiment: Varying hidden sizes for LSTM network
    hidden_sizes = [128, 192, 256, 384, 1024]

    for hidden_size in hidden_sizes:
        print(f"\n>>> Running experiment with hidden_size = {hidden_size}")
        run_command([
            PYTHON, "run.py", "train",
            "--network-type", "lstm",
            "--epochs", "20000",
            "--batch-size", "64",
            "--lr", "0.0002",
            "--dynamic-complexity",
            "--curriculum-stages", "basic",
            "--test-task-class", "basic",
            "--test-complexity-level", "1.0",
            "--algorithm", "ppo",
            "--ppo-intra-epochs", "2",
            "--mini-batch-size", "64",
            "--grid-size", "19",
            "--max-steps", "200",
            "--auxiliary-tasks",
            "--hidden-size", str(hidden_size),
            "--experiment-name", "networks_experiment"
        ])

    print("\n" + "=" * 50)
    print("All experiments completed successfully.")
    print("=" * 50)

if __name__ == "__main__":
    main()