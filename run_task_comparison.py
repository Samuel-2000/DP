#!/usr/bin/env python3
"""
run_task_comparison.py - Run task curriculum comparison experiments.
Usage: python run_task_comparison.py
"""

import subprocess
import sys
import os

# Ensure we're in the project root (where run.py is)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

PYTHON = sys.executable

def run_command(cmd):
    """Run a shell command and exit if it fails."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    print("=" * 50)
    print("Starting task curriculum comparison experiments")
    print("=" * 50)

    # Common arguments for all runs
    base_args = [
        PYTHON, "run.py", "train",
        "--network-type", "lstm",
        "--epochs", "20000",
        "--batch-size", "64",
        "--lr", "0.0002",
        "--dynamic-complexity",
        "--algorithm", "ppo",
        "--ppo-intra-epochs", "2",
        "--mini-batch-size", "64",
        "--grid-size", "19",
        "--max-steps", "200",
        "--auxiliary-tasks",
        "--experiment-name", "tasks_comparison_experiment"
    ]

    # Define the four experiment configurations
    experiments = [
        #{
        #    "name": "Doors only",
        #    "curriculum": ["--curriculum-stages", "doors"],
        #    "test": ["--test-task-class", "doors", "--test-complexity-level", "1.0"]
        #},
        {
            "name": "Buttons only",
            "curriculum": ["--curriculum-stages", "buttons"],
            "test": ["--test-task-class", "buttons", "--test-complexity-level", "1.0"]
        },
        {
            "name": "Complex only",
            "curriculum": ["--curriculum-stages", "complex"],
            "test": ["--test-task-class", "complex", "--test-complexity-level", "1.0"]
        },
        {
            "name": "Mixed (basic, doors, buttons, complex)",
            "curriculum": ["--curriculum-stages", "basic", "doors", "buttons", "complex"],
            "test": ["--test-task-class", "complex", "--test-complexity-level", "1.0"]
        }
    ]

    for exp in experiments:
        print(f"\n>>> Running experiment: {exp['name']}")
        cmd = base_args + exp["curriculum"] + exp["test"]
        run_command(cmd)

    print("\n" + "=" * 50)
    print("All task comparison experiments completed successfully.")
    print("=" * 50)

if __name__ == "__main__":
    main()