#!/usr/bin/env python3
"""
replot_experiments.py - Re-generate all training plots for existing experiments.
Usage: python tools/replot_experiments.py [--dry-run] [--verbose]

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2026)
"""

import sys
import subprocess
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

def find_metrics_dirs(root_dir: Path) -> list[Path]:
    """
    Find all directories that contain a 'metrics/metrics.npz' file.
    Returns a list of the timestamp directories (the parent of 'metrics').
    """
    metrics_dirs = []
    for metrics_npz in root_dir.glob("**/metrics/metrics.npz"):
        # The timestamp directory is the parent of the 'metrics' folder
        timestamp_dir = metrics_npz.parent.parent
        metrics_dirs.append(timestamp_dir)
    return metrics_dirs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-plot all experiment metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done.")
    parser.add_argument("--verbose", action="store_true", help="Print each command before running.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.resolve()
    experiments_dir = project_root / "experiments"

    if not experiments_dir.exists():
        print(f"❌ experiments directory not found: {experiments_dir}")
        return 1

    metrics_dirs = find_metrics_dirs(experiments_dir)
    if not metrics_dirs:
        print("⚠️  No metrics.npz files found in experiments/.")
        return 0

    print(f"Found {len(metrics_dirs)} experiment(s) with metrics.")
    python_exe = sys.executable
    run_py = project_root / "run.py"

    # Set environment to force UTF-8 encoding (fixes Unicode checkmark error on Windows)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    for exp_dir in sorted(metrics_dirs):
        if args.verbose or args.dry_run:
            print(f"Processing: {exp_dir.relative_to(project_root)}")
        if args.dry_run:
            continue

        cmd = [python_exe, str(run_py), "plot", "--metrics-path", str(exp_dir)]
        try:
            subprocess.run(cmd, check=True, capture_output=not args.verbose, env=env)
            if args.verbose:
                print(f"  ✅ Plots updated in {exp_dir}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed for {exp_dir}: {e}")
            if not args.verbose:
                stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else 'none'
                print(f"     stderr: {stderr}")

    print("\n✅ Re-plotting finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())