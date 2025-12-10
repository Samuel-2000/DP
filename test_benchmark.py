#!/usr/bin/env python3
"""
test_benchmark.py - Test the benchmark functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.benchmark import Benchmark

def main():
    print("Testing benchmark functionality...")
    
    # Create benchmark instance
    benchmark = Benchmark(models_dir="models", output_dir="results/benchmarks")
    
    # Run benchmark (will work even if no models exist)
    results = benchmark.run(episodes_per_model=5, verbose=True)
    
    if results is not None:
        print("\nBenchmark completed successfully!")
        print(f"Benchmarked {len(results)} models")
        
        # Show top model
        if len(results) > 0:
            best = results.iloc[0]
            print(f"\nBest model: {best['model_file']}")
            print(f"  Reward: {best['avg_reward']:.2f}")
            print(f"  Success rate: {best['success_rate']:.1f}%")
    else:
        print("\nNo models found to benchmark.")
        print("Train some models first:")
        print("  python run.py train --network-type lstm --epochs 1000")
        print("  python run.py train --network-type transformer --epochs 1000")

if __name__ == "__main__":
    main()