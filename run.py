#!/usr/bin/env python3
"""
Unified Runner for Memory Maze RL Experiments


TODOs:
    

"""

import sys
from pathlib import Path
from parser import parse_args

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.evaluation.benchmark import Benchmark
from src.core.utils import load_config


def main():
    args = parse_args()
    
    # Setup directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmarks").mkdir(parents=True, exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    Path("results/videos").mkdir(parents=True, exist_ok=True)
    
    if args.command == "train":
        from src.training.trainer import Trainer
        
        # Load config
        config = load_config(args.config)
        
        # Override config with command line args
        if args.network_type:
            config["model"]["type"] = args.network_type
        if args.auxiliary_tasks:
            config["training"]["auxiliary_tasks"] = True
            config["model"]["use_auxiliary"] = True
        if args.epochs:
            config["training"]["epochs"] = args.epochs
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.lr:
            config["training"]["learning_rate"] = args.lr
        
        # Create trainer and train
        trainer = Trainer(config)
        trainer.train()
        
    elif args.command == "test":
        from src.core.agent import Agent
        from src.core.environment import GridMazeWorld
        
        # Load agent
        print(f"Loading agent from {args.model}...")
        agent = Agent.load(args.model)
        
        # Create environment with default parameters
        env = GridMazeWorld(
            grid_size=11,
            max_steps=100,
            obstacle_fraction=0.25,
            n_food_sources=4,
            render_size=512
        )
        
        # Run test
        print(f"Testing agent for {args.episodes} episodes...")
        test_results = agent.test(
            env=env,
            episodes=args.episodes,
            visualize=args.visualize,
            save_video=args.save_video
        )
        
        print(f"\n{'='*50}")
        print("TEST RESULTS")
        print(f"{'='*50}")
        print(f"Average Reward: {test_results['avg_reward']:.2f}")
        print(f"Success Rate: {test_results['success_rate']:.1f}%")
        print(f"Average Steps: {test_results['avg_steps']:.1f}")
        print(f"Std Reward: {test_results['std_reward']:.2f}")
        print(f"{'='*50}")
        
    elif args.command == "benchmark":
        print(f"Running benchmark on models in {args.models_dir}...")
        
        benchmark = Benchmark(
            models_dir=args.models_dir,
            output_dir=args.output_dir
        )
        
        # Run benchmark
        results = benchmark.run(
            episodes_per_model=args.benchmark_episodes,
            verbose=True
        )
        
        if results is not None and not results.empty:
            # Show summary
            print(f"\n{'='*60}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*60}")
            
            # Print top models
            print("\nTop 5 models by average reward:")
            print(f"{'Model':<30} {'Reward':<10} {'Success':<10} {'Steps':<10}")
            print("-" * 70)
            
            for idx, row in results.head(5).iterrows():
                print(f"{row['model_file']:<30} "
                      f"{row['avg_reward']:>9.2f} "
                      f"{row['success_rate']:>9.1f}% "
                      f"{row['avg_steps']:>9.1f}")
        
    elif args.command == "visualize":
        from src.evaluation.visualization import Visualizer
        
        print(f"Loading model from {args.model}...")
        visualizer = Visualizer(args.model)
        
        # Run visualization
        visualizer.run(
            episodes=args.episodes,
            save_video=args.save_video,
            save_gif=args.save_gif
        )
        
    elif args.command == "compare":
        from experiments.compare import run_comparison
        
        print(f"Comparing architectures: {args.architectures}")
        
        run_comparison(
            architectures=args.architectures,
            epochs=args.epochs,
            trials=args.trials,
            output_dir=args.output_dir
        )
        
    else:
        print("Please specify a command. Use --help for usage information.")


if __name__ == "__main__":
    main()