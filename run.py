#!/usr/bin/env python3
"""
Unified Runner for Memory Maze RL Experiments
"""

import sys
from pathlib import Path
from parser import parse_args

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.trainer import Trainer
from src.evaluation.benchmark import Benchmark
from src.evaluation.visualization import Visualizer
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
        # Load config
        config = load_config(args.config)
        
        # Override config with command line args
        if args.network_type:
            config["model"]["type"] = args.network_type
        if args.auxiliary_tasks:
            config["training"]["auxiliary_tasks"] = True
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
        agent = Agent.load(args.model)
        
        # Create environment with default parameters
        env = GridMazeWorld(
            grid_size=11,
            max_steps=100,
            obstacle_fraction=0.25,
            n_food_sources=4
        )
        
        # Run test
        test_results = agent.test(
            env=env,  # Pass the environment
            episodes=args.episodes,
            visualize=args.visualize,
            save_video=args.save_video
        )
        
        print(f"\nTest Results:")
        print(f"Average Reward: {test_results['avg_reward']:.2f}")
        print(f"Success Rate: {test_results['success_rate']:.1f}%")
        print(f"Average Steps: {test_results['avg_steps']:.1f}")
        
    elif args.command == "benchmark":
        benchmark = Benchmark(args.models_dir, args.output_dir)
        results = benchmark.run(args.benchmark_episodes)
        benchmark.save_results(results)
        benchmark.plot_results(results)
        
    elif args.command == "visualize":
        visualizer = Visualizer(args.model)
        visualizer.run(
            episodes=args.episodes,
            save_video=args.save_video,
            save_gif=args.save_gif
        )
        
    elif args.command == "compare":
        from experiments.compare import run_comparison
        
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