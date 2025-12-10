import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory Maze RL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python run.py train --config configs/transformer.yaml
            python run.py train --network-type transformer --auxiliary-tasks
            python run.py test --model models/transformer_best.pt
            python run.py benchmark --benchmark-episodes 50
            python run.py visualize --model models/transformer_best.pt --save-video
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", 
                        type=str, 
                        default="configs/default.yaml",
                        help="Path to config file"
                    )
    train_parser.add_argument("--network-type", 
                        choices=["lstm", "transformer", "multimemory"],
                        default="lstm", 
                        help="Network architecture"
                    )
    train_parser.add_argument("--auxiliary-tasks", 
                        action="store_true",
                        help="Use auxiliary tasks"
                    )
    train_parser.add_argument("--epochs", 
                        type=int, 
                        default=10000,
                        help="Training epochs"
                    )
    train_parser.add_argument("--batch-size", 
                        type=int, 
                        default=64,
                        help="Batch size"
                    )
    train_parser.add_argument("--lr", 
                        type=float, 
                        default=0.0005,
                        help="Learning rate"
                    )
    train_parser.add_argument("--save-dir", 
                        type=str, 
                        default="models",
                        help="Directory to save models"
                    )
    train_parser.add_argument("--experiment-name", 
                        type=str, 
                        default=None,
                        help="Experiment name for logging"
                    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test a trained model")
    test_parser.add_argument("--model", 
                        type=str, 
                        required=True, 
                        help="Path to trained model"
                    )
    test_parser.add_argument("--episodes", 
                        type=int, 
                        default=10, 
                        help="Number of test episodes"
                    )
    test_parser.add_argument("--visualize", 
                        action="store_true", 
                        default=True, 
                        help="Show visualization"
                    )
    test_parser.add_argument("--save-video", 
                        action="store_true", 
                        help="Save test video",
                        default=True
                    )
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    bench_parser.add_argument("--models-dir", 
                        type=str, 
                        default="models",
                        help="Directory containing models"
                    )
    bench_parser.add_argument("--benchmark-episodes", 
                        type=int, 
                        default=20,
                        help="Episodes per model"
                    )
    bench_parser.add_argument("--output-dir", 
                        type=str, 
                        default="results/benchmarks",
                        help="Output directory for results"
                    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize model performance")
    viz_parser.add_argument("--model", 
                        type=str, 
                        required=True,
                        help="Path to model"
                    )
    viz_parser.add_argument("--episodes", 
                        type=int, 
                        default=3,
                        help="Number of episodes to visualize"
                    )
    viz_parser.add_argument("--save-video", 
                        action="store_true",
                        help="Save visualization as video"
                    )
    viz_parser.add_argument("--save-gif", 
                        action="store_true",
                        help="Save as GIF"
                    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare architectures")
    compare_parser.add_argument("--architectures", 
                        nargs="+",
                        default=["lstm", "transformer", "multimemory"],
                        help="Architectures to compare"
                    )
    compare_parser.add_argument("--epochs", 
                        type=int, 
                        default=5000,
                        help="Training epochs per architecture"
                    )
    compare_parser.add_argument("--trials", 
                        type=int, 
                        default=3,
                        help="Number of trials per architecture"
                    )
    compare_parser.add_argument("--output-dir", 
                        type=str, 
                        default="results/comparisons",
                        help="Output directory"
                    )
    
    return parser.parse_args()