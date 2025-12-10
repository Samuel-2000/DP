#!/usr/bin/env python3
"""
batch_train.py - Clean batch training using tqdm
"""

import os
import sys
import subprocess
import time
import argparse
import re
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm


def find_existing_model(model_dir, arch):
    """Find existing trained model, ignoring timestamp"""
    
    if not model_dir.exists():
        return None
    
    # Pattern to match: {arch}_*b_*lr_*_best.pt
    # We ignore the timestamp part
    pattern = f"{arch}_*b_*lr_*_best.pt"
    
    # Find all matching files
    model_files = list(model_dir.glob(pattern))
    
    if not model_files:
        # Also try final.pt files
        pattern_final = f"{arch}_*b_*lr_*_final.pt"
        model_files = list(model_dir.glob(pattern_final))
    
    if model_files:
        # Get the most recent one by modification time
        latest = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Check if it's a reasonable size (not corrupted)
        if latest.stat().st_size > 100_000:  # > 100KB
            return latest
    
    return None


def train_model(arch, auxiliary=False, epochs=10000, batch_size=64, lr=0.0005, 
                skip_if_exists=True, force=False, quiet=False):
    """Train a single model using run.py"""
    
    # Create save directory
    save_dir = Path("models") / arch
    if auxiliary:
        save_dir = Path("models") / f"{arch}_aux"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists (unless forcing)
    if not force and skip_if_exists:
        existing_model = find_existing_model(save_dir, arch)
        if existing_model:
            model_age = (datetime.now() - 
                        datetime.fromtimestamp(existing_model.stat().st_mtime)).total_seconds() / 3600
            print(f"✓ Model exists: {existing_model.name}")
            print(f"  Age: {model_age:.1f}h, Size: {existing_model.stat().st_size/1_000_000:.1f}MB")
            print(f"  Skipping {arch}{' (aux)' if auxiliary else ''}")
            return True, save_dir.name
    
    # Build experiment name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{arch}_{batch_size}b_{lr}lr_{timestamp}"
    if auxiliary:
        exp_name += "_aux"
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Training: {exp_name}")
        print(f"  Architecture: {arch}")
        print(f"  Auxiliary tasks: {auxiliary}")
        print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
        print(f"  Save dir: {save_dir}")
        print(f"{'='*60}")
    
    # Build command
    cmd = [
        sys.executable, "run.py", "train",
        "--network-type", arch,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--save-dir", str(save_dir),
        "--experiment-name", exp_name
    ]
    
    if auxiliary:
        cmd.append("--auxiliary-tasks")
    
    if quiet:
        print(f"Training {arch}{' (aux)' if auxiliary else ''}...")
    
    # Run training
    try:
        start_time = time.time()
        
        # Run subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Capture output but don't print all the tqdm updates
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            # Only print important messages
            if "epoch/s" in line and not quiet:
                # Extract progress from tqdm line
                if "Training:" in line:
                    # This is a tqdm progress line - we can show a simplified version
                    # Extract progress percentage if possible
                    match = re.search(r'(\d+)%\|', line)
                    if match:
                        percent = match.group(1)
                        sys.stdout.write(f"\rTraining: {percent}% complete")
                        sys.stdout.flush()
            elif "New best model" in line or "Saved model" in line:
                print(f"\n{line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            training_time = time.time() - start_time
            
            # Check if model was created
            model_files = list(save_dir.glob(f"{exp_name}_best.pt"))
            if model_files:
                if not quiet:
                    print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
                    print(f"✓ Model saved: {model_files[0].name}")
            else:
                print(f"\n⚠️  Warning: No best model file found")
            
            return True, exp_name
        else:
            print(f"\n✗ Training failed with exit code {process.returncode}")
            # Print last few lines of output for debugging
            if output_lines:
                print("Last 10 lines of output:")
                for line in output_lines[-10:]:
                    print(f"  {line.strip()}")
            return False, exp_name
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False, exp_name
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return False, exp_name


def list_existing_models():
    """List all existing trained models"""
    print("\n📁 Existing trained models:")
    print("-" * 50)
    
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found")
        return
    
    found_any = False
    
    # Check each architecture
    for arch in ['lstm', 'transformer', 'multimemory']:
        for auxiliary in [False, True]:
            save_dir = Path("models") / arch
            if auxiliary:
                save_dir = Path("models") / f"{arch}_aux"
            
            if not save_dir.exists():
                continue
            
            existing_model = find_existing_model(save_dir, arch)
            if existing_model:
                found_any = True
                model_age = (datetime.now() - 
                            datetime.fromtimestamp(existing_model.stat().st_mtime)).total_seconds() / 3600
                size_mb = existing_model.stat().st_size / 1_000_000
                
                aux_text = " (aux)" if auxiliary else ""
                print(f"{arch}{aux_text}/")
                print(f"  Model: {existing_model.name}")
                print(f"  Age: {model_age:.1f}h, Size: {size_mb:.1f}MB")
                print()
    
    if not found_any:
        print("No trained models found")


def main():
    parser = argparse.ArgumentParser(
        description="Batch train all model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_train.py                     # Train all 6 models
  python batch_train.py --list              # List existing models
  python batch_train.py --test              # Test existing models
  python batch_train.py --force             # Retrain all models
  python batch_train.py --arch lstm         # Train only LSTM
  python batch_train.py --quiet             # Quiet mode with progress
  python batch_train.py --epochs 1000       # Train for 1000 epochs
        
Architectures: lstm, transformer, multimemory
        """
    )
    
    parser.add_argument("--arch", 
                       type=str,
                       choices=['lstm', 'transformer', 'multimemory', 'all'],
                       default='all',
                       help="Architecture to train (default: all)")
    parser.add_argument("--auxiliary", 
                       action="store_true",
                       help="Train with auxiliary tasks")
    parser.add_argument("--no-auxiliary", 
                       action="store_true",
                       help="Train without auxiliary tasks")
    parser.add_argument("--epochs", 
                       type=int, 
                       default=10000,
                       help="Training epochs per model")
    parser.add_argument("--batch-size", 
                       type=int, 
                       default=64,
                       help="Batch size for training")
    parser.add_argument("--lr", 
                       type=float, 
                       default=0.0005,
                       help="Learning rate")
    parser.add_argument("--force", 
                       action="store_true",
                       help="Force retrain even if model exists")
    parser.add_argument("--skip-existing", 
                       action="store_true",
                       default=True,
                       help="Skip existing models (default: True)")
    parser.add_argument("--list", 
                       action="store_true",
                       help="List existing models without training")
    parser.add_argument("--quiet", 
                       action="store_true",
                       help="Quiet mode with minimal output")
    parser.add_argument("--test-after", 
                       action="store_true",
                       help="Test models after training")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    # List existing models
    if args.list:
        list_existing_models()
        return
    
    # Determine which experiments to run
    experiments = []
    
    architectures = []
    if args.arch == 'all':
        architectures = ['lstm', 'transformer', 'multimemory']
    else:
        architectures = [args.arch]
    
    # Determine auxiliary settings
    auxiliary_settings = []
    if args.auxiliary and not args.no_auxiliary:
        auxiliary_settings = [True]
    elif args.no_auxiliary and not args.auxiliary:
        auxiliary_settings = [False]
    else:
        # Default: train both with and without auxiliary
        auxiliary_settings = [False, True]
    
    # Create experiment list
    for arch in architectures:
        for auxiliary in auxiliary_settings:
            experiments.append((arch, auxiliary))
    
    if not args.quiet:
        print(f"""
╔═══════════════════════════════════════════════════╗
║          Maze RL - Batch Training                 ║
║          {len(experiments):1d} model{'s' if len(experiments) != 1 else ''}, {args.epochs} epochs each    ║
╚═══════════════════════════════════════════════════╝
    
Training parameters:
  Epochs:      {args.epochs}
  Batch size:  {args.batch_size}
  Learning rate: {args.lr}
  Force retrain: {args.force}
  Skip existing: {args.skip_existing}
  
Models to train:
""")
        
        for arch, auxiliary in experiments:
            print(f"  • {arch}{' (with auxiliary)' if auxiliary else ''}")
        
        print(f"\nEstimated time: ~{args.epochs * 0.3 / 60:.1f} minutes per model")
        print("Starting in 3 seconds...")
        time.sleep(3)
    
    # Train each model with tqdm progress
    results = []
    
    pbar = tqdm(experiments, desc="Training models", disable=args.quiet)
    for i, (arch, auxiliary) in enumerate(pbar, 1):
        if not args.quiet:
            pbar.set_description(f"Training {arch}{' (aux)' if auxiliary else ''}")
        else:
            print(f"[{i}/{len(experiments)}] Training {arch}{' (aux)' if auxiliary else ''}...")
        
        success, exp_name = train_model(
            arch=arch,
            auxiliary=auxiliary,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            skip_if_exists=args.skip_existing,
            force=args.force,
            quiet=args.quiet
        )
        
        results.append({
            "architecture": arch,
            "auxiliary": auxiliary,
            "success": success,
            "experiment": exp_name,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update progress bar
        if success:
            pbar.set_postfix({"status": "✓"})
        else:
            pbar.set_postfix({"status": "✗"})
        
        # Pause between trainings (unless it's the last one)
        if i < len(experiments) and success and not args.quiet:
            print(f"\n⏸️  Waiting 2 seconds before next training...")
            time.sleep(2)
    
    # Generate summary
    print(f"\n{'='*60}")
    print("BATCH TRAINING COMPLETE")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n✅ Successful: {len(successful)}/{len(experiments)}")
    for r in successful:
        aux_text = " (aux)" if r["auxiliary"] else ""
        print(f"  ✓ {r['architecture']}{aux_text}: {r['experiment']}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(experiments)}")
        for r in failed:
            aux_text = " (aux)" if r["auxiliary"] else ""
            print(f"  ✗ {r['architecture']}{aux_text}: {r['experiment']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("results") / f"batch_training_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "parameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "force": args.force,
                "skip_existing": args.skip_existing
            },
            "experiments": experiments,
            "results": results
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # Show next steps
    print(f"\n🎯 Next steps:")
    print(f"  1. Test a model: python run.py test --model models/lstm/lstm_*_best.pt")
    print(f"  2. Benchmark all: python run.py benchmark")
    print(f"  3. Visualize best: python run.py visualize --model models/lstm/lstm_*_best.pt")
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())