import os
import argparse
import subprocess
import torch
import platform
import time
import sys


def check_requirements():
    """Check if required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'sklearn': 'Scikit-learn'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {required_packages[package]} ({package})")
        print("\nPlease install required packages with:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("All required packages are installed")
    return True


def show_system_info():
    """Display system information"""
    print("\nSystem Information:")
    print(f"Python version: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes (version {torch.version.cuda})")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("CUDA available: No")
    
    print(f"PyTorch version: {torch.__version__}")
    print()


def run_prepare_data(args):
    """Run data preparation script for semiconductor data"""
    print("\n" + "="*40)
    print("Preparing Semiconductor Data")
    print("="*40)
    
    cmd = [
        sys.executable, "prepare_data.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir
    ]
    
    if args.analyze:
        cmd.append("--analyze")
    if args.preprocess:
        cmd.append("--preprocess")
    if args.normalize:
        cmd.append("--normalize")
    if args.create_indices:
        cmd.append("--create_indices")
    
    cmd.extend(["--test_ratio", str(args.test_ratio)])
    cmd.extend(["--seed", str(args.seed)])
    
    subprocess.run(cmd)


def run_train(args):
    """Run training script for semiconductor leakage detection model"""
    print("\n" + "="*40)
    print("Training Semiconductor Leakage Detection Model")
    print("="*40)
    
    cmd = [
        sys.executable, "train.py",
        "--data_dir", os.path.join(args.output_dir, "preprocessed"),
        "--indices_dir", args.output_dir,
        "--output_dir", os.path.join(args.output_dir, "models"),
        "--use_columns", args.use_columns,
        "--seq_length", str(args.seq_length),
        "--model_type", args.model_type,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--min_lr", str(args.min_lr),
        "--scheduler", args.scheduler,
        "--seed", str(args.seed),
        "--device", args.device,
        "--num_workers", str(args.num_workers)
    ]
    
    if args.use_time_reverse:
        cmd.append("--use_time_reverse")
    
    if args.use_swa:
        cmd.append("--use_swa")
        cmd.extend(["--swa_start", str(args.swa_start)])
        cmd.extend(["--swa_freq", str(args.swa_freq)])
        cmd.extend(["--swa_lr", str(args.swa_lr)])
    
    subprocess.run(cmd)


def run_test(args):
    """Run testing script for semiconductor leakage detection model"""
    print("\n" + "="*40)
    print("Testing Semiconductor Leakage Detection Model")
    print("="*40)
    
    # Find best model (prioritize FB score model)
    model_dir = os.path.join(args.output_dir, "models")
    model_path = os.path.join(model_dir, "best_model_fb.pth")
    
    # If FB score model doesn't exist, use best accuracy model
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "best_model_acc.pth")
    
    # If no best model, use SWA model
    if not os.path.exists(model_path) and args.use_swa:
        model_path = os.path.join(model_dir, "swa_model.pth")
    
    # If still no model, use final model
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.pth")
    
    if not os.path.exists(model_path):
        print("No trained model found. Please run training first.")
        return
    
    print(f"📂 Using model: {model_path}")
    
    cmd = [
        sys.executable, "test.py",
        "--data_dir", os.path.join(args.output_dir, "preprocessed"),
        "--indices_dir", args.output_dir,
        "--output_dir", os.path.join(args.output_dir, "test_results"),
        "--model_path", model_path,
        "--use_columns", args.use_columns,
        "--seq_length", str(args.seq_length),
        "--model_type", args.model_type,
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--num_workers", str(args.num_workers)
    ]
    
    if args.analyze_errors:
        cmd.append("--analyze_errors")
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Semiconductor Leakage Detection Complete Pipeline')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./dataset_3',
                        help='Directory containing original CSV files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output files')
    parser.add_argument('--use_columns', type=str, default='t,q',
                        help='Columns to use as features (comma separated)')
    parser.add_argument('--seq_length', type=int, default=1002,
                        help='Sequence length (rows in each CSV file)')
    
    # Data preparation parameters
    parser.add_argument('--analyze', action='store_true', default=True,
                        help='Analyze dataset')
    parser.add_argument('--preprocess', action='store_true', default=True,
                        help='Preprocess dataset')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize features')
    parser.add_argument('--create_indices', action='store_true', default=True,
                        help='Create train and test indices files')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='default', choices=['default', 'small'],
                        help='Model type to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0001,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Data augmentation parameters
    parser.add_argument('--use_time_reverse', action='store_true',
                        help='Use time reversal data augmentation')
    
    # SWA parameters
    parser.add_argument('--use_swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=10,
                        help='Epoch to start SWA')
    parser.add_argument('--swa_freq', type=int, default=5,
                        help='SWA update frequency')
    parser.add_argument('--swa_lr', type=float, default=0.0001,
                        help='SWA learning rate')
    
    # Test parameters
    parser.add_argument('--analyze_errors', action='store_true',
                        help='Analyze error predictions')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (e.g., cuda:0, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Flow control
    parser.add_argument('--skip_prepare', action='store_true',
                        help='Skip data preparation')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip testing')
    parser.add_argument('--system_info', action='store_true',
                        help='Display system information')
    
    args = parser.parse_args()
    
    # Check required packages
    if not check_requirements():
        return
    
    # Show system information
    if args.system_info:
        show_system_info()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Run workflow
    if not args.skip_prepare:
        run_prepare_data(args)
    
    if not args.skip_train:
        run_train(args)
    
    if not args.skip_test:
        run_test(args)
    
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    
    print("\n" + "="*40)
    print(f"Semiconductor Leakage Detection Pipeline Completed! Total time: {minutes}m {seconds}s")
    print("="*40)


if __name__ == '__main__':
    main()