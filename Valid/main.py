import argparse
import os
from prepare_data import prepare_data
from train import train

def main():
    """
    Main entry point for semiconductor leakage detection pipeline.
    Handles data preparation and model training/evaluation.
    """
    parser = argparse.ArgumentParser(description='Semiconductor Leakage Detection')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='./Valid/dataset_3',
                       help='Directory with original CSV files')
    parser.add_argument('--output_dir', type=str, default='./Valid',
                       help='Base output directory')
    parser.add_argument('--model_dir', type=str, default='./model_output',
                       help='Directory to save model checkpoints')
    
    # Data preprocessing
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip data preprocessing if already done')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Data augmentation
    parser.add_argument('--flip_signal', action='store_true', default=True,
                       help='Enable signal flipping augmentation')
    parser.add_argument('--add_noise', action='store_true', default=True,
                       help='Enable noise addition augmentation')
    parser.add_argument('--flip_time', action='store_true', default=False,
                       help='Enable time reversal augmentation')
    
    # SWA parameters
    parser.add_argument('--use_swa', action='store_true',
                       help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=10,
                       help='Epoch to start SWA from')
    parser.add_argument('--swa_freq', type=int, default=5,
                       help='SWA model collection frequency')
    parser.add_argument('--swa_lr', type=float, default=0.0001,
                       help='SWA learning rate')
    
    # Evaluation parameters
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate the model, no training')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the model to evaluate')
    
    # Logging
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to the log file')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training phase and only do preprocessing')

    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Data preparation
    if not args.skip_preprocessing:
        print("Step 1: Data Preparation")
        indices_dir = os.path.join(args.output_dir, 'indices')
        preprocessed_dir = os.path.join(indices_dir, 'preprocessed')
        
        prepare_data(
            data_dir=args.data_dir,
            output_dir=indices_dir,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:
        print("Skipping data preprocessing...")
        indices_dir = os.path.join(args.output_dir, 'indices')
        preprocessed_dir = os.path.join(indices_dir, 'preprocessed')
    
    # Model training/evaluation
    if not args.skip_training:
        print("\nStep 2: Model Training/Evaluation")
        
        # Create training arguments
        train_args = argparse.Namespace(
            data_dir=preprocessed_dir,
            indices_dir=indices_dir,
            output_dir=args.model_dir,
            seq_length=1002,  # Fixed for semiconductor data
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            workers=args.workers,
            device=args.device,
            no_cuda=args.no_cuda,
            flip_signal=args.flip_signal,
            add_noise=args.add_noise,
            flip_time=args.flip_time,
            use_swa=args.use_swa,
            swa_start=args.swa_start,
            swa_freq=args.swa_freq,
            swa_lr=args.swa_lr,
            evaluate_only=args.evaluate_only,
            model_path=args.model_path,
            log_file=args.log_file
        )
        
        # Train or evaluate model
        train(train_args)
    
    print("\nPipeline completed successfully!")

if __name__ == '__main__':
    main()