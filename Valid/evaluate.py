import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from tqdm import tqdm
import logging
from datetime import datetime
import sys
import traceback

from model import SemiconductorModel
from dataset import get_dataloaders, get_transforms
from utils import set_seed, stats_report, ACC, PPV, NPV, Sensitivity, Specificity, BAC, F1, FB

print("Starting evaluation script")

def setup_logging(output_dir, log_file=None):
    """Setup logging configuration"""
    # Create timestamp for log file if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"evaluation_{timestamp}.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file)  # File output
        ]
    )
    
    logging.info(f"Logging to {log_file}")
    return logging.getLogger()

def detect_model_architecture(model_path, device='cpu'):
    """
    Inspect a saved model to determine its architecture
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
    
    Returns:
        Dictionary with model architecture parameters
    """
    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Determine architecture from state dict keys and shapes
    architecture = {}
    
    # Get convolutional layer filters
    if 'conv1.weight' in state_dict:
        architecture['conv_filters'] = state_dict['conv1.weight'].shape[0]
    else:
        architecture['conv_filters'] = 3  # default
    
    # Get fully connected layer sizes
    if 'fc1.weight' in state_dict:
        architecture['fc1_size'] = state_dict['fc1.weight'].shape[0]
    else:
        architecture['fc1_size'] = 20  # default
    
    if 'fc2.weight' in state_dict:
        architecture['fc2_size'] = state_dict['fc2.weight'].shape[0]
    else:
        architecture['fc2_size'] = 10  # default
    
    # Use default dropout rates as they don't affect model loading
    architecture['dropout1'] = 0.3
    architecture['dropout2'] = 0.1
    
    # Use default sequence length
    architecture['seq_length'] = 1002
    
    return architecture

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on data loader and return metrics"""
    print("Running evaluate_model function...")
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0
    
    # Initialize confusion matrix counters
    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            # Update counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for p, l in zip(predicted, labels):
                if l.item() == 0:  # Non-leaky (negative class)
                    if p.item() == 0:
                        segs_TN += 1
                    else:
                        segs_FP += 1
                else:  # Leaky (positive class)
                    if p.item() == 1:
                        segs_TP += 1
                    else:
                        segs_FN += 1
    
    # Calculate metrics
    avg_loss = eval_loss / len(data_loader.dataset)
    acc = 100.0 * correct / total
    cm = [segs_TP, segs_FN, segs_FP, segs_TN]
    
    return avg_loss, acc, cm

def plot_confusion_matrix(cm, output_dir):
    """Plot confusion matrix"""
    tp, fn, fp, tn = cm
    
    # Reshape to 2x2 matrix for visualization
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-leaky (0)', 'Leaky (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    fmt = 'd'
    thresh = cm_matrix.max() / 2.
    for i in range(cm_matrix.shape[0]):
        for j in range(cm_matrix.shape[1]):
            plt.text(j, i, format(cm_matrix[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(description='Evaluate a semiconductor leakage detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with preprocessed CSV files')
    parser.add_argument('--indices_dir', type=str, required=True,
                       help='Directory with indices CSV files')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model file to evaluate')
    parser.add_argument('--seq_length', type=int, default=1002,
                       help='Length of input sequence')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for evaluation')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Logging parameters
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to save the log file')
    
    # Verbose output
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    
    try:
        # Create output directory
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup logging
        logger = setup_logging(args.output_dir, args.log_file)
        logger.info("Starting evaluation")
        
        # Log all arguments
        logger.info("Arguments:")
        for arg, value in vars(args).items():
            logger.info(f"  {arg}: {value}")
        
        # Set random seed
        print(f"Setting random seed: {args.seed}")
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Set device
        if args.no_cuda or not torch.cuda.is_available():
            device = torch.device("cpu")
            print("Using CPU for evaluation")
            logger.info("Using CPU for evaluation")
        else:
            device = torch.device(args.device)
            print(f"Using device: {device}")
            logger.info(f"Using device: {device}")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            err_msg = f"Model file not found: {args.model_path}"
            print(err_msg)
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        
        # Create model
        print("Creating model...")
        logger.info("Creating model...")
        
        # # To match the model architecture with what was used during training
        # model = SemiconductorModel(
        #     seq_length=args.seq_length,
        #     conv_filters=3,  # Default values
        #     fc1_size=20,
        #     fc2_size=10,
        #     dropout1=0.3,
        #     dropout2=0.1
        # ).to(device)
        
        # Detect architecture from saved model
        print(f"Detecting architecture from {args.model_path}...")
        logger.info(f"Detecting architecture from {args.model_path}...")
        architecture = detect_model_architecture(args.model_path, device)

        print(f"Detected architecture: {architecture}")
        logger.info(f"Detected architecture: {architecture}")

        # Create model with detected architecture
        model = SemiconductorModel(
            seq_length=architecture['seq_length'],
            conv_filters=architecture['conv_filters'],
            fc1_size=architecture['fc1_size'],
            fc2_size=architecture['fc2_size'],
            dropout1=architecture['dropout1'],
            dropout2=architecture['dropout2']
        ).to(device)

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
        
        # Load model weights
        print(f"Loading model from {args.model_path}")
        logger.info(f"Loading model from {args.model_path}")
        
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("Model loaded successfully")
            logger.info("Model loaded successfully")
        except Exception as e:
            err_msg = f"Failed to load model: {str(e)}"
            print(err_msg)
            logger.error(err_msg)
            raise
        
        # Get data loader
        print("Creating data loaders...")
        logger.info("Creating data loaders...")
        
        # Check if directories exist
        if not os.path.exists(args.data_dir):
            err_msg = f"Data directory not found: {args.data_dir}"
            print(err_msg)
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        
        if not os.path.exists(args.indices_dir):
            err_msg = f"Indices directory not found: {args.indices_dir}"
            print(err_msg)
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        
        try:
            # Get data loaders without transforms for evaluation
            loaders = get_dataloaders(
                data_dir=args.data_dir,
                indices_dir=args.indices_dir,
                batch_size=args.batch_size,
                num_workers=args.workers,
                transforms=None  # No transforms for evaluation
            )
            
            print("Data loaders created")
            logger.info("Data loaders created")
            
            # Determine which loader to use for evaluation
            if len(loaders) == 3:
                _, _, test_loader = loaders
                print("Using test loader from train/val/test split")
                logger.info("Using test loader from train/val/test split")
            else:
                _, test_loader = loaders
                print("Using test loader from train/test split")
                logger.info("Using test loader from train/test split")
            
        except Exception as e:
            err_msg = f"Failed to create data loaders: {str(e)}"
            print(err_msg)
            logger.error(err_msg)
            raise
        
        # Loss function for evaluation
        criterion = nn.CrossEntropyLoss()
        
        # Start timer for evaluation
        print("Starting evaluation...")
        logger.info("Starting evaluation...")
        eval_start_time = time.time()
        
        # Evaluate loaded model
        test_loss, test_acc, cm = evaluate_model(model, test_loader, criterion, device)
        fb = stats_report(cm)
        
        eval_time = time.time() - eval_start_time
        
        # Log results
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        logger.info(f"FB Score: {fb:.4f}")
        logger.info(f"Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")
        
        # Print results to console
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"FB Score: {fb:.4f}")
        print(f"Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")
        
        # Create visualization
        print("Creating visualization...")
        logger.info("Creating visualization...")
        plot_confusion_matrix(cm, args.output_dir)
        
        # Save results to CSV
        print("Saving detailed results...")
        import pandas as pd
        
        # Create a summary results file
        results_summary = {
            'metric': ['Loss', 'Accuracy', 'FB Score', 'TP', 'FN', 'FP', 'TN'],
            'value': [test_loss, test_acc, fb, cm[0], cm[1], cm[2], cm[3]]
        }
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(os.path.join(args.output_dir, 'evaluation_summary.csv'), index=False)
        
        print(f"Evaluation results saved to {args.output_dir}")
        logger.info(f"Evaluation results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Error during evaluation: {str(e)}")
            logger.error(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Uncaught exception: {str(e)}")
        traceback.print_exc()
        sys.exit(1)