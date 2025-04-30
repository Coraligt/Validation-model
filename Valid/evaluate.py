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

import os
import pandas as pd
import seaborn as sns

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

# Enhanced confusion matrix function (add to both files)
def plot_confusion_matrix(cm, output_dir, filename='confusion_matrix.png'):
    """
    Create an enhanced visualization of confusion matrix with additional metrics
    
    Args:
        cm: Confusion matrix as [TP, FN, FP, TN]
        output_dir: Directory to save visualization
        filename: Filename for the saved visualization
    """
    tp, fn, fp, tn = cm
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    fbeta = (1 + 4) * (precision * sensitivity) / (4 * precision + sensitivity) if (4 * precision + sensitivity) > 0 else 0
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Confusion matrix subplot
    ax1 = plt.subplot(1, 2, 1)
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Leaky (0)', 'Leaky (1)'],
               yticklabels=['Non-Leaky (0)', 'Leaky (1)'], ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Metrics subplot
    ax2 = plt.subplot(1, 2, 2)
    metrics = {
        'Accuracy': accuracy,
        'Sensitivity/Recall': sensitivity,
        'Specificity': specificity,
        'Precision/PPV': precision,
        'NPV': npv,
        'F1 Score': f1,
        'F-β Score (β=2)': fbeta
    }
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(metrics)))
    ax2.barh(list(metrics.keys()), list(metrics.values()), color=colors)
    
    # Add text labels
    for i, (metric, value) in enumerate(metrics.items()):
        ax2.text(value + 0.01, i, f'{value:.4f}', va='center')
    
    ax2.set_title('Model Performance Metrics')
    ax2.set_xlim(0, 1.1)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    
    print(f"Confusion matrix saved to {os.path.join(output_dir, filename)}")

# Add 
def visualize_results(model, test_loader, device, output_dir):
    """
    Create detailed visualizations of model performance and sample predictions
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        device: Device to run inference on
        output_dir: Directory to save visualizations
    """
    # Create visualization subdirectory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store results
    all_labels = []
    all_probs = []
    all_preds = []
    
    # Store some sample data for detailed visualization
    leaky_samples = []
    non_leaky_samples = []
    fp_samples = []  # False positives
    fn_samples = []  # False negatives
    
    # Process test data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Convert to numpy for easier processing
            labels_np = labels.cpu().numpy()
            probs_np = probs[:, 1].cpu().numpy()  # Probability of class 1 (leaky)
            preds_np = preds.cpu().numpy()
            
            # Collect overall results
            all_labels.extend(labels_np)
            all_probs.extend(probs_np)
            all_preds.extend(preds_np)
            
            # Collect example samples (limit to first few we find)
            for j in range(len(labels_np)):
                sample = {
                    'input': inputs[j].cpu().numpy(),
                    'label': labels_np[j],
                    'pred': preds_np[j],
                    'prob': probs_np[j]
                }
                
                # Correctly classified leaky sample
                if labels_np[j] == 1 and preds_np[j] == 1 and len(leaky_samples) < 5:
                    leaky_samples.append(sample)
                
                # Correctly classified non-leaky sample
                elif labels_np[j] == 0 and preds_np[j] == 0 and len(non_leaky_samples) < 5:
                    non_leaky_samples.append(sample)
                
                # False positive (predicted leaky when it's not)
                elif labels_np[j] == 0 and preds_np[j] == 1 and len(fp_samples) < 5:
                    fp_samples.append(sample)
                
                # False negative (predicted non-leaky when it is leaky)
                elif labels_np[j] == 1 and preds_np[j] == 0 and len(fn_samples) < 5:
                    fn_samples.append(sample)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # 1. Create ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'roc_curve.png'))
    plt.close()
    
    # 2. Create Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'Precision-Recall curve (area = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # 3. Create probability distribution histogram
    plt.figure(figsize=(12, 6))
    
    # Separate by true label
    leaky_probs = all_probs[all_labels == 1]
    non_leaky_probs = all_probs[all_labels == 0]
    
    plt.hist(non_leaky_probs, bins=20, alpha=0.5, color='blue', label='Non-Leaky (0)')
    plt.hist(leaky_probs, bins=20, alpha=0.5, color='red', label='Leaky (1)')
    
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Decision Boundary')
    plt.xlabel('Leaky Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Predictions by True Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'probability_distribution.png'))
    plt.close()
    
    # 4. Visualize sample waveforms from each category
    def plot_sample(sample, category, sample_idx):
        # Reshape input back to original shape
        q_data = sample['input'][0]  # First channel (q values)
        
        plt.figure(figsize=(12, 8))
        
        # Create a dummy time array (since the actual time information is not retained)
        t = np.arange(len(q_data))
        
        # Plot Q vs T
        plt.subplot(2, 1, 1)
        plt.plot(t, q_data, 'b-')
        plt.title(f"{category} - True: {sample['label']}, Pred: {sample['pred']} (Prob: {sample['prob']:.4f})")
        plt.xlabel('Time (index)')
        plt.ylabel('Charge (Q) - Normalized')
        plt.grid(True, alpha=0.3)
        
        # Plot Q changes (like a phase plot)
        plt.subplot(2, 1, 2)
        plt.plot(q_data[:-1], q_data[1:], 'r-')
        plt.title('Q(t) vs Q(t+1) - Phase Plot')
        plt.xlabel('Q(t)')
        plt.ylabel('Q(t+1)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{category}_sample_{sample_idx}.png"))
        plt.close()
    
    # Plot samples from each category
    categories = [
        ("True_Positive", leaky_samples),
        ("True_Negative", non_leaky_samples),
        ("False_Positive", fp_samples),
        ("False_Negative", fn_samples)
    ]
    
    for category_name, samples in categories:
        for i, sample in enumerate(samples):
            plot_sample(sample, category_name, i)
    
    print(f"All visualizations saved to {vis_dir}")
    
    return all_labels, all_probs, all_preds

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
        # After evaluating model
        all_labels, all_probs, all_preds = visualize_results(model, test_loader, device, args.output_dir)

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