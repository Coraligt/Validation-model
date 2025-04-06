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

from model import SemiconductorModel
from dataset import get_dataloaders, get_transforms
from utils import set_seed, stats_report, ACC, PPV, NPV, Sensitivity, Specificity, BAC, F1, FB

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

def evaluate_model(model, data_loader, device, criterion=None):
    """
    Comprehensive model evaluation
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        criterion: Loss function (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
            
            # Store raw outputs (before softmax) for ROC curve
            all_outputs.append(outputs.cpu().numpy())
            
            # Get predicted class
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    # Calculate softmax probabilities
    softmax_outputs = torch.nn.functional.softmax(torch.tensor(all_outputs), dim=1).numpy()
    pos_probs = softmax_outputs[:, 1]  # Probability of positive class (leaky)
    
    # Confusion matrix [TP, FN, FP, TN]
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    cm = [tp, fn, fp, tn]
    
    # Calculate metrics
    metrics = {
        'accuracy': ACC(cm),
        'precision': PPV(cm),
        'recall': Sensitivity(cm),
        'specificity': Specificity(cm),
        'npv': NPV(cm),
        'balanced_accuracy': BAC(cm),
        'f1': F1(cm),
        'fb': FB(cm),
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': pos_probs
    }
    
    # Calculate loss if criterion provided
    if criterion is not None:
        metrics['loss'] = total_loss / len(data_loader.dataset)
    
    return metrics

def plot_roc_curve(labels, probabilities, output_dir):
    """
    Plot ROC curve
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities for positive class
        output_dir: Directory to save the plot
    """
    fpr, tpr, _ = roc_curve(labels, probabilities)
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
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_precision_recall_curve(labels, probabilities, output_dir):
    """
    Plot Precision-Recall curve
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities for positive class
        output_dir: Directory to save the plot
    """
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

def main():
    """Main evaluation function"""
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
    
    args = parser.parse_args()
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.log_file)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    # model = SemiconductorModel(seq_length=args.seq_length).to(device)
    model = SemiconductorModel(
        seq_length=args.seq_length,
        conv_filters=3,  # Match the values from your tuning output
        fc1_size=80,     # Use the values you see in your tuning logs
        fc2_size=20,
        dropout1=0.3,
        dropout2=0.2
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Get data loader
    _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms=None  # No transforms for evaluation
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, test_loader, device, criterion)
    
    evaluation_time = time.time() - start_time
    logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Log metrics
    tp, fn, fp, tn = metrics['confusion_matrix']
    logger.info(f"Test Loss: {metrics.get('loss', 'N/A')}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision (PPV): {metrics['precision']:.4f}")
    logger.info(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']:.4f}")
    logger.info(f"Negative Predictive Value (NPV): {metrics['npv']:.4f}")
    logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"FB Score: {metrics['fb']:.4f}")
    logger.info(f"Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    plot_roc_curve(metrics['labels'], metrics['probabilities'], args.output_dir)
    plot_precision_recall_curve(metrics['labels'], metrics['probabilities'], args.output_dir)
    plot_confusion_matrix(metrics['confusion_matrix'], args.output_dir)
    
    # Save detailed results to CSV
    import pandas as pd
    results_df = pd.DataFrame({
        'true_label': metrics['labels'],
        'predicted_label': metrics['predictions'],
        'probability': metrics['probabilities']
    })
    results_df.to_csv(os.path.join(args.output_dir, 'evaluation_results.csv'), index=False)
    
    logger.info(f"Evaluation results saved to {args.output_dir}")

def plot_confusion_matrix(cm, output_dir):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix [TP, FN, FP, TN]
        output_dir: Directory to save the plot
    """
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
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()