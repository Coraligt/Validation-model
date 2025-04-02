import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from model import SemiconductorModel
from dataset import get_dataloaders
from utils import set_seed

def setup_output_dir(output_dir):
    """Create output directory for analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_model(model_path, seq_length=1002, device='cuda'):
    """Load a trained model from checkpoint"""
    model = SemiconductorModel(seq_length=seq_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def evaluate_detailed(model, dataloader, device):
    """Perform detailed evaluation of model"""
    y_true = []
    y_pred = []
    y_scores = []
    incorrect_samples = []
    
    eval_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store actual and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probabilities[:, 1].cpu().numpy())  # Positive class probability
            
            # Store indices of incorrectly classified samples
            incorrect_mask = (predicted != labels)
            if incorrect_mask.any():
                for j in range(len(labels)):
                    if incorrect_mask[j]:
                        incorrect_samples.append({
                            'batch_idx': i,
                            'sample_idx': j,
                            'true_label': labels[j].item(),
                            'predicted': predicted[j].item(),
                            'confidence': probabilities[j, predicted[j]].item()
                        })
    
    eval_time = time.time() - eval_time
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Results dictionary
    results = {
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'incorrect_samples': incorrect_samples,
        'evaluation_time': eval_time,
        'raw': {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_scores': y_scores,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision_curve': precision_curve.tolist(),
            'recall_curve': recall_curve.tolist()
        }
    }
    
    return results

def plot_results(results, output_dir):
    """Create visualizations for model results"""
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    cm_display = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    plt.imshow(cm_display, cmap='Blues')
    plt.colorbar()
    
    # Add labels and values
    classes = ['Non-Leaky (0)', 'Leaky (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            value = cm_display[i, j]
            plt.text(j, i, f"{int(value)}", 
                     ha="center", va="center", 
                     color="white" if value > cm_display.max()/2 else "black")
    
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(results['raw']['fpr'], results['raw']['tpr'], 
             label=f'ROC curve (AUC = {results["metrics"]["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(results['raw']['recall_curve'], results['raw']['precision_curve'],
             label=f'PR curve (AUC = {results["metrics"]["pr_auc"]:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()
    
    # 4. Metrics Summary
    plt.figure(figsize=(10, 6))
    metrics = results['metrics']
    metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    values = [metrics[m] for m in metrics_to_plot]
    
    plt.bar(metrics_to_plot, values)
    plt.ylim([0, 1.1])
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
    plt.close()
    
    # 5. Confidence Distribution
    plt.figure(figsize=(10, 6))
    scores = np.array(results['raw']['y_scores'])
    true_labels = np.array(results['raw']['y_true'])
    
    # Histogram for correct and incorrect predictions
    correct_mask = (np.array(results['raw']['y_pred']) == true_labels)
    
    plt.hist(scores[correct_mask & (true_labels==1)], alpha=0.5, bins=20, 
             label='Correct Leaky', color='green')
    plt.hist(scores[correct_mask & (true_labels==0)], alpha=0.5, bins=20, 
             label='Correct Non-Leaky', color='blue')
    plt.hist(scores[~correct_mask & (true_labels==1)], alpha=0.5, bins=20, 
             label='Incorrect Leaky', color='red')
    plt.hist(scores[~correct_mask & (true_labels==0)], alpha=0.5, bins=20, 
             label='Incorrect Non-Leaky', color='orange')
    
    plt.xlabel('Confidence Score (Probability of Leaky)')
    plt.ylabel('Count')
    plt.title('Distribution of Confidence Scores')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze model performance')
    
    # Model and data parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with preprocessed CSV files')
    parser.add_argument('--indices_dir', type=str, required=True,
                       help='Directory with indices CSV files')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Directory to save analysis results')
    parser.add_argument('--seq_length', type=int, default=1002,
                       help='Length of input sequence')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for evaluation')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Test parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = setup_output_dir(args.output_dir)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.seq_length, device)
    
    # Create dataloader
    print(f"Loading test data from {args.data_dir}")
    _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms=None  # No transforms for evaluation
    )
    
    # Evaluate model
    print("Performing detailed evaluation...")
    eval_start_time = time.time()
    results = evaluate_detailed(model, test_loader, device)
    eval_time = time.time() - eval_start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Save raw results
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = results.copy()
        # Remove raw data to make file smaller
        serializable_results.pop('raw', None)
        json.dump(serializable_results, f, indent=2)
    print(f"Saved detailed results to {results_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    plot_results(results, output_dir)
    print(f"Saved visualizations to {output_dir}")
    
    # Print summary metrics
    print("\nPerformance Summary:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"Sensitivity (Recall): {results['metrics']['sensitivity']:.4f}")
    print(f"Specificity: {results['metrics']['specificity']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"Precision-Recall AUC: {results['metrics']['pr_auc']:.4f}")
    
    # Print confusion matrix
    cm = results['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"True Positive (TP): {cm['tp']}")
    print(f"False Negative (FN): {cm['fn']}")
    print(f"False Positive (FP): {cm['fp']}")
    print(f"True Negative (TN): {cm['tn']}")
    
    # Print evaluation time
    print(f"\nEvaluation Time: {results['evaluation_time']:.2f} seconds")
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Performance Summary:\n")
        f.write(f"Accuracy: {results['metrics']['accuracy']:.4f}\n")
        f.write(f"Sensitivity (Recall): {results['metrics']['sensitivity']:.4f}\n")
        f.write(f"Specificity: {results['metrics']['specificity']:.4f}\n")
        f.write(f"Precision: {results['metrics']['precision']:.4f}\n")
        f.write(f"F1 Score: {results['metrics']['f1_score']:.4f}\n")
        f.write(f"ROC AUC: {results['metrics']['roc_auc']:.4f}\n")
        f.write(f"Precision-Recall AUC: {results['metrics']['pr_auc']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"True Positive (TP): {cm['tp']}\n")
        f.write(f"False Negative (FN): {cm['fn']}\n")
        f.write(f"False Positive (FP): {cm['fp']}\n")
        f.write(f"True Negative (TN): {cm['tn']}\n\n")
        
        f.write(f"Evaluation Time: {results['evaluation_time']:.2f} seconds\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Evaluated on: {device}\n")
    
    print(f"Saved summary to {summary_path}")
    print("\nAnalysis completed!")

if __name__ == '__main__':
    main()