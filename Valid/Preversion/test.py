import os
import argparse
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import get_model
from dataset import get_dataloaders
from utils import evaluate_model, visualize_results, set_seed


def test_model(args):
    """
    Test model and generate evaluation metrics for semiconductor leakage detection
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get test dataloader
    _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        use_columns=args.use_columns.split(','),
        num_workers=args.num_workers
    )
    
    # Load model
    in_channels = len(args.use_columns.split(','))
    model = get_model(
        model_type=args.model_type,
        in_channels=in_channels,
        seq_length=args.seq_length
    )
    
    # Load model weights
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"FB Score: {metrics['fb_score']:.4f}")
    print(f"Sensitivity/Recall: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Confusion Matrix [TP, FN, FP, TN]: {metrics['confusion_matrix']}")
    print(f"ROC AUC: {metrics['roc']['auc']:.4f}")
    
    # Visualize results
    visualize_results(metrics, args.output_dir)
    
    # Save evaluation metrics
    metrics_summary = {
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'fb_score': metrics['fb_score'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'precision': metrics['precision'],
        'npv': metrics['npv'],
        'bac': metrics['bac'],
        'roc_auc': metrics['roc']['auc'],
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    # If specified, analyze prediction errors
    if args.analyze_errors:
        analyze_errors(model, test_loader, metrics, args.output_dir, device)
    
    print(f"\nTest results saved to {args.output_dir}")


def analyze_errors(model, dataloader, metrics, output_dir, device):
    """
    Analyze semiconductor samples where the model made incorrect predictions
    
    Args:
        model (nn.Module): Model
        dataloader (DataLoader): Test data loader
        metrics (dict): Evaluation metrics
        output_dir (str): Output directory
        device: Device
    """
    print("\nAnalyzing error predictions...")
    
    # Create error analysis directory
    error_dir = os.path.join(output_dir, 'error_analysis')
    os.makedirs(error_dir, exist_ok=True)
    
    # Get indices of error predictions
    errors_idx = np.where(metrics['predictions'] != metrics['labels'])[0]
    
    if len(errors_idx) == 0:
        print("No error predictions found!")
        return
    
    print(f"Found {len(errors_idx)} error predictions")
    
    # Collect error samples, labels and predictions
    error_samples = []
    error_labels = []
    error_preds = []
    error_probs = []
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            # Skip batches not in error indices
            batch_start = i * dataloader.batch_size
            batch_end = batch_start + inputs.size(0)
            batch_errors = [idx for idx in errors_idx if batch_start <= idx < batch_end]
            
            if not batch_errors:
                continue
                
            # Generate predictions
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Get error samples in batch
            for idx in batch_errors:
                local_idx = idx - batch_start
                error_samples.append(inputs[local_idx].cpu().numpy())
                error_labels.append(targets[local_idx].item())
                error_preds.append(preds[local_idx].item())
                error_probs.append(probs[local_idx].cpu().numpy())
    
    # Create visualizations of error samples
    num_samples = min(10, len(error_samples))  # Show at most 10 samples
    
    for i in range(num_samples):
        plt.figure(figsize=(10, 6))
        
        # Get current sample
        sample = error_samples[i]
        label = error_labels[i]
        pred = error_preds[i]
        prob = error_probs[i]
        
        # Plot sample data
        for j in range(sample.shape[0]):
            plt.plot(sample[j], label=f'Feature {j}')
        
        plt.title(f'Error Sample #{i+1}: True Label={label} ("{"Leaky" if label == 1 else "Non-leaky"}"), '
                 f'Prediction={pred} ("{"Leaky" if pred == 1 else "Non-leaky"}")\n'
                 f'Prediction Probabilities: Non-leaky={prob[0]:.4f}, Leaky={prob[1]:.4f}')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(error_dir, f'error_sample_{i+1}.png'))
        plt.close()
    
    # Create error summary
    error_summary = pd.DataFrame({
        'True_Label': error_labels,
        'Predicted': error_preds,
        'Non_Leaky_Prob': [p[0] for p in error_probs],
        'Leaky_Prob': [p[1] for p in error_probs]
    })
    
    # Analyze error types
    fn_count = sum(1 for l, p in zip(error_labels, error_preds) if l == 1 and p == 0)  # Leaky predicted as non-leaky
    fp_count = sum(1 for l, p in zip(error_labels, error_preds) if l == 0 and p == 1)  # Non-leaky predicted as leaky
    
    # Save error summary
    error_summary.to_csv(os.path.join(error_dir, 'error_summary.csv'), index=False)
    
    # Create error analysis report
    with open(os.path.join(error_dir, 'error_analysis.txt'), 'w') as f:
        f.write(f"Error Prediction Analysis\n")
        f.write(f"===================\n\n")
        f.write(f"Total errors: {len(errors_idx)}\n")
        f.write(f"False Negatives (FN): {fn_count} (Leaky predicted as non-leaky)\n")
        f.write(f"False Positives (FP): {fp_count} (Non-leaky predicted as leaky)\n\n")
        
        # Add label distribution
        f.write(f"True label distribution in error samples:\n")
        f.write(f"- Leaky (1): {sum(1 for l in error_labels if l == 1)}\n")
        f.write(f"- Non-leaky (0): {sum(1 for l in error_labels if l == 0)}\n\n")
        
        # Add probability analysis
        f.write(f"Prediction probability analysis of error samples:\n")
        f.write(f"- Average non-leaky probability: {np.mean([p[0] for p in error_probs]):.4f}\n")
        f.write(f"- Average leaky probability: {np.mean([p[1] for p in error_probs]):.4f}\n")
        f.write(f"- Low confidence predictions (highest prob < 0.7): {sum(1 for p in error_probs if max(p) < 0.7)}\n")
    
    print(f"Error analysis saved to {error_dir}")


def check_model_summary(args):
    """
    Print summary information about the model
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model
    in_channels = len(args.use_columns.split(','))
    model = get_model(
        model_type=args.model_type,
        in_channels=in_channels,
        seq_length=args.seq_length
    )
    
    # Print model structure
    print(model)
    
    # Load model weig