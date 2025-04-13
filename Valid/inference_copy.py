#!/usr/bin/env python
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import argparse
from scipy import interpolate
from model import SemiconductorModel
from utils import stats_report

import pickle

def load_q_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers['q']  # Only need 'q'

    
def load_test_summary(summary_file):
    """
    Load test summary file containing labels and filenames
    
    Args:
        summary_file (str): Path to summary file
    
    Returns:
        dict: Dictionary mapping filenames to labels
    """
    file_labels = {}
    
    try:
        with open(summary_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    label = int(parts[0])
                    filename = parts[1]
                    # Add .csv extension if not already present
                    if not filename.endswith('.csv'):
                        filename = filename + '.csv'
                    # Only add this single entry to avoid duplicates
                    file_labels[filename] = label
        
        print(f"Loaded {len(file_labels)} file-label pairs from {summary_file}")
        # Print stats
        labels = list(file_labels.values())
        label_stats = {
            0: labels.count(0),
            1: labels.count(1)
        }
        print(f"Label distribution: Non-leaky (0): {label_stats[0]}, Leaky (1): {label_stats[1]}")
        
        return file_labels
    except Exception as e:
        print(f"Error loading test summary: {e}")
        return {}


def preprocess_real_data(filepath, target_length=1002, column='q', interp_method='linear'):
    """
    Preprocess a real measurement file to match the training data format
    
    Args:
        filepath: Path to the CSV file
        target_length: Target sequence length for model input
        column: Column to use as feature (q by default)
        interp_method: Interpolation method ('linear', 'cubic', 'pchip', or 'pv')
        
    Returns:
        Preprocessed q values ready for normalization
    """
    try:
        # Read CSV file
        df = pd.read_csv(filepath, header=None)
        
        # Assign column names if not present
        if len(df.columns) >= 4:
            df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
        elif len(df.columns) == 2:
            df.columns = ['t', 'q']
        
        # Extract q column (charge)
        q_values = None
        if column in df.columns:
            q_values = df[column].values
        else:
            # Fallback to position-based selection
            if len(df.columns) >= 3 and column == 'q':
                q_values = df.iloc[:, 2].values
            elif len(df.columns) >= 1 and column == 't':
                q_values = df.iloc[:, 0].values
            else:
                print(f"Warning: Could not find column {column}, using first column")
                q_values = df.iloc[:, 0].values
        
        # Handle sequence length discrepancy through interpolation
        current_length = len(q_values)
        
        if current_length == target_length:
            # No adjustment needed
            return q_values
            
        if interp_method == 'linear':
            # Linear interpolation (default)
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            q_resampled = np.interp(new_indices, original_indices, q_values)
            
        elif interp_method == 'cubic' and current_length > 3:
            try:
                # Cubic spline interpolation
                from scipy.interpolate import CubicSpline
                original_indices = np.arange(current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                cs = CubicSpline(original_indices, q_values)
                q_resampled = cs(new_indices)
            except Exception as e:
                print(f"Error with cubic interpolation: {e}, falling back to linear")
                original_indices = np.arange(current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                q_resampled = np.interp(new_indices, original_indices, q_values)
            
        elif interp_method == 'pchip' and current_length > 3:
            try:
                # PCHIP interpolation (preserves monotonicity)
                from scipy.interpolate import PchipInterpolator
                original_indices = np.arange(current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                pchip = PchipInterpolator(original_indices, q_values)
                q_resampled = pchip(new_indices)
            except Exception as e:
                print(f"Error with pchip interpolation: {e}, falling back to linear")
                original_indices = np.arange(current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                q_resampled = np.interp(new_indices, original_indices, q_values)
            
        elif interp_method == 'pv' and 't' in df.columns and 'v' in df.columns:
            try:
                # Special handling for PV loops - preserve relationship between v and q
                # First interpolate voltage to new time points
                t_values = df['t'].values
                v_values = df['v'].values
                
                # Create new time points
                new_t = np.linspace(t_values.min(), t_values.max(), target_length)
                
                # Interpolate v and q to new time points
                from scipy.interpolate import interp1d
                v_interp = interp1d(t_values, v_values, bounds_error=False, fill_value='extrapolate')
                q_interp = interp1d(t_values, q_values, bounds_error=False, fill_value='extrapolate')
                
                new_v = v_interp(new_t)
                new_q = q_interp(new_t)
                
                # Create interpolated PV relationship
                q_resampled = new_q
            except Exception as e:
                print(f"Error with pv interpolation: {e}, falling back to linear")
                original_indices = np.arange(current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                q_resampled = np.interp(new_indices, original_indices, q_values)
            
        else:
            # Fallback to linear interpolation
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            q_resampled = np.interp(new_indices, original_indices, q_values)
        
        return q_resampled
        
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        return None

def apply_scaler_or_norm(q_values, scaler_path=None, norm_type='per_file'):
    """
    Apply normalization to q values based on specified method
    
    Args:
        q_values: Array of q values
        scaler_path: Path to the saved scaler (optional)
        norm_type: Type of normalization ('per_file', 'scaler')
        
    Returns:
        Normalized q values
    """
    # Check if we should use saved scaler
    if norm_type == 'scaler' and scaler_path and os.path.exists(scaler_path):
        try:
            # Try to load the scaler
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                
            # Get the q scaler if it exists
            q_scaler = scalers.get('q')
            if q_scaler is not None:
                print(f"Using trained scaler from {scaler_path}")
                q_values_reshaped = q_values.reshape(-1, 1)
                q_normalized = q_scaler.transform(q_values_reshaped).flatten()
                return q_normalized
            else:
                print("Warning: 'q' scaler not found in scalers.pkl, falling back to per-file normalization")
        except Exception as e:
            print(f"Error using scaler: {e}, falling back to per-file normalization")
    
    # Apply per-file min-max normalization
    min_val = np.min(q_values)
    max_val = np.max(q_values)
    if max_val > min_val:
        return (q_values - min_val) / (max_val - min_val)
    else:
        # Edge case: all values are the same
        return np.ones_like(q_values) * 0.5

def load_model(model_path, device='cpu'):
    """
    Load the trained model
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Detect model architecture from state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Determine architecture parameters
    conv_filters = state_dict['conv1.weight'].shape[0] if 'conv1.weight' in state_dict else 3
    fc1_size = state_dict['fc1.weight'].shape[0] if 'fc1.weight' in state_dict else 20
    fc2_size = state_dict['fc2.weight'].shape[0] if 'fc2.weight' in state_dict else 10
    
    # Create model with detected architecture
    model = SemiconductorModel(
        seq_length=1002,
        conv_filters=conv_filters,
        fc1_size=fc1_size,
        fc2_size=fc2_size
    ).to(device)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded with: conv_filters={conv_filters}, fc1_size={fc1_size}, fc2_size={fc2_size}")
    return model

def run_inference(model, data_dir, file_labels, output_dir, device, 
                  scaler_path=None, norm_type='per_file', interp_method='linear',
                  batch_size=32):
    """
    Run inference on real measurement data
    
    Args:
        model: Trained PyTorch model
        data_dir: Directory containing real measurement files
        file_labels: Dictionary mapping filenames to labels
        output_dir: Directory to save results
        device: Device to run inference on
        scaler_path: Path to the training scaler (optional)
        norm_type: Type of normalization ('per_file', 'scaler')
        interp_method: Interpolation method ('linear', 'cubic', 'pchip', or 'pv')
        batch_size: Batch size for inference
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files to process
    all_files = list(file_labels.keys())
    total_files = len(all_files)
    
    if total_files == 0:
        print("No files to process")
        return {}
    
    # Print information about scaler and normalization
    if norm_type == 'scaler' and scaler_path:
        print(f"Will attempt to use scaler from {scaler_path}")
    else:
        print(f"Using {norm_type} normalization")
        
    # Print information about interpolation method
    print(f"Using {interp_method} interpolation for sequence length adjustment")
    
    # Initialize results
    results = []
    
    # Process all files
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1} ({len(batch_files)} files)")
        
        # Initialize lists for batch processing
        batch_inputs = []
        filenames = []
        true_labels = []
        
        # Preprocess each file
        for filename in tqdm(batch_files, desc="Preprocessing"):
            filepath = os.path.join(data_dir, filename)
            
            # Skip if file doesn't exist
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
            
            # Preprocess file
            q_values = preprocess_real_data(filepath, target_length=1002, 
                                           interp_method=interp_method)
            
            if q_values is None:
                print(f"Error preprocessing {filename}, skipping")
                continue
            
            # Apply normalization
            q_normalized = apply_scaler_or_norm(q_values, scaler_path, norm_type)
            
            # Add to batch
            batch_inputs.append(q_normalized.reshape(1, 1, -1))
            filenames.append(filename)
            true_labels.append(file_labels[filename])
        
        if not batch_inputs:
            print("No valid inputs in batch, skipping")
            continue
        
        # Convert to tensors
        inputs = torch.from_numpy(np.concatenate(batch_inputs, axis=0)).float().to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
            probs = probabilities.cpu().numpy()
        
        # Store results
        for j in range(len(filenames)):
            results.append({
                'filename': filenames[j],
                'true_label': true_labels[j],
                'prediction': int(predictions[j]),
                'confidence': float(confidences[j]),
                'leaky_prob': float(probs[j][1])
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_path = os.path.join(output_dir, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Calculate and save metrics
    calculate_metrics(results_df, output_dir)
    
    # Create visualizations
    create_visualizations(results_df, data_dir, output_dir)
    
    return results_df

def calculate_metrics(results_df, output_dir):
    """
    Calculate and save performance metrics
    
    Args:
        results_df: DataFrame with inference results
        output_dir: Directory to save metrics
    """
    # Extract true labels and predictions
    true_labels = results_df['true_label'].values
    predictions = results_df['prediction'].values
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Make sure it's a 2x2 matrix (binary classification)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle case where not all classes are represented
        tn, fp, fn, tp = 0, 0, 0, 0
        if true_labels.size > 0 and predictions.size > 0:
            # Calculate manually
            tp = np.sum((predictions == 1) & (true_labels == 1))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))
    
    # Calculate metrics
    cm_list = [tp, fn, fp, tn]
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    
    # Calculate metrics using utils functions
    fb = stats_report(cm_list)
    
    # Calculate additional metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F-Beta Score (β=2): {fb:.4f}\n")
        f.write(f"Sensitivity/Recall: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision/PPV: {ppv:.4f}\n")
        f.write(f"NPV: {npv:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow([[tn, fp], [fn, tp]], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-leaky (0)', 'Leaky (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2 if cm.size > 0 and cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            value = [[tn, fp], [fn, tp]][i][j]
            plt.text(j, i, str(value), ha="center", va="center",
                     color="white" if value > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

def create_visualizations(results_df, data_dir, output_dir):
    """
    Create visualizations of results
    
    Args:
        results_df: DataFrame with inference results
        data_dir: Directory containing data files
        output_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create confidence distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['confidence'], bins=20)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.savefig(os.path.join(vis_dir, 'confidence_distribution.png'))
    plt.close()
    
    # Create leaky probability distribution plot
    plt.figure(figsize=(10, 6))
    
    # Split by true label
    leaky = results_df[results_df['true_label'] == 1]
    non_leaky = results_df[results_df['true_label'] == 0]
    
    if len(leaky) > 0:
        plt.hist(leaky['leaky_prob'], bins=20, alpha=0.7, label='True Leaky')
    if len(non_leaky) > 0:
        plt.hist(non_leaky['leaky_prob'], bins=20, alpha=0.7, label='True Non-Leaky')
    
    plt.title('Leaky Probability Distribution')
    plt.xlabel('Probability of Being Leaky')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(vis_dir, 'leaky_probability.png'))
    plt.close()
    
    # Create ROC curve if sklearn is available
    try:
        from sklearn.metrics import roc_curve, auc
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(results_df['true_label'], results_df['leaky_prob'])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(vis_dir, 'roc_curve.png'))
        plt.close()
    except ImportError:
        print("sklearn.metrics ROC curve functions not available, skipping ROC curve")
    
    # Visualize some example files with predictions
    try:
        # Select a few examples of each category (TP, TN, FP, FN)
        results_df['correct'] = results_df['prediction'] == results_df['true_label']
        
        # True positives (correctly predicted leaky)
        tp_samples = results_df[(results_df['true_label'] == 1) & (results_df['prediction'] == 1)]
        # True negatives (correctly predicted non-leaky)
        tn_samples = results_df[(results_df['true_label'] == 0) & (results_df['prediction'] == 0)]
        # False positives (incorrectly predicted as leaky)
        fp_samples = results_df[(results_df['true_label'] == 0) & (results_df['prediction'] == 1)]
        # False negatives (incorrectly predicted as non-leaky)
        fn_samples = results_df[(results_df['true_label'] == 1) & (results_df['prediction'] == 0)]
        
        # Function to visualize a sample
        def visualize_sample(filepath, filename, pred_label, true_label, confidence, category):
            try:
                # Read CSV file
                df = pd.read_csv(filepath, header=None)
                
                # Assign column names
                if len(df.columns) >= 4:
                    df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
                elif len(df.columns) == 2:
                    df.columns = ['t', 'q']
                
                # Create plot
                plt.figure(figsize=(12, 10))
                
                # Plot time vs q if available
                if 't' in df.columns and 'q' in df.columns:
                    plt.subplot(2, 1, 1)
                    plt.plot(df['t'], df['q'])
                    plt.xlabel('Time')
                    plt.ylabel('Charge (q)')
                    plt.title('Time vs Charge')
                
                # Plot v vs q (PV loop) if available
                if 'v' in df.columns and 'q' in df.columns:
                    plt.subplot(2, 1, 2)
                    plt.plot(df['v'], df['q'])
                    plt.xlabel('Voltage (v)')
                    plt.ylabel('Charge (q)')
                    plt.title('Voltage vs Charge (PV Loop)')
                
                # Add prediction information
                pred_text = 'Leaky' if pred_label == 1 else 'Non-Leaky'
                true_text = 'Leaky' if true_label == 1 else 'Non-Leaky'
                plt.suptitle(f"{category}: {filename}\nTrue: {true_text}, Predicted: {pred_text}, Confidence: {confidence:.4f}")
                
                # Save plot
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(vis_dir, f"{category}_{os.path.basename(filename).replace('.csv', '')}.png"))
                plt.close()
            except Exception as e:
                print(f"Error visualizing {filepath}: {e}")
        
        # Visualize examples from each category
        categories = [
            ('TP', tp_samples), 
            ('TN', tn_samples), 
            ('FP', fp_samples), 
            ('FN', fn_samples)
        ]
        
        for category, samples in categories:
            if len(samples) > 0:
                # Get up to 3 samples
                for _, row in samples.head(3).iterrows():
                    filepath = os.path.join(data_dir, row['filename'])
                    if os.path.exists(filepath):
                        visualize_sample(
                            filepath=filepath,
                            filename=row['filename'],
                            pred_label=row['prediction'],
                            true_label=row['true_label'],
                            confidence=row['confidence'],
                            category=category
                        )
    except Exception as e:
        print(f"Error creating sample visualizations: {e}")
    
    print(f"Visualizations saved to {vis_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on real semiconductor data')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing real measurement files')
    parser.add_argument('--summary_file', type=str, required=True,
                        help='Path to summary file with labels')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save results')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='Path to the training scaler (optional)')
    parser.add_argument('--normalization', type=str, default='per_file',
                        choices=['per_file', 'scaler'],
                        help='Normalization method to use')
    parser.add_argument('--interpolation', type=str, default='linear',
                        choices=['linear', 'cubic', 'pchip', 'pv'],
                        help='Interpolation method to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load file labels from summary file
    file_labels = load_test_summary(args.summary_file)
    
    if not file_labels:
        print("No labels loaded. Please check the summary file.")
        return
    
    # If scaler path is not provided, try to find it
    if args.normalization == 'scaler' and args.scaler_path is None:
        # Try to find scalers.pkl in the same directory as the model
        model_dir = os.path.dirname(args.model_path)
        potential_scaler_path = os.path.join(model_dir, 'scalers.pkl')
        
        if os.path.exists(potential_scaler_path):
            args.scaler_path = potential_scaler_path
            print(f"Found scaler at {args.scaler_path}")
        else:
            # Try to find in indices directory
            for indices_dir in ['./indices', './Valid/indices', os.path.join(args.output_dir, '..', 'indices')]:
                potential_scaler_path = os.path.join(indices_dir, 'scalers.pkl')
                if os.path.exists(potential_scaler_path):
                    args.scaler_path = potential_scaler_path
                    print(f"Found scaler at {args.scaler_path}")
                    break
            
            if args.scaler_path is None:
                print("Warning: No scaler found. Using per-file normalization instead.")
                args.normalization = 'per_file'
    
    # Run inference
    results_df = run_inference(
        model=model,
        data_dir=args.data_dir,
        file_labels=file_labels,
        output_dir=args.output_dir,
        device=device,
        scaler_path=args.scaler_path,
        norm_type=args.normalization,
        interp_method=args.interpolation,
        batch_size=args.batch_size
    )
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()