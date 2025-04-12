import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from model import SemiconductorModel

def load_model(model_path, device='cpu'):
    """Load the trained model from file"""
    print(f"Loading model from {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Detect architecture
    conv_filters = state_dict['conv1.weight'].shape[0] if 'conv1.weight' in state_dict else 3
    fc1_size = state_dict['fc1.weight'].shape[0] if 'fc1.weight' in state_dict else 20
    fc2_size = state_dict['fc2.weight'].shape[0] if 'fc2.weight' in state_dict else 10
    
    # Create model
    model = SemiconductorModel(
        seq_length=1002,
        conv_filters=conv_filters,
        fc1_size=fc1_size,
        fc2_size=fc2_size,
        dropout1=0.3,
        dropout2=0.1
    ).to(device)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully (conv_filters={conv_filters}, fc1_size={fc1_size}, fc2_size={fc2_size})")
    return model

def load_indices_file(indices_file):
    """Load test indices CSV file (label, filename)"""
    file_labels = {}
    
    try:
        with open(indices_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    label = int(row[0])
                    filename = row[1]
                    file_labels[filename] = label
        
        print(f"Loaded {len(file_labels)} file-label pairs from {indices_file}")
        
        # Print stats
        labels = list(file_labels.values())
        label_stats = {
            0: labels.count(0),
            1: labels.count(1)
        }
        print(f"Label distribution: Non-leaky (0): {label_stats[0]}, Leaky (1): {label_stats[1]}")
        
        return file_labels
    except Exception as e:
        print(f"Error loading indices file: {e}")
        return {}

def preprocess_file(filepath, target_length=1002, column_idx=2):
    """
    Preprocess a single file for inference
    
    Args:
        filepath (str): Path to the CSV file
        target_length (int): Target sequence length for model input
        column_idx (int): Index of column to use (default: 2 for q)
    """
    try:
        # Read CSV file (no header)
        df = pd.read_csv(filepath, header=None)
        
        # Extract column (default is q column at index 2)
        if column_idx < len(df.columns):
            data = df.iloc[:, column_idx].values
        else:
            print(f"Warning: Column index {column_idx} out of range for {filepath}. Using first column.")
            data = df.iloc[:, 0].values
        
        # Apply min-max normalization
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val > min_val:
            data = (data - min_val) / (max_val - min_val)
        else:
            data = np.ones_like(data) * 0.5
        
        # Handle sequence length
        current_length = len(data)
        
        if current_length == target_length:
            # No adjustment needed
            pass
        elif current_length > target_length:
            # Downsample using interpolation
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            data = np.interp(new_indices, original_indices, data)
        else:
            # Upsample using interpolation
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            data = np.interp(new_indices, original_indices, data)
        
        # Reshape for model input: [batch, channels, sequence]
        tensor = torch.FloatTensor(data).reshape(1, 1, target_length)
        return tensor
    
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        return None

def run_inference_on_test_set(model, data_dir, file_labels, output_dir, device, batch_size=32):
    """Run inference using the test_indices.csv file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files to process
    all_files = list(file_labels.keys())
    total_files = len(all_files)
    
    if total_files == 0:
        print("No files to process")
        return {}
    
    results = []
    
    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1} ({len(batch_files)} files)")
        
        inputs = []
        filenames = []
        true_labels = []
        
        # Preprocess each file
        for filename in tqdm(batch_files, desc="Preprocessing"):
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
            
            input_tensor = preprocess_file(filepath)
            
            if input_tensor is not None:
                inputs.append(input_tensor)
                filenames.append(filename)
                true_labels.append(file_labels[filename])
        
        if not inputs:
            print("No valid inputs in batch, skipping")
            continue
        
        # Stack inputs for batch processing
        batch_tensor = torch.cat(inputs, dim=0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
        
        # Store results
        for j in range(len(filenames)):
            results.append({
                'filename': filenames[j],
                'true_label': true_labels[j],
                'prediction': int(predictions[j]),
                'confidence': float(confidences[j])
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_path = os.path.join(output_dir, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    return results_df

def calculate_metrics(results_df, output_dir):
    """Calculate performance metrics"""
    true_labels = results_df['true_label'].values
    predictions = results_df['prediction'].values
    
    # Basic metrics
    accuracy = np.mean(true_labels == predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate additional metrics
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = sensitivity = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 1.0
    
    balanced_acc = (sensitivity + specificity) / 2
    
    # Calculate F-beta score (beta=2)
    beta = 2
    if precision + recall > 0:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f_beta = 0
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall/Sensitivity: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F-Beta Score (β=2): {f_beta:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall/Sensitivity: {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"F-Beta Score (β=2): {f_beta:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-leaky (0)', 'Leaky (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'f_beta': f_beta,
        'balanced_acc': balanced_acc,
        'confusion_matrix': cm
    }

def main():
    parser = argparse.ArgumentParser(description='Run inference with existing test_indices.csv')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing data files')
    parser.add_argument('--indices_file', type=str, required=True,
                       help='Path to test_indices.csv file')
    parser.add_argument('--output_dir', type=str, default='./test_inference_results',
                       help='Directory to save inference results')
    
    # Inference parameters
    parser.add_argument('--column_idx', type=int, default=2,
                       help='Index of column to use as feature (default: 2 for q)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cpu') if args.no_cuda else torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load test indices
    file_labels = load_indices_file(args.indices_file)
    
    if not file_labels:
        print("No file-label pairs loaded. Exiting.")
        return
    
    # Run inference
    results_df = run_inference_on_test_set(
        model=model,
        data_dir=args.data_dir,
        file_labels=file_labels,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results_df, args.output_dir)
    
    print("\nInference completed successfully!")

if __name__ == '__main__':
    main()