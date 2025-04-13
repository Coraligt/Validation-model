import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import csv
import re
import pickle

from model import SemiconductorModel


def load_global_scaler(indices_dir, column='q'):
    scaler_path = os.path.join(indices_dir, 'scalers.pkl')
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    return scalers[column]


def load_model(model_path, device='cpu'):
    """
    Load trained model from file
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to run inference on
    
    Returns:
        model: Loaded PyTorch model
    """
    print(f"Loading model from {model_path}")
    
    # Load state dict to detect architecture
    state_dict = torch.load(model_path, map_location=device)
    
    # Detect architecture
    conv_filters = state_dict['conv1.weight'].shape[0] if 'conv1.weight' in state_dict else 3
    fc1_size = state_dict['fc1.weight'].shape[0] if 'fc1.weight' in state_dict else 20
    fc2_size = state_dict['fc2.weight'].shape[0] if 'fc2.weight' in state_dict else 10
    
    # Create model with detected architecture
    model = SemiconductorModel(
        seq_length=1002,  # Default expected length, will be handled by preprocessing
        conv_filters=conv_filters,
        fc1_size=fc1_size,
        fc2_size=fc2_size,
        dropout1=0.3,
        dropout2=0.1
    ).to(device)
    
    # Load weights
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully (conv_filters={conv_filters}, fc1_size={fc1_size}, fc2_size={fc2_size})")
    return model

# Replace the load_test_summary function with this updated version
def load_test_indices(indices_file):
    """
    Load test indices CSV file containing labels and filenames
    
    Args:
        indices_file (str): Path to test indices CSV file
    
    Returns:
        dict: Dictionary mapping filenames to labels
    """
    file_labels = {}
    
    try:
        with open(indices_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # Skip header
            for row in csvreader:
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
        print(f"Error loading test indices: {e}")
        return {}
    
# In the load_test_summary function, modify to append '.csv' to each filename
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
                    filename = parts[1] + '.csv'  # Add .csv extension here
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


# Preprocess a single file for inference
def preprocess_file(filepath, target_length=1002, column='q'):
    """
    Preprocess a single file for inference
    
    Args:
        filepath (str): Path to the CSV file
        target_length (int): Target sequence length for model input
        column (str): Column to use as feature (q by default)
    
    Returns:
        tensor: Preprocessed tensor for model input
    """
    try:
        # Read CSV file (no header)
        df = pd.read_csv(filepath, header=None)
        
        # Assign column names based on expected columns (t, v, q, i)
        if len(df.columns) >= 4:
            df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
        elif len(df.columns) == 2:
            df.columns = ['t', 'q']
        
        # Extract q column (3rd column)
        q_values = None
        if column in df.columns:
            q_values = df[column].values
        else:
            # If column names don't match, use the 3rd column (index 2) for q
            if len(df.columns) >= 3:
                q_values = df.iloc[:, 2].values
            elif len(df.columns) >= 1:
                # Use whatever is available
                q_values = df.iloc[:, 0].values
                print(f"Warning: Using first column as charge for {filepath}")
        
        if q_values is None:
            print(f"Error: Could not extract data from {filepath}")
            return None
        
        # Apply min-max normalization to the file
        min_val = np.min(q_values)
        max_val = np.max(q_values)
        if max_val > min_val:
            q_values = (q_values - min_val) / (max_val - min_val)
        else:
            # If all values are the same, normalize to 0.5
            q_values = np.ones_like(q_values) * 0.5
        # Normalize q using global scaler
        
        # indices_dir = os.path.dirname(filepath)
        # scaler = load_global_scaler(indices_dir, column)
        # q_values = scaler.transform(q_values.reshape(-1, 1)).flatten()

        # Handle sequence length discrepancy
        current_length = len(q_values)
        
        if current_length == target_length:
            # No adjustment needed
            pass
        elif current_length > target_length:
            # Downsample using interpolation
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            q_values = np.interp(new_indices, original_indices, q_values)
        else:
            # Upsample using interpolation
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            q_values = np.interp(new_indices, original_indices, q_values)
        
        # Reshape for model input: [batch, channels, sequence]
        tensor = torch.FloatTensor(q_values).reshape(1, 1, target_length)
        
        return tensor
    
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        return None

#   Run inference on all files in directory
def run_inference(data_dir, model, file_labels, output_dir, device, batch_size=32):
    """
    Run inference on all files in directory
    
    Args:
        data_dir (str): Directory containing measured data files
        model: PyTorch model
        file_labels (dict): Dictionary mapping filenames to labels
        output_dir (str): Directory to save results
        device: PyTorch device
        batch_size (int): Batch size for inference
    
    Returns:
        dict: Dictionary with inference results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files to process
    all_files = list(file_labels.keys())
    total_files = len(all_files)
    
    if total_files == 0:
        print("No files to process")
        return {}
    
    # Initialize results
    results = []
    
    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files-1)//batch_size + 1} ({len(batch_files)} files)")
        
        # Initialize lists for batch processing
        inputs = []
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
    """
    Calculate performance metrics and save them
    
    Args:
        results_df (DataFrame): DataFrame with inference results
        output_dir (str): Directory to save results
    """
    # Extract true labels and predictions
    true_labels = results_df['true_label'].values
    predictions = results_df['prediction'].values
    
    # Calculate basic metrics
    accuracy = np.mean(true_labels == predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = sensitivity = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate additional metrics
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
    
    # Create visualization of confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['confidence'], bins=20)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    conf_path = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(conf_path)
    plt.close()
    
    # Create visualization for correct vs. incorrect predictions
    plt.figure(figsize=(12, 6))
    
    # Correct predictions
    correct = results_df[results_df['true_label'] == results_df['prediction']]
    plt.hist(correct['confidence'], bins=20, alpha=0.7, label='Correct Predictions')
    
    # Incorrect predictions
    incorrect = results_df[results_df['true_label'] != results_df['prediction']]
    if len(incorrect) > 0:
        plt.hist(incorrect['confidence'], bins=20, alpha=0.7, label='Incorrect Predictions')
    
    plt.title('Prediction Confidence by Correctness')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    acc_conf_path = os.path.join(output_dir, 'confidence_by_correctness.png')
    plt.savefig(acc_conf_path)
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

def create_sample_visualizations(results_df, data_dir, output_dir, num_samples=5):
    """
    Create visualizations of sample predictions
    
    Args:
        results_df (DataFrame): DataFrame with inference results
        data_dir (str): Directory containing data files
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    # Create directory for sample visualizations
    samples_dir = os.path.join(output_dir, 'sample_visualizations')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Select random samples (stratified by correct/incorrect and leaky/non-leaky)
    correct = results_df[results_df['true_label'] == results_df['prediction']]
    incorrect = results_df[results_df['true_label'] != results_df['prediction']]
    
    # Correct predictions (both leaky and non-leaky)
    correct_leaky = correct[correct['true_label'] == 1]
    correct_non_leaky = correct[correct['true_label'] == 0]
    
    # Incorrect predictions (both leaky and non-leaky)
    incorrect_leaky = incorrect[incorrect['true_label'] == 1]
    incorrect_non_leaky = incorrect[incorrect['true_label'] == 0]
    
    # Sample from each category
    samples = []
    
    # Function to sample from a dataframe
    def sample_from_df(df, n):
        if len(df) <= n:
            return df
        else:
            return df.sample(n)
    
    # Add samples from each category
    samples.append(sample_from_df(correct_leaky, num_samples))
    samples.append(sample_from_df(correct_non_leaky, num_samples))
    samples.append(sample_from_df(incorrect_leaky, num_samples))
    samples.append(sample_from_df(incorrect_non_leaky, num_samples))
    
    # Combine all samples
    samples_df = pd.concat(samples)
    
    # Create visualizations for each sample
    for _, row in samples_df.iterrows():
        filename = row['filename']
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath, header=None)
            
            # Assign column names based on expected columns
            if len(df.columns) >= 4:
                df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
            elif len(df.columns) == 2:
                df.columns = ['t', 'q']
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot time vs charge if both columns exist
            if 't' in df.columns and 'q' in df.columns:
                plt.subplot(2, 1, 1)
                plt.plot(df['t'], df['q'])
                plt.xlabel('Time (t)')
                plt.ylabel('Charge (q)')
                plt.title('Time vs Charge')
            
            # Plot voltage vs charge if both columns exist (PV loop)
            if 'v' in df.columns and 'q' in df.columns:
                plt.subplot(2, 1, 2)
                plt.plot(df['v'], df['q'])
                plt.xlabel('Voltage (v)')
                plt.ylabel('Charge (q)')
                plt.title('Voltage vs Charge (PV Loop)')
            
            # Add prediction information
            correct_or_incorrect = "Correct" if row['true_label'] == row['prediction'] else "Incorrect"
            true_label = "Leaky" if row['true_label'] == 1 else "Non-Leaky"
            pred_label = "Leaky" if row['prediction'] == 1 else "Non-Leaky"
            
            plt.suptitle(f"Sample: {filename}\n" +
                         f"{correct_or_incorrect} Prediction: True: {true_label}, Pred: {pred_label}, Conf: {row['confidence']:.4f}")
            
            plt.tight_layout()
            
            # Save figure
            sample_filename = filename.replace('.csv', '').replace('.', '_')
            fig_path = os.path.join(samples_dir, f"sample_{sample_filename}.png")
            plt.savefig(fig_path)
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization for {filepath}: {e}")
    
    print(f"Sample visualizations saved to {samples_dir}")


def convert_summary_to_indices(summary_file, output_dir):
    """
    Convert summary file to indices files for compatibility with other scripts
    
    Args:
        summary_file (str): Path to summary file
        output_dir (str): Directory to save indices files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_labels = {}
    
    # Read summary file
    with open(summary_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                label = int(parts[0])
                filename = parts[1] + '.csv'  # Add .csv extension here
                file_labels[filename] = label
    
    # Write test indices file
    with open(os.path.join(output_dir, 'test_indices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'Filename'])  # header
        for filename, label in file_labels.items():
            writer.writerow([label, filename])
    
    print(f"Converted summary to indices file in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on measured semiconductor data')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing measured data files')
    parser.add_argument('--summary_file', type=str, required=True,
                       help='Path to summary file with labels')
    parser.add_argument('--output_dir', type=str, default='./measured_data_results',
                       help='Directory to save inference results')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--feature_column', type=str, default='q',
                       help='Column to use as feature (default: q)')
    
    # Visualization parameters
    parser.add_argument('--create_samples', action='store_true',
                       help='Create sample visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize per category')
    
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
    
    # Load test summary
    file_labels = load_test_summary(args.summary_file)
    
    if not file_labels:
        print("No file-label pairs loaded. Exiting.")
        return
    
    # Convert summary to indices file for compatibility
    convert_summary_to_indices(args.summary_file, args.output_dir)
    
    # Run inference
    results_df = run_inference(
        data_dir=args.data_dir,
        model=model,
        file_labels=file_labels,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results_df, args.output_dir)
    
    # Create sample visualizations if requested
    if args.create_samples:
        create_sample_visualizations(
            results_df=results_df,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
    
    print("\nInference on measured data completed successfully!")

if __name__ == '__main__':
    main()