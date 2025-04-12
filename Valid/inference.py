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
from model import SemiconductorModel
from utils import stats_report

def preprocess_real_data(filepath, target_length=1002, column='q'):
    """
    Preprocess a real measurement file to match the training data format
    
    Args:
        filepath: Path to the CSV file
        target_length: Target sequence length for model input
        column: Column to use as feature (q by default)
        
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
            elif len(df.columns) >= 2 and column == 't':
                q_values = df.iloc[:, 0].values
            else:
                print(f"Warning: Could not find column {column}, using first column")
                q_values = df.iloc[:, 0].values
        
        # Handle sequence length discrepancy through interpolation
        current_length = len(q_values)
        
        if current_length != target_length:
            # Use numpy interpolation to resample to target length
            original_indices = np.arange(current_length)
            new_indices = np.linspace(0, current_length - 1, target_length)
            q_values = np.interp(new_indices, original_indices, q_values)
        
        return q_values
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        return None

def normalize_with_training_scaler(q_values, scaler_path):
    """
    Normalize data using the scaler fitted on training data
    
    Args:
        q_values: Array of q values
        scaler_path: Path to the saved scaler
        
    Returns:
        Normalized q values
    """
    try:
        # Load the scaler used during training
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        
        q_scaler = scalers.get('q')
        if q_scaler is not None:
            # Reshape for scaler transform
            q_values_reshaped = q_values.reshape(-1, 1)
            q_normalized = q_scaler.transform(q_values_reshaped).flatten()
            return q_normalized
        else:
            # Fall back to min-max if scaler not found
            print("Warning: 'q' scaler not found, falling back to min-max normalization")
            min_val = np.min(q_values)
            max_val = np.max(q_values)
            if max_val > min_val:
                return (q_values - min_val) / (max_val - min_val)
            else:
                return np.ones_like(q_values) * 0.5
    except Exception as e:
        print(f"Error during normalization: {e}")
        # Fall back to min-max normalization
        min_val = np.min(q_values)
        max_val = np.max(q_values)
        if max_val > min_val:
            return (q_values - min_val) / (max_val - min_val)
        else:
            return np.ones_like(q_values) * 0.5

def apply_min_max_normalization(q_values):
    """
    Apply min-max normalization to q values
    
    Args:
        q_values: Array of q values
        
    Returns:
        Normalized q values
    """
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

def run_inference_on_real_data(model, data_dir, output_dir, scaler_path=None, 
                              normalization_type='global', device='cpu', batch_size=32):
    """
    Run inference on real measurement data
    
    Args:
        model: Trained PyTorch model
        data_dir: Directory containing real measurement files
        output_dir: Directory to save results
        scaler_path: Path to the training scaler (optional)
        normalization_type: Type of normalization to use ('global', 'per_file', 'scaler')
        device: Device to run inference on
        batch_size: Batch size for inference
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files in the directory
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    total_files = len(file_paths)
    
    if total_files == 0:
        print(f"No CSV files found in {data_dir}")
        return None
    
    print(f"Found {total_files} files in {data_dir}")
    
    # Extract pattern for device and label (if available)
    pattern = re.compile(r'dev(\d+)_(\d+)\.csv')
    
    # Store all preprocessed data for batch inference
    all_data = []
    filenames = []
    true_labels = []
    
    # First pass: preprocess all files
    print("Preprocessing files...")
    for filepath in tqdm(file_paths):
        filename = os.path.basename(filepath)
        filenames.append(filename)
        
        # Extract label if available
        match = pattern.match(filename)
        label = int(match.group(2)) if match else None
        true_labels.append(label)
        
        # Preprocess the file
        q_values = preprocess_real_data(filepath, target_length=1002)
        
        if q_values is not None:
            all_data.append(q_values)
        else:
            # Skip this file if preprocessing failed
            print(f"Skipping {filename} due to preprocessing error")
            filenames.pop()
            true_labels.pop()
    
    # Check if we have any valid data
    if not all_data:
        print("No valid data found after preprocessing")
        return None
    
    # Normalize based on selected approach
    normalized_data = []
    if normalization_type == 'global':
        # Global min-max normalization across all files
        all_values = np.concatenate(all_data)
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        print(f"Global min: {global_min}, Global max: {global_max}")
        
        for q_values in all_data:
            if global_max > global_min:
                normalized = (q_values - global_min) / (global_max - global_min)
            else:
                normalized = np.ones_like(q_values) * 0.5
            normalized_data.append(normalized)
    
    elif normalization_type == 'scaler' and scaler_path:
        # Use saved scaler from training
        print(f"Using training scaler from {scaler_path}")
        for q_values in all_data:
            normalized = normalize_with_training_scaler(q_values, scaler_path)
            normalized_data.append(normalized)
    
    else:
        # Per-file normalization (default fallback)
        print("Using per-file min-max normalization")
        for q_values in all_data:
            normalized = apply_min_max_normalization(q_values)
            normalized_data.append(normalized)
    
    # Save a few examples of normalized data for inspection
    os.makedirs(os.path.join(output_dir, 'examples'), exist_ok=True)
    for i in range(min(5, len(normalized_data))):
        plt.figure(figsize=(10, 4))
        plt.plot(normalized_data[i])
        plt.title(f"Normalized data example: {filenames[i]}")
        plt.xlabel("Sample Index")
        plt.ylabel("Normalized Value")
        plt.savefig(os.path.join(output_dir, 'examples', f"example_{i}.png"))
        plt.close()
    
    # Run inference in batches
    all_predictions = []
    all_confidences = []
    
    print("Running inference...")
    for i in range(0, len(normalized_data), batch_size):
        batch_data = normalized_data[i:i+batch_size]
        
        # Convert to tensor
        batch_tensor = torch.FloatTensor(batch_data).unsqueeze(1).to(device)  # [batch, channels, sequence]
        
        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
        
        all_predictions.extend(predictions)
        all_confidences.extend(confidences)
    
    # Create results dataframe
    results = []
    for i in range(len(filenames)):
        results.append({
            'filename': filenames[i],
            'true_label': true_labels[i],
            'prediction': all_predictions[i],
            'confidence': all_confidences[i]
        })
    
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(output_dir, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Calculate metrics if true labels are available
    if all(label is not None for label in true_labels):
        calculate_metrics(results_df, output_dir)
    
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
        print("Warning: Confusion matrix is not 2x2, some classes may not be represented")
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape[0] == 1 and cm.shape[1] == 1:
            # Only one class predicted and present
            if true_labels[0] == 0:  # Only non-leaky class
                tn = cm[0, 0]
            else:  # Only leaky class
                tp = cm[0, 0]
        elif cm.shape == (2, 1):  # Two classes present, one class predicted
            if predictions[0] == 0:  # Only non-leaky predicted
                tn = cm[0, 0]  # True non-leaky correctly predicted
                fn = cm[1, 0]  # True leaky predicted as non-leaky
            else:  # Only leaky predicted
                fp = cm[0, 0]  # True non-leaky predicted as leaky
                tp = cm[1, 0]  # True leaky correctly predicted
        elif cm.shape == (1, 2):  # One class present, two classes predicted
            if true_labels[0] == 0:  # Only non-leaky present
                tn = cm[0, 0]  # True non-leaky correctly predicted
                fp = cm[0, 1]  # True non-leaky predicted as leaky
            else:  # Only leaky present
                fn = cm[0, 0]  # True leaky predicted as non-leaky
                tp = cm[0, 1]  # True leaky correctly predicted
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    
    # Calculate metrics from utils
    cm_list = [tp, fn, fp, tn]
    fb = stats_report(cm_list)
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall/Sensitivity: {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"F-Beta Score (β=2): {fb:.4f}\n\n")
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
    for i in range(2):
        for j in range(2):
            value = [[tn, fp], [fn, tp]][i][j]
            plt.text(j, i, str(value), ha="center", va="center", 
                     color="white" if value > (tp + tn + fp + fn)/4 else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    # Create confidence distribution plots
    plt.figure(figsize=(10, 6))
    
    # Separate by correctness
    correct = results_df[results_df['true_label'] == results_df['prediction']]
    incorrect = results_df[results_df['true_label'] != results_df['prediction']]
    
    plt.hist(correct['confidence'], bins=10, alpha=0.7, label='Correct Predictions')
    if len(incorrect) > 0:
        plt.hist(incorrect['confidence'], bins=10, alpha=0.7, label='Incorrect Predictions')
    
    plt.title('Model Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    
    conf_path = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(conf_path)
    plt.close()
    
    # Separate by true class
    plt.figure(figsize=(10, 6))
    
    leaky = results_df[results_df['true_label'] == 1]
    non_leaky = results_df[results_df['true_label'] == 0]
    
    if len(leaky) > 0:
        plt.hist(leaky['confidence'], bins=10, alpha=0.7, label='Leaky Devices')
    if len(non_leaky) > 0:
        plt.hist(non_leaky['confidence'], bins=10, alpha=0.7, label='Non-Leaky Devices')
    
    plt.title('Confidence by True Class')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    
    class_conf_path = os.path.join(output_dir, 'class_confidence.png')
    plt.savefig(class_conf_path)
    plt.close()
    
    print(f"Metrics and visualizations saved to {output_dir}")

def visualize_sample_data(data_dir, output_dir, use_column='q', max_samples=5):
    """
    Visualize sample data from real measurement files
    
    Args:
        data_dir: Directory containing real measurement files
        output_dir: Directory to save visualizations
        use_column: Column to visualize (q by default)
        max_samples: Maximum number of samples to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # Extract pattern for device and label
    pattern = re.compile(r'dev(\d+)_(\d+)\.csv')
    
    # Find some leaky and non-leaky samples
    leaky_samples = []
    non_leaky_samples = []
    
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        
        if match:
            label = int(match.group(2))
            if label == 1 and len(leaky_samples) < max_samples:
                leaky_samples.append(filepath)
            elif label == 0 and len(non_leaky_samples) < max_samples:
                non_leaky_samples.append(filepath)
        
        if len(leaky_samples) >= max_samples and len(non_leaky_samples) >= max_samples:
            break
    
    # Visualize samples
    for i, filepath in enumerate(leaky_samples):
        try:
            df = pd.read_csv(filepath, header=None)
            
            # Assign column names if not present
            if len(df.columns) >= 4:
                df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
            
            plt.figure(figsize=(12, 5))
            
            # Plot time vs charge if both columns exist
            if 't' in df.columns and 'q' in df.columns:
                plt.subplot(1, 2, 1)
                plt.plot(df['t'], df['q'])
                plt.title('Time vs Charge')
                plt.xlabel('Time')
                plt.ylabel('Charge')
            
            # Plot voltage vs charge (PV loop) if both columns exist
            if 'v' in df.columns and 'q' in df.columns:
                plt.subplot(1, 2, 2)
                plt.plot(df['v'], df['q'])
                plt.title('Voltage vs Charge (PV Loop)')
                plt.xlabel('Voltage')
                plt.ylabel('Charge')
            
            plt.suptitle(f"Leaky Sample: {os.path.basename(filepath)}")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"leaky_sample_{i}.png"))
            plt.close()
        except Exception as e:
            print(f"Error visualizing {filepath}: {e}")
    
    for i, filepath in enumerate(non_leaky_samples):
        try:
            df = pd.read_csv(filepath, header=None)
            
            # Assign column names if not present
            if len(df.columns) >= 4:
                df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
            
            plt.figure(figsize=(12, 5))
            
            # Plot time vs charge if both columns exist
            if 't' in df.columns and 'q' in df.columns:
                plt.subplot(1, 2, 1)
                plt.plot(df['t'], df['q'])
                plt.title('Time vs Charge')
                plt.xlabel('Time')
                plt.ylabel('Charge')
            
            # Plot voltage vs charge (PV loop) if both columns exist
            if 'v' in df.columns and 'q' in df.columns:
                plt.subplot(1, 2, 2)
                plt.plot(df['v'], df['q'])
                plt.title('Voltage vs Charge (PV Loop)')
                plt.xlabel('Voltage')
                plt.ylabel('Charge')
            
            plt.suptitle(f"Non-Leaky Sample: {os.path.basename(filepath)}")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"non_leaky_sample_{i}.png"))
            plt.close()
        except Exception as e:
            print(f"Error visualizing {filepath}: {e}")
    
    print(f"Sample visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run inference on real measurement data')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing real measurement files')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./real_data_results',
                      help='Directory to save results')
    parser.add_argument('--scaler_path', type=str, default=None,
                      help='Path to the training scaler (optional)')
    parser.add_argument('--normalization', type=str, default='per_file',
                      choices=['per_file', 'global', 'scaler'],
                      help='Normalization type to use')
    parser.add_argument('--visualize_samples', action='store_true',
                      help='Visualize sample data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize sample data if requested
    if args.visualize_samples:
        print("Visualizing sample data...")
        visualize_sample_dir = os.path.join(args.output_dir, 'sample_visualizations')
        visualize_sample_data(args.data_dir, visualize_sample_dir)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    device = torch.device(args.device)
    model = load_model(args.model_path, device)
    
    # Run inference
    print(f"Running inference with {args.normalization} normalization...")
    results_df = run_inference_on_real_data(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        scaler_path=args.scaler_path,
        normalization_type=args.normalization,
        device=device,
        batch_size=args.batch_size
    )
    
    print("Inference completed!")

if __name__ == "__main__":
    main()