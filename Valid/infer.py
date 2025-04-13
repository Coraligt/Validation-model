import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pickle
from scipy import interpolate

from model import SemiconductorModel
from utils import stats_report

def load_test_summary(summary_file):
    """
    Load test summary file containing labels and filenames
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

def load_model(model_path, device='cpu'):
    """
    Load the trained model
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

def preprocess_file(filepath, target_length=1002, column='q', interp_method='pchip', invert_q=True, flip_horizontal=True):
    """
    Preprocess a single file for inference
    
    Args:
        filepath: Path to the CSV file
        target_length: Target sequence length for model input
        column: Column to use as feature (q by default)
        interp_method: Interpolation method ('linear', 'cubic', 'pchip', or 'pv')
        invert_q: Whether to invert q values (multiply by -1)
        flip_horizontal: Whether to flip the sequence horizontally (reverse the order)
    
    Returns:
        Preprocessed q values, original length, and target length
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
        
        # Verify we have data
        if q_values is None or len(q_values) == 0:
            return None, None, None
        
        # Store original length for debugging
        original_length = len(q_values)
        
        # Apply transformations:
        
        # 1. Invert q values if requested (multiply by -1)
        if invert_q:
            q_values = -q_values
            
        # 2. Flip sequence horizontally if requested (reverse the order)
        if flip_horizontal:
            q_values = np.flip(q_values)
        
        # Handle sequence length discrepancy through interpolation
        if original_length != target_length:
            if interp_method == 'linear':
                # Linear interpolation
                original_indices = np.arange(original_length)
                new_indices = np.linspace(0, original_length - 1, target_length)
                q_resampled = np.interp(new_indices, original_indices, q_values)
                
            elif interp_method == 'cubic' and original_length > 3:
                # Cubic spline interpolation
                try:
                    from scipy.interpolate import CubicSpline
                    original_indices = np.arange(original_length)
                    new_indices = np.linspace(0, original_length - 1, target_length)
                    cs = CubicSpline(original_indices, q_values)
                    q_resampled = cs(new_indices)
                except Exception as e:
                    print(f"Error with cubic interpolation: {e}, falling back to linear")
                    original_indices = np.arange(original_length)
                    new_indices = np.linspace(0, original_length - 1, target_length)
                    q_resampled = np.interp(new_indices, original_indices, q_values)
                
            elif interp_method == 'pchip' and original_length > 3:
                # PCHIP interpolation (preserves monotonicity)
                try:
                    from scipy.interpolate import PchipInterpolator
                    original_indices = np.arange(original_length)
                    new_indices = np.linspace(0, original_length - 1, target_length)
                    pchip = PchipInterpolator(original_indices, q_values)
                    q_resampled = pchip(new_indices)
                except Exception as e:
                    print(f"Error with pchip interpolation: {e}, falling back to linear")
                    original_indices = np.arange(original_length)
                    new_indices = np.linspace(0, original_length - 1, target_length)
                    q_resampled = np.interp(new_indices, original_indices, q_values)
                
            elif interp_method == 'pv' and 't' in df.columns and 'v' in df.columns:
                # Special handling for PV loops - preserve relationship between v and q
                try:
                    t_values = df['t'].values
                    v_values = df['v'].values
                    
                    # If we flipped horizontally, we need to flip t and v too
                    if flip_horizontal:
                        t_values = np.flip(t_values)
                        v_values = np.flip(v_values)
                    
                    # Create new time points
                    new_t = np.linspace(t_values.min(), t_values.max(), target_length)
                    
                    # Interpolate v and q to new time points
                    from scipy.interpolate import interp1d
                    v_interp = interp1d(t_values, v_values, bounds_error=False, fill_value='extrapolate')
                    q_interp = interp1d(t_values, q_values, bounds_error=False, fill_value='extrapolate')
                    
                    new_v = v_interp(new_t)
                    q_resampled = q_interp(new_t)
                except Exception as e:
                    print(f"Error with pv interpolation: {e}, falling back to linear")
                    original_indices = np.arange(original_length)
                    new_indices = np.linspace(0, original_length - 1, target_length)
                    q_resampled = np.interp(new_indices, original_indices, q_values)
            else:
                # Fallback to linear interpolation
                original_indices = np.arange(original_length)
                new_indices = np.linspace(0, original_length - 1, target_length)
                q_resampled = np.interp(new_indices, original_indices, q_values)
                
            print(f"Interpolated from {original_length} to {target_length} points using {interp_method}")
        else:
            # No interpolation needed
            q_resampled = q_values
            
        return q_resampled, original_length, target_length
        
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        return None, None, None
    

def apply_normalization(q_values, scaler_path=None, norm_type='scaler'):
    """
    Apply normalization to q values
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

def run_inference(args):
    """
    Run inference on real measurement data
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load file labels from summary file
    file_labels = load_test_summary(args.summary_file)
    
    # Collect all CSV files to process
    if args.specific_file:
        # Process a single file
        all_files = [args.specific_file]
        print(f"Processing single file: {args.specific_file}")
    else:
        # Process all CSV files in directory
        all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        print(f"Found {len(all_files)} CSV files in {args.data_dir}")
    
    # Initialize results storage
    results = []
    
    # Process each file
    for filename in tqdm(all_files, desc="Processing files"):
        filepath = os.path.join(args.data_dir, filename)
        
        # Skip if file doesn't exist
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        # Get true label if available
        true_label = file_labels.get(filename, -1)  # -1 if unknown
        
        # Preprocess file
        q_values, original_length, target_length = preprocess_file(
            filepath,
            target_length=args.seq_length,
            column='q',
            interp_method=args.interpolation,
            invert_q=args.invert_q,
            flip_horizontal=args.flip_horizontal
        )
        
        if q_values is None:
            print(f"Error preprocessing {filename}, skipping")
            continue
        
        # Apply normalization
        q_normalized = apply_normalization(q_values, args.scaler_path, args.normalization)
        
        # Convert to tensor and reshape for model
        q_tensor = torch.FloatTensor(q_normalized).reshape(1, 1, -1)
        
        # Run model prediction
        with torch.no_grad():
            outputs = model(q_tensor.to(device))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
            leaky_prob = probabilities[0, 1].item()  # Probability of being leaky (class 1)
        
        # Store results
        results.append({
            'filename': filename,
            'true_label': true_label,
            'prediction': prediction,
            'confidence': confidence,
            'leaky_prob': leaky_prob,
            'original_length': original_length,
            'target_length': target_length
        })
        
        # Optionally save processed data
        if args.save_processed:
            output_dir = os.path.join(args.output_dir, 'processed_data')
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a DataFrame with the processed q values
            processed_df = pd.DataFrame({'q': q_normalized})
            processed_df.to_csv(os.path.join(output_dir, filename), index=False, header=False)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'inference_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Calculate and visualize metrics
    if 'true_label' in results_df.columns and not results_df['true_label'].isnull().all():
        calculate_metrics(results_df, args.output_dir)
    
    return results_df


def calculate_metrics(results_df, output_dir):
    """
    Calculate and visualize evaluation metrics
    """
    # Filter out rows without true labels
    valid_results = results_df[results_df['true_label'] >= 0]
    
    if len(valid_results) == 0:
        print("No valid results with true labels for evaluation")
        return
    
    # Extract true labels and predictions
    true_labels = valid_results['true_label'].values
    predictions = valid_results['prediction'].values
    
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
    fb = stats_report(cm_list)
    
    # Print results
    print("\nConfusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Negatives (FN): {fn}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    
    # Save confusion matrix visualization
    plt.figure(figsize=(10, 8))
    plt.imshow([[tn, fp], [fn, tp]], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-leaky (0)', 'Leaky (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = (np.max([tn, fp, fn, tp]) if np.max([tn, fp, fn, tp]) > 0 else 0) / 2
    for i in range(2):
        for j in range(2):
            value = [[tn, fp], [fn, tp]][i][j]
            plt.text(j, i, str(value), ha="center", va="center",
                     color="white" if value > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the figure
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Create probability distribution visualization
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
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    prob_path = os.path.join(output_dir, 'probability_distribution.png')
    plt.savefig(prob_path)
    plt.close()
    print(f"Probability distribution saved to {prob_path}")
    
    # Visualize examples of each category
    visualize_examples(results_df, output_dir)

def visualize_examples(results_df, output_dir):
    """
    Create visualizations of example predictions
    """
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'example_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Try to visualize a few examples from each category (TP, TN, FP, FN)
    categories = [
        ('TP', (results_df['true_label'] == 1) & (results_df['prediction'] == 1)),
        ('TN', (results_df['true_label'] == 0) & (results_df['prediction'] == 0)),
        ('FP', (results_df['true_label'] == 0) & (results_df['prediction'] == 1)),
        ('FN', (results_df['true_label'] == 1) & (results_df['prediction'] == 0))
    ]
    
    for category_name, mask in categories:
        subset = results_df[mask]
        if len(subset) > 0:
            # Take up to 3 samples
            for _, row in subset.head(3).iterrows():
                filename = os.path.basename(row['filename'])
                # Create a summary of the prediction
                plt.figure(figsize=(8, 6))
                plt.bar(['Non-Leaky (0)', 'Leaky (1)'], [1 - row['leaky_prob'], row['leaky_prob']])
                plt.title(f"{category_name}: {filename}\nTrue: {int(row['true_label'])}, Pred: {int(row['prediction'])}, Conf: {row['confidence']:.4f}")
                plt.ylabel('Probability')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(vis_dir, f"{category_name}_{filename}.png"))
                plt.close()
    
    print(f"Example visualizations saved to {vis_dir}")

def visualize_transformation(data_dir, output_dir, sample_count=3, seq_length=1002):
    """
    Visualize the effect of various transformations on the data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get CSV files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not all_files:
        print("No CSV files found!")
        return
    
    # Select a sample of files
    if len(all_files) > sample_count:
        import random
        sample_files = random.sample(all_files, sample_count)
    else:
        sample_files = all_files
    
    for filename in sample_files:
        filepath = os.path.join(data_dir, filename)
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath, header=None)
            
            # Assign column names if not present
            if len(df.columns) >= 4:
                df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
            elif len(df.columns) == 2:
                df.columns = ['t', 'q']
            
            # Extract q column
            q_values = df['q'].values if 'q' in df.columns else df.iloc[:, min(2, len(df.columns)-1)].values
            
            # Create different transformations
            original = q_values
            inverted = -q_values
            flipped = np.flip(q_values)
            inverted_flipped = np.flip(-q_values)
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(original)
            plt.title('Original')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(inverted)
            plt.title('Inverted (q * -1)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            plt.plot(flipped)
            plt.title('Flipped (horizontal)')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.plot(inverted_flipped)
            plt.title('Inverted & Flipped')
            plt.grid(True, alpha=0.3)
            
            plt.suptitle(f'Transformations for {filename}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save visualization
            plt.savefig(os.path.join(output_dir, f"transformations_{filename}.png"))
            plt.close()
            
            # Create PV Loop visualization if v is available
            if 'v' in df.columns:
                plt.figure(figsize=(12, 10))
                
                v_values = df['v'].values
                
                plt.subplot(2, 2, 1)
                plt.plot(v_values, original)
                plt.title('Original PV Loop')
                plt.xlabel('V')
                plt.ylabel('q')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 2)
                plt.plot(v_values, inverted)
                plt.title('Inverted PV Loop')
                plt.xlabel('V')
                plt.ylabel('-q')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 3)
                plt.plot(np.flip(v_values), flipped)
                plt.title('Flipped PV Loop')
                plt.xlabel('V (flipped)')
                plt.ylabel('q (flipped)')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 2, 4)
                plt.plot(np.flip(v_values), inverted_flipped)
                plt.title('Inverted & Flipped PV Loop')
                plt.xlabel('V (flipped)')
                plt.ylabel('-q (flipped)')
                plt.grid(True, alpha=0.3)
                
                plt.suptitle(f'PV Loop Transformations for {filename}', fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # Save visualization
                plt.savefig(os.path.join(output_dir, f"pv_transformations_{filename}.png"))
                plt.close()
                
        except Exception as e:
            print(f"Error visualizing {filepath}: {e}")
    
    print(f"Visualizations saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Semiconductor Leakage Detection Inference')
    
    # Required parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing real measurement CSV files')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file')
    
    # Optional parameters
    parser.add_argument('--summary_file', type=str, default=None,
                        help='Path to summary file with true labels')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='Path to the scaler used during training')
    parser.add_argument('--specific_file', type=str, default=None,
                        help='Process only this specific file (optional)')
    
    # Preprocessing parameters
    parser.add_argument('--seq_length', type=int, default=1002,
                        help='Target sequence length (should match training)')
    parser.add_argument('--interpolation', type=str, default='pv',
                        choices=['linear', 'cubic', 'pchip', 'pv'],
                        help='Interpolation method for real data')
    parser.add_argument('--normalization', type=str, default='scaler',
                        choices=['per_file', 'scaler'],
                        help='Normalization method')
    parser.add_argument('--invert_q', action='store_true',
                        help='Invert q values (multiply by -1)')
    parser.add_argument('--flip_horizontal', action='store_true',
                        help='Flip the sequence horizontally (reverse order)')
    parser.add_argument('--save_processed', action='store_true',
                        help='Save processed data files')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for inference')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    parser.add_argument('--visualize_transformations', action='store_true',
                    help='Visualize different data transformations')
    
    args = parser.parse_args()

    # In main() after parsing arguments
    if args.visualize_transformations:
        visualize_transformation(args.data_dir, os.path.join(args.output_dir, 'transformations'), 
                                 sample_count=5, seq_length=args.seq_length)
    
    # Check if scaler path is provided
    if args.normalization == 'scaler' and args.scaler_path is None:
        # Try to find scaler in common locations
        for potential_path in [
            './indices/scalers.pkl',
            './Valid/indices/scalers.pkl',
            './indices/preprocessed/scalers.pkl',
            './Valid/indices/preprocessed/scalers.pkl',
            './model_output/scalers.pkl',
            os.path.join(os.path.dirname(args.output_dir), 'scalers.pkl'),
            os.path.join(os.path.dirname(args.model_path), 'scalers.pkl') if args.model_path else None
        ]:
            if potential_path and os.path.exists(potential_path):
                args.scaler_path = potential_path
                print(f"Found scaler at {args.scaler_path}")
                break
        
        if args.scaler_path is None:
            print("Warning: No scaler found. Using per-file normalization instead.")
            args.normalization = 'per_file'
    
    # Run inference
    results_df = run_inference(args)
    
    print("Inference completed successfully!")
    return results_df

if __name__ == "__main__":
    main()