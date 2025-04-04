import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SemiconductorModel

def load_model(model_path, config=None, device="cpu"):
    """
    Load a trained model from path
    
    Args:
        model_path: Path to model file (.pth)
        config: Optional dictionary with model configuration
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    if config is None:
        # Default configuration
        config = {
            'seq_length': 1002,
            'conv_filters': 3,
            'fc1_size': 20,
            'fc2_size': 10,
            'dropout1': 0.3,
            'dropout2': 0.1
        }
    
    # Create model
    model = SemiconductorModel(
        seq_length=config['seq_length'],
        conv_filters=config['conv_filters'],
        fc1_size=config['fc1_size'],
        fc2_size=config['fc2_size'],
        dropout1=config['dropout1'],
        dropout2=config['dropout2']
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    model.eval()
    
    return model

def preprocess_file(file_path, scalers_path=None, device="cpu"):
    """
    Preprocess a CSV file for model inference
    
    Args:
        file_path: Path to CSV file
        scalers_path: Path to scalers.pkl for normalization
        device: Device to use
    
    Returns:
        Tensor ready for model input
    """
    # Read CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Extract q column (usually the 3rd column, index 2)
    q_values = df.iloc[:, 2].values
    
    # Apply normalization if scalers provided
    if scalers_path is not None and os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
        
        if 'q' in scalers:
            q_values = scalers['q'].transform(q_values.reshape(-1, 1)).flatten()
    
    # Reshape and convert to tensor
    q_tensor = torch.FloatTensor(q_values.reshape(1, 1, -1))
    
    # Make sure length is 1002
    seq_len = q_tensor.shape[2]
    if seq_len < 1002:
        padding = torch.zeros(1, 1, 1002 - seq_len)
        q_tensor = torch.cat([q_tensor, padding], dim=2)
    elif seq_len > 1002:
        q_tensor = q_tensor[:, :, :1002]
    
    return q_tensor.to(device)

def predict_file(model, file_path, scalers_path=None, device="cpu"):
    """
    Predict leakage for a single file
    
    Args:
        model: Trained model
        file_path: Path to CSV file
        scalers_path: Path to scalers.pkl for normalization
        device: Device to use
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess file
    q_tensor = preprocess_file(file_path, scalers_path, device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(q_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    # Extract probabilities
    class_probs = probabilities[0].cpu().numpy()
    
    return {
        'file': os.path.basename(file_path),
        'predicted_class': predicted_class,
        'leaky_probability': class_probs[1],
        'non_leaky_probability': class_probs[0]
    }

def batch_predict(model, dir_path, pattern="*.csv", scalers_path=None, device="cpu"):
    """
    Predict leakage for multiple files in a directory
    
    Args:
        model: Trained model
        dir_path: Directory containing CSV files
        pattern: File pattern to match
        scalers_path: Path to scalers.pkl for normalization
        device: Device to use
    
    Returns:
        DataFrame with prediction results
    """
    # Get list of files
    file_paths = glob.glob(os.path.join(dir_path, pattern))
    
    if not file_paths:
        print(f"No files found matching pattern '{pattern}' in '{dir_path}'")
        return None
    
    # Make predictions
    results = []
    for file_path in tqdm(file_paths, desc="Making predictions"):
        try:
            result = predict_file(model, file_path, scalers_path, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

def analyze_results(results_df, true_labels=None):
    """
    Analyze prediction results
    
    Args:
        results_df: DataFrame with prediction results
        true_labels: Optional dictionary mapping filenames to true labels
    
    Returns:
        Dict with performance metrics (if true_labels provided)
    """
    # Calculate metrics if true labels provided
    if true_labels is not None:
        # Add true labels column
        results_df['true_class'] = results_df['file'].map(lambda x: true_labels.get(x, None))
        
        # Calculate metrics
        correct_predictions = results_df.dropna(subset=['true_class'])
        if len(correct_predictions) > 0:
            accuracy = (correct_predictions['predicted_class'] == correct_predictions['true_class']).mean()
            
            # Confusion matrix
            tp = ((correct_predictions['predicted_class'] == 1) & (correct_predictions['true_class'] == 1)).sum()
            tn = ((correct_predictions['predicted_class'] == 0) & (correct_predictions['true_class'] == 0)).sum()
            fp = ((correct_predictions['predicted_class'] == 1) & (correct_predictions['true_class'] == 0)).sum()
            fn = ((correct_predictions['predicted_class'] == 0) & (correct_predictions['true_class'] == 1)).sum()
            
            # Additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': {
                    'true_positive': tp,
                    'true_negative': tn,
                    'false_positive': fp,
                    'false_negative': fn
                }
            }
            
            return metrics
    
    # Basic statistics
    print(f"Total files processed: {len(results_df)}")
    print(f"Predicted leaky: {(results_df['predicted_class'] == 1).sum()}")
    print(f"Predicted non-leaky: {(results_df['predicted_class'] == 0).sum()}")
    
    # Plot probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['leaky_probability'], bins=20)
    plt.xlabel('Probability of Leakage')
    plt.ylabel('Count')
    plt.title('Distribution of Leakage Probabilities')
    plt.grid(True)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Inference with trained semiconductor leakage detection model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model file (.pth)')
    parser.add_argument('--scalers_path', type=str, default=None,
                       help='Path to scalers.pkl file')
    
    # Input parameters
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to input CSV file for single prediction')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing CSV files for batch prediction')
    parser.add_argument('--pattern', type=str, default="*.csv",
                       help='File pattern for batch prediction')
    parser.add_argument('--output_file', type=str, default="predictions.csv",
                       help='Path to save prediction results')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for inference')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Model configuration (only needed if not saved with model)
    parser.add_argument('--conv_filters', type=int, default=3,
                       help='Number of filters in convolutional layer')
    parser.add_argument('--fc1_size', type=int, default=20,
                       help='Size of first fully connected layer')
    parser.add_argument('--fc2_size', type=int, default=10,
                       help='Size of second fully connected layer')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if args.no_cuda:
        device = 'cpu'
    elif not torch.cuda.is_available() and device.startswith('cuda'):
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Model configuration
    config = {
        'seq_length': 1002,
        'conv_filters': args.conv_filters,
        'fc1_size': args.fc1_size,
        'fc2_size': args.fc2_size,
        'dropout1': 0.0,  # No dropout during inference
        'dropout2': 0.0   # No dropout during inference
    }
    
    # Load model
    model = load_model(args.model_path, config, device)
    print(f"Model loaded from {args.model_path}")
    
    # Single file prediction
    if args.input_file is not None:
        if not os.path.exists(args.input_file):
            print(f"Input file {args.input_file} does not exist")
            return
        
        result = predict_file(model, args.input_file, args.scalers_path, device)
        print("\nPrediction result:")
        print(f"File: {result['file']}")
        print(f"Predicted class: {'Leaky' if result['predicted_class'] == 1 else 'Non-leaky'}")
        print(f"Leaky probability: {result['leaky_probability']:.4f}")
        print(f"Non-leaky probability: {result['non_leaky_probability']:.4f}")
    
    # Batch prediction
    elif args.input_dir is not None:
        if not os.path.exists(args.input_dir):
            print(f"Input directory {args.input_dir} does not exist")
            return
        
        results_df = batch_predict(model, args.input_dir, args.pattern, args.scalers_path, device)
        
        if results_df is not None and not results_df.empty:
            # Save results
            results_df.to_csv(args.output_file, index=False)
            print(f"Results saved to {args.output_file}")
            
            # Analyze results
            analyze_results(results_df)
    
    else:
        print("Either --input_file or --input_dir must be specified")

if __name__ == '__main__':
    main()