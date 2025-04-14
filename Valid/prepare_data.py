import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re
import csv
import shutil
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy import interpolate
import torch  

def analyze_dataset(data_dir, output_dir):
    """
    Analyze the dataset and create visualizations
    
    Args:
        data_dir (str): Directory containing CSV files
        output_dir (str): Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing dataset in {data_dir}...")
    
    # Filenames dev#_label.csv
    pattern = re.compile(r'dev(\d+)_(\d)\.csv')
    
    # Initialize counters and lists for analysis
    total_files = 0
    leaky_count = 0
    non_leaky_count = 0
    num_rows = []
    t_min, t_max = float('inf'), float('-inf')
    v_min, v_max = float('inf'), float('-inf')
    q_min, q_max = float('inf'), float('-inf')
    i_min, i_max = float('inf'), float('-inf')
    
    # Get all device files and analyze
    file_paths = glob.glob(os.path.join(data_dir, 'dev*_*.csv'))
    
    # Analyze the original dataset
    for filepath in tqdm(file_paths, desc="Analyzing all files"): 
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        
        if match:
            total_files += 1
            label = int(match.group(2))
            
            if label == 1:
                leaky_count += 1
            else:
                non_leaky_count += 1
            
            # Read CSV
            try:
                df = pd.read_csv(filepath, names=['t', 'v', 'q', 'i'], header=None)

                num_rows.append(len(df))
                
                # Update min/max values
                if 't' in df.columns:
                    t_min = min(t_min, df['t'].min())
                    t_max = max(t_max, df['t'].max())
                if 'v' in df.columns:
                    v_min = min(v_min, df['v'].min())
                    v_max = max(v_max, df['v'].max())
                if 'q' in df.columns:
                    q_min = min(q_min, df['q'].min())
                    q_max = max(q_max, df['q'].max())
                if 'i' in df.columns:
                    i_min = min(i_min, df['i'].min())
                    i_max = max(i_max, df['i'].max())
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    avg_rows = float(np.mean(num_rows)) if num_rows else 0.0

    # Print summary
    print(f"Total files: {total_files}")
    print(f"Leaky files: {leaky_count} ({100 * leaky_count / total_files:.2f}%)")
    print(f"Non-leaky files: {non_leaky_count} ({100 * non_leaky_count / total_files:.2f}%)")
    print(f"Average number of rows: {avg_rows:.2f}")
    print(f"Min rows: {min(num_rows)}, Max rows: {max(num_rows)}")
    print(f"Time t range: [{t_min} to {t_max}]")
    print(f"Voltage V range: [{v_min} to {v_max}]")
    print(f"Charge q range: [{q_min} to {q_max}]")
    print(f"Current i range: [{i_min} to {i_max}]")

    # Save summary to a text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Total files: {total_files}\n")
        f.write(f"Leaky files: {leaky_count} ({100 * leaky_count / total_files:.2f}%) \n")
        f.write(f"Non-leaky files: {non_leaky_count} ({100 * non_leaky_count / total_files:.2f}%)\n")
        f.write(f"Average number of rows: {avg_rows:.2f}\n")
        f.write(f"Min rows: {min(num_rows)}, Max rows: {max(num_rows)}\n")
        f.write(f"t range: [{t_min} to {t_max}]\n")
        f.write(f"v range: [{v_min} to {v_max}]\n")
        f.write(f"q range: [{q_min} to {q_max}]\n")
        f.write(f"i range: [{i_min} to {i_max}]\n")

    # Visualize distributions
    plt.figure(figsize=(12, 6))
    plt.hist(num_rows, bins=20)
    plt.xlabel('Number of Rows')
    plt.ylabel('Count')
    plt.title('Distribution of Rows per File')
    plt.savefig(os.path.join(output_dir, 'rows_distribution.png'))
    plt.close()

    # Visualize leaky vs non-leaky
    labels = ['Leaky', 'Non-Leaky']
    counts = [leaky_count, non_leaky_count]
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title('Leaky vs Non-Leaky Distribution')
    plt.savefig(os.path.join(output_dir, 'leaky_vs_non_leaky.png'))
    plt.close()

    # Sample time series, a sample leaky and non-leaky file
    sample_leaky = None
    sample_non_leaky = None

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)

        if match:
            label = int(match.group(2))
            if label == 1 and sample_leaky is None:
                sample_leaky = filepath
            elif label == 0 and sample_non_leaky is None:
                sample_non_leaky = filepath
            
            if sample_leaky is not None and sample_non_leaky is not None:
                break
                
    if sample_leaky is not None and sample_non_leaky is not None:
        try:
            df_leaky = pd.read_csv(sample_leaky, names=['t', 'v', 'q', 'i'], header=None)
            df_non_leaky = pd.read_csv(sample_non_leaky, names=['t', 'v', 'q', 'i'], header=None)

            plt.figure(figsize=(12, 6))

            # t vs q for leaky
            plt.subplot(1, 2, 1)
            plt.plot(df_leaky['t'], df_leaky['q'])
            plt.xlabel('Time (t)')
            plt.ylabel('Charge (q)')
            plt.title(f'Leaky Sample - t vs q : {os.path.basename(sample_leaky)}')

            # t vs q for non-leaky
            plt.subplot(1, 2, 2)
            plt.plot(df_non_leaky['t'], df_non_leaky['q'])
            plt.xlabel('Time (t)')
            plt.ylabel('Charge (q)')
            plt.title(f'Non-Leaky Sample - t vs q : {os.path.basename(sample_non_leaky)}')
            plt.savefig(os.path.join(output_dir, 'sample_t_q_comparison.png'))
            plt.close()

            # Copy sample files to output directory for reference
            shutil.copy(sample_leaky, os.path.join(output_dir, 'sample_leaky.csv'))
            shutil.copy(sample_non_leaky, os.path.join(output_dir, 'sample_non_leaky.csv'))
        except Exception as e:
            print(f"Error creating sample visualizations: {e}")
    
    print(f"Analysis completed. Results saved to {output_dir}")


def create_indices_files(data_dir, output_dir, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Create indices files for training, validation and testing
    
    Args:
        data_dir (str): Directory containing CSV files
        output_dir (str): Directory to save indices files
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        random_state (int): Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating indices files in {data_dir}...")

    pattern = re.compile(r'dev(\d+)_(\d)\.csv')  # dev#_label.csv
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get all device files
    files_info = []
    file_paths = glob.glob(os.path.join(data_dir, 'dev*_*.csv'))
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            device_id = int(match.group(1))
            label = int(match.group(2))
            files_info.append((filename, device_id, label))

    # Group by device_id to ensure all files for a device are in the same set
    devices = {}
    for filename, device_id, label in files_info:
        if device_id not in devices:
            devices[device_id] = []
        devices[device_id].append((filename, label))

    # Split into train, validation and test sets
    device_ids = list(devices.keys())
    np.random.shuffle(device_ids)
    
    test_size = int(len(device_ids) * test_ratio)
    val_size = int(len(device_ids) * val_ratio)
    
    test_device_ids = device_ids[:test_size]
    val_device_ids = device_ids[test_size:test_size+val_size]
    train_device_ids = device_ids[test_size+val_size:]

    # Create indices files
    train_files = []
    val_files = []
    test_files = []

    for device_id in train_device_ids:
        for filename, label in devices[device_id]:
            train_files.append((label, filename))

    for device_id in val_device_ids:
        for filename, label in devices[device_id]:
            val_files.append((label, filename))

    for device_id in test_device_ids:
        for filename, label in devices[device_id]:
            test_files.append((label, filename))

    train_labels = [label for label, _ in train_files]
    val_labels = [label for label, _ in val_files]
    test_labels = [label for label, _ in test_files]

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    test_counter = Counter(test_labels)

    print(f"Train set size: {len(train_files)} files from {len(train_device_ids)} devices")
    print(f"Validation set size: {len(val_files)} files from {len(val_device_ids)} devices")
    print(f"Test set size: {len(test_files)} files from {len(test_device_ids)} devices")

    print(f"Train set: {train_counter[0]} non-leaky, {train_counter[1]} leaky")
    print(f"Validation set: {val_counter[0]} non-leaky, {val_counter[1]} leaky")
    print(f"Test set: {test_counter[0]} non-leaky, {test_counter[1]} leaky")

    # Write indices to files
    with open(os.path.join(output_dir, 'train_indices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'Filename'])  # header
        for label, filename in train_files:
            writer.writerow([label, filename])

    with open(os.path.join(output_dir, 'val_indices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'Filename'])  # header
        for label, filename in val_files:
            writer.writerow([label, filename])

    with open(os.path.join(output_dir, 'test_indices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'Filename']) 
        for label, filename in test_files:
            writer.writerow([label, filename])  
    
    print(f"Indices files created in {output_dir}")


def preprocess_dataset(data_dir, output_dir, train_indices_file, use_columns=['t', 'q']):
    """
    Preprocess the dataset with proper normalization
    
    Args:
        data_dir (str): Directory containing original CSV files
        output_dir (str): Directory to save preprocessed files
        train_indices_file (str): Path to train indices CSV file
        use_columns (list): Columns to use as features
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load train indices to fit scalers only on training data
    train_files = []
    with open(train_indices_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                label, filename = int(row[0]), row[1]
                train_files.append(filename)
    
    print(f"Will fit scalers on {len(train_files)} training files")
    
    # Initialize scalers for each column
    scalers = {column: MinMaxScaler() for column in use_columns}
    
    # First pass: collect data from training set to fit scalers
    print("Collecting data from training set to fit scalers...")
    for column in use_columns:
        column_data = []
        
        for filename in tqdm(train_files, desc=f"Processing {column} column"):
            try:
                # Read CSV file
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath, header=None)
                
                # If we have 4 columns, assume they are t, v, q, i
                if len(df.columns) >= 4:
                    df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
                
                # If column exists, add data to list
                if column in df.columns:
                    column_data.append(df[column].values)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        # Combine all data for this column and fit scaler
        if column_data:
            combined_data = np.concatenate(column_data).reshape(-1, 1)
            scalers[column].fit(combined_data)
            print(f"Fitted scaler for {column}: min={scalers[column].data_min_[0]}, max={scalers[column].data_max_[0]}")
    
    # Save scalers for future use - ENSURE DIRECTORY EXISTS
    os.makedirs(os.path.dirname(os.path.join(output_dir, 'scalers.pkl')), exist_ok=True)
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to {os.path.join(output_dir, 'scalers.pkl')}")
    
    # Also save to parent directory for easier access during inference
    parent_dir = os.path.dirname(output_dir)
    with open(os.path.join(parent_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalers also saved to {os.path.join(parent_dir, 'scalers.pkl')}")
    
    # Second pass: transform all files (train, validation, and test) 
    # Get all files
    all_files = glob.glob(os.path.join(data_dir, 'dev*_*.csv'))
    
    print(f"Transforming {len(all_files)} files with fitted scalers...")
    for filepath in tqdm(all_files, desc="Transforming files"):
        try:
            filename = os.path.basename(filepath)
            
            # Read CSV file
            df = pd.read_csv(filepath, header=None)
            
            # If we have 4 columns, assume they are t, v, q, i
            if len(df.columns) >= 4:
                df.columns = ['t', 'v', 'q', 'i'][:len(df.columns)]
            
            # Transform each column with its scaler
            for column in use_columns:
                if column in df.columns:
                    df[column] = scalers[column].transform(df[column].values.reshape(-1, 1))
            
            # Save transformed file
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False, header=False)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"Preprocessing completed. Files saved to {output_dir}")


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
            return q_values, df
            
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
                
                # Update dataframe with interpolated values
                new_df = pd.DataFrame()
                new_df['t'] = new_t
                new_df['v'] = new_v
                new_df['q'] = new_q
                if 'i' in df.columns:
                    i_values = df['i'].values
                    i_interp = interp1d(t_values, i_values, bounds_error=False, fill_value='extrapolate')
                    new_i = i_interp(new_t)
                    new_df['i'] = new_i
                
                df = new_df
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
        
        return q_resampled, df
        
    except Exception as e:
        print(f"Error preprocessing {filepath}: {e}")
        return None, None

def apply_scaler(q_values, scalers, column='q'):
    """
    Apply trained scaler to q values
    
    Args:
        q_values: Array of q values
        scalers: Dictionary of trained scalers
        column: Column name to use (default: 'q')
        
    Returns:
        Normalized q values
    """
    q_scaler = scalers.get(column)
    if q_scaler is not None:
        q_values_reshaped = q_values.reshape(-1, 1)
        q_normalized = q_scaler.transform(q_values_reshaped).flatten()
        return q_normalized
    else:
        # Fallback to min-max normalization if no scaler found
        print(f"Warning: No scaler found for {column}, using per-file normalization")
        min_val = np.min(q_values)
        max_val = np.max(q_values)
        if max_val > min_val:
            return (q_values - min_val) / (max_val - min_val)
        else:
            # Edge case: all values are the same
            return np.ones_like(q_values) * 0.5

def prepare_real_data(real_data_dir, output_dir, scaler_path=None, target_length=1002, 
                      column='q', interp_method='linear', summary_file=None):
    """
    Prepare real measurement data for inference
    
    Args:
        real_data_dir (str): Directory containing real measurement files
        output_dir (str): Directory to save preprocessed files
        scaler_path (str): Path to trained scalers
        target_length (int): Target sequence length
        column (str): Column to use as feature
        interp_method (str): Interpolation method
        summary_file (str): Path to summary file with labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load trained scalers if available
    scalers = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
            print(f"Loaded scalers from {scaler_path}")
        except Exception as e:
            print(f"Error loading scalers: {e}")
    
    # Load summary file with labels if available
    file_labels = {}
    if summary_file and os.path.exists(summary_file):
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
        except Exception as e:
            print(f"Error loading summary file: {e}")
    
    # Get list of files to process
    all_files = glob.glob(os.path.join(real_data_dir, "*.csv"))
    print(f"Found {len(all_files)} CSV files in {real_data_dir}")
    
    # Create a CSV file to track preprocessing results
    results_file = os.path.join(output_dir, 'preprocessing_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Original Length', 'Target Length', 'Label', 'Status'])
    
    # Process each file
    for filepath in tqdm(all_files, desc="Preprocessing real data"):
        filename = os.path.basename(filepath)
        
        try:
            # Get label if available
            label = file_labels.get(filename, -1)  # -1 if no label found
            
            # Step 1: Interpolate to target length
            q_values, df = preprocess_real_data(
                filepath, 
                target_length=target_length,
                column=column,
                interp_method=interp_method
            )
            
            if q_values is None or df is None:
                # Record error
                with open(results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([filename, 'N/A', target_length, label, 'Error: Preprocessing failed'])
                continue
            
            original_length = len(df)
            
            # Step 2: Apply normalization
            if scalers is not None:
                # Use trained scaler
                q_normalized = apply_scaler(q_values, scalers, column)
                
                # Update q column in dataframe
                if column in df.columns:
                    df[column] = q_normalized
            else:
                # Use per-file normalization
                min_val = np.min(q_values)
                max_val = np.max(q_values)
                if max_val > min_val:
                    q_normalized = (q_values - min_val) / (max_val - min_val)
                else:
                    q_normalized = np.ones_like(q_values) * 0.5
                
                # Update q column in dataframe
                if column in df.columns:
                    df[column] = q_normalized
            
            # Save preprocessed file
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False, header=False)
            
            # Record success
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, original_length, target_length, label, 'Success'])
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
            # Record error
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([filename, 'N/A', target_length, label, f'Error: {str(e)}'])
    
    print(f"Real data preprocessing completed. Files saved to {output_dir}")
    
    # Create indices file for inference if labels are available
    if file_labels:
        indices_file = os.path.join(output_dir, 'inference_indices.csv')
        with open(indices_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'Filename'])  # header
            for filename, label in file_labels.items():
                writer.writerow([label, filename])
        print(f"Inference indices file created at {indices_file}")


def prepare_data(data_dir, output_dir, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Master function for data preparation
    
    Args:
        data_dir (str): Directory containing CSV files
        output_dir (str): Directory to save processed files
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Analyze the dataset
    analysis_dir = os.path.join(output_dir, 'analysis')
    analyze_dataset(data_dir, analysis_dir)
    
    # Step 2: Create train/val/test indices
    create_indices_files(data_dir, output_dir, val_ratio, test_ratio, seed)
    
    # Step 3: Preprocess the dataset
    preprocessed_dir = os.path.join(output_dir, 'preprocessed')
    train_indices_file = os.path.join(output_dir, 'train_indices.csv')
    
    # If train indices file doesn't exist, raise error
    if not os.path.exists(train_indices_file):
        raise FileNotFoundError(f"Train indices file {train_indices_file} not found")
    
    # Preprocess with only q column (similar to IEGM model)
    preprocess_dataset(data_dir, preprocessed_dir, train_indices_file, use_columns=['q'])
    
    print(f"Data preparation completed. Processed files saved to {output_dir}")


def prepare_test_data(real_data_dir, output_dir, scaler_path=None, summary_file=None, 
                      interp_method='pv', target_length=1002):
    """
    Prepare real test measurement data for inference with appropriate interpolation and normalization
    
    Args:
        real_data_dir (str): Directory containing real measurement files
        output_dir (str): Directory to save preprocessed files
        scaler_path (str): Path to trained scalers
        summary_file (str): Path to summary file with labels
        interp_method (str): Interpolation method ('linear', 'cubic', 'pchip', or 'pv')
        target_length (int): Target sequence length
    """
    if scaler_path is None:
        # Try to find scalers.pkl in common locations
        for potential_path in [
            './indices/scalers.pkl',
            './Valid/indices/scalers.pkl',
            './indices/preprocessed/scalers.pkl',
            './Valid/indices/preprocessed/scalers.pkl',
            './model_output/scalers.pkl'
        ]:
            if os.path.exists(potential_path):
                scaler_path = potential_path
                print(f"Found scaler file at {scaler_path}")
                break
    
    # Create directory for preprocessed test data
    preprocessed_dir = os.path.join(output_dir, 'preprocessed_real_data')
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Prepare real data using found scaler
    prepare_real_data(
        real_data_dir=real_data_dir,
        output_dir=preprocessed_dir,
        scaler_path=scaler_path,
        target_length=target_length,
        column='q',
        interp_method=interp_method,
        summary_file=summary_file
    )
    
    print(f"Test data preparation completed. Files saved to {preprocessed_dir}")
    
    # Return preprocessed directory path for convenience
    return preprocessed_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess semiconductor device data')
    
    # Main data processing parameters
    parser.add_argument('--data_dir', type=str, default='./dataset_3',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data',
                        help='Directory to save preprocessed files')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Test data preprocessing parameters
    parser.add_argument('--prepare_test', action='store_true',
                        help='Prepare real test data for inference')
    parser.add_argument('--test_data_dir', type=str, default=None,
                        help='Directory containing real test data CSV files')
    parser.add_argument('--summary_file', type=str, default=None,
                        help='Path to summary file with labels for test data')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='Path to trained scaler')
    parser.add_argument('--interp_method', type=str, default='pv',
                        choices=['linear', 'cubic', 'pchip', 'pv'],
                        help='Interpolation method for test data')
    parser.add_argument('--target_length', type=int, default=1002,
                        help='Target sequence length for test data')
    
    args = parser.parse_args()
    
    # Run standard data preparation if not preparing test data
    if not args.prepare_test:
        prepare_data(args.data_dir, args.output_dir, args.val_ratio, args.test_ratio, args.seed)
    else:
        # Prepare test data for inference
        if args.test_data_dir is None:
            parser.error("--test_data_dir is required when --prepare_test is set")
        
        prepare_test_data(
            real_data_dir=args.test_data_dir,
            output_dir=args.output_dir,
            scaler_path=args.scaler_path,
            summary_file=args.summary_file,
            interp_method=args.interp_method,
            target_length=args.target_length
        )

        