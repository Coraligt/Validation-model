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
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import pickle

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


def create_indices_files(data_dir, output_dir, test_ratio=0.15, random_state=42):
    """
    Create indices files for training and testing
    
    Args:
        data_dir (str): Directory containing CSV files
        output_dir (str): Directory to save indices files
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

    # Split into train and test sets
    device_ids = list(devices.keys())
    np.random.shuffle(device_ids)
    test_size = int(len(device_ids) * test_ratio)
    test_device_ids = device_ids[:test_size]
    train_device_ids = device_ids[test_size:]

    # Create indices files
    train_files = []
    test_files = []

    for device_id in train_device_ids:
        for filename, label in devices[device_id]:
            train_files.append((label, filename))

    for device_id in test_device_ids:
        for filename, label in devices[device_id]:
            test_files.append((label, filename))

    train_labels = [label for label, _ in train_files]
    test_labels = [label for label, _ in test_files]

    train_counter = Counter(train_labels)
    test_counter = Counter(test_labels)

    print(f"Train set size: {len(train_files)} files from {len(train_device_ids)} devices")
    print(f"Test set size: {len(test_files)} files from {len(test_device_ids)} devices")

    print(f"Train set: {train_counter[0]} non-leaky, {train_counter[1]} leaky")
    print(f"Test set: {test_counter[0]} non-leaky, {test_counter[1]} leaky")

    # Write indices to files
    with open(os.path.join(output_dir, 'train_indices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'Filename'])  # header
        for label, filename in train_files:
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
    
    # Save scalers for future use
    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
        pickle.dump(scalers, f)
    
    # Second pass: transform all files (both train and test) 
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


def main():
    parser = argparse.ArgumentParser(description='Preprocess semiconductor device data')
    
    parser.add_argument('--data_dir', type=str, default='./dataset_3',
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data',
                        help='Directory to save preprocessed files')
    parser.add_argument('--analyze', action='store_true', default=True,
                        help='Analyze the dataset')
    parser.add_argument('--preprocess', action='store_true', default=True,
                        help='Preprocess the dataset')
    parser.add_argument('--create_indices', action='store_true', default=True,
                        help='Create train and test indices files')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_columns', type=str, default='t,q',
                        help='Columns to use (comma-separated)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse columns to use
    use_columns = args.use_columns.split(',')
    
    # Create train/test indices first
    if args.create_indices:
        create_indices_files(
            args.data_dir,
            args.output_dir,
            args.test_ratio,
            args.seed
        )
    
    # Analyze the dataset
    if args.analyze:
        analysis_dir = os.path.join(args.output_dir, 'analysis')
        analyze_dataset(args.data_dir, analysis_dir)
    
    # Preprocess the dataset
    if args.preprocess:
        preprocessed_dir = os.path.join(args.output_dir, 'preprocessed')
        train_indices_file = os.path.join(args.output_dir, 'train_indices.csv')
        
        # Check if train indices file exists
        if not os.path.exists(train_indices_file):
            print(f"Error: Train indices file {train_indices_file} not found. Run with --create_indices first.")
            return
        
        preprocess_dataset(
            args.data_dir,
            preprocessed_dir,
            train_indices_file,
            use_columns
        )


if __name__ == '__main__':
    main()