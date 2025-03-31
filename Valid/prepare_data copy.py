import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re
import csv
import argparse
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

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
    
    # Analyze the original dataset_3
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
    print(f"Columns in {filepath}: {df.columns.tolist()}")  # Print columns of the last file processed
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


def preprocess_dataset(data_dir, output_dir, normalize=True):
    """
    Preprocess the dataset and save the processed files
    
    Args:
        data_dir (str): Directory containing CSV files
        output_dir (str): Directory to save processed files
        normalize (bool): Whether to normalize the features
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Preprocessing dataset in {data_dir}...")

    pattern  = re.compile(r'dev(\d+)_(\d)\.csv') # dev#_label.csv
    
    # Get all test data files
    file_paths = glob.glob(os.path.join(data_dir, 'dev*_*.csv'))

    if normalize:
         # Initialize scalers for each column
        t_scaler = MinMaxScaler()
        v_scaler = MinMaxScaler()
        q_scaler = MinMaxScaler()
        i_scaler = MinMaxScaler()
        
        # Collect data for fitting scalers
        t_data = []
        v_data = []
        q_data = []
        i_data = []

        #sample_size = min(1000, len(file_paths)) # Do not exceed 1000 samples
        #sample_files = np.random.choice(file_paths, sample_size, replace=False)
        sample_files = file_paths  # Use all files for normalization

        for filepath in tqdm(file_paths, desc="Analyzing all files"):  # Analyze all files
            filename = os.path.basename(filepath)
            match = pattern.match(filename)
        
        for filepath in tqdm(sample_files, desc="Collecting data for normalization"):
            try:
                df = pd.read_csv(filepath, names=['t', 'v', 'q', 'i'], header=None)

                if 't' in df.columns:
                    t_data.append(df['t'].values)
                if 'v' in df.columns:
                    v_data.append(df['v'].values)
                if 'q' in df.columns:
                    q_data.append(df['q'].values)
                if 'i' in df.columns:
                    i_data.append(df['i'].values)
            except Exception as e:
                print(f"Error processing {filepath} for normalization: {e}")
        
        # Fit scalers
        if t_data:
            t_scaler.fit(np.concatenate(t_data).reshape(-1, 1))
        if v_data:
            v_scaler.fit(np.concatenate(v_data).reshape(-1, 1))
        if q_data:
            q_scaler.fit(np.concatenate(q_data).reshape(-1, 1))
        if i_data:
            i_scaler.fit(np.concatenate(i_data).reshape(-1, 1))
    
    # Process all files
    for filepath in tqdm(file_paths, desc="Preprocessing files"):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        
        if match:
            try:
                # Read CSV
                df = pd.read_csv(filepath, names=['t', 'v', 'q', 'i'], header=None)

                # Normalize if requested
                if normalize:
                    if 't' in df.columns:
                        df['t'] = t_scaler.transform(df['t'].values.reshape(-1, 1)).flatten()
                    if 'v' in df.columns:
                        df['v'] = v_scaler.transform(df['v'].values.reshape(-1, 1)).flatten()
                    if 'q' in df.columns:
                        df['q'] = q_scaler.transform(df['q'].values.reshape(-1, 1)).flatten()
                    if 'i' in df.columns:
                        df['i'] = i_scaler.transform(df['i'].values.reshape(-1, 1)).flatten()
                
                # Save to output directory
                output_filepath = os.path.join(output_dir, filename)
                df.to_csv(output_filepath, index=False)
            except Exception as e:
                print(f"Error preprocessing {filepath}: {e}")
    
    # Save scalers for future use
    if normalize:
        import pickle
        
        scalers = {
            't': t_scaler,
            'v': v_scaler,
            'q': q_scaler,
            'i': i_scaler
        }
        
        with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump(scalers, f)
    
    print(f"Preprocessing completed. Files saved to {output_dir}")


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

    pattern  = re.compile(r'dev(\d+)_(\d)\.csv') # dev#_label.csv
    
    # Get all device files
    files_info = []
    file_paths = glob.glob(os.path.join(data_dir, 'dev*_*.csv'))
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            device_id = int(match.group(1))
            label = int(match.group(2))
            files_info.append((filepath, device_id, label))

    # Group by device_id to ensure all files for a device are in the same set
    devices = {}
    for filepath, device_id, label in files_info:
        if device_id not in devices:
            devices[device_id] = []
        devices[device_id].append((filepath, label))

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
        for filepath, label in devices[device_id]:
            train_files.append((filepath, label))

    for device_id in test_device_ids:
        for filepath, label in devices[device_id]:
            test_files.append((filepath, label))

    train_labels = [label for _, label in train_files]
    test_labels = [label for _, label in test_files]

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
        for filepath, label in train_files:
            writer.writerow([label, os.path.basename(filepath)])

    with open(os.path.join(output_dir, 'test_indices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'Filename']) 
        for filepath, label in test_files:
            writer.writerow([label, os.path.basename(filepath)])  
    
    print(f"Indices files created in {output_dir}")


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
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize the features during preprocessing')
    parser.add_argument('--create_indices', action='store_true', default=True,
                        help='Create train and test indices files')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze the dataset
    if args.analyze:
        analysis_dir = os.path.join(args.output_dir, 'analysis')
        analyze_dataset(args.data_dir, analysis_dir)
    
    # Preprocess the dataset
    if args.preprocess:
        preprocess_dir = os.path.join(args.output_dir, 'preprocessed')
        preprocess_dataset(args.data_dir, preprocess_dir, args.normalize)
    
    # Create indices files
    if args.create_indices:
        indices_dir = os.path.join(args.output_dir)
        data_source = os.path.join(args.output_dir, 'preprocessed') if (args.preprocess and args.normalize and 
            os.path.exists(os.path.join(args.output_dir, 'preprocessed'))) else args.data_dir
            
        create_indices_files(
            data_source,
            indices_dir, 
            args.test_ratio, 
            args.seed
        )


if __name__ == '__main__':
    main() 

   

