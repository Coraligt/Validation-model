import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import csv


def load_csv_indices(csv_file):
    """
    Load indices CSV file with format:
    label, Filename
    
    Args:
        csv_file (str): Path to CSV file
        
    Returns:
        list: List of (filename, label) tuples
    """
    file_labels = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:  # Ensure we have both label and filename
                label, filename = int(row[0]), row[1]
                file_labels.append((filename, label))
    return file_labels


class SemiconductorDataset(Dataset):
    """
    Dataset for semiconductor device data
    """
    def __init__(self, root_dir, indices_file, use_columns=['t', 'q'], transform=None):
        """
        Initialize dataset
        
        Args:
            root_dir (str): Directory containing CSV files
            indices_file (str): Path to indices file
            use_columns (list): Columns to use as features (default: ['t', 'q'])
            transform (callable, optional): Transform to apply to samples
        """
        self.root_dir = root_dir
        self.file_labels = load_csv_indices(indices_file)
        self.use_columns = use_columns
        self.transform = transform
        
        # Map column names to indices
        self.column_map = {
            't': 0,
            'v': 1,
            'q': 2,
            'i': 3
        }
        
    def __len__(self):
        return len(self.file_labels)
    
    def __getitem__(self, idx):
        filename, label = self.file_labels[idx]
        
        # Load CSV data
        filepath = os.path.join(self.root_dir, filename)
        try:
            # Read CSV with no header, assuming columns order is t, v, q, i
            df = pd.read_csv(filepath, header=None)
            
            # Extract only the columns we need
            data = []
            for col in self.use_columns:
                if col in self.column_map:
                    col_idx = self.column_map[col]
                    if col_idx < len(df.columns):
                        data.append(df.iloc[:, col_idx].values)
            
            # Convert to numpy array and stack columns
            data = np.stack(data, axis=0)  # Shape: [n_features, seq_len]
            
            # Apply transform if provided
            if self.transform:
                data = self.transform(data)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(data)
            
            # Pad or truncate to fixed length (1002)
            seq_len = data_tensor.shape[1]
            if seq_len < 1002:
                # Pad with zeros at the end
                padding = torch.zeros((data_tensor.shape[0], 1002 - seq_len))
                data_tensor = torch.cat([data_tensor, padding], dim=1)
            elif seq_len > 1002:
                # Truncate to 1002
                data_tensor = data_tensor[:, :1002]
                
            return data_tensor, label
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return dummy tensor in case of error
            dummy_data = torch.zeros((len(self.use_columns), 1002))
            return dummy_data, label


def get_dataloaders(data_dir, indices_dir, batch_size=32, use_columns=['t', 'q'], 
                   num_workers=4, train_transform=None, test_transform=None):
    """
    Get train and test dataloaders
    
    Args:
        data_dir (str): Directory containing CSV files
        indices_dir (str): Directory containing indices CSV files
        batch_size (int): Batch size for DataLoader
        use_columns (list): Columns to use as features
        num_workers (int): Number of workers for data loading
        train_transform (callable): Transform for training data
        test_transform (callable): Transform for test data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = SemiconductorDataset(
        root_dir=data_dir,
        indices_file=os.path.join(indices_dir, 'train_indices.csv'),
        use_columns=use_columns,
        transform=train_transform
    )
    
    test_dataset = SemiconductorDataset(
        root_dir=data_dir,
        indices_file=os.path.join(indices_dir, 'test_indices.csv'),
        use_columns=use_columns,
        transform=test_transform
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Data augmentation transforms
class FlipSignal:
    """Flip signal values (multiply by -1)"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            return -x
        return x


class AddGaussianNoise:
    """Add Gaussian noise to signal"""
    def __init__(self, mean=0., std_factor=0.05, p=0.5):
        self.mean = mean
        self.std_factor = std_factor
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            # Calculate noise standard deviation based on signal amplitude
            max_val = np.max(np.abs(x))
            noise_std = max_val * self.std_factor
            
            # Generate random factor (0-1) to vary noise level, as in original model
            factor = np.random.random()
            
            # Add noise
            noise = np.random.normal(self.mean, factor * noise_std, x.shape)
            return x + noise
        return x


class TimeReverse:
    """
    Reverse signal in time dimension
    Similar to flip_time augmentation mentioned in original model
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            return x[:, ::-1]  # Reverse along time dimension
        return x


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


def get_default_transforms(use_time_reverse=False):
    """
    Get default transforms matching original model
    
    Args:
        use_time_reverse (bool): Whether to use time reversal augmentation
        
    Returns:
        dict: Dictionary containing train and test transforms
    """
    # Training transforms include augmentation
    transforms = [
        FlipSignal(p=0.5),              # 50% chance to flip signal
        AddGaussianNoise(std_factor=0.05, p=0.5)  # 50% chance to add noise
    ]
    
    if use_time_reverse:
        transforms.append(TimeReverse(p=0.5))  # 50% chance of time reversal
    
    train_transform = Compose(transforms)
    
    # Test transform is None (no augmentation for testing)
    test_transform = None
    
    return {
        'train': train_transform,
        'test': test_transform
    }