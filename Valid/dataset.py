import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# def load_csv_indices(csv_file):
#     """
#     Load CSV file with indices.
#     Format: label, filename
#     """
#     file_labels = []
#     with open(csv_file) as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=',')
#         next(csvreader, None)  # Skip header
#         for row in csvreader:
#             label = int(row[0])
#             filename = row[1]
#             file_labels.append((filename, label))
#     return file_labels

# Change the function to load CSV file with indices for the new format
# Format: leaky_label, voltage_label, filename
def load_csv_indices(csv_file):
    """
    Load CSV file with indices for the new format
    Format: leaky_label, voltage_label, filename
    """
    file_labels = []
    with open(csv_file) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # Skip header
        for row in csvreader:
            if len(row) >= 3:
                leaky_label = int(row[0])
                voltage_label = int(row[1])
                filename = row[2]
                file_labels.append((filename, leaky_label, voltage_label))
    return file_labels

class SemiconductorDataset(Dataset):
    """
    Dataset for semiconductor device data, using only q values as features.
    """
    def __init__(self, root_dir, indices_file, transform=None):
        """
        Args:
            root_dir: Directory containing CSV files
            indices_file: Path to indices CSV file
            transform: Optional transform to be applied to samples
        """
        self.root_dir = root_dir
        self.file_labels = load_csv_indices(indices_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.file_labels)
    
    def __getitem__(self, idx):
        filename, label = self.file_labels[idx]
        filepath = os.path.join(self.root_dir, filename)
        
        try:
            # Read CSV file - assume column order is t, v, q, i
            df = pd.read_csv(filepath, header=None)
            
            # Extract q column (usually the 3rd column, index 2)
            q_values = df.iloc[:, 2].values
            
            # Reshape to match model input format [channels, sequence_length]
            q_values = q_values.reshape(1, -1)
            
            # Apply transform if provided
            if self.transform:
                q_values = self.transform(q_values)
            
            # Convert to tensor
            q_tensor = torch.FloatTensor(q_values)
            
            # Make sure length is 1002
            seq_len = q_tensor.shape[1]
            if seq_len < 1002:
                padding = torch.zeros(1, 1002 - seq_len)
                q_tensor = torch.cat([q_tensor, padding], dim=1)
            elif seq_len > 1002:
                q_tensor = q_tensor[:, :1002]
            
            return q_tensor, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return dummy tensor in case of error
            return torch.zeros(1, 1002), torch.tensor(label, dtype=torch.long)

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
    def __init__(self, std_factor=0.05, p=0.5):
        self.std_factor = std_factor
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            max_val = np.max(np.abs(x))
            noise_std = max_val * self.std_factor
            factor = np.random.random()
            noise = np.random.normal(0, factor * noise_std, x.shape)
            return x + noise
        return x

class TimeReverse:
    """Reverse signal in time dimension"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, x):
        if np.random.random() < self.p:
            return x[:, ::-1]
        return x

class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

# def get_transforms(use_time_reverse=False):
#     """Get data transforms for training and testing"""
#     train_transforms = [
#         FlipSignal(p=0.5),
#         AddGaussianNoise(std_factor=0.05, p=0.5)
#     ]
    
#     if use_time_reverse:
#         train_transforms.append(TimeReverse(p=0.5))
    
#     return {
#         'train': Compose(train_transforms),
#         'val': None,  # No transforms for validation data
#         'test': None  # No transforms for test data
#     }

def get_transforms(use_time_reverse=False, flip_prob=0.5, noise_prob=0.5, noise_factor=0.05):
    """Get data transforms for training and testing"""
    train_transforms = [
        FlipSignal(p=flip_prob),
        AddGaussianNoise(std_factor=noise_factor, p=noise_prob)
    ]
    
    if use_time_reverse:
        train_transforms.append(TimeReverse(p=flip_prob))
    
    return {
        'train': Compose(train_transforms),
        'val': None,  # No transforms for validation data
        'test': None  # No transforms for test data
    }

def get_dataloaders(data_dir, indices_dir, batch_size=32, num_workers=4, transforms=None):
    """
    Create DataLoaders for training, validation and testing
    
    Args:
        data_dir: Directory with preprocessed CSV files
        indices_dir: Directory with indices files
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        transforms: Dictionary with 'train', 'val', and 'test' transforms
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if transforms is None:
        transforms = get_transforms()
    
    # Create datasets
    train_dataset = SemiconductorDataset(
        root_dir=data_dir,
        indices_file=os.path.join(indices_dir, 'train_indices.csv'),
        transform=transforms['train']
    )
    
    val_dataset = None
    val_indices_path = os.path.join(indices_dir, 'val_indices.csv')
    if os.path.exists(val_indices_path):
        val_dataset = SemiconductorDataset(
            root_dir=data_dir,
            indices_file=val_indices_path,
            transform=transforms['val']
        )
    
    test_dataset = SemiconductorDataset(
        root_dir=data_dir,
        indices_file=os.path.join(indices_dir, 'test_indices.csv'),
        transform=transforms['test']
    )
    
    # Print dataset sizes
    print(f"Train dataset: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
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
    
    if val_loader:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader