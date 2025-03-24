import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv
import random

def loadCSV(csvf):
    """
    Load a CSV file into a dictionary mapping labels to filenames.
    Exact copy of the functionality from the original code.
    
    Args:
        csvf: CSV file path
    Returns:
        dictLabels: Dict mapping label to list of filenames
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip header
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]
            
            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels

def txt_to_numpy(filename, row):
    """
    Convert a text file to a numpy array.
    Exact copy of the functionality from the original code.
    
    Args:
        filename: Text file path
        row: Number of rows to read
    Returns:
        datamat: Numpy array with the data
    """
    file = open(filename)
    lines = file.readlines()
    datamat = np.zeros(row, dtype=np.float32)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = float(line[0])
        row_count += 1
    
    return datamat

class IEGM_DataSET(Dataset):
    """
    PyTorch Dataset for IEGM data.
    Follows the original implementation closely.
    """
    def __init__(self, root_dir, indice_dir, mode, size, transform=None, data_aug=False):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform
        self.data_aug = data_aug
        
        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))
        
        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))
    
    def __len__(self):
        return len(self.names_list)
    
    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]
        
        if not os.path.isfile(text_path):
            print(text_path + ' does not exist')
            return None
        
        # Load data
        IEGM_seg = txt_to_numpy(text_path, self.size)
        label = int(self.names_list[idx].split(' ')[1])
        
        # Apply data augmentation if enabled
        if self.data_aug:
            # Flip peak
            if random.random() < 0.5:
                IEGM_seg = -IEGM_seg
            
            # Add noise
            max_peak = np.max(np.abs(IEGM_seg)) * 0.05
            factor = random.random()
            noise = np.random.normal(0, factor * max_peak, self.size)
            IEGM_seg = IEGM_seg + noise
        
        # Reshape to [C, L] format for PyTorch (channels first)
        IEGM_seg = IEGM_seg.reshape(1, self.size)
        
        # Convert to tensor
        IEGM_tensor = torch.from_numpy(IEGM_seg).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return IEGM_tensor, label_tensor

def get_data_loaders(args):
    """
    Create data loaders for training and testing.
    
    Args:
        args: Arguments object with data paths and parameters
    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    # Create datasets
    train_dataset = IEGM_DataSET(
        root_dir=args.path_data,
        indice_dir=args.path_indices,
        mode='train',
        size=args.size,
        data_aug=True
    )
    
    test_dataset = IEGM_DataSET(
        root_dir=args.path_data,
        indice_dir=args.path_indices,
        mode='test',
        size=args.size,
        data_aug=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsz,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsz,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader