import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def loadCSV(csvf):
    """
    Load a CSV file and return a dictionary mapping labels to filenames.
    Identical to the loadCSV function in the original code.
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
    Identical to the txt_to_numpy function in the original code.
    """
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.float32)  # Changed to float32 for PyTorch
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat

class IEGMDataset(Dataset):
    """
    PyTorch Dataset for IEGM data.
    Equivalent to IEGM_DataSET in the original code.
    """
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + ' does not exist')
            # Return a default sample to avoid crashing
            return {'IEGM_seg': torch.zeros(1, self.size, 1), 'label': 0}

        IEGM_seg = txt_to_numpy(text_path, self.size)
        
        # Reshape for PyTorch: [channels, sequence_length]
        IEGM_seg = IEGM_seg.reshape(1, self.size)
        
        # Convert to PyTorch tensor
        IEGM_seg = torch.from_numpy(IEGM_seg).float()
        
        label = int(self.names_list[idx].split(' ')[1])
        label = torch.tensor(label, dtype=torch.long)
        
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample