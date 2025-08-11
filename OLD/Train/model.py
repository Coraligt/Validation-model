import torch
import torch.nn as nn
import torch.nn.functional as F

class IEGMModel(nn.Module):
    """
    PyTorch implementation of the IEGM classification model.
    Equivalent to the TensorFlow model_best() function in the original code.
    """
    def __init__(self):
        super(IEGMModel, self).__init__()
        # Conv1D layer (filters=3, kernel_size=85, strides=32)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=85, stride=32, padding=0)
        self.bn1 = nn.BatchNorm1d(3)
        
        # Calculate the size after Conv1D and flatten
        # For input size 1250, with kernel=85, stride=32, padding=0
        # Output size = (1250 - 85) / 32 + 1 = 37 (rounded up)
        # After flattening: 3 * 37 = 111
        self.flatten_size = 3 * 37
        
        # MLP layers
        self.fc1 = nn.Linear(self.flatten_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, 1250)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # First dense layer with dropout
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        
        # Second dense layer with dropout
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        
        # Output layer
        x = self.fc3(x)
        
        return x