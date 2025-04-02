import torch
import torch.nn as nn
import torch.nn.functional as F

class SemiconductorModel(nn.Module):
    """
    1D CNN model for semiconductor leakage detection.
    Structure identical to the original IEGM model.
    """
    def __init__(self, seq_length=1002):
        super(SemiconductorModel, self).__init__()
        
        # First convolutional layer
        # (in_channels=1, out_channels=3, kernel_size=85, stride=32)
        self.conv1 = nn.Conv1d(1, 3, kernel_size=85, stride=32)
        self.bn1 = nn.BatchNorm1d(3)
        
        # Calculate flattened feature size after convolution
        # For input size 1002, with kernel=85, stride=32:
        # Output size = (1002 - 85) / 32 + 1 = 29.28 -> 29
        # After flattening: 3 * 29 = 87
        self.flatten_size = 3 * ((seq_length - 85) // 32 + 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)  # Binary classification
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)
        
    def forward(self, x):
        # Input shape: [batch_size, 1, seq_length]
        
        # Apply convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # First dense layer with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Second dense layer with dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

def count_parameters(model):
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)