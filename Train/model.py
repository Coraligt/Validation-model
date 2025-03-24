import torch
import torch.nn as nn
import torch.nn.functional as F

def model_best():
    """
    PyTorch implementation of the model architecture from the original TensorFlow model.
    Exactly matches the architecture described in the figure with Conv1D
    with kernel size = 85 and stride = 32 followed by BatchNorm, ReLU, and MLP.
    """
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            # Conv1D layer with filters=3, kernel_size=85, strides=32
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=85, stride=32)
            self.bn1 = nn.BatchNorm1d(3)
            
            # Fully connected layers
            self.dropout1 = nn.Dropout(0.3)
            self.fc1 = nn.Linear(3 * 37, 20)  # (1250-85)//32 + 1 = 37 points after conv
            self.dropout2 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 2)
            
        def forward(self, x):
            # x shape: [batch_size, 1, 1250]
            x = self.conv1(x)      # -> [batch_size, 3, 37]
            x = self.bn1(x)
            x = F.relu(x)
            
            x = torch.flatten(x, 1)  # -> [batch_size, 3*37]
            
            x = self.dropout1(x)
            x = self.fc1(x)        # -> [batch_size, 20]
            x = F.relu(x)
            
            x = self.dropout2(x)
            x = self.fc2(x)        # -> [batch_size, 10]
            x = F.relu(x)
            
            x = self.fc3(x)        # -> [batch_size, 2]
            return x
            
    return Model()