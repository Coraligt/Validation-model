import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DMLP(nn.Module):
    """
    Semiconductor leakage detection model, adapted from the original TinyML contest model
    1D CNN + MLP head for binary classification
    """
    def __init__(self, in_channels=2, seq_length=1002):
        """
        Initialize model
        
        Args:
            in_channels (int): Number of input channels (features)
            seq_length (int): Length of input sequence
        """
        super(Conv1DMLP, self).__init__()
        
        # First layer: Conv1D with same parameters as original model
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=3, 
                              kernel_size=85, 
                              stride=32, 
                              padding=0)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(3)
        
        # Calculate size after convolution
        # size = (seq_length - kernel_size) / stride + 1
        # For seq_length=1002, kernel_size=85, stride=32: size = (1002 - 85) / 32 + 1 = 29.28 -> 29
        conv_output_size = ((seq_length - 85) // 32) + 1
        self.conv_output_size = conv_output_size * 3  # 3 filters * output size
        
        # Fully connected layers (same as original model)
        self.fc1 = nn.Linear(self.conv_output_size, 20)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(20, 10)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(10, 2)  # Binary classification: non-leaky/leaky
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, in_channels, seq_length]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, 2]
        """
        # Convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class SmallConv1DMLP(nn.Module):
    """
    Smaller version of Conv1DMLP model with fewer parameters
    Useful for testing or resource-constrained environments
    """
    def __init__(self, in_channels=2, seq_length=1002):
        super(SmallConv1DMLP, self).__init__()
        
        # First layer: Conv1D with 2 filters (fewer than standard model)
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=2, 
                              kernel_size=85, 
                              stride=32, 
                              padding=0)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(2)
        
        # Calculate size after convolution
        conv_output_size = ((seq_length - 85) // 32) + 1
        self.conv_output_size = conv_output_size * 2  # 2 filters * output size
        
        # Simplified fully connected layers
        self.fc1 = nn.Linear(self.conv_output_size, 10)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(10, 2)  # Binary classification
        
    def forward(self, x):
        # Convolutional layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


def count_parameters(model):
    """
    Count number of trainable parameters in model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type='default', in_channels=2, seq_length=1002):
    """
    Get model of specified type
    
    Args:
        model_type (str): Model type ('default' or 'small')
        in_channels (int): Number of input channels
        seq_length (int): Sequence length
        
    Returns:
        nn.Module: Model instance
    """
    if model_type == 'small':
        model = SmallConv1DMLP(in_channels=in_channels, seq_length=seq_length)
    else:
        model = Conv1DMLP(in_channels=in_channels, seq_length=seq_length)
    
    # Calculate and print model parameters
    num_params = count_parameters(model)
    print(f"Total model parameters: {num_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model
    model = Conv1DMLP(in_channels=2, seq_length=1002)
    batch_size = 8
    x = torch.randn(batch_size, 2, 1002)  # Batch size 8, 2 channels, sequence length 1002
    
    # Forward pass
    y = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Print model structure
    print(model)
    
    # Print parameter count
    num_params = count_parameters(model)
    print(f"Total model parameters: {num_params:,}")