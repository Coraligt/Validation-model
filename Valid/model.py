import torch
import torch.nn as nn
import torch.nn.functional as F

class SemiconductorModel(nn.Module):
    """
    1D CNN model for semiconductor leakage detection.
    Structure based on the original IEGM model with configurable hyperparameters.
    """
    def __init__(self, seq_length=1002, conv_filters=3, fc1_size=20, fc2_size=10, 
                 dropout1=0.3, dropout2=0.1):
        """
        Initialize the model with configurable hyperparameters.
        
        Args:
            seq_length: Length of input sequence
            conv_filters: Number of filters in convolutional layer
            fc1_size: Size of first fully connected layer
            fc2_size: Size of second fully connected layer
            dropout1: Dropout rate for first dropout layer
            dropout2: Dropout rate for second dropout layer
        """
        super(SemiconductorModel, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(1, conv_filters, kernel_size=85, stride=32)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        
        # Calculate flattened feature size after convolution
        # For input size 1002, with kernel=85, stride=32:
        # Output size = (1002 - 85) / 32 + 1 = 29.28 -> 29
        # After flattening: conv_filters * 29
        self.flatten_size = conv_filters * ((seq_length - 85) // 32 + 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 2)  # Binary classification
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, channels, seq_length]
        
        Returns:
            Output tensor of shape [batch_size, 2]
        """
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

def save_for_inference(model, path, input_shape=(1, 1, 1002)):
    """
    Save model in format suitable for inference (ONNX, TorchScript)
    
    Args:
        model: Trained PyTorch model
        path: Path to save the model
        input_shape: Expected input shape (batch_size, channels, seq_length)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Create example input
    dummy_input = torch.randn(input_shape, dtype=torch.float32)
    
    # Save as TorchScript
    torch_script_path = f"{path}.pt"
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(torch_script_path)
    print(f"TorchScript model saved to {torch_script_path}")
    
    # Save as ONNX
    try:
        onnx_path = f"{path}.onnx"
        torch.onnx.export(
            model,               # model being run
            dummy_input,         # model input (or a tuple for multiple inputs)
            onnx_path,           # where to save the model
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,    # the ONNX version to export the model to
            do_constant_folding=True,  # optimization
            input_names=['input'],   # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes={'input': {0: 'batch_size'},   # variable length axes
                          'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"Error saving ONNX model: {e}")

    
class BaselineSemiconductorModel(nn.Module):
    """
    Baseline model with multiple Conv1D layers based on GATECH-EIC LAB's TinyML contest model
    """
    def __init__(self, seq_length=1002):
        super(BaselineSemiconductorModel, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(1, 3, kernel_size=6, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(3)
        
        # Second conv block
        self.conv2 = nn.Conv1d(3, 5, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(5)
        
        # Third conv block
        self.conv3 = nn.Conv1d(5, 10, kernel_size=4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(10)
        
        # Fourth conv block
        self.conv4 = nn.Conv1d(10, 20, kernel_size=4, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(20)
        
        # Fifth conv block
        self.conv5 = nn.Conv1d(20, 20, kernel_size=4, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(20)
        
        # Calculate output size after 5 conv layers with padding
        # This will need adjustment based on exact padding and stride
        self.output_size = seq_length
        for _ in range(5):
            # Approximate size change per layer
            self.output_size = self.output_size // 1  # No change due to padding
        
        # Final feature size
        self.flatten_size = 20 * self.output_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 10)
        self.fc2 = nn.Linear(10, 2)  # Binary classification
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Fifth conv block
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x