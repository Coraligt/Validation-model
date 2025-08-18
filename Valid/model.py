import torch
import torch.nn as nn
import torch.nn.functional as F

class DualLabelSemiconductorModel(nn.Module):
    """
    Enhanced 1D CNN model for semiconductor leakage detection with voltage flag information.
    Can handle both leaky classification and voltage loop type.
    """
    def __init__(self, seq_length=1002, conv_filters=3, fc1_size=20, fc2_size=10, 
                 dropout1=0.3, dropout2=0.1, use_voltage_embedding=True):
        """
        Initialize the model with voltage embedding capability.
        
        Args:
            seq_length: Length of input sequence
            conv_filters: Number of filters in convolutional layer
            fc1_size: Size of first fully connected layer
            fc2_size: Size of second fully connected layer
            dropout1: Dropout rate for first dropout layer
            dropout2: Dropout rate for second dropout layer
            use_voltage_embedding: Whether to use voltage flag as additional information
        """
        super(DualLabelSemiconductorModel, self).__init__()
        
        self.use_voltage_embedding = use_voltage_embedding
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(1, conv_filters, kernel_size=85, stride=32)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        
        # Additional conv layer for richer feature extraction
        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_filters * 2)
        
        # Calculate flattened feature size
        conv1_out_length = (seq_length - 85) // 32 + 1
        conv2_out_length = conv1_out_length  # Due to padding=1 in conv2
        self.flatten_size = conv_filters * 2 * conv2_out_length
        
        # Voltage embedding layer (optional)
        if self.use_voltage_embedding:
            self.voltage_embedding = nn.Embedding(2, 8)  # 2 voltage types, 8-dim embedding
            fc_input_size = self.flatten_size + 8
        else:
            fc_input_size = self.flatten_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 2)  # Binary classification for leakage
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x, voltage_flag=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, channels, seq_length]
            voltage_flag: Optional voltage flag tensor [batch_size] (0: minor, 1: major)
        
        Returns:
            Output tensor of shape [batch_size, 2]
        """
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Add voltage embedding if available
        if self.use_voltage_embedding and voltage_flag is not None:
            voltage_emb = self.voltage_embedding(voltage_flag)
            x = torch.cat([x, voltage_emb], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class MultiTaskSemiconductorModel(nn.Module):
    """
    Multi-task learning model that predicts both leakage and voltage type.
    This can help the model learn better representations.
    """
    def __init__(self, seq_length=1002, conv_filters=3, fc1_size=20, fc2_size=10, 
                 dropout1=0.3, dropout2=0.1):
        super(MultiTaskSemiconductorModel, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv1d(1, conv_filters, kernel_size=85, stride=32)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        
        self.conv2 = nn.Conv1d(conv_filters, conv_filters * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_filters * 2)
        
        # Calculate flattened size
        conv1_out_length = (seq_length - 85) // 32 + 1
        self.flatten_size = conv_filters * 2 * conv1_out_length
        
        # Shared fully connected layer
        self.fc_shared = nn.Linear(self.flatten_size, fc1_size)
        self.dropout1 = nn.Dropout(dropout1)
        
        # Task-specific heads
        # Leakage detection head
        self.fc_leaky1 = nn.Linear(fc1_size, fc2_size)
        self.fc_leaky2 = nn.Linear(fc2_size, 2)
        
        # Voltage type detection head
        self.fc_voltage1 = nn.Linear(fc1_size, fc2_size)
        self.fc_voltage2 = nn.Linear(fc2_size, 2)
        
        self.dropout2 = nn.Dropout(dropout2)
        
    def forward(self, x):
        """
        Forward pass for multi-task learning
        
        Args:
            x: Input tensor [batch_size, channels, seq_length]
        
        Returns:
            tuple: (leaky_logits, voltage_logits)
        """
        # Shared feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared FC layer
        shared_features = F.relu(self.fc_shared(x))
        shared_features = self.dropout1(shared_features)
        
        # Leakage detection branch
        leaky_features = F.relu(self.fc_leaky1(shared_features))
        leaky_features = self.dropout2(leaky_features)
        leaky_output = self.fc_leaky2(leaky_features)
        
        # Voltage type detection branch
        voltage_features = F.relu(self.fc_voltage1(shared_features))
        voltage_features = self.dropout2(voltage_features)
        voltage_output = self.fc_voltage2(voltage_features)
        
        return leaky_output, voltage_output

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
        
        # Instead of calculating the output size mathematically, we'll determine the
        # actual size by doing a forward pass with a dummy tensor
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, seq_length)
            dummy_output = self._forward_features(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).size(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 10)
        self.fc2 = nn.Linear(10, 2)  # Binary classification
    
    def _forward_features(self, x):
        """Extract features through convolutional layers"""
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
        
        return x
        
    def forward(self, x):
        """Forward pass through the entire model"""
        # Extract features through conv layers
        x = self._forward_features(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
