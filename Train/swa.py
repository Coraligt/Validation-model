import torch
import copy
from collections import defaultdict

class SWA:
    """
    PyTorch implementation of Stochastic Weight Averaging.
    Based on the SWA implementation in the original code but simplified for PyTorch.
    """
    def __init__(self, model, swa_start, swa_freq=1, swa_lr=None, verbose=True):
        """
        Initialize SWA.
        
        Args:
            model: PyTorch model
            swa_start: epoch to start SWA from (0-indexed)
            swa_freq: frequency of models to include in average
            swa_lr: SWA learning rate (if None, uses the optimizer's learning rate)
            verbose: verbosity flag
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        self.verbose = verbose
        
        # Initialize SWA model with a deepcopy of the model
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0
    
    def update(self, epoch, optimizer=None):
        """
        Update SWA model parameters.
        
        Args:
            epoch: current epoch (0-indexed)
            optimizer: optimizer for SWA learning rate adjustment
        """
        # Check if we should update at this epoch
        if (epoch + 1) >= self.swa_start and ((epoch + 1 - self.swa_start) % self.swa_freq == 0):
            if self.verbose:
                print(f"SWA: Updating average model at epoch {epoch + 1}")
                
            # Adjust learning rate if specified
            if self.swa_lr is not None and optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.swa_lr
            
            # Update the parameters of swa_model
            if self.swa_n == 0:
                # First model to include in average
                for swa_p, p in zip(self.swa_model.parameters(), self.model.parameters()):
                    swa_p.data.copy_(p.data)
            else:
                # Update average
                for swa_p, p in zip(self.swa_model.parameters(), self.model.parameters()):
                    swa_p.data.mul_(self.swa_n / (self.swa_n + 1))
                    swa_p.data.add_(p.data / (self.swa_n + 1))
            
            self.swa_n += 1
            
    def get_model(self):
        """
        Return the SWA model.
        """
        return self.swa_model
    
    def update_bn(self, train_loader, device):
        """
        Update batch normalization statistics for the SWA model.
        
        Args:
            train_loader: training data loader
            device: device to run the model on
        """
        if self.verbose:
            print("SWA: Updating batch normalization statistics")
            
        # Set model to evaluation mode
        self.swa_model.train()
        
        # Reset batch norm statistics
        for module in self.swa_model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.reset_running_stats()
                module.momentum = None  # Use cumulative moving average
        
        # Forward pass through all training data to update BN statistics
        with torch.no_grad():
            for batch in train_loader:
                IEGM_seg = batch['IEGM_seg'].to(device)
                self.swa_model(IEGM_seg)  