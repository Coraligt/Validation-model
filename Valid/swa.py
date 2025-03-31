import torch
import math
from torch.optim.lr_scheduler import _LRScheduler


class StochasticWeightAveraging:
    """
    PyTorch implementation of Stochastic Weight Averaging (SWA)
    Based on the paper "Averaging Weights Leads to Wider Optima and Better Generalization"
    
    This implementation is adapted for the semiconductor leakage detection project.
    """
    def __init__(self, model, swa_start=10, swa_freq=5, swa_lr=0.0001):
        """
        Initialize SWA
        
        Args:
            model (nn.Module): PyTorch model
            swa_start (int): Epoch to start SWA
            swa_freq (int): Frequency of model averaging
            swa_lr (float): Learning rate after SWA starts
        """
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr
        
        # Initialize SWA model and counter
        self.swa_model = None
        self.swa_n = 0
        
    def update(self, optimizer, epoch):
        """
        Update SWA model
        
        Args:
            optimizer (Optimizer): PyTorch optimizer
            epoch (int): Current epoch
        """
        # Only update after swa_start and at specified frequency
        if epoch < self.swa_start or (epoch - self.swa_start) % self.swa_freq != 0:
            return
            
        # Update learning rate if specified
        if self.swa_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.swa_lr
                
        # Initialize SWA model with deep copy of current model if needed
        if self.swa_model is None:
            self.swa_model = {name: param.clone().detach() 
                             for name, param in self.model.state_dict().items()}
            self.swa_n = 1
        else:
            # Update running average of parameters
            for name, param in self.model.state_dict().items():
                self.swa_model[name].mul_(self.swa_n / (self.swa_n + 1.0))
                self.swa_model[name].add_(param.clone().detach() / (self.swa_n + 1.0))
            self.swa_n += 1
            
    def finalize(self):
        """
        Apply SWA model weights to the original model
        """
        if self.swa_model is None:
            print("Error: SWA model is not initialized.")
            return
            
        # Load SWA weights into model
        self.model.load_state_dict(self.swa_model)
        
        # Handle batch normalization if present
        if any(isinstance(m, torch.nn.BatchNorm1d) for m in self.model.modules()):
            print("Warning: Model contains batch normalization layers.")
            print("Running forward passes to update batch norm statistics is recommended.")


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warm restarts
    Adapted from PyTorch implementation to match behavior of original TensorFlow model
    """
    def __init__(self, optimizer, T_0=100, T_mult=1, eta_min=0.0001, last_epoch=-1):
        """
        Initialize scheduler
        
        Args:
            optimizer (Optimizer): PyTorch optimizer
            T_0 (int): First cycle length
            T_mult (int): Cycle length multiplication factor
            eta_min (float): Minimum learning rate
            last_epoch (int): The index of the last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """Calculate learning rate"""
        if self.T_cur == -1:
            return self.base_lrs
            
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_0)) / 2
                for base_lr in self.base_lrs]
                
    def step(self, epoch=None):
        """Step the scheduler"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0:
                self.T_cur = self.T_cur - self.T_0
                self.T_0 = self.T_0 * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_0 = self.T_0 * self.T_mult ** n
            else:
                self.T_cur = epoch
                
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def update_bn(model, dataloader, device):
    """
    Update batch normalization layer statistics using data from dataloader
    Call this function after applying SWA weights
    
    Args:
        model (nn.Module): Model
        dataloader (DataLoader): Data loader
        device: Device to run model on
    """
    # Set to training mode to update BN layers, but don't update model parameters
    model.train()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Reset running statistics in BN layers
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = 0.1  # Use default momentum
            m.training = True
    
    # Compute new BN statistics
    n = 0
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            b = inputs.size(0)
            
            # Forward pass
            model(inputs)
            n += b
            
            # Prevent processing too much data (usually just a few batches needed)
            if n > 1000:
                break
    
    # Restore parameters gradients state
    for param in model.parameters():
        param.requires_grad = True