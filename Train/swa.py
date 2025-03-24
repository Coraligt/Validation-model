import torch
import torch.nn as nn
import copy
import warnings

class SWA:
    """
    PyTorch implementation of Stochastic Weight Averaging (SWA).
    This replicates the functionality of the original SWA implementation.
    """
    def __init__(self, model, start_epoch, lr_schedule="manual", swa_lr="auto", 
                 swa_lr2="auto", swa_freq=1, verbose=0):
        self.model = model
        self.start_epoch = start_epoch - 1
        self.lr_schedule = lr_schedule
        self.swa_lr = swa_lr
        self.swa_lr2 = swa_lr2 if swa_lr2 is not None else (10 * swa_lr if swa_lr != "auto" else "auto")
        self.swa_freq = swa_freq
        self.verbose = verbose
        
        # Check arguments
        if start_epoch < 2:
            raise ValueError('"start_epoch" attribute cannot be lower than 2.')
        
        schedules = ["manual", "constant", "cyclic"]
        if self.lr_schedule not in schedules:
            raise ValueError(f'"{self.lr_schedule}" is not a valid learning rate schedule')
        
        if self.lr_schedule == "cyclic" and self.swa_freq < 2:
            raise ValueError('"swa_freq" must be higher than 1 for cyclic schedule.')
        
        if self.swa_lr == "auto" and self.swa_lr2 != "auto":
            raise ValueError('"swa_lr2" cannot be manually set if "swa_lr" is automatic.')
        
        if (self.lr_schedule == "cyclic" and self.swa_lr != "auto" 
            and self.swa_lr2 != "auto" and self.swa_lr > self.swa_lr2):
            raise ValueError('"swa_lr" must be lower than "swa_lr2".')
        
        self.device = next(model.parameters()).device
        self.swa_model = None
        self.init_lr = None
    
    def on_train_begin(self, optimizer):
        """Called at the beginning of training"""
        self.init_lr = optimizer.param_groups[0]['lr']
        
        # Automatic swa_lr
        if self.swa_lr == "auto":
            self.swa_lr = 0.1 * self.init_lr
        
        if self.init_lr < self.swa_lr:
            raise ValueError('"swa_lr" must be lower than rate set in optimizer.')
        
        # Automatic swa_lr2 between initial lr and swa_lr
        if self.lr_schedule == "cyclic" and self.swa_lr2 == "auto":
            self.swa_lr2 = self.swa_lr + (self.init_lr - self.swa_lr) * 0.25
    
    def on_epoch_begin(self, optimizer, epoch):
        """Called at the beginning of each epoch"""
        self.current_epoch = epoch
        self._scheduler(epoch)
        
        # Constant schedule is updated epoch-wise
        if self.lr_schedule == "constant":
            self._update_lr(optimizer, epoch)
        
        if self.is_swa_start_epoch:
            # Create a copy of the model for SWA
            self.swa_model = copy.deepcopy(self.model)
            
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: starting stochastic weight averaging')
    
    def on_batch_begin(self, optimizer, epoch, batch):
        """Called at the beginning of each batch"""
        # Update lr each batch for cyclic lr schedule
        if self.lr_schedule == "cyclic":
            self._update_lr(optimizer, epoch, batch)
    
    def on_epoch_end(self, epoch):
        """Called at the end of each epoch"""
        if self.is_swa_epoch and not self.is_batch_norm_epoch:
            self._average_weights(epoch)
    
    def on_train_end(self):
        """Called at the end of training"""
        # Set the model weights to the SWA weights
        if self.swa_model is not None:
            self._set_swa_weights()
    
    def _scheduler(self, epoch):
        """Determine if current epoch uses SWA"""
        swa_epoch = epoch - self.start_epoch
        
        self.is_swa_epoch = epoch >= self.start_epoch and swa_epoch % self.swa_freq == 0
        self.is_swa_start_epoch = epoch == self.start_epoch
        self.is_batch_norm_epoch = False  # Not using batch norm updating in PyTorch version
    
    def _average_weights(self, epoch):
        """Average model weights with SWA model weights"""
        # Get the number of models to average
        n_models = (epoch - self.start_epoch) // self.swa_freq + 1
        
        # For each parameter in the model
        for p_swa, p_model in zip(self.swa_model.parameters(), self.model.parameters()):
            device = p_swa.device
            # Update the SWA weights
            p_swa.data = (p_swa.data * (n_models - 1) + p_model.data.to(device)) / n_models
    
    def _update_lr(self, optimizer, epoch, batch=None):
        """Update learning rate based on schedule"""
        if self.lr_schedule == "constant":
            lr = self._constant_schedule(epoch)
        elif self.lr_schedule == "cyclic":
            lr = self._cyclic_schedule(epoch, batch)
        else:
            return
        
        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    def _constant_schedule(self, epoch):
        """Constant learning rate schedule"""
        t = epoch / self.start_epoch
        lr_ratio = self.swa_lr / self.init_lr
        
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
            
        return self.init_lr * factor
    
    def _cyclic_schedule(self, epoch, batch):
        """Cyclic learning rate schedule"""
        # Calculate the current step in the cycle
        steps_per_epoch = 1000  # Approximate steps per epoch
        
        swa_epoch = (epoch - self.start_epoch) % self.swa_freq
        cycle_length = self.swa_freq * steps_per_epoch
        
        i = (swa_epoch * steps_per_epoch) + (batch + 1 if batch is not None else 0)
        
        if epoch >= self.start_epoch:
            t = (((i - 1) % cycle_length) + 1) / cycle_length
            return (1 - t) * self.swa_lr2 + t * self.swa_lr
        else:
            return self._constant_schedule(epoch)
    
    def _set_swa_weights(self):
        """Set the model weights to the SWA weights"""
        for p_model, p_swa in zip(self.model.parameters(), self.swa_model.parameters()):
            p_model.data.copy_(p_swa.data)
        
        if self.verbose > 0:
            print('\nFinal model weights set to stochastic weight average')
    
    def update_bn(self, loader, model=None):
        """
        Update batch normalization statistics for the SWA model.
        This is a simplified version since PyTorch handles most of this internally.
        """
        if model is None:
            model = self.swa_model
            
        if model is None:
            return
        
        model.train()
        
        # Reset batch norm statistics
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.reset_running_stats()
                module.momentum = 0.1  # Use default momentum
        
        # Update batch norm statistics
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                model(inputs)