import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingScheduler(_LRScheduler):
    """
    PyTorch implementation of the CosineAnnealing scheduler from the original code.
    """
    def __init__(self, optimizer, T_max, eta_max, eta_min=0, verbose=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        super(CosineAnnealingScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        
        if self.verbose > 0:
            print(f'Epoch {self.last_epoch+1}: CosineAnnealingScheduler setting learning rate to {lr:.6f}')
            
        return [lr for _ in self.optimizer.param_groups]