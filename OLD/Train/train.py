import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our modules
from model import IEGMModel
from dataset import IEGMDataset
from cosine_annealing import CosineAnnealingLR
from swa import SWA
from utils import stats_report

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataAugmentation:
    """Data augmentation for IEGM data."""
    def __init__(self, flip_peak=True, flip_time=False, add_noise=True):
        self.flip_peak = flip_peak
        self.flip_time = flip_time
        self.add_noise = add_noise
    
    def __call__(self, sample):
        IEGM_seg, label = sample['IEGM_seg'], sample['label']
        
        # Convert to numpy for easier manipulation
        IEGM_np = IEGM_seg.numpy()
        
        # Flip peak (invert signal)
        if self.flip_peak and random.random() < 0.5:
            IEGM_np = -IEGM_np
        
        # Flip time (reverse signal)
        if self.flip_time and random.random() < 0.5:
            IEGM_np = np.flip(IEGM_np, axis=1)
        
        # Add Gaussian noise
        if self.add_noise:
            max_peak = np.max(np.abs(IEGM_np)) * 0.05
            factor = random.random()
            noise = np.random.normal(0, factor * max_peak, IEGM_np.shape)
            IEGM_np = IEGM_np + noise
        
        # Convert back to torch tensor
        IEGM_seg = torch.from_numpy(IEGM_np).float()
        
        return {'IEGM_seg': IEGM_seg, 'label': label}

def save_model(model, file_path):
    """Save PyTorch model to file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def run_once(count, args):
    """Train and evaluate the model once."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets and data loaders
    train_dataset = IEGMDataset(
        root_dir=args.path_data, 
        indice_dir=args.path_indices, 
        mode='train', 
        size=args.size
    )
    
    test_dataset = IEGMDataset(
        root_dir=args.path_data, 
        indice_dir=args.path_indices, 
        mode='test', 
        size=args.size
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batchsz, 
        shuffle=True,
        num_workers=args.workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batchsz, 
        shuffle=False,
        num_workers=args.workers
    )
    
    # Create model
    model = IEGMModel().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epoch, 
        eta_max=args.lr, 
        eta_min=args.lr * 0.01
    )
    
    # Initialize SWA
    swa_start = 10  # Start SWA from epoch 10
    swa = SWA(model, swa_start=swa_start, swa_freq=5, swa_lr=0.0001)
    
    # Create directory for saving checkpoints
    save_name = f'random_{count}'
    checkpoint_filepath = os.path.join('./checkpoints', save_name)
    os.makedirs(checkpoint_filepath, exist_ok=True)
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        
        # Apply data augmentation directly within the training loop
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch}")):
            IEGM_seg = batch['IEGM_seg'].to(device)
            labels = batch['label'].to(device)
            
            # Data augmentation
            if args.data_aug:
                # Create augmented samples
                IEGM_np = IEGM_seg.cpu().numpy()
                
                # Flip peak (invert signal)
                if args.flip_peak:
                    flip_mask = (torch.rand(IEGM_seg.size(0)) < 0.5).numpy()
                    IEGM_np[flip_mask] = -IEGM_np[flip_mask]
                
                # Flip time (reverse signal)
                if args.flip_time:
                    flip_mask = (torch.rand(IEGM_seg.size(0)) < 0.5).numpy()
                    for idx in np.where(flip_mask)[0]:
                        IEGM_np[idx] = np.flip(IEGM_np[idx], axis=1)
                
                # Add Gaussian noise
                if args.add_noise:
                    for idx in range(IEGM_np.shape[0]):
                        max_peak = np.max(np.abs(IEGM_np[idx])) * 0.05
                        factor = random.random()
                        noise = np.random.normal(0, factor * max_peak, IEGM_np[idx].shape)
                        IEGM_np[idx] = IEGM_np[idx] + noise
                
                # Convert back to torch tensor
                IEGM_seg = torch.from_numpy(IEGM_np).float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(IEGM_seg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
        
        # Update SWA if in SWA phase
        swa.update(epoch, optimizer)
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                IEGM_seg = batch['IEGM_seg'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(IEGM_seg)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            save_model(model, os.path.join(checkpoint_filepath, 'best_model.pth'))
    
    # Use SWA model for final evaluation
    if epoch >= swa_start:
        # Update batch normalization statistics
        swa.update_bn(train_loader, device)
        swa_model = swa.get_model()
        
        # Evaluate SWA model
        swa_model.eval()
        correct = 0
        total = 0
        
        # Initialize confusion matrix counters
        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0
        
        with torch.no_grad():
            for batch in test_loader:
                IEGM_seg = batch['IEGM_seg'].to(device)
                labels = batch['label'].to(device)
                
                outputs = swa_model(IEGM_seg)
                _, predicted = torch.max(outputs, 1)
                
                # Update confusion matrix
                for p, l in zip(predicted, labels):
                    if l.item() == 0:  # Negative class
                        if p.item() == 0:
                            segs_TN += 1
                        else:
                            segs_FP += 1
                    else:  # Positive class
                        if p.item() == 1:
                            segs_TP += 1
                        else:
                            segs_FN += 1
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        swa_acc = 100 * correct / total
        print(f"SWA Model Test Accuracy: {swa_acc:.2f}%")
        
        # Calculate metrics
        FB = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
        
        # Save SWA model
        save_model(swa_model, os.path.join(checkpoint_filepath, 'swa_model.pth'))
        
        return FB, swa_model
    else:
        # If SWA not used, evaluate regular model
        model.eval()
        
        # Initialize confusion matrix counters
        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0
        
        with torch.no_grad():
            for batch in test_loader:
                IEGM_seg = batch['IEGM_seg'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(IEGM_seg)
                _, predicted = torch.max(outputs, 1)
                
                # Update confusion matrix
                for p, l in zip(predicted, labels):
                    if l.item() == 0:  # Negative class
                        if p.item() == 0:
                            segs_TN += 1
                        else:
                            segs_FP += 1
                    else:  # Positive class
                        if p.item() == 1:
                            segs_TP += 1
                        else:
                            segs_FN += 1
        
        # Calculate metrics
        FB = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
        
        return FB, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='epoch number', default=50)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    parser.add_argument('--batchsz', type=int, help='total batch size for all GPUs', default=32)
    parser.add_argument('--size', type=int, default=1250)
    parser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    parser.add_argument('--path_indices', type=str, default='./data_indices')
    parser.add_argument('--no-cuda', action='store_true', help='disable CUDA')
    parser.add_argument('--data_aug', action='store_true', help='enable data augmentation', default=True)
    parser.add_argument('--flip_peak', action='store_true', help='flip peak in data augmentation', default=True)
    parser.add_argument('--flip_time', action='store_true', help='flip time in data augmentation', default=False)
    parser.add_argument('--add_noise', action='store_true', help='add noise in data augmentation', default=True)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--seed', type=int, help='random seed', default=42)
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    best_FB = 0.0
    best_model = None
    
    # Run multiple times with different random initializations
    for i in range(10):
        print(f"Run {i+1}/10")
        FB, model = run_once(i, args)
        if FB > best_FB:
            best_FB = FB
            best_model = model
            save_model(best_model, f'./checkpoints/best_{i}.pth')
            print(f'Current Best FB: {best_FB}')
        print(f'Run {i+1} FB: {FB}')
    
    print(f'Final Best FB: {best_FB}')
    
    # Export to ONNX format for potential conversion to TFLite later
    if best_model is not None:
        dummy_input = torch.randn(1, 1, args.size, device='cpu')
        best_model = best_model.to('cpu')
        best_model.eval()
        torch.onnx.export(
            best_model,
            dummy_input,
            './checkpoints/best_model.onnx',
            verbose=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("Model exported to ONNX format")

if __name__ == '__main__':
    main()