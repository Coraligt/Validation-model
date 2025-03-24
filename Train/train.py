import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LearningRateScheduler
import time

# Import local modules
from model import model_best
from data_loader import get_data_loaders
from cosine_annealing import CosineAnnealingScheduler
from swa import SWA
from utils import stats_report

def save_model(model, save_path):
    """Save model to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    
def save_to_tflite(model, save_path):
    """
    Placeholder for saving to TFLite.
    In PyTorch, we'll save to ONNX or TorchScript instead.
    """
    print(f"Saving model to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_scripted = torch.jit.script(model)
    model_scripted.save(save_path)
    print(f"Model saved to {save_path}")

def run_once(args, count):
    # Set random seeds for reproducibility
    random.seed(count)
    np.random.seed(count)
    torch.manual_seed(count)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(count)
        torch.backends.cudnn.deterministic = True
    
    # Device configuration
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and args.cuda >= 0 else 'cpu')
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(args)
    
    # Build model
    model = model_best()
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingScheduler(optimizer, T_max=args.epoch, eta_max=4e-4, eta_min=2e-4)
    else:
        scheduler = LearningRateScheduler(lambda epoch: args.lr * 0.5 ** (epoch // 10))
    
    # SWA configuration
    swa = SWA(model, start_epoch=10, lr_schedule='cyclic', 
              swa_lr=0.0001, swa_lr2=0.0005, swa_freq=5, verbose=1)
    swa.on_train_begin(optimizer)
    
    # Checkpoint path
    save_name = f'random_{count}'
    checkpoint_filepath = f'./20_10/{save_name}/'
    os.makedirs(checkpoint_filepath, exist_ok=True)
    
    # Best validation accuracy
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.epoch):
        model.train()
        running_loss = 0.0
        
        # SWA on epoch begin
        swa.on_epoch_begin(optimizer, epoch)
        
        # Train for one epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # SWA on batch begin
            swa.on_batch_begin(optimizer, epoch, batch_idx)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Update learning rate
        scheduler.step()
        
        # Validate model
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = correct / total
        print(f'Epoch [{epoch+1}/{args.epoch}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(checkpoint_filepath, 'best_model.pth'))
        
        # SWA on epoch end
        swa.on_epoch_end(epoch)
    
    # SWA on train end
    swa.on_train_end()
    swa.update_bn(train_loader)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(checkpoint_filepath, 'best_model.pth')))
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    
    # For confusion matrix
    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update confusion matrix
            for p, t in zip(predicted, target):
                if t.item() == 0:
                    if p.item() == 0:
                        segs_TN += 1
                    else:
                        segs_FP += 1
                else:
                    if p.item() == 1:
                        segs_TP += 1
                    else:
                        segs_FN += 1
    
    score = correct / total
    print(f'Model: {save_name}')
    print(f'Accuracy: {score:.4f}')
    
    # Save model to TorchScript (as a placeholder for TFLite)
    save_to_tflite(model, f'./ckpt/{save_name}.pt')
    
    # Calculate and print statistics
    FB = stats_report([segs_TP, segs_FN, segs_FP, segs_TN])
    
    return FB, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='epoch number', default=100)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0002)
    parser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--size', type=int, default=1250)
    parser.add_argument('--path_data', type=str, default='./tinyml_contest_data_training/')
    parser.add_argument('--path_indices', type=str, default='./data_indices')
    parser.add_argument('--lr_scheduler', type=str, default='cosine')
    
    args = parser.parse_args()
    
    best_FB = 0.0
    for i in range(10):
        FB, model = run_once(args, i)
        if FB > best_FB:
            best_FB = FB
            save_to_tflite(model, f'./20_10/best_{i}.pt')
            print(f'Current Best: {best_FB}')
        print(FB)
    print(f'Current Best: {best_FB}')

if __name__ == '__main__':
    main()