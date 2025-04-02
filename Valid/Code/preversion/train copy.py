import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from model import SemiconductorModel, count_parameters
from dataset import get_dataloaders, get_transforms
from utils import set_seed, stats_report
from swa import SWA
from cosine_annealing import CosineAnnealingLR

def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )

    # Add file handler if log file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logging.info(f"Logging to {log_file}")
    
    return logging.getLogger()

def train(args):
    """Main training function"""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Get data transforms
    transforms = get_transforms(use_time_reverse=args.flip_time)
    
    # Create dataloaders
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms=transforms
    )
    
    # Create model
    model = SemiconductorModel(seq_length=args.seq_length).to(device)
    print(model)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_max=args.learning_rate,
        eta_min=args.learning_rate * 0.01
    )
    
    # Initialize SWA if enabled
    if args.use_swa:
        swa = SWA(model, 
                 swa_start=args.swa_start, 
                 swa_freq=args.swa_freq, 
                 swa_lr=args.swa_lr)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training metrics
    best_acc = 0.0
    best_fb = 0.0
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        # Training
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Average training loss for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        
        # Update SWA if enabled
        if args.use_swa:
            swa.update(epoch, optimizer)
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        # Initialize confusion matrix counters
        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Update counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update confusion matrix
                for p, l in zip(predicted, labels):
                    if l.item() == 0:  # Non-leaky (negative class)
                        if p.item() == 0:
                            segs_TN += 1
                        else:
                            segs_FP += 1
                    else:  # Leaky (positive class)
                        if p.item() == 1:
                            segs_TP += 1
                        else:
                            segs_FN += 1
        
        # Calculate metrics
        acc = 100.0 * correct / total
        cm = [segs_TP, segs_FN, segs_FP, segs_TN]
        fb = stats_report(cm)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Accuracy: {acc:.2f}%")
        print(f"Confusion Matrix: TP={segs_TP}, FN={segs_FN}, FP={segs_FP}, TN={segs_TN}")
        
        # Save best model (by accuracy)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_acc.pth'))
            print(f"Saved new best accuracy model: {best_acc:.2f}%")
        
        # Save best model (by FB score)
        if fb > best_fb:
            best_fb = fb
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_fb.pth'))
            print(f"Saved new best FB-score model: {best_fb:.4f}")
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Final evaluation with SWA model if enabled
    if args.use_swa:
        print("\nEvaluating SWA model...")
        
        # Update batch normalization statistics
        swa.update_bn(train_loader, device)
        
        # Get SWA model
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
            for inputs, labels in tqdm(test_loader, desc="SWA Evaluation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = swa_model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Update counters
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update confusion matrix
                for p, l in zip(predicted, labels):
                    if l.item() == 0:  # Non-leaky (negative class)
                        if p.item() == 0:
                            segs_TN += 1
                        else:
                            segs_FP += 1
                    else:  # Leaky (positive class)
                        if p.item() == 1:
                            segs_TP += 1
                        else:
                            segs_FN += 1
        
        # Calculate metrics
        swa_acc = 100.0 * correct / total
        cm = [segs_TP, segs_FN, segs_FP, segs_TN]
        swa_fb = stats_report(cm)
        
        print(f"SWA Model Accuracy: {swa_acc:.2f}%")
        
        # Save SWA model
        torch.save(swa_model.state_dict(), os.path.join(args.output_dir, 'swa_model.pth'))
        
        # Compare with best model
        if swa_acc > best_acc:
            print(f"SWA improved accuracy: {best_acc:.2f}% -> {swa_acc:.2f}%")
        else:
            print(f"Best model accuracy: {best_acc:.2f}% (SWA: {swa_acc:.2f}%)")
        
        if swa_fb > best_fb:
            print(f"SWA improved FB score: {best_fb:.4f} -> {swa_fb:.4f}")
        else:
            print(f"Best model FB score: {best_fb:.4f} (SWA: {swa_fb:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    print("\nTraining Summary:")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best FB Score: {best_fb:.4f}")
    print(f"Models saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train semiconductor leakage detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./Valid/indices/preprocessed',
                       help='Directory with preprocessed CSV files')
    parser.add_argument('--indices_dir', type=str, default='./Valid/indices',
                       help='Directory with indices CSV files')
    parser.add_argument('--output_dir', type=str, default='./model_output',
                       help='Directory to save model checkpoints')
    parser.add_argument('--seq_length', type=int, default=1002,
                       help='Length of input sequence')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Data augmentation
    parser.add_argument('--flip_signal', action='store_true', default=True,
                       help='Enable signal flipping augmentation')
    parser.add_argument('--add_noise', action='store_true', default=True,
                       help='Enable noise addition augmentation')
    parser.add_argument('--flip_time', action='store_true', default=False,
                       help='Enable time reversal augmentation')
    
    # SWA parameters
    parser.add_argument('--use_swa', action='store_true',
                       help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=10,
                       help='Epoch to start SWA from')
    parser.add_argument('--swa_freq', type=int, default=5,
                       help='SWA model collection frequency')
    parser.add_argument('--swa_lr', type=float, default=0.0001,
                       help='SWA learning rate')
    
    args = parser.parse_args()
    
    # Print arguments
    print("Training with the following parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Train model
    train(args)

if __name__ == '__main__':
    main()

