import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import sys
from datetime import datetime

from model import SemiconductorModel, count_parameters
from dataset import get_dataloaders, get_transforms
from utils import set_seed, stats_report
from swa import SWA
from cosine_annealing import CosineAnnealingLR

def setup_logging(output_dir, log_file=None):
    """Setup logging configuration"""
    # Create timestamp for log file if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
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
            logging.FileHandler(log_file)  # File output
        ]
    )
    
    logging.info(f"Logging to {log_file}")
    return logging.getLogger()

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model on data loader and return metrics"""
    model.eval()
    correct = 0
    total = 0
    eval_loss = 0.0
    
    # Initialize confusion matrix counters
    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * inputs.size(0)
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
    avg_loss = eval_loss / len(data_loader.dataset)
    acc = 100.0 * correct / total
    cm = [segs_TP, segs_FN, segs_FP, segs_TN]
    
    return avg_loss, acc, cm

def train(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_file)
    
    # Log all arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device - ensure we're using the correct device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("Using CPU for training/evaluation")
    else:
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")
        
        # Print CUDA device properties if using CUDA
        if device.type == 'cuda':
            cuda_id = device.index if device.index is not None else 0
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(cuda_id)}")
            logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(cuda_id).total_memory / 1e9:.2f} GB")
    
    # Get data transforms
    transforms = get_transforms(use_time_reverse=args.flip_time)
    
    # Create dataloaders
    logger.info("Loading data...")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms=transforms
    )
    
    # Check if we have validation set
    has_val_set = len(loaders) == 3
    if has_val_set:
        train_loader, val_loader, test_loader = loaders
        logger.info("Using separate validation and test sets")
    else:
        train_loader, test_loader = loaders
        val_loader = test_loader  # Use test set as validation if no validation set
        logger.info("Using test set as validation (no separate validation set found)")
    
    # Create model
    logger.info("Creating model...")
    model = SemiconductorModel(seq_length=args.seq_length).to(device)
    logger.info(str(model))
    logger.info(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate-only mode
    if args.evaluate_only:
        if not args.model_path:
            logger.error("Model path must be specified for evaluation mode")
            return
        
        logger.info(f"Loading model from {args.model_path} for evaluation")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Start timer for evaluation
        eval_start_time = time.time()
        
        # Evaluate loaded model
        test_loss, test_acc, cm = evaluate_model(model, test_loader, criterion, device)
        fb = stats_report(cm)
        
        eval_time = time.time() - eval_start_time
        
        logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        logger.info(f"FB Score: {fb:.4f}")
        logger.info(f"Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")
        
        return
    
    # If not evaluation-only, proceed with training
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
    
    # Training metrics
    best_val_acc = 0.0
    best_val_fb = 0.0
    best_epoch = 0
    
    # Start timer for overall training
    start_time = time.time()
    
    # Save training stats for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_fbs = []
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Average training loss and accuracy for this epoch
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        
        # Save for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Update SWA if enabled
        if args.use_swa:
            swa.update(epoch, optimizer)
        
        # Update learning rate
        scheduler.step()
        
        # Validation phase
        val_loss, val_acc, val_cm = evaluate_model(model, val_loader, criterion, device)
        val_fb = stats_report(val_cm)
        
        # Save for plotting
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_fbs.append(val_fb)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log statistics
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        logger.info(f"  Val FB Score: {val_fb:.4f}")
        logger.info(f"  Val Confusion Matrix: TP={val_cm[0]}, FN={val_cm[1]}, FP={val_cm[2]}, TN={val_cm[3]}")
        
        # Save best model (by validation FB score)
        if val_fb > best_val_fb:
            best_val_fb = val_fb
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_fb.pth'))
            logger.info(f"  Saved new best model (FB: {best_val_fb:.4f}, Acc: {best_val_acc:.2f}%)")
    
    # Training complete
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    logger.info("\nEvaluating best model on test set...")
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model_fb.pth'), map_location=device))
    test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, device)
    test_fb = stats_report(test_cm)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test FB Score: {test_fb:.4f}")
    logger.info(f"Test Confusion Matrix: TP={test_cm[0]}, FN={test_cm[1]}, FP={test_cm[2]}, TN={test_cm[3]}")
    
    # Final evaluation with SWA model if enabled
    if args.use_swa:
        logger.info("\nEvaluating SWA model...")
        
        # Update batch normalization statistics
        swa.update_bn(train_loader, device)
        
        # Get SWA model
        swa_model = swa.get_model()
        
        # Evaluate SWA model on test set
        test_loss, swa_acc, cm = evaluate_model(swa_model, test_loader, criterion, device)
        swa_fb = stats_report(cm)
        
        logger.info(f"SWA Model Loss: {test_loss:.4f}")
        logger.info(f"SWA Model Accuracy: {swa_acc:.2f}%")
        logger.info(f"SWA FB Score: {swa_fb:.4f}")
        logger.info(f"SWA Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")
        
        # Save SWA model
        torch.save(swa_model.state_dict(), os.path.join(args.output_dir, 'swa_model.pth'))
        
        # Compare with best model
        if swa_acc > best_val_acc:
            logger.info(f"SWA improved accuracy: {best_val_acc:.2f}% -> {swa_acc:.2f}%")
        else:
            logger.info(f"Best model accuracy: {best_val_acc:.2f}% (SWA: {swa_acc:.2f}%)")
        
        if swa_fb > best_val_fb:
            logger.info(f"SWA improved FB score: {best_val_fb:.4f} -> {swa_fb:.4f}")
        else:
            logger.info(f"Best model FB score: {best_val_fb:.4f} (SWA: {swa_fb:.4f})")
    
    # Save training history
    import numpy as np
    import pandas as pd
    history = pd.DataFrame({
        'epoch': np.arange(1, args.epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'val_fb': val_fbs
    })
    history.to_csv(os.path.join(args.output_dir, 'training_history.csv'), index=False)
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, args.epochs + 1), train_losses, label='Train Loss')
        plt.plot(np.arange(1, args.epochs + 1), val_losses, label='Val Loss')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
        
        # Accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, args.epochs + 1), train_accs, label='Train Accuracy')
        plt.plot(np.arange(1, args.epochs + 1), val_accs, label='Val Accuracy')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'accuracy_curve.png'))
        
        # FB Score curve
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, args.epochs + 1), val_fbs, label='Validation FB Score')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('FB Score')
        plt.title('Validation FB Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, 'fb_score_curve.png'))
        
        logger.info("Training curves saved to output directory")
    except Exception as e:
        logger.warning(f"Error plotting training curves: {e}")
    
    # Final summary
    logger.info("\nTraining Summary:")
    logger.info(f"Best Epoch: {best_epoch}/{args.epochs}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best Validation FB Score: {best_val_fb:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test FB Score: {test_fb:.4f}")
    logger.info(f"Total Training Time: {training_time:.2f} seconds")
    logger.info(f"Average Time per Epoch: {training_time/args.epochs:.2f} seconds")
    logger.info(f"Models saved to: {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate semiconductor leakage detection model')
    
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
    
    # Evaluation-only mode
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate the model, no training')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the model to evaluate')
    
    # Logging parameters
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to save the log file')
    
    args = parser.parse_args()
    
    # Train or evaluate model
    train(args)

if __name__ == '__main__':
    main()