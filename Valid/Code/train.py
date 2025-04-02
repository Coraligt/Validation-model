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

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test set and return metrics"""
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    # Initialize confusion matrix counters
    segs_TP = 0
    segs_TN = 0
    segs_FP = 0
    segs_FN = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
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
    avg_loss = test_loss / len(test_loader.dataset)
    acc = 100.0 * correct / total
    cm = [segs_TP, segs_FN, segs_FP, segs_TN]
    
    return avg_loss, acc, cm

def train(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    # Log all arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Get data transforms
    transforms = get_transforms(use_time_reverse=args.flip_time)
    
    # Create dataloaders
    logger.info("Loading data...")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms=transforms
    )
    
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
    best_acc = 0.0
    best_fb = 0.0
    
    # Start timer for overall training
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Training phase
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
        
        # Evaluation phase
        test_loss, test_acc, cm = evaluate_model(model, test_loader, criterion, device)
        fb = stats_report(cm)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log statistics
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.2f}%")
        logger.info(f"  FB Score: {fb:.4f}")
        logger.info(f"  Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")
        
        # Save best model (by accuracy)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_acc.pth'))
            logger.info(f"  Saved new best accuracy model: {best_acc:.2f}%")
        
        # Save best model (by FB score)
        if fb > best_fb:
            best_fb = fb
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_fb.pth'))
            logger.info(f"  Saved new best FB-score model: {best_fb:.4f}")
    
    # Training complete
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Final evaluation with SWA model if enabled
    if args.use_swa:
        logger.info("\nEvaluating SWA model...")
        
        # Update batch normalization statistics
        swa.update_bn(train_loader, device)
        
        # Get SWA model
        swa_model = swa.get_model()
        
        # Evaluate SWA model
        test_loss, swa_acc, cm = evaluate_model(swa_model, test_loader, criterion, device)
        swa_fb = stats_report(cm)
        
        logger.info(f"SWA Model Loss: {test_loss:.4f}")
        logger.info(f"SWA Model Accuracy: {swa_acc:.2f}%")
        logger.info(f"SWA FB Score: {swa_fb:.4f}")
        logger.info(f"SWA Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")
        
        # Save SWA model
        torch.save(swa_model.state_dict(), os.path.join(args.output_dir, 'swa_model.pth'))
        
        # Compare with best model
        if swa_acc > best_acc:
            logger.info(f"SWA improved accuracy: {best_acc:.2f}% -> {swa_acc:.2f}%")
        else:
            logger.info(f"Best model accuracy: {best_acc:.2f}% (SWA: {swa_acc:.2f}%)")
        
        if swa_fb > best_fb:
            logger.info(f"SWA improved FB score: {best_fb:.4f} -> {swa_fb:.4f}")
        else:
            logger.info(f"Best model FB score: {best_fb:.4f} (SWA: {swa_fb:.4f})")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    logger.info("\nTraining Summary:")
    logger.info(f"Best Accuracy: {best_acc:.2f}%")
    logger.info(f"Best FB Score: {best_fb:.4f}")
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