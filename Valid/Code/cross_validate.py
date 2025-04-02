import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from model import SemiconductorModel, count_parameters
from dataset import SemiconductorDataset, get_transforms
from utils import set_seed, stats_report, ACC, F1, FB
from cosine_annealing import CosineAnnealingLR

def setup_logging(output_dir, log_file=None):
    """Setup logging configuration"""
    # Create timestamp for log file if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"cross_validation_{timestamp}.log")
    
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

def train_and_evaluate_fold(model, train_loader, val_loader, device, criterion, optimizer, 
                           scheduler, epochs, fold_idx, logger):
    """
    Train and evaluate model on a single fold
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to use for training
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epochs: Number of training epochs
        fold_idx: Current fold index
        logger: Logger
    
    Returns:
        Dictionary with training results
    """
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_fb = 0.0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training time tracking
    fold_start_time = time.time()
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Fold {fold_idx+1}, Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Fold {fold_idx+1}, Epoch {epoch+1}/{epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # Confusion matrix for validation set
        tp = np.sum((val_labels == 1) & (val_preds == 1))
        fn = np.sum((val_labels == 1) & (val_preds == 0))
        fp = np.sum((val_labels == 0) & (val_preds == 1))
        tn = np.sum((val_labels == 0) & (val_preds == 0))
        cm = [tp, fn, fp, tn]
        
        val_acc = ACC(cm)
        val_f1 = F1(cm)
        val_fb = FB(cm, beta=2)
        
        # Record metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc * 100)  # Convert to percentage
        
        # Save best model
        if val_fb > best_val_fb:
            best_val_fb = val_fb
            best_model_state = model.state_dict().copy()
            
        # Track best metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Log epoch results
        logger.info(f"Fold {fold_idx+1}, Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc*100:.2f}%, F1: {val_f1:.4f}, FB: {val_fb:.4f}")
        logger.info(f"  Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    
    # Calculate total fold time
    fold_time = time.time() - fold_start_time
    avg_epoch_time = np.mean(epoch_times)
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    # Return fold results
    return {
        'model': model,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,
        'best_val_fb': best_val_fb,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'fold_time': fold_time,
        'avg_epoch_time': avg_epoch_time
    }

def cross_validate(args):
    """
    Perform k-fold cross-validation
    
    Args:
        args: Command line arguments
    """
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.log_file)
    
    # Log arguments
    logger.info("Cross-validation with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info("Loading dataset...")
    train_transforms = get_transforms(use_time_reverse=args.flip_time)['train']
    
    # Use only train indices for cross-validation
    train_dataset = SemiconductorDataset(
        root_dir=args.data_dir,
        indices_file=os.path.join(args.indices_dir, 'train_indices.csv'),
        transform=train_transforms
    )
    
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    
    # Setup k-fold cross-validation
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    # Track metrics across all folds
    fold_accuracies = []
    fold_f1_scores = []
    fold_fb_scores = []
    fold_train_times = []
    fold_epoch_times = []
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    
    # Start timer for overall cross-validation
    cv_start_time = time.time()
    
    # Perform k-fold cross-validation
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(np.arange(len(train_dataset)))):
        logger.info(f"\nStarting fold {fold_idx+1}/{args.k_folds}")
        
        # Create data loaders for this fold
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        
        logger.info(f"Fold {fold_idx+1}: {len(train_subset)} training samples, {len(val_subset)} validation samples")
        
        # Create new model for this fold
        model = SemiconductorModel(seq_length=args.seq_length).to(device)
        
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
        
        # Train and evaluate on this fold
        fold_results = train_and_evaluate_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=args.epochs,
            fold_idx=fold_idx,
            logger=logger
        )
        
        # Save fold model
        torch.save(fold_results['model'].state_dict(), 
                  os.path.join(args.output_dir, f'fold_{fold_idx+1}_model.pth'))
        
        # Record fold metrics
        fold_accuracies.append(fold_results['best_val_acc'])
        fold_f1_scores.append(fold_results['best_val_f1'])
        fold_fb_scores.append(fold_results['best_val_fb'])
        fold_train_times.append(fold_results['fold_time'])
        fold_epoch_times.append(fold_results['avg_epoch_time'])
        
        # Collect learning curves
        all_train_losses.append(fold_results['train_losses'])
        all_val_losses.append(fold_results['val_losses'])
        all_train_accs.append(fold_results['train_accs'])
        all_val_accs.append(fold_results['val_accs'])
        
        # Log fold results
        logger.info(f"Fold {fold_idx+1} completed in {fold_results['fold_time']:.2f} seconds")
        logger.info(f"Best accuracy: {fold_results['best_val_acc']*100:.2f}%")
        logger.info(f"Best F1 score: {fold_results['best_val_f1']:.4f}")
        logger.info(f"Best FB score: {fold_results['best_val_fb']:.4f}")
    
    # Calculate cross-validation time
    cv_time = time.time() - cv_start_time
    
    # Calculate average metrics across all folds
    avg_accuracy = np.mean(fold_accuracies)
    avg_f1_score = np.mean(fold_f1_scores)
    avg_fb_score = np.mean(fold_fb_scores)
    avg_fold_time = np.mean(fold_train_times)
    avg_epoch_time = np.mean(fold_epoch_times)
    
    # Log cross-validation results
    logger.info("\nCross-validation completed")
    logger.info(f"Total time: {cv_time:.2f} seconds")
    logger.info(f"Average fold time: {avg_fold_time:.2f} seconds")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    logger.info(f"Average accuracy: {avg_accuracy*100:.2f}% (±{np.std(fold_accuracies)*100:.2f}%)")
    logger.info(f"Average F1 score: {avg_f1_score:.4f} (±{np.std(fold_f1_scores):.4f})")
    logger.info(f"Average FB score: {avg_fb_score:.4f} (±{np.std(fold_fb_scores):.4f})")
    
    # Save individual fold metrics
    fold_metrics = pd.DataFrame({
        'Fold': range(1, args.k_folds + 1),
        'Accuracy': [acc * 100 for acc in fold_accuracies],
        'F1_Score': fold_f1_scores,
        'FB_Score': fold_fb_scores,
        'Training_Time': fold_train_times,
        'Avg_Epoch_Time': fold_epoch_times
    })
    fold_metrics.to_csv(os.path.join(args.output_dir, 'fold_metrics.csv'), index=False)
    
    # Plot learning curves
    plot_learning_curves(all_train_losses, all_val_losses, 'Loss', args.output_dir)
    plot_learning_curves(all_train_accs, all_val_accs, 'Accuracy', args.output_dir)
    
    # Log summary
    logger.info("\nCross-validation summary saved to output directory")
    
    return avg_accuracy, avg_f1_score, avg_fb_score

def plot_learning_curves(train_metrics, val_metrics, metric_name, output_dir):
    """
    Plot learning curves for each fold and the average
    
    Args:
        train_metrics: List of training metrics for each fold
        val_metrics: List of validation metrics for each fold
        metric_name: Name of the metric (Loss or Accuracy)
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate epochs
    epochs = len(train_metrics[0])
    x = np.arange(1, epochs + 1)
    
    # Plot each fold
    for i, (train_curve, val_curve) in enumerate(zip(train_metrics, val_metrics)):
        plt.plot(x, train_curve, linestyle='-', alpha=0.3, label=f'Fold {i+1} Train')
        plt.plot(x, val_curve, linestyle='--', alpha=0.3, label=f'Fold {i+1} Val')
    
    # Plot average
    avg_train = np.mean(train_metrics, axis=0)
    avg_val = np.mean(val_metrics, axis=0)
    plt.plot(x, avg_train, 'b-', linewidth=2, label='Avg Train')
    plt.plot(x, avg_val, 'r--', linewidth=2, label='Avg Val')
    
    plt.title(f'{metric_name} Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'{metric_name.lower()}_learning_curves.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Cross-validation for semiconductor leakage model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./Valid/indices/preprocessed',
                       help='Directory with preprocessed CSV files')
    parser.add_argument('--indices_dir', type=str, default='./Valid/indices',
                       help='Directory with indices CSV files')
    parser.add_argument('--output_dir', type=str, default='./cv_results',
                       help='Directory to save cross-validation results')
    parser.add_argument('--seq_length', type=int, default=1002,
                       help='Length of input sequence')
    
    # Cross-validation parameters
    parser.add_argument('--k_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs per fold')
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
    
    # Logging parameters
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to save the log file')
    
    args = parser.parse_args()
    
    # Run cross-validation
    cross_validate(args)

if __name__ == '__main__':
    main()