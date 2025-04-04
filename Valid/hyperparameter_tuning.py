import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid, ParameterSampler

from model import SemiconductorModel, count_parameters
from dataset import get_dataloaders, get_transforms
from utils import set_seed, stats_report
from cosine_annealing import CosineAnnealingLR

def setup_logging(output_dir, log_file=None):
    """Setup logging configuration"""
    # Create timestamp for log file if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"hyperparameter_tuning_{timestamp}.log")
    
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
        for inputs, labels in data_loader:
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
    fb = stats_report(cm)
    
    return avg_loss, acc, cm, fb

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                     scheduler, device, epochs, early_stopping=5):
    """Train and evaluate model with early stopping"""
    best_val_fb = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    no_improve_epochs = 0
    
    train_losses = []
    val_losses = []
    val_fbs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
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
        
        # Average training loss
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Validation phase
        val_loss, val_acc, val_cm, val_fb = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_fbs.append(val_fb)
        
        # Check if this is the best model so far
        if val_fb > best_val_fb:
            best_val_fb = val_fb
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # Early stopping
        if early_stopping and no_improve_epochs >= early_stopping:
            print(f"Early stopping at epoch {epoch+1} as no improvement for {early_stopping} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'model': model,
        'best_epoch': best_epoch,
        'best_val_fb': best_val_fb,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_fbs': val_fbs,
        'epochs_trained': len(train_losses)
    }

def hyperparameter_tuning(args):
    """Perform hyperparameter tuning"""
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.log_file)
    
    # Log all arguments
    logger.info("Hyperparameter tuning with the following parameters:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        logger.info("Using CPU for training/evaluation")
    else:
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")
    
    # Get data transforms
    transforms = get_transforms(use_time_reverse=False)
    
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
    
    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [0.0005, 0.0002, 0.0001],
        'batch_size': [16, 32, 64],
        'dropout1': [0.2, 0.3, 0.4],
        'dropout2': [0.1, 0.2, 0.3],
        'conv_filters': [3, 6, 12],
        'fc1_size': [20, 40, 80],
        'fc2_size': [10, 20, 30]
    }
    
    # Convert to a list of parameter dictionaries
    if args.grid_search:
        param_list = list(ParameterGrid(param_grid))
        logger.info(f"Grid search will evaluate {len(param_list)} hyperparameter combinations")
    else:
        # Random search
        param_list = list(ParameterSampler(param_grid, n_iter=args.n_iter, random_state=args.seed))
        logger.info(f"Random search will evaluate {len(param_list)} hyperparameter combinations")
    
    # Results storage
    results = []
    best_fb = 0.0
    best_params = None
    best_model_state = None
    
    # Start timer
    start_time = time.time()
    
    # Evaluate each hyperparameter combination
    for i, params in enumerate(param_list):
        try:
            logger.info(f"\nHyperparameter combination {i+1}/{len(param_list)}")
            for param_name, param_value in params.items():
                logger.info(f"  {param_name}: {param_value}")
            
            # Update batch size for loaders if needed
            if params['batch_size'] != args.batch_size:
                logger.info(f"Recreating dataloaders with batch size {params['batch_size']}")
                loaders = get_dataloaders(
                    data_dir=args.data_dir,
                    indices_dir=args.indices_dir,
                    batch_size=params['batch_size'],
                    num_workers=args.workers,
                    transforms=transforms
                )
                
                if has_val_set:
                    train_loader, val_loader, test_loader = loaders
                else:
                    train_loader, test_loader = loaders
                    val_loader = test_loader
            
            # Create model with specific hyperparameters
            model = SemiconductorModel(
                seq_length=args.seq_length,
                conv_filters=params['conv_filters'],
                fc1_size=params['fc1_size'],
                fc2_size=params['fc2_size'],
                dropout1=params['dropout1'],
                dropout2=params['dropout2']
            ).to(device)
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Learning rate scheduler
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_max=params['learning_rate'],
                eta_min=params['learning_rate'] * 0.01
            )
            
            # Train and evaluate model
            combo_start_time = time.time()
            train_result = train_and_evaluate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=args.epochs,
                early_stopping=args.early_stopping
            )
            combo_time = time.time() - combo_start_time
            
            # Evaluate on test set
            test_loss, test_acc, test_cm, test_fb = evaluate_model(model, test_loader, criterion, device)
            
            # Store results
            result = {
                'params': params,
                'best_epoch': train_result['best_epoch'],
                'val_fb': train_result['best_val_fb'],
                'val_acc': train_result['best_val_acc'],
                'test_fb': test_fb,
                'test_acc': test_acc,
                'epochs_trained': train_result['epochs_trained'],
                'training_time': combo_time
            }
            results.append(result)
            
            # Check if this is the best model so far
            if test_fb > best_fb:
                best_fb = test_fb
                best_params = params.copy()
                best_model_state = model.state_dict().copy()
                
                # Save best model immediately
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                
                logger.info(f"New best model found! FB score: {best_fb:.4f}")
            
            # Log results
            logger.info(f"Results for combination {i+1}/{len(param_list)}:")
            logger.info(f"  Best epoch: {train_result['best_epoch']}/{train_result['epochs_trained']}")
            logger.info(f"  Validation FB score: {train_result['best_val_fb']:.4f}")
            logger.info(f"  Validation accuracy: {train_result['best_val_acc']:.2f}%")
            logger.info(f"  Test FB score: {test_fb:.4f}")
            logger.info(f"  Test accuracy: {test_acc:.2f}%")
            logger.info(f"  Training time: {combo_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error evaluating hyperparameter combination {i+1}: {e}")
            continue
    
    # Training complete
    tuning_time = time.time() - start_time
    logger.info(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, 'hyperparameter_results.csv'), index=False)
    
    # Log best hyperparameters
    logger.info("\nBest hyperparameters:")
    for param_name, param_value in best_params.items():
        logger.info(f"  {param_name}: {param_value}")
    
    logger.info(f"Best test FB score: {best_fb:.4f}")
    
    # Create model with best hyperparameters for final training
    if args.final_train and best_model_state is not None:
        logger.info("\nTraining final model with best hyperparameters on all training data...")
        
        # Combine training and validation data if available
        if has_val_set:
            # This would require creating a new combined dataloader
            logger.info("Training on combined training and validation data not implemented yet")
        
        # Create model with best hyperparameters
        final_model = SemiconductorModel(
            seq_length=args.seq_length,
            conv_filters=best_params['conv_filters'],
            fc1_size=best_params['fc1_size'],
            fc2_size=best_params['fc2_size'],
            dropout1=best_params['dropout1'],
            dropout2=best_params['dropout2']
        ).to(device)
        
        # Load best model state
        final_model.load_state_dict(best_model_state)
        
        # Save final model
        torch.save(final_model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
        
        # Evaluate on test set
        test_loss, test_acc, test_cm, test_fb = evaluate_model(final_model, test_loader, criterion, device)
        
        logger.info(f"Final model test FB score: {test_fb:.4f}")
        logger.info(f"Final model test accuracy: {test_acc:.2f}%")
    
    # Return best hyperparameters
    return best_params, best_fb

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for semiconductor leakage detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with preprocessed CSV files')
    parser.add_argument('--indices_dir', type=str, required=True,
                       help='Directory with indices files')
    parser.add_argument('--output_dir', type=str, default='./tuning_results',
                       help='Directory to save tuning results')
    parser.add_argument('--seq_length', type=int, default=1002,
                       help='Length of input sequence')
    
    # Search strategy
    parser.add_argument('--grid_search', action='store_true',
                       help='Use grid search instead of random search')
    parser.add_argument('--n_iter', type=int, default=20,
                       help='Number of combinations to try for random search')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Default batch size')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs per combination')
    parser.add_argument('--early_stopping', type=int, default=5,
                       help='Number of epochs with no improvement after which training stops')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for training')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    # Final training
    parser.add_argument('--final_train', action='store_true',
                       help='Train final model with best hyperparameters')
    
    # Logging parameters
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to save the log file')
    
    args = parser.parse_args()
    
    # Run hyperparameter tuning
    best_params, best_fb = hyperparameter_tuning(args)
    
    # Print best parameters
    print("\nBest hyperparameters:")
    for param_name, param_value in best_params.items():
        print(f"  {param_name}: {param_value}")
    print(f"Best FB score: {best_fb:.4f}")

if __name__ == '__main__':
    main()