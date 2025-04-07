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
import json
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from model import SemiconductorModel, count_parameters, save_for_inference
from dataset import get_dataloaders, get_transforms
from utils import set_seed, stats_report
from swa import SWA
from cosine_annealing import CosineAnnealingLR

def setup_logging(output_dir, log_file=None, run_id=None):
    """Setup logging configuration"""
    # Create timestamp for log file if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_id is not None:
            log_file = os.path.join(output_dir, f"training_run{run_id}_{timestamp}.log")
        else:
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

def setup_model_and_optimizer(args, device, logger):
    """Create and initialize model and optimizer"""
    # Create model
    logger.info("Creating model...")
    model = SemiconductorModel(
        seq_length=args.seq_length,
        conv_filters=args.conv_filters,
        fc1_size=args.fc1_size,
        fc2_size=args.fc2_size,
        dropout1=args.dropout1,
        dropout2=args.dropout2
    ).to(device)
    
    logger.info(f"Model architecture:\n{str(model)}")
    logger.info(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Initialize optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_max=args.learning_rate,
        eta_min=args.learning_rate * args.min_lr_factor
    )
    
    return model, optimizer, scheduler

def apply_data_augmentation(inputs, args):
    """Apply data augmentation to input batch"""
    augmented_inputs = inputs.clone()
    batch_size = inputs.size(0)
    
    # Apply random flipping
    if args.flip_signal:
        # Create a mask that matches the dimensions of the input
        flip_mask = (torch.rand(batch_size, 1, 1) < args.flip_prob).expand_as(inputs)
        augmented_inputs[flip_mask] = -augmented_inputs[flip_mask]
    
    # Apply random time reversal
    if args.flip_time:
        flip_mask = torch.rand(batch_size) < args.flip_prob
        for i in range(batch_size):
            if flip_mask[i]:
                augmented_inputs[i] = torch.flip(augmented_inputs[i], [1])
    
    # Apply random noise
    if args.add_noise:
        for i in range(batch_size):
            if torch.rand(1) < args.noise_prob:
                max_val = torch.max(torch.abs(augmented_inputs[i]))
                noise_std = max_val * args.noise_factor
                noise = torch.randn_like(augmented_inputs[i]) * noise_std * torch.rand(1)
                augmented_inputs[i] = augmented_inputs[i] + noise
    
    return augmented_inputs

def train_single_run(args, device, loaders, run_id=0):
    """Train and evaluate a single model run"""
    # Setup run-specific logging
    run_dir = os.path.join(args.output_dir, f'run_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(run_dir, run_id=run_id)
    
    # Log run configuration
    logger.info(f"Starting training run {run_id}")
    logger.info("Run configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Unpack loaders
    has_val_set = len(loaders) == 3
    if has_val_set:
        train_loader, val_loader, test_loader = loaders
        logger.info("Using separate validation and test sets")
    else:
        train_loader, test_loader = loaders
        val_loader = test_loader  # Use test set as validation if no validation set
        logger.info("Using test set as validation (no separate validation set found)")
    
    # Create model, optimizer and scheduler
    model, optimizer, scheduler = setup_model_and_optimizer(args, device, logger)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize SWA if enabled
    if args.use_swa:
        swa = SWA(
            model, 
            swa_start=args.swa_start, 
            swa_freq=args.swa_freq, 
            swa_lr=args.swa_lr
        )
    
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
            
            # Apply data augmentation
            if args.do_augmentation:
                inputs = apply_data_augmentation(inputs, args)
            
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
        if args.use_swa and epoch >= args.swa_start:
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
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (by validation FB score)
        if val_fb > best_val_fb:
            best_val_fb = val_fb
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model_fb.pth'))
            logger.info(f"  Saved new best model (FB: {best_val_fb:.4f}, Acc: {best_val_acc:.2f}%)")
        
        # Also save by accuracy as alternative metric
        if val_acc > best_val_acc and val_fb <= best_val_fb:
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_model_acc.pth'))
            logger.info(f"  Saved new best accuracy model (Acc: {val_acc:.2f}%)")
        
        # Save checkpoint for resuming
        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_fb': best_val_fb,
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch
            }
            torch.save(checkpoint, os.path.join(run_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Training complete
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    logger.info("\nEvaluating best model on test set...")
    # Load best model
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model_fb.pth'), map_location=device))
    test_loss, test_acc, test_cm = evaluate_model(model, test_loader, criterion, device)
    test_fb = stats_report(test_cm)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test FB Score: {test_fb:.4f}")
    logger.info(f"Test Confusion Matrix: TP={test_cm[0]}, FN={test_cm[1]}, FP={test_cm[2]}, TN={test_cm[3]}")
    
    # Final evaluation with SWA model if enabled
    swa_test_acc = None
    swa_test_fb = None
    
    if args.use_swa:
        logger.info("\nEvaluating SWA model...")
        
        # Update batch normalization statistics
        swa.update_bn(train_loader, device)
        
        # Get SWA model
        swa_model = swa.get_model()
        
        # Evaluate SWA model on test set
        swa_test_loss, swa_test_acc, swa_cm = evaluate_model(swa_model, test_loader, criterion, device)
        swa_test_fb = stats_report(swa_cm)
        
        logger.info(f"SWA Model Loss: {swa_test_loss:.4f}")
        logger.info(f"SWA Model Accuracy: {swa_test_acc:.2f}%")
        logger.info(f"SWA FB Score: {swa_test_fb:.4f}")
        logger.info(f"SWA Confusion Matrix: TP={swa_cm[0]}, FN={swa_cm[1]}, FP={swa_cm[2]}, TN={swa_cm[3]}")
        
        # Save SWA model
        torch.save(swa_model.state_dict(), os.path.join(run_dir, 'swa_model.pth'))
        
        # Compare with best model
        if swa_test_acc > test_acc:
            logger.info(f"SWA improved accuracy: {test_acc:.2f}% -> {swa_test_acc:.2f}%")
        else:
            logger.info(f"Best model accuracy: {test_acc:.2f}% (SWA: {swa_test_acc:.2f}%)")
        
        if swa_test_fb > test_fb:
            logger.info(f"SWA improved FB score: {test_fb:.4f} -> {swa_test_fb:.4f}")
        else:
            logger.info(f"Best model FB score: {test_fb:.4f} (SWA: {swa_test_fb:.4f})")
    
    # Save training history
    history = {
        'epoch': list(range(1, args.epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'val_fb': val_fbs
    }
    
    with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Plot training curves
    try:
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['epoch'], train_losses, label='Train Loss')
        plt.plot(history['epoch'], val_losses, label='Val Loss')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'loss_curve.png'))
        plt.close()
        
        # Accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['epoch'], train_accs, label='Train Accuracy')
        plt.plot(history['epoch'], val_accs, label='Val Accuracy')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'accuracy_curve.png'))
        plt.close()
        
        # FB Score curve
        plt.figure(figsize=(10, 6))
        plt.plot(history['epoch'], val_fbs, label='Validation FB Score')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('FB Score')
        plt.title('Validation FB Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_dir, 'fb_score_curve.png'))
        plt.close()
        
        logger.info("Training curves saved to output directory")
    except Exception as e:
        logger.warning(f"Error plotting training curves: {e}")
    
    # Save optimized model for inference
    try:
        model_path = os.path.join(run_dir, 'optimized_model')
        save_for_inference(model, model_path, input_shape=(1, 1, args.seq_length))
        logger.info(f"Optimized model saved for inference at {model_path}")
    except Exception as e:
        logger.warning(f"Error saving optimized model: {e}")
    
    # Return metrics for model selection
    result = {
        'run_id': run_id,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_val_fb': best_val_fb,
        'test_acc': test_acc,
        'test_fb': test_fb,
        'swa_test_acc': swa_test_acc,
        'swa_test_fb': swa_test_fb,
        'training_time': training_time,
        'model_path': os.path.join(run_dir, 'best_model_fb.pth'),
        'swa_model_path': os.path.join(run_dir, 'swa_model.pth') if args.use_swa else None
    }
    
    # Save metrics
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def train_baseline(args):
    """Train a baseline model without augmentation or SWA"""
    # Override arguments for baseline
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.do_augmentation = False
    baseline_args.use_swa = False
    baseline_args.output_dir = os.path.join(args.output_dir, 'baseline')
    
    # Create device
    if baseline_args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(baseline_args.device)
    
    # Get transforms - no transforms for baseline
    transforms = get_transforms(
        use_time_reverse=False,
        flip_prob=0.0,
        noise_prob=0.0,
        noise_factor=0.0
    )
    
    # Create dataloaders
    loaders = get_dataloaders(
        data_dir=baseline_args.data_dir,
        indices_dir=baseline_args.indices_dir,
        batch_size=baseline_args.batch_size,
        num_workers=baseline_args.workers,
        transforms=transforms
    )
    
    # Run training with baseline settings
    return train_single_run(baseline_args, device, loaders, run_id=0)

def train(args):
    """Main training function with multiple runs"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup main logging
    logger = setup_logging(args.output_dir)
    
    # Log all arguments
    logger.info("Starting semiconductor leakage detection training")
    logger.info("Global configuration:")
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
    transforms = get_transforms(
        use_time_reverse=args.flip_time,
        flip_prob=args.flip_prob,
        noise_prob=args.noise_prob,
        noise_factor=args.noise_factor
    )
    
    # Create dataloaders
    logger.info("Loading data...")
    loaders = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        transforms=transforms
    )
    
    # Evaluation-only mode
    if args.evaluate_only:
        if not args.model_path:
            logger.error("Model path must be specified for evaluation mode")
            return
        
        logger.info(f"Loading model from {args.model_path} for evaluation")
        model = SemiconductorModel(
            seq_length=args.seq_length,
            conv_filters=args.conv_filters,
            fc1_size=args.fc1_size,
            fc2_size=args.fc2_size,
            dropout1=args.dropout1,
            dropout2=args.dropout2
        ).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Choose test loader
        has_val_set = len(loaders) == 3
        if has_val_set:
            _, _, test_loader = loaders
        else:
            _, test_loader = loaders
        
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
    
    # Multiple training runs
    all_results = []
    
    for run in range(args.num_runs):
        logger.info(f"\n{'='*50}\nStarting Run {run+1}/{args.num_runs}\n{'='*50}")
        
        # Set different seed for each run
        run_seed = args.seed + run
        set_seed(run_seed)
        
        # Modify hyperparameters for tuning if specified
        run_args = argparse.Namespace(**vars(args))
        run_args.seed = run_seed
        
        # Random hyperparameter tuning if enabled
        if args.random_hyperparams:
            # Randomly adjust learning rate
            lr_factor = random.uniform(0.5, 2.0)
            run_args.learning_rate *= lr_factor
            
            # Randomly adjust dropout
            run_args.dropout1 = random.uniform(0.2, 0.5)
            run_args.dropout2 = random.uniform(0.1, 0.3)
            
            # Randomly adjust network size
            size_factor = random.uniform(0.75, 1.5)
            run_args.fc1_size = max(10, int(run_args.fc1_size * size_factor))
            run_args.fc2_size = max(5, int(run_args.fc2_size * size_factor))
            
            # Randomly adjust data augmentation settings
            if args.do_augmentation:
                run_args.flip_prob = random.uniform(0.3, 0.7)
                run_args.noise_prob = random.uniform(0.3, 0.7)
                run_args.noise_factor = random.uniform(0.03, 0.1)
        
        # Run training
        result = train_single_run(run_args, device, loaders, run_id=run)
        all_results.append(result)
    
    # Analyze results from all runs
    logger.info(f"\n{'='*50}\nResults Summary for All Runs\n{'='*50}")
    
    best_fb = -1
    best_acc = -1
    best_run_fb = None
    best_run_acc = None
    
    # Sort runs by FB score and accuracy
    all_results_by_fb = sorted(all_results, key=lambda x: x['test_fb'], reverse=True)
    all_results_by_acc = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    
    # Find best runs
    for result in all_results:
        # Check normal models
        if result['test_fb'] > best_fb:
            best_fb = result['test_fb']
            best_run_fb = result
        
        if result['test_acc'] > best_acc:
            best_acc = result['test_acc']
            best_run_acc = result
        
        # Check SWA models
        if result['swa_test_fb'] is not None and result['swa_test_fb'] > best_fb:
            best_fb = result['swa_test_fb']
            best_run_fb = {**result, 'is_swa': True}
            
        if result['swa_test_acc'] is not None and result['swa_test_acc'] > best_acc:
            best_acc = result['swa_test_acc']
            best_run_acc = {**result, 'is_swa': True}
    
    # Print results for all runs
    logger.info("\nAll Runs (sorted by FB score):")
    logger.info(f"{'Run':^5}{'Test FB':^12}{'Test Acc':^12}{'SWA FB':^12}{'SWA Acc':^12}{'Time(s)':^12}")
    logger.info('-' * 65)
    
    for idx, result in enumerate(all_results_by_fb):
        swa_fb = result['swa_test_fb'] if result['swa_test_fb'] is not None else 'N/A'
        swa_acc = result['swa_test_acc'] if result['swa_test_acc'] is not None else 'N/A'
        
        logger.info(f"{result['run_id']:^5}{result['test_fb']:^12.4f}{result['test_acc']:^12.2f}" + 
                    f"{swa_fb:^12}{swa_acc:^12}{result['training_time']:^12.1f}")
    
    # Copy best models to output directory
    logger.info("\nBest Models:")
    
    # Best FB model
    if best_run_fb is not None:
        is_swa = best_run_fb.get('is_swa', False)
        source_path = best_run_fb['swa_model_path'] if is_swa else best_run_fb['model_path']
        target_path = os.path.join(args.output_dir, 'best_model_fb.pth')
        try:
            import shutil
            shutil.copy2(source_path, target_path)
            logger.info(f"Best FB model ({best_fb:.4f}) from Run {best_run_fb['run_id']} " + 
                        f"{'(SWA)' if is_swa else ''} copied to {target_path}")
        except Exception as e:
            logger.error(f"Error copying best FB model: {e}")
    
    # Best Accuracy model
    if best_run_acc is not None and best_run_acc != best_run_fb:
        is_swa = best_run_acc.get('is_swa', False)
        source_path = best_run_acc['swa_model_path'] if is_swa else best_run_acc['model_path']
        target_path = os.path.join(args.output_dir, 'best_model_acc.pth')
        try:
            import shutil
            shutil.copy2(source_path, target_path)
            logger.info(f"Best Accuracy model ({best_acc:.2f}%) from Run {best_run_acc['run_id']} " + 
                        f"{'(SWA)' if is_swa else ''} copied to {target_path}")
        except Exception as e:
            logger.error(f"Error copying best Accuracy model: {e}")
    
    # Save all results
    with open(os.path.join(args.output_dir, 'all_runs_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info("\nTraining complete!")
    logger.info(f"Best FB Score: {best_fb:.4f}")
    logger.info(f"Best Accuracy: {best_acc:.2f}%")
    logger.info(f"Results saved to {args.output_dir}")

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
    
    # Multiple runs parameters
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of training runs with different seeds')
    parser.add_argument('--random_hyperparams', action='store_true',
                       help='Randomly tune hyperparameters between runs')
    
    # Model parameters
    parser.add_argument('--conv_filters', type=int, default=3,
                       help='Number of filters in convolutional layer')
    parser.add_argument('--fc1_size', type=int, default=20,
                       help='Size of first fully connected layer')
    parser.add_argument('--fc2_size', type=int, default=10,
                       help='Size of second fully connected layer')
    parser.add_argument('--dropout1', type=float, default=0.3,
                       help='Dropout rate for first dropout layer')
    parser.add_argument('--dropout2', type=float, default=0.1,
                       help='Dropout rate for second dropout layer')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay for optimizer')
    parser.add_argument('--min_lr_factor', type=float, default=0.01,
                       help='Minimum learning rate factor (fraction of initial lr)')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                       help='Interval for saving checkpoints')
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
    parser.add_argument('--do_augmentation', action='store_true', default=True,
                       help='Apply data augmentation during training')
    parser.add_argument('--flip_signal', action='store_true', default=True,
                       help='Enable signal flipping augmentation')
    parser.add_argument('--add_noise', action='store_true', default=True,
                       help='Enable noise addition augmentation')
    parser.add_argument('--flip_time', action='store_true', default=False,
                       help='Enable time reversal augmentation')
    parser.add_argument('--flip_prob', type=float, default=0.5,
                       help='Probability of flipping signal or time')
    parser.add_argument('--noise_prob', type=float, default=0.5,
                       help='Probability of adding noise')
    parser.add_argument('--noise_factor', type=float, default=0.05,
                       help='Noise factor (relative to signal amplitude)')
    
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
    
    # Flag for baseline model
    parser.add_argument('--train_baseline', action='store_true',
                       help='Train only a baseline model with no augmentation and no SWA')
    
    args = parser.parse_args()
    
    # If baseline flag is set, override other settings
    if args.train_baseline:
        args.do_augmentation = False
        args.use_swa = False
        args.output_dir = os.path.join(args.output_dir, 'baseline')
        args.num_runs = 1
        print("Training baseline model with no augmentation and no SWA")
    
    # Train or evaluate model
    train(args)

if __name__ == '__main__':
    main()