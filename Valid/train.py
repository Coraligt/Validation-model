import os
import argparse
import json
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from dataset import get_dataloaders, get_default_transforms
from swa import StochasticWeightAveraging, CosineAnnealingWarmRestarts, update_bn
from utils import evaluate_model, visualize_results, set_seed


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train model for one epoch on semiconductor data
    
    Args:
        model (nn.Module): Model
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device for training
        
    Returns:
        tuple: (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    # Calculate average loss and accuracy
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(args):
    """
    Train semiconductor leakage detection model
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data augmentation
    transforms = get_default_transforms(use_time_reverse=args.use_time_reverse)
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        indices_dir=args.indices_dir,
        batch_size=args.batch_size,
        use_columns=args.use_columns.split(','),
        num_workers=args.num_workers,
        train_transform=transforms['train'],
        test_transform=transforms['test']
    )
    
    # Create model
    model = get_model(
        model_type=args.model_type,
        in_channels=len(args.use_columns.split(',')),
        seq_length=args.seq_length
    )
    model = model.to(device)
    
    # Print model structure
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.epochs,
            eta_min=args.min_lr
        )
    else:
        scheduler = None
    
    # Stochastic Weight Averaging
    if args.use_swa:
        swa = StochasticWeightAveraging(
            model=model,
            swa_start=args.swa_start,
            swa_freq=args.swa_freq,
            swa_lr=args.swa_lr
        )
    else:
        swa = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'fb_score': []
    }
    
    best_fb = 0.0
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Update SWA if enabled
        if swa is not None:
            swa.update(optimizer, epoch)
        
        # Validate
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device
        )
        
        val_loss = criterion(
            torch.tensor(val_metrics['probabilities']).float().to(device), 
            torch.tensor(val_metrics['labels']).long().to(device)
        ).item()
        
        val_acc = val_metrics['accuracy']
        fb_score = val_metrics['fb_score']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, FB Score: {fb_score:.4f}")
        print(f"TP: {val_metrics['confusion_matrix'][0]}, FN: {val_metrics['confusion_matrix'][1]}, "
              f"FP: {val_metrics['confusion_matrix'][2]}, TN: {val_metrics['confusion_matrix'][3]}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['fb_score'].append(fb_score)
        
        # Save best model based on FB score (the metric used in the original model)
        if fb_score > best_fb:
            best_fb = fb_score
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_fb.pth'))
            print(f"New best model saved! (FB: {best_fb:.4f})")
        
        # Also save best model based on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model_acc.pth'))
            print(f"New best accuracy model saved! (Acc: {best_acc:.2f}%)")
    
    # Apply SWA weights if enabled
    if swa is not None:
        print("\nApplying SWA weights...")
        swa.finalize()
        
        # Update batch normalization layer statistics
        print("Updating batch normalization statistics...")
        update_bn(model, train_loader, device)
        
        # Validate with SWA weights
        swa_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device
        )
        
        print(f"SWA Val Acc: {swa_metrics['accuracy']:.2f}%")
        print(f"SWA FB Score: {swa_metrics['fb_score']:.4f}")
        
        # Save SWA model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'swa_model.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    # Training complete
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print(f"\nTraining completed in {minutes}m {seconds}s")
    print(f"Best FB score: {best_fb:.4f} at epoch {best_epoch+1}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    
    # Visualize final results
    print("\nEvaluating final model...")
    final_metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device
    )
    
    visualize_results(
        metrics=final_metrics,
        output_dir=os.path.join(args.output_dir, 'final_eval')
    )


def main():
    parser = argparse.ArgumentParser(description='Train semiconductor leakage detection model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./Valid/dataset_3',
                        help='Directory containing preprocessed CSV files')
    parser.add_argument('--indices_dir', type=str, default='./Valid/indices',
                        help='Directory containing train and test indices files')
    parser.add_argument('--output_dir', type=str, default='./model_output',
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--use_columns', type=str, default='t,q',
                        help='Columns to use as features (comma separated)')
    parser.add_argument('--seq_length', type=int, default=1002,
                        help='Sequence length (number of rows in each CSV file)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='default', choices=['default', 'small'],
                        help='Model type to use')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0001,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'],
                        help='Learning rate scheduler')
    
    # Data augmentation parameters
    parser.add_argument('--use_time_reverse', action='store_true',
                        help='Use time reversal data augmentation')
    
    # SWA parameters
    parser.add_argument('--use_swa', action='store_true',
                        help='Use Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=10,
                        help='Epoch to start SWA')
    parser.add_argument('--swa_freq', type=int, default=5,
                        help='SWA update frequency')
    parser.add_argument('--swa_lr', type=float, default=0.0001,
                        help='SWA learning rate')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (e.g., cuda:0, cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Train the model
    train_model(args)


if __name__ == '__main__':
    main()