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


#  if not args.model_path:
#             logger.error("Model path must be specified for evaluation mode")
#             return
        
#         logger.info(f"Loading model from {args.model_path} for evaluation")
#         model.load_state_dict(torch.load(args.model_path, map_location=device))
        
#         # Start timer for evaluation
#         eval_start_time = time.time()
        
#         # Evaluate loaded model
#         test_loss, test_acc, cm = evaluate_model(model, test_loader, criterion, device)
#         fb = stats_report(cm)
        
#         eval_time = time.time() - eval_start_time
        
#         logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
#         logger.info(f"Test Loss: {test_loss:.4f}")
#         logger.info(f"Test Accuracy: {test_acc:.2f}%")
#         logger.info(f"FB Score: {fb:.4f}")
#         logger.info(f"Confusion Matrix: TP={cm[0]}, FN={cm[1]}, FP={cm[2]}, TN={cm[3]}")