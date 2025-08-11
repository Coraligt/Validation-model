import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix


def ACC(mylist):
    """
    Calculate Accuracy
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: Accuracy
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    """
    Calculate Positive Predictive Value / Precision
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: Positive Predictive Value
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no leaky samples in the dataset, ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # Special case: there are leaky samples, but all predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    return ppv


def NPV(mylist):
    """
    Calculate Negative Predictive Value
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: Negative Predictive Value
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no non-leaky samples in the dataset, npv should be 1
    if tn + fp == 0:
        npv = 1
    # Special case: there are non-leaky samples, but all predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return npv


def Sensitivity(mylist):
    """
    Calculate Sensitivity / Recall
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: Sensitivity
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no leaky samples in the dataset, sensitivity should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    """
    Calculate Specificity
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: Specificity
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no non-leaky samples in the dataset, specificity should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    """
    Calculate Balanced Accuracy
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: Balanced Accuracy
    """
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    """
    Calculate F1 Score
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: F1 Score
    """
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    """
    Calculate F-Beta Score, default Beta=2
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        beta (float): Beta parameter, default is 2
        
    Returns:
        float: F-Beta Score
    """
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        fb = 0
    else:
        fb = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return fb


def stats_report(mylist):
    """
    Generate complete statistics report
    
    Args:
        mylist (list): [TP, FN, FP, TN]
        
    Returns:
        float: FB Score (also prints all metrics)
    """
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print("F-1 = ", f1)
    print("F-B = ", fb)
    print("SEN = ", se)
    print("SPE = ", sp)
    print("BAC = ", bac)
    print("ACC = ", acc)
    print("PPV = ", ppv)
    print("NPV = ", npv)
    return fb


def evaluate_model(model, dataloader, device):
    """
    Evaluate model on given dataloader
    
    Args:
        model (nn.Module): PyTorch model
        dataloader (DataLoader): Data loader
        device: Device for evaluation
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Evaluate without gradient tracking
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (leaky)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute confusion matrix based metrics
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate TP, FN, FP, TN
    if cm.shape == (2, 2):  # Ensure confusion matrix is 2x2
        tn, fp, fn, tp = cm.ravel()
    else:  # If not 2x2, calculate manually
        tp = fn = fp = tn = 0
        for i in range(len(all_labels)):
            if all_labels[i] == 1 and all_preds[i] == 1:
                tp += 1
            elif all_labels[i] == 1 and all_preds[i] == 0:
                fn += 1
            elif all_labels[i] == 0 and all_preds[i] == 1:
                fp += 1
            elif all_labels[i] == 0 and all_preds[i] == 0:
                tn += 1
    
    # Calculate metrics using utility functions
    stats = [tp, fn, fp, tn]
    fb_score = FB(stats)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Return all metrics
    metrics = {
        'accuracy': ACC(stats) * 100,  # Convert to percentage
        'f1_score': F1(stats),
        'fb_score': fb_score,
        'sensitivity': Sensitivity(stats),
        'specificity': Specificity(stats),
        'precision': PPV(stats),
        'npv': NPV(stats),
        'bac': BAC(stats),
        'confusion_matrix': stats,
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics


def visualize_results(metrics, output_dir):
    """
    Visualize evaluation results
    
    Args:
        metrics (dict): Evaluation metrics
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    accuracy = metrics['accuracy']
    f1_score = metrics['f1_score']
    fb_score = metrics['fb_score']
    sensitivity = metrics['sensitivity']
    specificity = metrics['specificity']
    precision = metrics['precision']
    tp, fn, fp, tn = metrics['confusion_matrix']
    fpr, tpr, roc_auc = metrics['roc']['fpr'], metrics['roc']['tpr'], metrics['roc']['auc']
    
    # Print summary
    print("\nTest Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"FB Score: {fb_score:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Visualize confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-leaky', 'Leaky'])
    plt.yticks(tick_marks, ['Non-leaky', 'Leaky'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Visualize ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Metrics bar chart
    metrics_data = {
        'F1 Score': f1_score,
        'FB Score': fb_score,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'ACC': accuracy / 100  # Convert percentage back to 0-1 range
    }
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_data.keys(), metrics_data.values())
    plt.title('Evaluation Metrics')
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed
    """
    import random
    import torch
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False