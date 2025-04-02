import numpy as np
import random
import torch

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ACC(mylist):
    """Calculate accuracy from confusion matrix"""
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc

def PPV(mylist):
    """Calculate positive predictive value (precision)"""
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no leaky samples in dataset
    if tp + fn == 0:
        ppv = 1
    # Special case: leaky samples exist but all predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    return ppv

def NPV(mylist):
    """Calculate negative predictive value"""
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no non-leaky samples in dataset
    if tn + fp == 0:
        npv = 1
    # Special case: non-leaky samples exist but all predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return npv

def Sensitivity(mylist):
    """Calculate sensitivity (recall)"""
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no leaky samples in dataset
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity

def Specificity(mylist):
    """Calculate specificity"""
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # Special case: no non-leaky samples in dataset
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity

def BAC(mylist):
    """Calculate balanced accuracy"""
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc

def F1(mylist):
    """Calculate F1 score"""
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def FB(mylist, beta=2):
    """Calculate F-beta score, with beta=2 by default"""
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        fb = 0
    else:
        fb = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return fb

def stats_report(mylist):
    """Print and return evaluation metrics"""
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    print("F-1 = ", f1)
    print("F-B = ", fb)
    print("SEN = ", se)
    print("SPE = ", sp)
    print("BAC = ", bac)
    print("ACC = ", acc)
    print("PPV = ", ppv)
    print("NPV = ", npv)
    return fb