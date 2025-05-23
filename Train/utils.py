import numpy as np

def ACC(mylist):
    """
    Calculate accuracy from confusion matrix.
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc

def PPV(mylist):
    """
    Calculate positive predictive value (precision) from confusion matrix.
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv

def NPV(mylist):
    """
    Calculate negative predictive value from confusion matrix.
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv

def Sensitivity(mylist):
    """
    Calculate sensitivity (recall) from confusion matrix.
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity

def Specificity(mylist):
    """
    Calculate specificity from confusion matrix.
    """
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity

def BAC(mylist):
    """
    Calculate balanced accuracy from confusion matrix.
    """
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc

def F1(mylist):
    """
    Calculate F1 score from confusion matrix.
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
    Calculate FB score from confusion matrix.
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
    Print and return various evaluation metrics.
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
    return FB(mylist)