import torch
import numpy as np

def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate Dice, IoU, Precision, and Recall.
    preds: probabilities from the model (after sigmoid)
    targets: ground truth masks
    """
    # Binarize predictions
    preds = (preds > threshold).float()
    targets = targets.float()
    
    smooth = 1e-6
    
    # True Positives, False Positives, False Negatives
    tp = (preds * targets).sum()
    fp = ((1 - targets) * preds).sum()
    fn = (targets * (1 - preds)).sum()
    
    # Intersection and Union
    intersection = tp
    union = preds.sum() + targets.sum() - intersection
    
    # Metrics
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }
