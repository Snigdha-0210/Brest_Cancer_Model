import torch
import numpy as np

def calculate_metrics(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    targets = targets.float()
    smooth = 1e-06
    tp = (preds * targets).sum()
    fp = ((1 - targets) * preds).sum()
    fn = (targets * (1 - preds)).sum()
    intersection = tp
    union = preds.sum() + targets.sum() - intersection
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    iou = (intersection + smooth) / (union + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return {'dice': dice.item(), 'iou': iou.item(), 'precision': precision.item(), 'recall': recall.item()}
