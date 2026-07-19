import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_dice_loss(inputs, targets, bce_weight=0.5):
    """
    Combines BCE loss and Dice loss.
    inputs: logits from the model (before sigmoid)
    targets: ground truth masks
    """
    # Compute BCE Loss
    bce = F.binary_cross_entropy_with_logits(inputs, targets)
    
    # Compute Dice Loss
    preds = torch.sigmoid(inputs)
    smooth = 1e-6
    
    # Flatten label and prediction tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    intersection = (preds * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    
    return bce_weight * bce + (1 - bce_weight) * dice_loss

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        
    def forward(self, inputs, targets):
        return bce_dice_loss(inputs, targets, self.bce_weight)
