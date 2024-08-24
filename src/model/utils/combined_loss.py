import torch
import torch.nn.functional as F
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, logits, targets):
        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        # Apply sigmoid to logits for Dice calculation
        preds = torch.sigmoid(logits)
        smooth = 1.0  # for numerical stability

        # Flatten the tensors for Dice calculation
        preds = preds.view(-1)
        targets = targets.view(-1)

        # Compute Dice loss
        intersection = (preds * targets).sum()
        dice = 1 - (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

        return bce + dice
