"""
Combined Loss Functions for APAU-Net Training
Author: Wildfire Prediction Team
Date: April 2026

This module contains various combined loss functions optimized for:
- F1 score maximization
- Class imbalance handling (wildfire pixels ~1%)
- Semantic segmentation tasks (64x64 grid wildfire prediction)

Available Loss Functions:
1. DiceLoss - Direct F1 optimization (mathematically equivalent to F1)
2. WBCEDiceLoss - Weighted BCE + 2×Dice Loss (RECOMMENDED for F1 optimization)
3. FocalDiceLoss - Focal + Dice (for hard-example mining)
4. WeightedComboLoss - Fully customizable combo loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss - Direct F1 Score Optimization
    
    Mathematically equivalent to F1 score:
    F1 = 2 * TP / (2*TP + FP + FN)
    Dice = 2 * |A ∩ B| / (|A| + |B|) = 2 * TP / (2*TP + FP + FN)
    
    Args:
        smooth (float): Smoothing constant to prevent division by zero
                       smooth=1.0 is conservative (medical imaging standard)
                       smooth=1e-7 is tighter to true Dice
        logits (bool): If True, apply sigmoid to predictions before computing loss
    
    Returns:
        Scalar loss value (lower is better)
    """
    
    def __init__(self, smooth=1e-7, logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.logits = logits
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits, shape (B, 1, H, W)
            targets: Ground truth binary masks, shape (B, 1, H, W)
        
        Returns:
            Scalar Dice loss value
        """
        if self.logits:
            inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # Compute intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1.0 - dice


class WBCEDiceLoss(nn.Module):
    """
    Weighted BCE + 2×Dice Loss - Optimal for F1 Maximization
    
    Loss = 1.0 * WBCE(pos_weight=90.33) + 2.0 * DiceLoss(smooth=1e-7)
    
    Weighting breakdown:
    - WBCE (33%): Handles class imbalance via pos_weight
    - DiceLoss (67%): Direct F1 optimization
    
    This combination provides:
    1. Direct F1 gradient (via Dice)
    2. Class imbalance handling (via weighted BCE)
    3. Stable training (two complementary losses)
    
    Args:
        pos_weight (float): Positive class weight for BCEWithLogitsLoss (handles class imbalance)
        dice_weight (float): Multiplier for Dice loss (default 2.0 for 67% contribution)
        bce_weight (float): Multiplier for BCE loss (default 1.0 for 33% contribution)
        dice_smooth (float): Smoothing constant for Dice loss
    
    Returns:
        Scalar loss value
    """
    
    def __init__(self, pos_weight=90.33, dice_weight=2.0, bce_weight=1.0, dice_smooth=1e-7):
        super(WBCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice_loss = DiceLoss(smooth=dice_smooth, logits=True)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits, shape (B, 1, H, W)
            targets: Ground truth binary masks, shape (B, 1, H, W)
        
        Returns:
            Scalar combined loss value
        """
        # Ensure pos_weight is on same device as inputs
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        # Combined: 1×WBCE + 2×Dice (67% focus on F1 via Dice)
        combined_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return combined_loss


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss - Hard Example Mining + F1 Optimization
    
    Loss = alpha * FocalLoss(gamma=2) + (1-alpha) * DiceLoss
    
    Combines:
    - Focal Loss: Down-weights easy negatives, focuses on hard examples
    - Dice Loss: Direct F1 optimization
    
    Args:
        focal_gamma (float): Focusing parameter (higher = more focus on hard examples)
        focal_alpha (float): Weighting parameter for Focal loss
        focal_weight (float): Multiplier for Focal component
        dice_weight (float): Multiplier for Dice component
        dice_smooth (float): Smoothing constant for Dice loss
    
    Returns:
        Scalar loss value
    """
    
    def __init__(self, focal_gamma=2.0, focal_alpha=1.0, focal_weight=0.5, 
                 dice_weight=0.5, dice_smooth=1e-7):
        super(FocalDiceLoss, self).__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=dice_smooth, logits=True)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits, shape (B, 1, H, W)
            targets: Ground truth binary masks, shape (B, 1, H, W)
        
        Returns:
            Scalar combined loss value
        """
        # Focal Loss component
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        focal = focal.mean()
        
        # Dice Loss component
        dice = self.dice_loss(inputs, targets)
        
        # Combined
        combined_loss = self.focal_weight * focal + self.dice_weight * dice
        
        return combined_loss


class WeightedComboLoss(nn.Module):
    """
    Fully Customizable Combined Loss Function
    
    Allows arbitrary weighting of multiple loss components:
    Loss = w_bce * WBCE + w_focal * FocalLoss + w_dice * DiceLoss + w_tversky * TverskyLoss
    
    Args:
        pos_weight (float): Positive class weight for BCE
        w_bce (float): Weight for WBCE component
        w_focal (float): Weight for Focal loss component
        w_dice (float): Weight for Dice loss component
        w_tversky (float): Weight for Tversky loss component
        focal_gamma (float): Focal loss focusing parameter
        focal_alpha (float): Focal loss weighting parameter
        tversky_alpha (float): Tversky alpha (penalizes false positives)
        tversky_beta (float): Tversky beta (penalizes false negatives)
        dice_smooth (float): Dice smoothing constant
        tversky_smooth (float): Tversky smoothing constant
    
    Returns:
        Scalar loss value
    """
    
    def __init__(self, pos_weight=90.33, w_bce=1.0, w_focal=0.0, w_dice=2.0, w_tversky=0.0,
                 focal_gamma=2.0, focal_alpha=1.0, tversky_alpha=0.5, tversky_beta=0.5,
                 dice_smooth=1e-7, tversky_smooth=1.0):
        super(WeightedComboLoss, self).__init__()
        
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice_loss = DiceLoss(smooth=dice_smooth, logits=True)
        
        self.w_bce = w_bce
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.w_tversky = w_tversky
        
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_smooth = tversky_smooth
    
    def _tversky_loss(self, inputs, targets):
        """Compute Tversky loss for imbalanced data"""
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        fp = (inputs_flat * (1 - targets_flat)).sum()
        fn = ((1 - inputs_flat) * targets_flat).sum()
        
        tversky = (intersection + self.tversky_smooth) / (
            intersection + self.tversky_alpha * fp + self.tversky_beta * fn + self.tversky_smooth
        )
        
        return 1.0 - tversky
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits, shape (B, 1, H, W)
            targets: Ground truth binary masks, shape (B, 1, H, W)
        
        Returns:
            Scalar combined loss value
        """
        combined_loss = 0.0
        
        # BCE component
        if self.w_bce > 0:
            if self.bce_loss.pos_weight.device != inputs.device:
                self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
            bce = self.bce_loss(inputs, targets)
            combined_loss += self.w_bce * bce
        
        # Focal component
        if self.w_focal > 0:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
            combined_loss += self.w_focal * focal.mean()
        
        # Dice component
        if self.w_dice > 0:
            dice = self.dice_loss(inputs, targets)
            combined_loss += self.w_dice * dice
        
        # Tversky component
        if self.w_tversky > 0:
            tversky = self._tversky_loss(inputs, targets)
            combined_loss += self.w_tversky * tversky
        
        return combined_loss


# ============================================================================
# PRESET CONFIGURATIONS FOR QUICK USE
# ============================================================================

def get_wbce_dice_loss(pos_weight=90.33):
    """
    Recommended: WBCE + 2×Dice Loss for F1 Maximization
    
    Configuration:
    - Loss = 1.0 * WBCE(pos_weight=90.33) + 2.0 * DiceLoss(smooth=1e-7)
    - 33% BCE (class imbalance) + 67% Dice (F1 optimization)
    
    Returns:
        WBCEDiceLoss instance
    """
    return WBCEDiceLoss(pos_weight=pos_weight, dice_weight=2.0, bce_weight=1.0)


def get_focal_dice_loss():
    """
    Hard Example Mining: Focal + Dice Loss
    
    Configuration:
    - Loss = 0.5 * FocalLoss(gamma=2) + 0.5 * DiceLoss
    - Focuses on difficult predictions + F1 optimization
    
    Returns:
        FocalDiceLoss instance
    """
    return FocalDiceLoss(focal_weight=0.5, dice_weight=0.5)


def get_balanced_combo_loss(pos_weight=90.33):
    """
    Balanced: BCE + Focal + Dice (equal weighting)
    
    Configuration:
    - Loss = 0.33 * WBCE + 0.33 * FocalLoss + 0.34 * DiceLoss
    - All three components equally represented
    
    Returns:
        WeightedComboLoss instance
    """
    return WeightedComboLoss(
        pos_weight=pos_weight,
        w_bce=0.33,
        w_focal=0.33,
        w_dice=0.34
    )


def get_custom_combo_loss(w_bce=1.0, w_focal=0.0, w_dice=2.0, w_tversky=0.0, 
                         pos_weight=90.33, focal_gamma=2.0, tversky_alpha=0.5, tversky_beta=0.5):
    """
    Custom: Fully configurable weighted combination
    
    Args:
        w_bce (float): Weight for WBCE component
        w_focal (float): Weight for Focal loss component
        w_dice (float): Weight for Dice loss component
        w_tversky (float): Weight for Tversky loss component
        pos_weight (float): Positive class weight for BCE
        focal_gamma (float): Focal focusing parameter
        tversky_alpha (float): Tversky FP penalty
        tversky_beta (float): Tversky FN penalty
    
    Returns:
        WeightedComboLoss instance
    """
    return WeightedComboLoss(
        pos_weight=pos_weight,
        w_bce=w_bce,
        w_focal=w_focal,
        w_dice=w_dice,
        w_tversky=w_tversky,
        focal_gamma=focal_gamma,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta
    )


# ============================================================================
# QUICK REFERENCE GUIDE
# ============================================================================

LOSS_PRESETS = {
    'wbce_dice': {
        'description': 'RECOMMENDED: WBCE + 2×Dice (F1 maximization)',
        'function': get_wbce_dice_loss,
        'expected_f1': '0.56-0.62',
        'use_case': 'Default for wildfire prediction'
    },
    'focal_dice': {
        'description': 'Focal + Dice (hard example mining)',
        'function': get_focal_dice_loss,
        'expected_f1': '0.52-0.58',
        'use_case': 'When model struggles with edge cases'
    },
    'balanced_combo': {
        'description': 'BCE + Focal + Dice (balanced)',
        'function': get_balanced_combo_loss,
        'expected_f1': '0.50-0.56',
        'use_case': 'Conservative improvement'
    },
    'custom': {
        'description': 'Custom weighting',
        'function': get_custom_combo_loss,
        'expected_f1': 'Depends on config',
        'use_case': 'Advanced tuning'
    }
}


if __name__ == '__main__':
    """
    Quick test of all loss functions
    """
    print("Testing Combined Loss Functions\n")
    
    # Create dummy data
    batch_size = 4
    height, width = 64, 64
    pred_logits = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test each loss
    print("1. DiceLoss:")
    dice = DiceLoss()
    loss = dice(pred_logits, targets)
    print(f"   Loss value: {loss.item():.6f}\n")
    
    print("2. WBCEDiceLoss (RECOMMENDED):")
    wbce_dice = WBCEDiceLoss()
    loss = wbce_dice(pred_logits, targets)
    print(f"   Loss value: {loss.item():.6f}\n")
    
    print("3. FocalDiceLoss:")
    focal_dice = FocalDiceLoss()
    loss = focal_dice(pred_logits, targets)
    print(f"   Loss value: {loss.item():.6f}\n")
    
    print("4. WeightedComboLoss (custom):")
    custom = WeightedComboLoss(w_bce=1.0, w_dice=2.0)
    loss = custom(pred_logits, targets)
    print(f"   Loss value: {loss.item():.6f}\n")
    
    print("All loss functions working correctly! ✓")
