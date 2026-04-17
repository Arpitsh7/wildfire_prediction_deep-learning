import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7, logits=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.logits = logits
    
    def forward(self, inputs, targets):
        if self.logits:
            inputs = torch.sigmoid(inputs)
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - Generalized Dice Loss
    
    Tversky Index = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    
    - alpha = 0.3: Penalizes false positives less
    - beta = 0.7: Penalizes false negatives more (better for imbalanced data)
    
    Args:
        alpha (float): FP penalty (default 0.3)
        beta (float): FN penalty (default 0.7)
        smooth (float): Smoothing constant
        logits (bool): Apply sigmoid if True
    
    Returns:
        Scalar loss value (lower is better)
    """
    
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, logits=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.logits = logits
    
    def forward(self, inputs, targets):
        if self.logits:
            inputs = torch.sigmoid(inputs)
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        tp = (inputs_flat * targets_flat).sum()
        fp = (inputs_flat * (1 - targets_flat)).sum()
        fn = ((1 - inputs_flat) * targets_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1.0 - tversky


class WBCETverskyLoss(nn.Module):
    def __init__(self, pos_weight=3.0, alpha=0.3, beta=0.7, w_bce=1.0, w_tversky=1.0,
                 bce_smooth=1e-7, tversky_smooth=1.0):
        super(WBCETverskyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta, smooth=tversky_smooth, logits=True)
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.beta = beta
        self.w_bce = w_bce
        self.w_tversky = w_tversky
    
    def forward(self, inputs, targets):
        if self.bce_loss.pos_weight.device != inputs.device:
            self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(inputs.device)
        bce = self.bce_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        return self.w_bce * bce + self.w_tversky * tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal + Tversky Combined Loss
    
    Loss = w_focal * FocalLoss(gamma) + w_tversky * TverskyLoss
    
    Configuration:
    - gamma = 2.0 or 2.5 (Focal focusing parameter)
    - alpha = 0.3 (Tversky: penalizes FP less)
    - beta = 0.7 (Tversky: penalizes FN more)
    
    Args:
        gamma (float): Focal loss focusing parameter (2.0 or 2.5)
        alpha (float): Tversky FP penalty
        beta (float): Tversky FN penalty
        w_focal (float): Weight for Focal component
        w_tversky (float): Weight for Tversky component
        focal_alpha (float): Focal alpha parameter
    
    Returns:
        Scalar loss value
    """
    
    def __init__(self, gamma=2.0, tversky_alpha=0.3, tversky_beta=0.7, 
                 w_focal=1.0, w_tversky=1.0, focal_alpha=1.0, tversky_smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.w_focal = w_focal
        self.w_tversky = w_tversky
        self.focal_alpha = focal_alpha
        self.tversky_smooth = tversky_smooth
    
    def _focal_loss(self, inputs, targets):
        """Focal loss for hard example mining"""
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = self.focal_alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()
    
    def _tversky_loss(self, inputs, targets):
        """Tversky loss for imbalanced data"""
        inputs_sigmoid = torch.sigmoid(inputs)
        
        tp = (inputs_sigmoid * targets).sum()
        fp = (inputs_sigmoid * (1 - targets)).sum()
        fn = ((1 - inputs_sigmoid) * targets).sum()
        
        tversky = (tp + self.tversky_smooth) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.tversky_smooth
        )
        
        return 1.0 - tversky
    
    def forward(self, inputs, targets):
        focal = self._focal_loss(inputs, targets)
        tversky = self._tversky_loss(inputs, targets)
        
        return self.w_focal * focal + self.w_tversky * tversky


if __name__ == '__main__':
    batch_size = 4
    height, width = 64, 64
    
    pred_logits = torch.randn(batch_size, 1, height, width)
    targets = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    print("Testing WBCETverskyLoss")
    print("=" * 50)
    print(f"pos_weight = 3")
    print(f"alpha = 0.3 (FP penalty)")
    print(f"beta = 0.7 (FN penalty)")
    print(f"w_bce = 1.0, w_tversky = 1.0")
    print("=" * 50)
    
    loss_fn = WBCETverskyLoss(pos_weight=3.0, alpha=0.3, beta=0.7)
    loss = loss_fn(pred_logits, targets)
    print(f"Loss value: {loss.item():.6f}")
    print("Loss function working correctly!")