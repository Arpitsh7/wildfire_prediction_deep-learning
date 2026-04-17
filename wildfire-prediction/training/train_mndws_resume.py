"""
Resume training MNDWS model - quick training with existing checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import numpy as np
from datetime import datetime
from scipy import ndimage

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss
from data.load_mndws import MNDWSDataset


def log_to_file(log_file, message):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def remove_small_blobs(pred_binary, min_area=25):
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.cpu().numpy()
    
    if pred_binary.ndim == 4:
        batch_size = pred_binary.shape[0]
        cleaned = np.zeros_like(pred_binary)
        for i in range(batch_size):
            cleaned[i] = remove_small_blobs(pred_binary[i], min_area)
        return cleaned
    else:
        labeled, num_features = ndimage.label(pred_binary > 0.5)
        if num_features == 0:
            return pred_binary
        sizes = ndimage.sum(pred_binary > 0.5, labeled, range(num_features + 1))
        mask = sizes > min_area
        cleaned = mask[labeled]
        return cleaned.astype(np.float32)


def compute_metrics_with_postproc(outputs, targets, threshold=0.85, min_area=25):
    probs = torch.sigmoid(outputs)
    preds_binary = (probs > threshold).float()
    preds_postproc = torch.tensor(
        remove_small_blobs(preds_binary.cpu().numpy(), min_area),
        dtype=torch.float32,
        device=outputs.device
    )
    targets = targets.float()
    
    tp = ((preds_postproc == 1) & (targets == 1)).sum().float().item()
    fp = ((preds_postproc == 1) & (targets == 0)).sum().float().item()
    fn = ((preds_postproc == 0) & (targets == 1)).sum().float().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed_mndws")
    log_file = os.path.join(base_dir, "training_mndws_resume.log")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold = 0.85
    min_area = 25
    
    log_to_file(log_file, f"Device: {device}")
    log_to_file(log_file, f"Threshold: {threshold}, Min Area: {min_area}")
    
    # Load dataset
    dataset = MNDWSDataset(data_dir)
    total_samples = len(dataset)
    log_to_file(log_file, f"Total samples: {total_samples}")
    
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.85)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    log_to_file(log_file, f"Train: {len(train_loader)}, Val: {len(val_loader)} batches")
    
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]
    log_to_file(log_file, f"Input channels: {in_channels}")
    
    # Load existing checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "apau_net_wbce_dice.pth")
    model = AttentionUNet(in_channels=in_channels, out_channels=1).to(device)
    
    # Start fresh - MNDWS has 22 channels vs previous 12
    log_to_file(log_file, f"Starting fresh with {in_channels} input channels")
    
    log_to_file(log_file, f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = WBCEDiceLoss(pos_weight=25, bce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    
    log_to_file(log_file, f"\nEpoch  Loss       Val F1     Best F1")
    log_to_file(log_file, "-" * 50)
    
    best_f1 = 0
    best_model_state = None
    epochs = 5
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                metrics = compute_metrics_with_postproc(outputs, y, threshold, min_area)
                total_metrics['tp'] += metrics['tp']
                total_metrics['fp'] += metrics['fp']
                total_metrics['fn'] += metrics['fn']
        
        tp, fp, fn = total_metrics['tp'], total_metrics['fp'], total_metrics['fn']
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        val_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "*"
        else:
            marker = ""
        
        log_to_file(log_file, f"{epoch:<6} {train_loss:<10.4f} {val_f1:<10.4f} {best_f1:<10.4f} {marker}")
        scheduler.step()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    log_to_file(log_file, "-" * 50)
    log_to_file(log_file, f"Best Val F1: {best_f1:.6f}")
    
    # Test evaluation
    log_to_file(log_file, "\nTEST EVALUATION:")
    model.eval()
    total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            metrics = compute_metrics_with_postproc(outputs, y, threshold, min_area)
            total_metrics['tp'] += metrics['tp']
            total_metrics['fp'] += metrics['fp']
            total_metrics['fn'] += metrics['fn']
    
    tp, fp, fn = total_metrics['tp'], total_metrics['fp'], total_metrics['fn']
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    log_to_file(log_file, f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}")
    log_to_file(log_file, f"TP: {tp}, FP: {fp}, FN: {fn}")
    
    target_f1 = 0.3725
    log_to_file(log_file, f"\nTarget F1: {target_f1:.4f} | Actual: {f1:.4f} | Diff: {f1 - target_f1:+.4f}")
    
    model_path = os.path.join(checkpoint_dir, "apau_net_mndws_final.pth")
    torch.save(model.state_dict(), model_path)
    log_to_file(log_file, f"\nSaved: {model_path}")
    
    log_to_file(log_file, "\nDONE!")


if __name__ == "__main__":
    main()