"""
MNDWS Quick Training - GPU optimized, fewer batches
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import json
import numpy as np
from datetime import datetime
from scipy import ndimage

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss
from data.load_mndws import MNDWSDataset


def log(msg):
    print(msg)
    with open("training_quick.log", 'a') as f:
        f.write(msg + "\n")


def remove_small_blobs(pred, min_area=25):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    labeled, n = ndimage.label(pred > 0.5)
    if n == 0:
        return pred
    sizes = ndimage.sum(pred > 0.5, labeled, range(n + 1))
    mask = sizes > min_area
    return (mask[labeled] if hasattr(mask, '__getitem__') else np.zeros_like(pred)).astype(np.float32)


def compute_metrics(outputs, targets, threshold=0.85, min_area=25):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    preds = torch.tensor(remove_small_blobs(preds.cpu().numpy(), min_area), device=outputs.device)
    
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed_mndws")
    
    device = torch.device('cuda')
    threshold, min_area = 0.85, 25
    
    log(f"Device: {device}")
    
    dataset = MNDWSDataset(data_dir)
    total = len(dataset)
    log(f"Total: {total}")
    
    np.random.seed(42)
    idx = np.random.permutation(total)
    train_idx, val_idx, test_idx = idx[:14000], idx[14000:17000], idx[17000:]
    
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, num_workers=0, pin_memory=True)
    
    in_ch = dataset[0][0].shape[0]
    log(f"Channels: {in_ch}, Train batches: {len(train_loader)}")
    
    model = AttentionUNet(in_channels=in_ch, out_channels=1).to(device)
    log(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = WBCEDiceLoss(pos_weight=25, bce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    log("\nEpoch  Loss     Val F1   Best")
    log("-" * 40)
    
    best_f1, best_state = 0, None
    
    for epoch in range(1, 8):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Quick eval on subset
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                _, p, r = compute_metrics(model(X), y, threshold, min_area)
        
        val_f1, val_p, val_r = 0, 0, 0
        count = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                f1, p, r = compute_metrics(model(X), y, threshold, min_area)
                val_f1 += f1
                count += 1
        val_f1 /= count
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "*"
        else:
            marker = ""
        
        log(f"{epoch:<6} {total_loss/len(train_loader):.4f}  {val_f1:.4f}  {best_f1:.4f} {marker}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Test
    log("\n--- TEST ---")
    tp, fp, fn = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            probs = torch.sigmoid(model(X))
            preds = (probs > threshold).float()
            preds = torch.tensor(np.array([remove_small_blobs(p, min_area) for p in preds.cpu().numpy()]), device=device)
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    log(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    log(f"TP: {tp}, FP: {fp}, FN: {fn}")
    log(f"Target F1: 0.3725 | Actual: {f1:.4f} | Diff: {f1 - 0.3725:+.4f}")
    
    torch.save(model.state_dict(), os.path.join(base_dir, "checkpoints", "apau_net_mndws.pth"))
    log("\nDONE!")


if __name__ == "__main__":
    main()