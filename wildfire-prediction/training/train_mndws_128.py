"""
MNDWS Training - 128x128 image size, batch_size=2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import json
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from scipy import ndimage

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss
from data.load_mndws import MNDWSDataset


def log(msg):
    print(msg)
    with open("training_128.log", 'a') as f:
        f.write(msg + "\n")


class ResizeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, size=128):
        self.dataset = dataset
        self.size = size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = F.interpolate(x.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=(self.size, self.size), mode='nearest').squeeze(0)
        return x, y


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
    preds_np = preds.cpu().numpy()
    preds_clean = np.array([remove_small_blobs(p, min_area) for p in preds_np])
    preds = torch.tensor(preds_clean, device=outputs.device, dtype=torch.float32)
    
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return f1, p, r, tp, fp, fn


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed_mndws")
    img_size = 128
    batch_size = 2
    
    device = torch.device('cuda')
    threshold, min_area = 0.85, 25
    
    log(f"Device: {device}")
    log(f"Image size: {img_size}, Batch size: {batch_size}")
    
    dataset = MNDWSDataset(data_dir)
    total = len(dataset)
    log(f"Total: {total}")
    
    np.random.seed(42)
    idx = np.random.permutation(total)
    train_idx, val_idx, test_idx = idx[:14000], idx[14000:17000], idx[17000:]
    
    train_ds = ResizeDataset(torch.utils.data.Subset(dataset, train_idx), img_size)
    val_ds = ResizeDataset(torch.utils.data.Subset(dataset, val_idx), img_size)
    test_ds = ResizeDataset(torch.utils.data.Subset(dataset, test_idx), img_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    in_ch = dataset[0][0].shape[0]
    log(f"Channels: {in_ch}, Train batches: {len(train_loader)}")
    
    model = AttentionUNet(in_channels=in_ch, out_channels=1).to(device)
    log(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = WBCEDiceLoss(pos_weight=25, bce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    log("\nEpoch  Loss     Val F1   Best")
    log("-" * 40)
    
    best_f1, best_state = 0, None
    
    for epoch in range(1, 10):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Eval on validation
        model.eval()
        total_tp, total_fp, total_fn = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                _, _, _, tp, fp, fn = compute_metrics(model(X), y, threshold, min_area)
                total_tp += tp
                total_fp += fp
                total_fn += fn
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        val_f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
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
    total_tp, total_fp, total_fn = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            _, _, _, tp, fp, fn = compute_metrics(model(X), y, threshold, min_area)
            total_tp += tp
            total_fp += fp
            total_fn += fn
    
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    
    log(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    log(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    log(f"Target F1: 0.3725 | Actual: {f1:.4f} | Diff: {f1 - 0.3725:+.4f}")
    
    torch.save(model.state_dict(), os.path.join(base_dir, "checkpoints", "apau_net_mndws_128.pth"))
    log("\nDONE!")


if __name__ == "__main__":
    main()