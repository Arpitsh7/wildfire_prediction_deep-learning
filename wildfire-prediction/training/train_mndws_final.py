"""
MNDWS Training - Quick finish, 3 more epochs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from scipy import ndimage

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss
from data.load_mndws import MNDWSDataset


def log(msg):
    print(msg)
    with open("training_final.log", 'a') as f:
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
    checkpoint_path = os.path.join(base_dir, "checkpoints", "apau_net_mndws.pth")
    
    device = torch.device('cuda')
    threshold, min_area = 0.85, 25
    
    log(f"Device: {device}")
    
    dataset = MNDWSDataset(data_dir)
    total = len(dataset)
    
    np.random.seed(42)
    idx = np.random.permutation(total)[:5000]
    train_idx, val_idx, test_idx = idx[:3500], idx[3500:4250], idx[4250:]
    
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, num_workers=0, pin_memory=True)
    
    in_ch = dataset[0][0].shape[0]
    log(f"Channels: {in_ch}")
    
    model = AttentionUNet(in_channels=in_ch, out_channels=1).to(device)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log("Loaded existing checkpoint")
    
    criterion = WBCEDiceLoss(pos_weight=25, bce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    log("\nEpoch  Loss     Val F1   Best")
    log("-" * 40)
    
    best_f1, best_state = 0, None
    
    # Quick training - 3 epochs
    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
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
    
    # Test evaluation
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
    log(f"IoU: {iou:.4f}")
    log(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    log(f"")
    log(f"Target F1: 0.3725 | Actual: {f1:.4f} | Diff: {f1 - 0.3725:+.4f}")
    log(f"Target Precision: 0.2968 | Actual: {precision:.4f}")
    log(f"Target Recall: 0.5002 | Actual: {recall:.4f}")
    
    torch.save(model.state_dict(), checkpoint_path)
    log(f"\nSaved: {checkpoint_path}")
    log("DONE!")


if __name__ == "__main__":
    main()