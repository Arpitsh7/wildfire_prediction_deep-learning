"""
APAU-Net Training on MNDWS Dataset - FAST VERSION
- Loss: WBCEDiceLoss (pos_weight=25, WBCE:Dice = 1:1.5)
- Threshold: 0.85 (fixed)
- Post-processing: min_area=25 pixels
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


def load_mndws_data(data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    print(f"Loading MNDWS dataset from {data_path}...")
    dataset = MNDWSDataset(data_path)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return dataset, train_indices, val_indices, test_indices


class SubsetDataset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


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
    elif pred_binary.ndim == 3:
        cleaned = np.zeros_like(pred_binary)
        for c in range(pred_binary.shape[0]):
            cleaned[c] = remove_small_blobs(pred_binary[c], min_area)
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
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_with_fixed_threshold(model, val_loader, device, threshold=0.85, min_area=25):
    model.eval()
    total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            metrics = compute_metrics_with_postproc(outputs, y, threshold=threshold, min_area=min_area)
            total_metrics['tp'] += metrics['tp']
            total_metrics['fp'] += metrics['fp']
            total_metrics['fn'] += metrics['fn']
    
    total_tp = total_metrics['tp']
    total_fp = total_metrics['fp']
    total_fn = total_metrics['fn']
    
    precision_total = total_tp / (total_tp + total_fp + 1e-8)
    recall_total = total_tp / (total_tp + total_fn + 1e-8)
    f1_total = 2 * precision_total * recall_total / (precision_total + recall_total + 1e-8)
    iou_total = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    
    return {
        'precision': float(precision_total),
        'recall': float(recall_total),
        'f1': float(f1_total),
        'iou': float(iou_total),
        'tp': int(total_tp),
        'fp': int(total_fp),
        'fn': int(total_fn)
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed_mndws")
    log_file = os.path.join(base_dir, "training_mndws_fast.log")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold = 0.85
    min_area = 25
    
    header = f"""
================================================================================
               APAU-NET FAST TRAINING ON MNDWS
                          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
Device: {device}
Loss: WBCEDiceLoss (pos_weight=25, dice_weight=1.5)
Threshold: {threshold}, Min Area: {min_area}
================================================================================
"""
    log_to_file(log_file, header)
    
    dataset, train_indices, val_indices, test_indices = load_mndws_data(data_dir)
    
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    test_dataset = SubsetDataset(dataset, test_indices)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    log_to_file(log_file, f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]
    log_to_file(log_file, f"Input channels: {in_channels}")
    
    model = AttentionUNet(in_channels=in_channels, out_channels=1).to(device)
    log_to_file(log_file, f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = WBCEDiceLoss(pos_weight=25, bce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    
    log_to_file(log_file, f"\n{'Epoch':<8} {'Loss':<12} {'Val F1':<10} {'Best F1':<10} {'LR':<12}")
    log_to_file(log_file, "-" * 60)
    
    best_f1 = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    epochs = 20
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_with_fixed_threshold(model, val_loader, device, threshold, min_area)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""
        
        log_to_file(log_file, f"{epoch:<8} {train_loss:<12.6f} {val_metrics['f1']:<10.6f} {best_f1:<10.6f} {current_lr:<12.2e} {marker}")
        
        scheduler.step()
        
        if patience_counter >= patience:
            log_to_file(log_file, f"\nEarly stopping at epoch {epoch}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    log_to_file(log_file, "-" * 60)
    log_to_file(log_file, f"\nBest Val F1: {best_f1:.6f}")
    
    log_to_file(log_file, "\n" + "="*60)
    log_to_file(log_file, "TEST SET EVALUATION")
    log_to_file(log_file, "="*60)
    
    test_metrics = evaluate_with_fixed_threshold(model, test_loader, device, threshold, min_area)
    
    log_to_file(log_file, f"Precision: {test_metrics['precision']:.6f}")
    log_to_file(log_file, f"Recall:    {test_metrics['recall']:.6f}")
    log_to_file(log_file, f"F1 Score:  {test_metrics['f1']:.6f}")
    log_to_file(log_file, f"IoU:       {test_metrics['iou']:.6f}")
    
    target_f1 = 0.3725
    target_precision = 0.2968
    target_recall = 0.5002
    
    log_to_file(log_file, f"\nTarget F1: {target_f1:.4f} | Actual: {test_metrics['f1']:.4f} | Diff: {test_metrics['f1'] - target_f1:+.4f}")
    log_to_file(log_file, f"Target Precision: {target_precision:.4f} | Actual: {test_metrics['precision']:.4f}")
    log_to_file(log_file, f"Target Recall:    {target_recall:.4f} | Actual: {test_metrics['recall']:.4f}")
    
    model_path = os.path.join(checkpoint_dir, "apau_net_mndws_new.pth")
    torch.save(model.state_dict(), model_path)
    log_to_file(log_file, f"\nModel saved: {model_path}")
    
    results = {
        'version': 'mndws_new',
        'loss_config': {'pos_weight': 25, 'dice_weight': 1.5},
        'threshold': threshold,
        'min_area': min_area,
        'best_val_f1': float(best_f1),
        'test_metrics': test_metrics,
        'target_metrics': {'f1': target_f1, 'precision': target_precision, 'recall': target_recall}
    }
    
    results_path = os.path.join(results_dir, "apau_net_mndws_new_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log_to_file(log_file, f"Results saved: {results_path}")
    
    log_to_file(log_file, "\n" + "="*60)
    log_to_file(log_file, "TRAINING COMPLETE!")
    log_to_file(log_file, "="*60)
    
    return results


if __name__ == "__main__":
    results = main()