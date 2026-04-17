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
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_at_threshold(model, val_loader, device, threshold=0.85, min_area=25):
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
    
    tp = total_metrics['tp']
    fp = total_metrics['fp']
    fn = total_metrics['fn']
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(base_dir, "training_ndws_filtered.log")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"""
================================================================================
         NDWS FILTERED TRAINING: Fire Patches Only + AttentionUNet
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
Device: {device}
Dataset: NDWS filtered (fire patches only)
Loss: WBCEDiceLoss (pos_weight=25, dice_weight=1.5)
Epochs: 35 with early stopping (patience=6)
Threshold: 0.85
Post-processing: Remove small blobs (min_area=25)
================================================================================
""")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    def log_to_file(message):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
        print(message)
    
    header = f"""
================================================================================
         NDWS FILTERED TRAINING: Fire Patches Only + AttentionUNet
                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
Device: {device}
Dataset: NDWS filtered (fire patches only)
Loss: WBCEDiceLoss (pos_weight=25, dice_weight=1.5)
Epochs: 35 with early stopping (patience=6)
Threshold: 0.85
Post-processing: Remove small blobs (min_area=25)
================================================================================
"""
    log_to_file(header)
    
    data_path = os.path.join(base_dir, "data", "processed_ndws_filtered")
    
    log_to_file("\nLoading NDWS filtered dataset...")
    full_dataset = MNDWSDataset(data_path)
    total_samples = len(full_dataset)
    log_to_file(f"Total samples: {total_samples}")
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    log_to_file(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    log_to_file(f"Batch size: {batch_size}")
    log_to_file(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    sample_x, sample_y = train_dataset[0]
    in_channels = sample_x.shape[0]
    log_to_file(f"Input channels: {in_channels}")
    
    log_to_file("\nInitializing AttentionUNet model...")
    model = AttentionUNet(in_channels=in_channels, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log_to_file(f"Total Parameters: {total_params:,}")
    
    pos_weight = 25
    dice_weight = 1.5
    threshold = 0.85
    min_area = 25
    
    criterion = WBCEDiceLoss(pos_weight=pos_weight, bce_weight=1.0, dice_weight=dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    log_to_file(f"\nLoss: WBCEDiceLoss")
    log_to_file(f"  - pos_weight: {pos_weight}")
    log_to_file(f"  - BCE weight: 1.0")
    log_to_file(f"  - Dice weight: {dice_weight}")
    log_to_file(f"Optimizer: Adam (lr=1e-4)")
    log_to_file(f"Scheduler: StepLR (decay=0.5 every 5 epochs)")
    log_to_file(f"Threshold: {threshold}")
    log_to_file(f"Min Area: {min_area}")
    
    best_f1 = 0
    best_model_state = None
    patience = 6
    patience_counter = 0
    epochs = 35
    
    log_to_file(f"\n{'Epoch':<8} {'Loss':<12} {'Val F1':<10} {'Best F1':<10} {'LR':<12}")
    log_to_file("-" * 60)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_at_threshold(model, val_loader, device, threshold=threshold, min_area=min_area)
        val_f1 = val_metrics['f1']
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            log_to_file(f"{epoch:<8} {train_loss:<12.6f} {val_f1:<10.6f} {best_f1:<10.6f} {current_lr:<12.2e} *")
        else:
            patience_counter += 1
            log_to_file(f"{epoch:<8} {train_loss:<12.6f} {val_f1:<10.6f} {best_f1:<10.6f} {current_lr:<12.2e}")
        
        scheduler.step()
        
        if patience_counter >= patience:
            log_to_file(f"\nEarly stopping at epoch {epoch}")
            break
    
    log_to_file("-" * 60)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log_to_file(f"\nLoaded best model (F1: {best_f1:.6f})")
    
    log_to_file("\n" + "="*60)
    log_to_file("TEST SET EVALUATION")
    log_to_file("="*60)
    
    test_metrics = evaluate_at_threshold(model, test_loader, device, threshold=threshold, min_area=min_area)
    
    log_to_file(f"\nThreshold: {threshold}")
    log_to_file(f"Min Area: {min_area}")
    log_to_file(f"True Positives:  {test_metrics['tp']}")
    log_to_file(f"False Positives: {test_metrics['fp']}")
    log_to_file(f"False Negatives: {test_metrics['fn']}")
    log_to_file(f"Precision:       {test_metrics['precision']:.6f}")
    log_to_file(f"Recall:          {test_metrics['recall']:.6f}")
    log_to_file(f"F1 Score:        {test_metrics['f1']:.6f}")
    log_to_file(f"IoU:             {test_metrics['iou']:.6f}")
    
    model_path = os.path.join(checkpoint_dir, "ndws_filtered_apau_net.pth")
    torch.save(model.state_dict(), model_path)
    log_to_file(f"\nModel saved to: {model_path}")
    
    results = {
        'dataset': 'NDWS_filtered',
        'filtering': 'fire_patches_only',
        'model': 'AttentionUNet',
        'loss': 'WBCEDiceLoss',
        'pos_weight': pos_weight,
        'dice_weight': dice_weight,
        'threshold': threshold,
        'min_area': min_area,
        'epochs_trained': epoch,
        'best_val_f1': float(best_f1),
        'test_metrics': {
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall']),
            'f1': float(test_metrics['f1']),
            'iou': float(test_metrics['iou']),
            'tp': int(test_metrics['tp']),
            'fp': int(test_metrics['fp']),
            'fn': int(test_metrics['fn'])
        },
        'total_parameters': int(total_params),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(results_dir, "ndws_filtered_training_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    log_to_file(f"Results saved to: {results_path}")
    
    log_to_file("\n" + "="*60)
    log_to_file("TRAINING COMPLETE!")
    log_to_file("="*60)
    
    return results


if __name__ == "__main__":
    results = main()