"""
APAU-Net Training v2: WBCEDiceLoss with Post-Processing
- Loss: WBCEDiceLoss (pos_weight=25, WBCE:Dice = 1:1.5)
- Epochs: 35 with early stopping (patience=5-7)
- Thresholds: [0.7, 0.8, 0.85, 0.9, 0.95]
- Post-processing: Remove small blobs (min_area=50-200)
- Real-time logging to training_apau_net_wbce_dice.log
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import numpy as np
from datetime import datetime
from scipy import ndimage

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss


def load_data():
    """Load and normalize dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    print("Loading dataset...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    
    # Convert to tensors
    X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    X_val = torch.tensor(X_val).permute(0, 3, 1, 2).float()
    y_val = torch.tensor(y_val).unsqueeze(1).float()
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float()
    y_test = torch.tensor(y_test).unsqueeze(1).float()
    
    # Normalize per channel (min-max)
    print("Normalizing data...")
    for i in range(X_train.shape[1]):
        min_val = X_train[:, i].min()
        max_val = X_train[:, i].max()
        if max_val > min_val:
            X_train[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
            X_val[:, i] = (X_val[:, i] - min_val) / (max_val - min_val)
            X_test[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def log_to_file(log_file, message):
    """Write message to log file and print."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def remove_small_blobs(pred_binary, min_area=100):
    """Remove small connected components (blobs) from prediction."""
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.cpu().numpy()
    
    if pred_binary.ndim == 4:
        # Batch processing
        batch_size = pred_binary.shape[0]
        cleaned = np.zeros_like(pred_binary)
        for i in range(batch_size):
            cleaned[i] = remove_small_blobs(pred_binary[i], min_area)
        return cleaned
    elif pred_binary.ndim == 3:
        # Single sample with channel dimension
        cleaned = np.zeros_like(pred_binary)
        for c in range(pred_binary.shape[0]):
            cleaned[c] = remove_small_blobs(pred_binary[c], min_area)
        return cleaned
    else:
        # Single 2D array
        labeled, num_features = ndimage.label(pred_binary > 0.5)
        sizes = ndimage.sum(pred_binary > 0.5, labeled, range(num_features + 1))
        mask = sizes > min_area
        cleaned = mask[labeled]
        return cleaned.astype(np.float32)


def compute_metrics_with_postproc(outputs, targets, threshold=0.5, min_area=100):
    """Compute metrics with post-processing."""
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
    """Train one epoch."""
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


def evaluate_threshold_sweep(model, val_loader, device, thresholds, min_area=100):
    """Test multiple thresholds and return best."""
    model.eval()
    
    best_f1 = 0
    best_threshold = 0.5
    all_results = []
    
    with torch.no_grad():
        for threshold in thresholds:
            total_metrics = {
                'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0,
                'tp': 0, 'fp': 0, 'fn': 0
            }
            num_batches = 0
            
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                
                metrics = compute_metrics_with_postproc(outputs, y, threshold=threshold, min_area=min_area)
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                num_batches += 1
            
            # Recalculate F1 from total counts
            total_tp = total_metrics['tp']
            total_fp = total_metrics['fp']
            total_fn = total_metrics['fn']
            
            precision_total = total_tp / (total_tp + total_fp + 1e-8)
            recall_total = total_tp / (total_tp + total_fn + 1e-8)
            f1_total = 2 * precision_total * recall_total / (precision_total + recall_total + 1e-8)
            
            all_results.append({
                'threshold': threshold,
                'precision': float(precision_total),
                'recall': float(recall_total),
                'f1': float(f1_total),
                'iou': float(total_tp / (total_tp + total_fp + total_fn + 1e-8))
            })
            
            if f1_total > best_f1:
                best_f1 = f1_total
                best_threshold = threshold
    
    return best_f1, best_threshold, all_results


def main():
    """Main training function."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(base_dir, "training_apau_net_wbce_dice.log")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    results_dir = os.path.join(base_dir, "results")
    
    # Clear log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    header = f"""
================================================================================
              APAU-NET TRAINING v2: WBCEDiceLoss + POST-PROCESSING
                         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
Device: {device}
Loss: WBCEDiceLoss (pos_weight=25, WBCE:Dice = 1:1.5)
Epochs: 35 (Early stopping patience: 5-7)
Thresholds: [0.7, 0.8, 0.85, 0.9, 0.95]
Post-processing: Remove small blobs (min_area=100 pixels)
================================================================================
"""
    log_to_file(log_file, header)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # Create data loaders
    batch_size = 32
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    log_to_file(log_file, f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    log_to_file(log_file, "\nInitializing APAU-Net model...")
    model = AttentionUNet(in_channels=12, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log_to_file(log_file, f"Total Parameters: {total_params:,}")
    
    # Loss: WBCEDiceLoss (pos_weight=25, WBCE:Dice = 1:1.5)
    criterion = WBCEDiceLoss(pos_weight=25, bce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    log_to_file(log_file, f"\nLoss: WBCEDiceLoss")
    log_to_file(log_file, f"  - pos_weight: 25")
    log_to_file(log_file, f"  - WBCE weight: 1.0")
    log_to_file(log_file, f"  - Dice weight: 1.5")
    log_to_file(log_file, f"Optimizer: Adam (lr=1e-4)")
    log_to_file(log_file, f"Scheduler: StepLR (decay=0.5 every 5 epochs)")
    log_to_file(log_file, f"Batch Size: {batch_size}")
    log_to_file(log_file, f"Max Epochs: 35, Early Stopping Patience: 5-7")
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    patience = 6  # 5-7 range
    patience_counter = 0
    epochs = 35
    thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
    min_area = 100
    
    log_to_file(log_file, f"\n{'Epoch':<8} {'Loss':<12} {'Val F1':<10} {'Best F1':<10} {'Best Thresh':<12} {'LR':<12}")
    log_to_file(log_file, "-" * 70)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_f1, best_threshold, _ = evaluate_threshold_sweep(model, val_loader, device, thresholds, min_area)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        log_to_file(log_file, f"{epoch:<8} {train_loss:<12.6f} {val_f1:<10.6f} {best_f1:<10.6f} {best_threshold:<12.4f} {current_lr:<12.2e}")
        
        scheduler.step()
        
        # Early stopping
        if patience_counter >= patience:
            log_to_file(log_file, f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    log_to_file(log_file, "-" * 70)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log_to_file(log_file, f"\nLoaded best model (F1: {best_f1:.6f})")
    
    # Threshold sweep on validation set with post-processing
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "THRESHOLD SWEEP ON VALIDATION SET (WITH POST-PROCESSING)")
    log_to_file(log_file, "="*70)
    
    _, opt_threshold, threshold_results = evaluate_threshold_sweep(model, val_loader, device, thresholds, min_area)
    
    log_to_file(log_file, f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
    log_to_file(log_file, "-" * 70)
    
    for result in threshold_results:
        log_to_file(log_file, f"{result['threshold']:<12.2f} {result['precision']:<12.6f} {result['recall']:<12.6f} {result['f1']:<12.6f} {result['iou']:<12.6f}")
    
    log_to_file(log_file, "-" * 70)
    log_to_file(log_file, f"\nOptimal Threshold: {opt_threshold:.2f} (F1: {max([r['f1'] for r in threshold_results]):.6f})")
    
    # Test set evaluation with post-processing
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "TEST SET EVALUATION (WITH POST-PROCESSING)")
    log_to_file(log_file, "="*70)
    
    model.eval()
    total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            metrics = compute_metrics_with_postproc(outputs, y, threshold=opt_threshold, min_area=min_area)
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
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    
    log_to_file(log_file, f"\nThreshold: {opt_threshold:.2f}")
    log_to_file(log_file, f"Min Area (post-processing): {min_area} pixels")
    log_to_file(log_file, f"True Positives:  {tp}")
    log_to_file(log_file, f"False Positives: {fp}")
    log_to_file(log_file, f"False Negatives: {fn}")
    log_to_file(log_file, f"Precision:       {precision:.6f}")
    log_to_file(log_file, f"Recall:          {recall:.6f}")
    log_to_file(log_file, f"F1 Score:        {f1:.6f}")
    log_to_file(log_file, f"IoU:             {iou:.6f}")
    log_to_file(log_file, f"Dice:            {dice:.6f}")
    
    # Comparison with previous versions
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "COMPARISON WITH PREVIOUS VERSIONS")
    log_to_file(log_file, "="*70)
    
    level1_f1 = 0.2331
    apau_v1_f1 = 0.3001
    improvement_v1 = ((f1 - level1_f1) / level1_f1) * 100
    improvement_v2 = ((f1 - apau_v1_f1) / apau_v1_f1) * 100
    
    log_to_file(log_file, f"Level 1 (BCEWithLogitsLoss):         F1 = 0.2331")
    log_to_file(log_file, f"APAU-Net v1 (BCEWithLogitsLoss):    F1 = 0.3001 (+28.75% vs Level 1)")
    log_to_file(log_file, f"APAU-Net v2 (WBCEDiceLoss):         F1 = {f1:.4f} ({improvement_v2:+.2f}% vs v1, {improvement_v1:+.2f}% vs Level 1)")
    
    if f1 > apau_v1_f1:
        log_to_file(log_file, f"Status: [PASS] v2 OUTPERFORMED v1")
    elif f1 == apau_v1_f1:
        log_to_file(log_file, f"Status: [TIED] v2 MATCHED v1")
    else:
        log_to_file(log_file, f"Status: [FAIL] v2 UNDERPERFORMED v1")
    
    # Save model
    model_path = os.path.join(checkpoint_dir, "apau_net_wbce_dice.pth")
    torch.save(model.state_dict(), model_path)
    log_to_file(log_file, f"\nModel saved to: {model_path}")
    
    # Save results
    results = {
        'version': 'v2_wbce_dice',
        'level': 'APAU-Net v2',
        'status': 'completed',
        'model': 'AttentionUNet (Atrous + Pyramid + Attention U-Net)',
        'loss_function': 'WBCEDiceLoss',
        'loss_config': {
            'pos_weight': 25,
            'bce_weight': 1.0,
            'dice_weight': 1.5
        },
        'epochs_trained': epoch,
        'best_val_f1': float(best_f1),
        'optimal_threshold': float(opt_threshold),
        'post_processing': {
            'enabled': True,
            'min_area': min_area
        },
        'total_parameters': int(total_params),
        'threshold_sweep': threshold_results,
        'test_metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'dice': float(dice),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        },
        'comparison': {
            'level1_f1': level1_f1,
            'apau_v1_f1': apau_v1_f1,
            'apau_v2_f1': float(f1),
            'improvement_vs_v1_percent': float(improvement_v2),
            'improvement_vs_level1_percent': float(improvement_v1)
        },
        'architecture': {
            'input_shape': [1, 12, 64, 64],
            'output_shape': [1, 1, 64, 64],
            'phases_completed': 8,
            'cbam_modules': 9,
            'attention_gates': 4
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(results_dir, "apau_net_wbce_dice_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    log_to_file(log_file, f"Results saved to: {results_path}")
    
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "TRAINING COMPLETE!")
    log_to_file(log_file, "="*70)
    
    return results


if __name__ == "__main__":
    results = main()
