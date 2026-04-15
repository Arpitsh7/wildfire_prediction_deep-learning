"""
APAU-Net Training Script - 25 Epochs with Optimizations
- Learning Rate Decay (Step scheduler every 5 epochs)  
- Early Stopping (patience=3)
- Threshold Sweep (0.3-0.8)
- Real-time logging to training_apau_net.log
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

from models.attention_unet import AttentionUNet


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


def compute_class_weight(y):
    """Compute class weight for imbalanced data."""
    total = y.numel()
    fire = (y > 0).sum().item()
    if fire == 0:
        return 1.0
    return (total - fire) / fire


def log_to_file(log_file, message):
    """Write message to log file and print."""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def compute_metrics_batch(outputs, targets, threshold=0.5):
    """Compute metrics on batch output (logits)."""
    # Convert logits to probabilities
    probs = torch.sigmoid(outputs)
    preds_binary = (probs > threshold).float()
    
    # Ensure targets are float
    targets = targets.float()
    
    # Compute TP, FP, FN using element-wise operations
    tp = ((preds_binary == 1) & (targets == 1)).sum().float().item()
    fp = ((preds_binary == 1) & (targets == 0)).sum().float().item()
    fn = ((preds_binary == 0) & (targets == 1)).sum().float().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
    
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8) if union > 0 else 0
    
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


def evaluate_threshold_sweep(model, val_loader, device, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    """Test multiple thresholds and return best."""
    model.eval()
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
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
                
                metrics = compute_metrics_batch(outputs, y, threshold=threshold)
                for key in total_metrics:
                    if key not in ['tp', 'fp', 'fn']:
                        total_metrics[key] += metrics[key]
                    else:
                        total_metrics[key] += metrics[key]
                num_batches += 1
            
            # Average metrics
            avg_metrics = {
                'precision': total_metrics['precision'] / num_batches,
                'recall': total_metrics['recall'] / num_batches,
                'f1': total_metrics['f1'] / num_batches,
                'iou': total_metrics['iou'] / num_batches
            }
            
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
                best_metrics = all_results[-1]
    
    return best_f1, best_threshold, best_metrics, all_results


def main():
    """Main training function."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(base_dir, "training_apau_net.log")
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
                    APAU-NET TRAINING - 25 EPOCHS OPTIMIZED
                         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
Device: {device}
Optimization Strategy: Learning Rate Decay + Early Stopping + Threshold Sweep
================================================================================
"""
    log_to_file(log_file, header)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # Compute class weight
    class_weight = compute_class_weight(y_train)
    log_to_file(log_file, f"\nClass Weight: {class_weight:.4f}")
    
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
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # LR decay every 5 epochs
    
    log_to_file(log_file, f"\nOptimizer: Adam (lr=1e-4)")
    log_to_file(log_file, f"Scheduler: StepLR (decay=0.5 every 5 epochs)")
    log_to_file(log_file, f"Loss: BCEWithLogitsLoss (pos_weight={class_weight:.4f})")
    log_to_file(log_file, f"Batch Size: {batch_size}")
    log_to_file(log_file, f"Max Epochs: 25, Early Stopping Patience: 3")
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    best_threshold_overall = 0.5
    patience = 3
    patience_counter = 0
    epochs = 25
    
    log_to_file(log_file, f"\n{'Epoch':<8} {'Loss':<12} {'Val F1':<10} {'Best F1':<10} {'Best Thresh':<12} {'LR':<12}")
    log_to_file(log_file, "-" * 70)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_f1, best_threshold, _, _ = evaluate_threshold_sweep(model, val_loader, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_threshold_overall = best_threshold
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
    
    # Threshold sweep on validation set
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "THRESHOLD SWEEP ON VALIDATION SET")
    log_to_file(log_file, "="*70)
    
    _, opt_threshold, _, threshold_results = evaluate_threshold_sweep(model, val_loader, device)
    
    log_to_file(log_file, f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
    log_to_file(log_file, "-" * 70)
    
    for result in threshold_results:
        log_to_file(log_file, f"{result['threshold']:<12.1f} {result['precision']:<12.6f} {result['recall']:<12.6f} {result['f1']:<12.6f} {result['iou']:<12.6f}")
    
    log_to_file(log_file, "-" * 70)
    log_to_file(log_file, f"\nOptimal Threshold: {opt_threshold:.1f} (F1: {max([r['f1'] for r in threshold_results]):.6f})")
    
    # Test set evaluation
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "TEST SET EVALUATION")
    log_to_file(log_file, "="*70)
    
    model.eval()
    total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            metrics = compute_metrics_batch(outputs, y, threshold=opt_threshold)
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
    
    log_to_file(log_file, f"\nThreshold: {opt_threshold:.1f}")
    log_to_file(log_file, f"True Positives:  {tp}")
    log_to_file(log_file, f"False Positives: {fp}")
    log_to_file(log_file, f"False Negatives: {fn}")
    log_to_file(log_file, f"Precision:       {precision:.6f}")
    log_to_file(log_file, f"Recall:          {recall:.6f}")
    log_to_file(log_file, f"F1 Score:        {f1:.6f}")
    log_to_file(log_file, f"IoU:             {iou:.6f}")
    log_to_file(log_file, f"Dice:            {dice:.6f}")
    
    # Comparison with Level 1
    level1_f1 = 0.2331
    improvement = ((f1 - level1_f1) / level1_f1) * 100
    
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "COMPARISON WITH LEVEL 1 BASELINE")
    log_to_file(log_file, "="*70)
    log_to_file(log_file, f"Level 1 F1:      {level1_f1:.6f}")
    log_to_file(log_file, f"APAU-Net F1:     {f1:.6f}")
    log_to_file(log_file, f"Improvement:     {improvement:+.2f}%")
    
    if f1 > level1_f1:
        log_to_file(log_file, f"Status: [PASS] APAU-NET OUTPERFORMED LEVEL 1")
    elif abs(f1 - level1_f1) < 0.01:
        log_to_file(log_file, f"Status: [TIED] APAU-NET MATCHED LEVEL 1")
    else:
        log_to_file(log_file, f"Status: [FAIL] APAU-NET UNDERPERFORMED LEVEL 1")
    
    # Save model
    model_path = os.path.join(checkpoint_dir, "apau_net.pth")
    torch.save(model.state_dict(), model_path)
    log_to_file(log_file, f"\nModel saved to: {model_path}")
    
    # Save results
    results = {
        'level': 'APAU-Net',
        'status': 'completed',
        'model': 'AttentionUNet (Atrous + Pyramid + Attention U-Net)',
        'epochs_trained': epoch,
        'best_val_f1': float(best_f1),
        'optimal_threshold': float(opt_threshold),
        'class_weight': float(class_weight),
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
        'comparison_with_level1': {
            'level1_f1': level1_f1,
            'apau_net_f1': float(f1),
            'improvement_percent': float(improvement)
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
    
    results_path = os.path.join(results_dir, "apau_net_metrics.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    log_to_file(log_file, f"Results saved to: {results_path}")
    
    log_to_file(log_file, "\n" + "="*70)
    log_to_file(log_file, "TRAINING COMPLETE!")
    log_to_file(log_file, "="*70)
    
    return results


if __name__ == "__main__":
    results = main()
