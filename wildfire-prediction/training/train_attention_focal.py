import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np

from models.attention_unet import AttentionUNet
from datasets.wildfire_dataset import WildfireDataset
from utils.metrics import MetricsTracker
from utils.losses import FocalLoss, TverskyLoss, ComboLoss


def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Use dataset with augmentation
    train_ds = WildfireDataset(
        os.path.join(data_dir, "X_train.npy"),
        os.path.join(data_dir, "Y_train.npy"),
        augment=True
    )
    val_ds = WildfireDataset(
        os.path.join(data_dir, "X_val.npy"),
        os.path.join(data_dir, "Y_val.npy"),
        augment=False
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    y_all = train_ds.Y
    cw = (y_all.numel() - y_all.sum().item()) / max(y_all.sum().item(), 1)
    print(f"Class weight for reference: {cw:.2f}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Use Attention U-Net
    model = AttentionUNet(in_channels=12, out_channels=1).to(device)
    
    # Try Focal Loss first (good for class imbalance and focusing on hard examples)
    criterion = FocalLoss(alpha=1, gamma=2, logits=True)
    # Alternative: ComboLoss for balancing focal and tversky
    # criterion = ComboLoss(focal_weight=0.7, tversky_weight=0.3, 
    #                      focal_alpha=1, focal_gamma=2,
    #                      tversky_alpha=0.3, tversky_beta=0.7)  # More weight on reducing FP
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler - cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    best_f1 = 0
    best_state = None
    epochs = 10
    
    print(f"Training {epochs} epochs with Attention U-Net + Focal Loss...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation with Test-Time Augmentation (TTA)
        model.eval()
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                # Test-Time Augmentation: average predictions over flips
                pred1 = model(x)
                pred2 = model(torch.flip(x, dims=[2]))  # horizontal flip
                pred3 = model(torch.flip(x, dims=[1]))  # vertical flip
                pred4 = model(torch.rot90(x, k=1, dims=[2,3]))  # rotate 90
                pred5 = model(torch.rot90(x, k=2, dims=[2,3]))  # rotate 180
                pred6 = model(torch.rot90(x, k=3, dims=[2,3]))  # rotate 270
                
                # Average predictions
                pred_avg = (pred1 + pred2 + pred3 + pred4 + pred5 + pred6) / 6
                pred_sigmoid = torch.sigmoid(pred_avg)
                
                metrics.update(pred_sigmoid, y, threshold=0.7)
        
        results = metrics.get_avg()
        avg_loss = train_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, P={results["precision"]:.4f}, R={results["recall"]:.4f}, F1={results["f1"]:.4f}, LR={current_lr:.6f}')
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  *** New best F1: {best_f1:.4f} ***')
        
        scheduler.step()
    
    print(f'\nBest F1 at threshold 0.7: {best_f1:.4f}')
    
    # Threshold tuning with TTA
    print('\nThreshold tuning with Test-Time Augmentation...')
    model.load_state_dict(best_state)
    model.eval()
    
    threshold_results = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                # Apply TTA for consistent evaluation
                pred1 = model(x)
                pred2 = model(torch.flip(x, dims=[2]))
                pred3 = model(torch.flip(x, dims=[1]))
                pred4 = model(torch.rot90(x, k=1, dims=[2,3]))
                pred5 = model(torch.rot90(x, k=2, dims=[2,3]))
                pred6 = model(torch.rot90(x, k=3, dims=[2,3]))
                
                pred_avg = (pred1 + pred2 + pred3 + pred4 + pred5 + pred6) / 6
                pred_sigmoid = torch.sigmoid(pred_avg)
                
                metrics.update(pred_sigmoid, y, threshold=thresh)
        r = metrics.get_avg()
        threshold_results.append({
            "threshold": thresh,
            "precision": r['precision'],
            "recall": r['recall'],
            "f1": r['f1'],
            "iou": r['iou']
        })
        print(f"  Threshold {thresh}: P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
    
    best_thresh_result = max(threshold_results, key=lambda x: x['f1'])
    print(f"\nBest threshold: {best_thresh_result['threshold']}, F1: {best_thresh_result['f1']:.4f}")
    
    level1_f1 = 0.2331
    if best_thresh_result['f1'] > level1_f1:
        improvement = best_thresh_result['f1'] - level1_f1
        print(f"*** New Model IMPROVED over Level 1! (+{improvement:.4f}) ***")
        # Save the improved model
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(best_state, os.path.join(ckpt_dir, "attention_focal.pth"))
    else:
        print(f"*** Model did not improve (diff: {best_thresh_result['f1'] - level1_f1:.4f}) ***")
        # Still save it for comparison
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(best_state, os.path.join(ckpt_dir, "attention_focal.pth"))
    
    # Save results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'level': 'attention_focal',
        'status': 'completed',
        'model': 'Attention U-Net + Focal Loss + TTA',
        'epochs': epochs,
        'class_weight': cw,
        'best_threshold': best_thresh_result['threshold'],
        'best_f1': best_thresh_result['f1'],
        'threshold_results': threshold_results
    }
    with open(os.path.join(results_dir, "attention_focal_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train()