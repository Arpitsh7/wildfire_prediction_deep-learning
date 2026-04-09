import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np

from models.resnet_unet import ResNetUNet
from datasets.wildfire_dataset import WildfireDataset
from utils.metrics import MetricsTracker


def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Use our improved dataset with augmentation
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
    print(f"Class weight: {cw:.2f}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Use our ResNet18 U-Net
    model = ResNetUNet(in_channels=12, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0
    best_state = None
    
    print("Training 5 epochs...")
    for epoch in range(5):
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
        
        # Validation
        model.eval()
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = torch.sigmoid(model(x))  # Apply sigmoid for evaluation
                metrics.update(pred, y, threshold=0.7)
        
        results = metrics.get_avg()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, P={results["precision"]:.4f}, R={results["recall"]:.4f}, F1={results["f1"]:.4f}')
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  *** New best F1: {best_f1:.4f} ***')
    
    print(f'\nBest F1 at threshold 0.7: {best_f1:.4f}')
    
    # Threshold tuning
    print('\nThreshold tuning...')
    model.load_state_dict(best_state)
    model.eval()
    
    threshold_results = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = torch.sigmoid(model(x))
                metrics.update(pred, y, threshold=thresh)
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
        print(f"*** Level 2 IMPROVED over Level 1! (+{improvement:.4f}) ***")
        # Save the improved model
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(best_state, os.path.join(ckpt_dir, "level2.pth"))
    else:
        print(f"*** Level 2 did not improve (diff: {best_thresh_result['f1'] - level1_f1:.4f}) ***")
        # Still save it for comparison
        ckpt_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(best_state, os.path.join(ckpt_dir, "level2.pth"))
    
    # Save results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'level': 2,
        'status': 'completed',
        'model': 'ResNet18 U-Net + Augmentation',
        'epochs': 5,
        'class_weight': cw,
        'best_threshold': best_thresh_result['threshold'],
        'best_f1': best_thresh_result['f1'],
        'threshold_results': threshold_results
    }
    with open(os.path.join(results_dir, "level2_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nLevel 2 training complete!")


if __name__ == "__main__":
    train()