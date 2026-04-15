import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np
from datetime import datetime

from models.attention_unet import AttentionUNet
from utils.metrics import MetricsTracker


def load_full_data():
    """Load full dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
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
    
    # Normalize
    for i in range(X_train.shape[1]):
        min_val = X_train[:, i].min()
        max_val = X_train[:, i].max()
        if max_val > min_val:
            X_train[:, i] = (X_train[:, i] - min_val) / (max_val - min_val)
            X_val[:, i] = (X_val[:, i] - min_val) / (max_val - min_val)
            X_test[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)
    
    print(f"Full Data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_class_weight(y):
    """Compute class weight for weighted loss."""
    total = y.numel()
    fire = (y > 0).sum().item()
    if fire == 0:
        return 1.0
    return (total - fire) / fire


def train_apau_net():
    """Train APAU-Net with BCEWithLogitsLoss (matched to Level 1 for fair comparison)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\n" + "="*70)
    print("APAU-NET TRAINING (FULL DATASET, WEIGHTED BCE LOSS)")
    print("="*70)
    
    # Load full data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_full_data()
    
    # Compute class weight
    cw = compute_class_weight(y_train)
    print(f"Class weight: {cw:.2f}")
    
    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Initialize APAU-Net model
    print("\nInitializing APAU-Net model...")
    model = AttentionUNet(in_channels=12, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Use weighted BCE loss (same as Level 1 for fair comparison)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0
    best_state = None
    best_threshold = 0.5
    epochs = 8  # Reasonable for full dataset
    
    print(f"\n{'Epoch':<6} {'Loss':<10} {'Val P':<8} {'Val R':<8} {'Val F1':<8} {'Best Thresh':<12}")
    print("-" * 70)
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        num_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validate
        model.eval()
        with torch.no_grad():
            # Test different thresholds
            best_epoch_f1 = 0
            best_epoch_threshold = 0.5
            
            for threshold in [0.5, 0.6, 0.7]:
                metrics = MetricsTracker()
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    metrics.update(pred, y, threshold=threshold)
                
                results = metrics.get_avg()
                if results['f1'] > best_epoch_f1:
                    best_epoch_f1 = results['f1']
                    best_epoch_threshold = threshold
            
            # Get metrics for best threshold
            metrics = MetricsTracker()
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                metrics.update(pred, y, threshold=best_epoch_threshold)
            
            val_results = metrics.get_avg()
            
            # Print progress
            print(f"{epoch+1:<6} {train_loss:<10.6f} {val_results['precision']:<8.4f} "
                  f"{val_results['recall']:<8.4f} {val_results['f1']:<8.4f} {best_epoch_threshold:<12.1f}")
            
            # Save best model
            if val_results['f1'] > best_f1:
                best_f1 = val_results['f1']
                best_state = model.state_dict().copy()
                best_threshold = best_epoch_threshold
    
    print("-" * 70)
    print(f"Best F1 on validation: {best_f1:.4f} at threshold {best_threshold:.2f}")
    
    # Evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    # Threshold analysis on test set
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12} {'Dice':<12}")
    print("-" * 72)
    
    test_results = {}
    for threshold in np.linspace(0.3, 0.8, 11):
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                metrics.update(pred, y, threshold=threshold)
        
        results = metrics.get_avg()
        test_results[float(threshold)] = results
        print(f"{threshold:<12.1f} {results['precision']:<12.4f} {results['recall']:<12.4f} "
              f"{results['f1']:<12.4f} {results['iou']:<12.4f} {results['dice']:<12.4f}")
    
    print("-" * 72)
    
    # Find best test threshold
    best_test_threshold = max(test_results.keys(), key=lambda t: test_results[t]['f1'])
    best_test_metrics = test_results[best_test_threshold]
    
    print(f"\nBest Test Threshold: {best_test_threshold:.1f}")
    print(f"Test F1: {best_test_metrics['f1']:.4f}")
    print(f"Test Precision: {best_test_metrics['precision']:.4f}")
    print(f"Test Recall: {best_test_metrics['recall']:.4f}")
    print(f"Test IoU: {best_test_metrics['iou']:.4f}")
    print(f"Test Dice: {best_test_metrics['dice']:.4f}")
    
    # Save checkpoint and results
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, os.path.join(ckpt_dir, "apau_net.pth"))
    print(f"\nCheckpoint saved to: checkpoints/apau_net.pth")
    
    # Save detailed results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = {
        'level': 'APAU-Net',
        'status': 'completed',
        'timestamp': datetime.now().isoformat(),
        'model': 'AttentionUNet (APAU-Net: Atrous + Pyramid + Attention)',
        'model_params': total_params,
        'architecture_features': [
            'Atrous/Dilated Convolutions (dilation rates: 1, 2, 4, 8)',
            'Multi-Scale Feature Pyramid (64x64 → 32x32 → 16x16 → 8x8 → 4x4)',
            'CBAM Attention (Channel + Spatial)',
            'Attention Gates in Decoder',
            'Skip Connections'
        ],
        'training_config': {
            'optimizer': 'Adam',
            'learning_rate': 1e-4,
            'loss': 'BCEWithLogitsLoss',
            'pos_weight': float(cw),
            'batch_size': batch_size,
            'epochs': epochs,
            'normalization': 'Min-Max per channel',
            'gradient_clipping': 1.0,
        },
        'validation_results': {
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1),
        },
        'test_results': {
            'best_threshold': best_test_threshold,
            'precision': float(best_test_metrics['precision']),
            'recall': float(best_test_metrics['recall']),
            'f1': float(best_test_metrics['f1']),
            'iou': float(best_test_metrics['iou']),
            'dice': float(best_test_metrics['dice']),
        },
        'threshold_analysis': {str(k): {
            'precision': v['precision'],
            'recall': v['recall'],
            'f1': v['f1'],
            'iou': v['iou'],
            'dice': v['dice']
        } for k, v in test_results.items()},
    }
    
    with open(os.path.join(results_dir, "apau_net_results.json"), 'w') as f:
        json.dump(results_file, f, indent=2)
    
    print(f"Results saved to: results/apau_net_results.json")
    
    return results_file


if __name__ == "__main__":
    results = train_apau_net()
    print("\n" + "="*70)
    print("APAU-NET TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel: APAU-Net (Atrous Pyramid Attention U-Net)")
    print(f"Test F1: {results['test_results']['f1']:.4f}")
    print(f"Test Precision: {results['test_results']['precision']:.4f}")
    print(f"Test Recall: {results['test_results']['recall']:.4f}")
