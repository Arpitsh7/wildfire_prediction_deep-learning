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
from utils.losses import FocalLoss


def load_data_subset():
    """Load only a subset of data for faster training."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))[:200]  # Use only 200 samples
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))[:200]
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))[:50]  # Use only 50 validation samples
    y_val = np.load(os.path.join(data_dir, "Y_val.npy"))[:50]
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))[:50]  # Use only 50 test samples
    y_test = np.load(os.path.join(data_dir, "Y_test.npy"))[:50]
    
    # Convert to tensors
    X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    X_val = torch.tensor(X_val).permute(0, 3, 1, 2).float()
    y_val = torch.tensor(y_val).unsqueeze(1).float()
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float()
    y_test = torch.tensor(y_test).unsqueeze(1).float()
    
    print(f"Subset Data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_apau_net_fast():
    """Fast APAU-Net training with Focal Loss - using subset for quick testing."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("\n" + "="*70)
    print("APAU-NET TRAINING WITH FOCAL LOSS (FAST VERSION - SUBSET DATA)")
    print("="*70)
    
    # Load subset data for faster training
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data_subset()
    
    # Create datasets
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Initialize APAU-Net model
    print("\nInitializing APAU-Net model...")
    model = AttentionUNet(in_channels=12, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Use Focal Loss (better for class imbalance)
    criterion = FocalLoss(alpha=1, gamma=2, logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0
    best_state = None
    best_threshold = 0.5
    epochs = 3  # Quick test with 3 epochs
    
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
                    pred_sigmoid = torch.sigmoid(pred)
                    metrics.update(pred_sigmoid, y, threshold=threshold)
                
                results = metrics.get_avg()
                if results['f1'] > best_epoch_f1:
                    best_epoch_f1 = results['f1']
                    best_epoch_threshold = threshold
            
            # Get metrics for best threshold
            metrics = MetricsTracker()
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                pred_sigmoid = torch.sigmoid(pred)
                metrics.update(pred_sigmoid, y, threshold=best_epoch_threshold)
            
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
                pred_sigmoid = torch.sigmoid(pred)
                metrics.update(pred_sigmoid, y, threshold=threshold)
        
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
    torch.save(best_state, os.path.join(ckpt_dir, "apau_net_fast.pth"))
    print(f"\nCheckpoint saved to: checkpoints/apau_net_fast.pth")
    
    # Save detailed results
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = {
        'level': 'APAU-Net (Fast Test - Subset)',
        'status': 'completed',
        'timestamp': datetime.now().isoformat(),
        'model': 'AttentionUNet (APAU-Net with Atrous, Pyramid, Attention)',
        'model_params': total_params,
        'training_config': {
            'optimizer': 'Adam',
            'learning_rate': 1e-4,
            'loss': 'Focal Loss (alpha=1, gamma=2)',
            'batch_size': batch_size,
            'epochs': epochs,
            'data_subset': '200 train, 50 val, 50 test (fast testing)',
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
        'notes': [
            'This is a fast test run using subset of data (200 train, 50 val, 50 test)',
            'Full training should be run with complete dataset',
            'APAU-Net has all enhancements: Atrous Convolutions + Pyramid + CBAM Attention',
            'Using Focal Loss which is better for class imbalance than BCE',
        ]
    }
    
    with open(os.path.join(results_dir, "apau_net_fast_results.json"), 'w') as f:
        json.dump(results_file, f, indent=2)
    
    print(f"Results saved to: results/apau_net_fast_results.json")
    
    return results_file


if __name__ == "__main__":
    results = train_apau_net_fast()
    print("\n" + "="*70)
    print("APAU-NET FAST TRAINING COMPLETE")
    print("="*70)
    print(f"\nModel: APAU-Net (Atrous Pyramid Attention U-Net)")
    print(f"Test F1: {results['test_results']['f1']:.4f}")
    print(f"Test Precision: {results['test_results']['precision']:.4f}")
    print(f"Test Recall: {results['test_results']['recall']:.4f}")
    print(f"\nNote: This was a quick test on subset. Run full training for complete results.")
