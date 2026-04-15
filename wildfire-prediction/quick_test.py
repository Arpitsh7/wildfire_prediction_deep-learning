import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np

from models.attention_unet import AttentionUNet
from datasets.wildfire_dataset import WildfireDataset
from utils.metrics import MetricsTracker
from utils.losses import FocalLoss

def quick_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    device = torch.device('cpu')  # Force CPU to avoid issues
    print(f"Device: {device}")
    
    # Small subset for quick testing
    train_ds = WildfireDataset(
        os.path.join(data_dir, "X_train.npy"),
        os.path.join(data_dir, "Y_train.npy"),
        augment=False  # No augmentation for speed
    )
    val_ds = WildfireDataset(
        os.path.join(data_dir, "X_val.npy"),
        os.path.join(data_dir, "Y_val.npy"),
        augment=False
    )
    
    # Use only first 20 samples for ultra-quick test
    from torch.utils.data import Subset
    train_indices = list(range(min(20, len(train_ds))))
    val_indices = list(range(min(10, len(val_ds))))
    train_ds = Subset(train_ds, train_indices)
    val_ds = Subset(val_ds, val_indices)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    
    # Use Attention U-Net
    model = AttentionUNet(in_channels=12, out_channels=1).to(device)
    
    # Use Focal Loss 
    criterion = FocalLoss(alpha=1, gamma=2, logits=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Testing 2 epochs...")
    best_f1 = 0
    
    for epoch in range(2):
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
            
            if batch_idx % 2 == 0:
                print(f'  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                pred_sigmoid = torch.sigmoid(pred)
                metrics.update(pred_sigmoid, y, threshold=0.7)
        
        results = metrics.get_avg()
        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, P={results["precision"]:.4f}, R={results["recall"]:.4f}, F1={results["f1"]:.4f}')
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
    
    print(f'\nBest F1: {best_f1:.4f}')
    print("Quick test completed!")

if __name__ == "__main__":
    quick_test()