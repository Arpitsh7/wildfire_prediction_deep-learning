import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np

from models.resnet_unet import ResNetUNet
from datasets.wildfire_dataset import WildfireDataset
from utils.metrics import MetricsTracker

def test_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    train_ds = WildfireDataset(
        os.path.join(data_dir, "X_train.npy"),
        os.path.join(data_dir, "Y_train.npy"),
        augment=False  # No augmentation for quick test
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
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    
    model = ResNetUNet(in_channels=12, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("Testing 2 batches...")
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        if batch_idx >= 2:  # Just test a few batches
            break
    
    print("\nEvaluating 1 batch...")
    model.eval()
    metrics = MetricsTracker()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = torch.sigmoid(model(x))
            metrics.update(pred, y, threshold=0.7)
            break  # Just one batch
    
    results = metrics.get_avg()
    print(f"\nResults:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"IoU: {results['iou']:.4f}")

if __name__ == "__main__":
    test_train()