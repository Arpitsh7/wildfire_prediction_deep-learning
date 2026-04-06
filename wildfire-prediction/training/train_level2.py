import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np

from models.resnet_unet import ResNetUNet
from utils.metrics import MetricsTracker


def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    X_train = torch.tensor(X_train).permute(0,3,1,2).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    X_val = torch.tensor(X_val).permute(0,3,1,2).float()
    y_val = torch.tensor(y_val).unsqueeze(1).float()
    
    y_train = torch.clamp(y_train, 0, 1)
    y_val = torch.clamp(y_val, 0, 1)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    cw = (y_train.numel() - y_train.sum().item()) / y_train.sum().item()
    print(f"Weight: {cw:.2f}")
    
    model = ResNetUNet(in_channels=12, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "level1.pth")))
    print("Loaded Level 1 weights")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    
    print("Training 1 epoch...")
    model.train()
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Done training")
    
    val_ds = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=128)
    
    model.eval()
    metrics = MetricsTracker()
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            metrics.update(pred, y, threshold=0.7)
    
    results = metrics.get_avg()
    print(f"\nResults:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"IoU: {results['iou']:.4f}")
    
    if results['f1'] > 0.2331:
        print("\n*** Level 2 IMPROVED over Level 1! ***")
        torch.save(model.state_dict(), os.path.join(base_dir, "checkpoints", "level2.pth"))
        with open(os.path.join(base_dir, "results", "level2_metrics.json"), 'w') as f:
            json.dump({
                'level': 2,
                'status': 'completed',
                'best_f1': results['f1'],
                'improvement': results['f1'] - 0.2331
            }, f, indent=2)
    else:
        print("\n*** Level 2 did not improve ***")
        print("Keeping Level 1 as best model")


if __name__ == "__main__":
    train()
