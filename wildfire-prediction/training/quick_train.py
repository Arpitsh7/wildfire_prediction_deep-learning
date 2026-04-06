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
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))[:200]
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))[:200]
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))[:50]
    y_val = np.load(os.path.join(data_dir, "Y_val.npy"))[:50]
    
    X_train = torch.tensor(X_train).permute(0,3,1,2).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    X_val = torch.tensor(X_val).permute(0,3,1,2).float()
    y_val = torch.tensor(y_val).unsqueeze(1).float()
    
    y_train = torch.clamp(y_train, 0, 1)
    y_val = torch.clamp(y_val, 0, 1)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    cw = (y_train.numel() - y_train.sum().item()) / y_train.sum().item()
    
    model = ResNetUNet(in_channels=12, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "level1.pth")))
    print("Loaded Level 1 weights")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw]))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model.train()
    for x, y in train_loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Trained 1 epoch")
    
    model.eval()
    metrics = MetricsTracker()
    with torch.no_grad():
        pred = model(X_val)
        metrics.update(pred, y_val, threshold=0.7)
    
    r = metrics.get_avg()
    print(f"Results: P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
    
    if r['f1'] > 0.2259:
        print("*** IMPROVED! ***")
        torch.save(model.state_dict(), os.path.join(base_dir, "checkpoints", "level2.pth"))
        with open(os.path.join(base_dir, "results", "level2_metrics.json"), 'w') as f:
            json.dump({'level': 2, 'status': 'completed', 'best_f1': r['f1']}, f, indent=2)
    else:
        print("No improvement")


if __name__ == "__main__":
    train()
