import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import numpy as np

from models.resnet_unet import ResNetUNet
from utils.metrics import MetricsTracker


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    X_val = torch.tensor(X_val).permute(0, 3, 1, 2).float()
    y_val = torch.tensor(y_val).unsqueeze(1).float()
    
    y_train = torch.clamp(y_train, 0, 1)
    y_val = torch.clamp(y_val, 0, 1)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    return (X_train, y_train), (X_val, y_val)


def compute_class_weight(y):
    total = y.numel()
    fire = (y > 0).sum().item()
    return (total - fire) / fire


def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    (X_train, y_train), (X_val, y_val) = load_data()
    
    cw = compute_class_weight(y_train)
    print(f"Class weight: {cw:.2f}")
    
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    model = ResNetUNet(in_channels=12, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cw]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        metrics = MetricsTracker()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                metrics.update(pred, y, threshold=0.7)
        
        results = metrics.get_avg()
        print(f"Epoch {epoch+1}: P={results['precision']:.4f}, R={results['recall']:.4f}, F1={results['f1']:.4f}")
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_state = model.state_dict().copy()
    
    print(f"\nBest F1: {best_f1:.4f}")
    
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(best_state, os.path.join(ckpt_dir, "level2.pth"))
    
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results = {
        'level': 2,
        'status': 'completed',
        'model': 'U-Net (larger) + Fine-tune',
        'epochs': 3,
        'class_weight': cw,
        'best_threshold': 0.7,
        'best_f1': best_f1
    }
    with open(os.path.join(results_dir, "level2_metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Level 2 complete!")


if __name__ == "__main__":
    train()