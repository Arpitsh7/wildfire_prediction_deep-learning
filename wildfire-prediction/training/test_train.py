import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from models.resnet_unet import ResNetUNet
from utils.metrics import MetricsTracker


def train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    X = np.load(os.path.join(data_dir, "X_val.npy"))[:32]
    y = np.load(os.path.join(data_dir, "Y_val.npy"))[:32]
    
    X = torch.tensor(X).permute(0,3,1,2).float()
    y = torch.tensor(y).unsqueeze(1).float()
    y = torch.clamp(y, 0, 1)
    
    print(f"Data: {X.shape}")
    
    model = ResNetUNet(in_channels=12, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "level1.pth")))
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([90.0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for i in range(3):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {i+1}: loss={loss.item():.4f}")
    
    model.eval()
    metrics = MetricsTracker()
    with torch.no_grad():
        pred = model(X)
        metrics.update(pred, y, threshold=0.7)
    
    r = metrics.get_avg()
    print(f"Results: P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")


if __name__ == "__main__":
    train()
