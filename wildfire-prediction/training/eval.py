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
    
    X = np.load(os.path.join(data_dir, "X_val.npy"))
    y = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    X = torch.tensor(X).permute(0,3,1,2).float()
    y = torch.tensor(y).unsqueeze(1).float()
    y = torch.clamp(y, 0, 1)
    
    print(f"Val: {X.shape}")
    
    model = ResNetUNet(in_channels=12, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "level1.pth")))
    
    model.eval()
    metrics = MetricsTracker()
    with torch.no_grad():
        for i in range(0, len(X), 32):
            x = X[i:i+32]
            y_batch = y[i:i+32]
            pred = model(x)
            metrics.update(pred, y_batch, threshold=0.7)
    
    results = metrics.get_avg()
    print(f"\n=== Level 1 Model Evaluation ===")
    print(f"P: {results['precision']:.4f}")
    print(f"R: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"IoU: {results['iou']:.4f}")


if __name__ == "__main__":
    train()
