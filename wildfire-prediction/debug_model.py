import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from datasets.wildfire_dataset import WildfireDataset
from torch.utils.data import DataLoader

def debug_model():
    print("Debugging model...")
    
    train_ds = WildfireDataset(
        "data/processed/X_train.npy",
        "data/processed/Y_train.npy",
        augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=2)
    
    from models.resnet_unet import ResNetUNet
    device = torch.device('cpu')
    model = ResNetUNet(in_channels=12, out_channels=1).to(device)
    
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Input shape: {x.shape}")
            print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
            
            pred = model(x)
            print(f"  Raw output shape: {pred.shape}")
            print(f"  Raw output range: [{pred.min():.4f}, {pred.max():.4f}]")
            
            pred_sigmoid = torch.sigmoid(pred)
            print(f"  Sigmoid output range: [{pred_sigmoid.min():.4f}, {pred_sigmoid.max():.4f}]")
            
            pred_binary = (pred_sigmoid > 0.5).float()
            print(f"  Binary output sum: {pred_binary.sum().item()}")
            print(f"  Target sum: {y.sum().item()}")
            
            if batch_idx >= 2:
                break

if __name__ == "__main__":
    debug_model()