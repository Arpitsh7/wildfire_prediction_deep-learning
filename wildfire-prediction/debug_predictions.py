import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.attention_unet import AttentionUNet
from datasets.wildfire_dataset import WildfireDataset
from utils.losses import FocalLoss

def debug_predictions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Test with one sample
    train_ds = WildfireDataset(
        os.path.join(data_dir, "X_train.npy"),
        os.path.join(data_dir, "Y_train.npy"),
        augment=False
    )
    
    x, y = train_ds[0]  # Get first sample
    x = x.unsqueeze(0)  # Add batch dimension
    y = y.unsqueeze(0)  # Add batch dimension
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Target unique values: {torch.unique(y)}")
    print(f"Target sum: {y.sum().item()}")  # Number of fire pixels
    
    # Create model
    model = AttentionUNet(in_channels=12, out_channels=1)
    model.eval()
    
    with torch.no_grad():
        pred = model(x)
        print(f"Raw prediction shape: {pred.shape}")
        print(f"Raw prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"Raw prediction mean: {pred.mean().item():.6f}")
        
        pred_sigmoid = torch.sigmoid(pred)
        print(f"Sigmoid prediction range: [{pred_sigmoid.min():.4f}, {pred_sigmoid.max():.4f}]")
        print(f"Sigmoid prediction mean: {pred_sigmoid.mean().item():.6f}")
        
        # Check what we get with different thresholds
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            pred_binary = (pred_sigmoid > thresh).float()
            print(f"Threshold {thresh}: prediction sum = {pred_binary.sum().item()}")
    
    # Compare with target
    print(f"\nTarget sum: {y.sum().item()}")

if __name__ == "__main__":
    debug_predictions()