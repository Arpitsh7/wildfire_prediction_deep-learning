import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datasets.wildfire_dataset import WildfireDataset

def debug_data():
    print("Debugging data...")
    
    train_ds = WildfireDataset(
        "data/processed/X_train.npy",
        "data/processed/Y_train.npy",
        augment=False
    )
    
    print(f"Dataset length: {len(train_ds)}")
    
    x, y = train_ds[0]
    print(f"First sample - X: {x.shape}, Y: {y.shape}")
    print(f"X min/max: {x.min():.4f}/{x.max():.4f}")
    print(f"Y min/max: {y.min():.4f}/{y.max():.4f}")
    print(f"Y unique: {torch.unique(y)}")
    
    # Check a few more samples
    fire_counts = []
    for i in range(min(10, len(train_ds))):
        _, y = train_ds[i]
        fire_counts.append((y > 0).sum().item())
    
    print(f"Fire counts in first 10 samples: {fire_counts}")
    print(f"Total fire pixels in first 10: {sum(fire_counts)}")

if __name__ == "__main__":
    debug_data()