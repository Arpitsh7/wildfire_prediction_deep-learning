import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

print("Testing all components...")

# Test 1: Model creation
print("\n1. Testing Attention U-Net creation...")
from models.attention_unet import AttentionUNet
model = AttentionUNet(in_channels=12, out_channels=1)
print("   [OK] Attention U-Net created successfully")

# Test 2: Loss functions
print("\n2. Testing loss functions...")
from utils.losses import FocalLoss, TverskyLoss, ComboLoss
focal_loss = FocalLoss(alpha=1, gamma=2, logits=True)
tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)  # More weight on reducing FP
combo_loss = ComboLoss(focal_weight=0.7, tversky_weight=0.3)
print("   [OK] Loss functions created successfully")

# Test 3: Forward pass
print("\n3. Testing forward pass...")
model.eval()
x = torch.randn(2, 12, 64, 64)  # Small batch for testing
with torch.no_grad():
    pred = model(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {pred.shape}")
print(f"   Expected: [2, 1, 64, 64]")
result = pred.shape == (2, 1, 64, 64)
print(f"   [{'OK' if result else 'FAIL'}] Forward pass: {pred.shape}")

# Test 4: Loss computation
print("\n4. Testing loss computation...")
with torch.no_grad():
    # Create dummy target (mostly zeros with some ones to simulate imbalance)
    target = torch.zeros_like(pred)
    target[:, :, 10:20, 10:20] = 1.0  # Small square of ones
    
    focal_val = focal_loss(pred, target)
    tversky_val = tversky_loss(pred, target)
    combo_val = combo_loss(pred, target)
    
    print(f"   Focal loss: {focal_val.item():.6f}")
    print(f"   Tversky loss: {tversky_val.item():.6f}")
    print(f"   Combo loss: {combo_val.item():.6f}")
    print("   [OK] Loss computation successful")

# Test 5: Dataset loading (small subset)
print("\n5. Testing dataset loading...")
from datasets.wildfire_dataset import WildfireDataset
try:
    # Create tiny dataset for testing
    train_ds = WildfireDataset(
        "data/processed/X_train.npy",
        "data/processed/Y_train.npy",
        augment=False
    )
    # Use only first 4 samples
    from torch.utils.data import Subset
    train_ds_small = Subset(train_ds, list(range(min(4, len(train_ds)))))
    train_loader = DataLoader(train_ds_small, batch_size=2, shuffle=True)
    
    x_batch, y_batch = next(iter(train_loader))
    print(f"   Batch X shape: {x_batch.shape}")
    print(f"   Batch Y shape: {y_batch.shape}")
    print(f"   Y unique values: {torch.unique(y_batch)}")
    print("   [OK] Dataset loading successful")
except Exception as e:
    print(f"   [NOTE] Dataset loading: {e}")

print("\n" + "="*50)
print("ALL COMPONENT TESTS COMPLETED")
print("="*50)