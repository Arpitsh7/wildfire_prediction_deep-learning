#!/usr/bin/env python3
"""
Train APAU-Net with different pos_weight values to find optimal configuration
Target: F1 >= 0.60
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import numpy as np
from scipy import ndimage
import logging
import time

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss

LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'training_optimization.log'
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    logger.info("Loading dataset...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    
    X_train = torch.tensor(X_train).permute(0, 3, 1, 2).float()
    y_train = torch.tensor(y_train).unsqueeze(1).float()
    X_val = torch.tensor(X_val).permute(0, 3, 1, 2).float()
    y_val = torch.tensor(y_val).unsqueeze(1).float()
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float()
    y_test = torch.tensor(y_test).unsqueeze(1).float()
    
    # Normalize
    logger.info("Normalizing data...")
    for i in range(X_train.shape[1]):
        train_min = X_train[:, i].min()
        train_max = X_train[:, i].max()
        if train_max > train_min:
            X_train[:, i] = (X_train[:, i] - train_min) / (train_max - train_min)
            X_val[:, i] = (X_val[:, i] - train_min) / (train_max - train_min)
            X_test[:, i] = (X_test[:, i] - train_min) / (train_max - train_min)
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def remove_small_blobs(pred_binary, min_area=100):
    if isinstance(pred_binary, torch.Tensor):
        pred_binary = pred_binary.cpu().numpy()
    
    if pred_binary.ndim == 4:
        batch_size = pred_binary.shape[0]
        cleaned = np.zeros_like(pred_binary)
        for i in range(batch_size):
            cleaned[i] = remove_small_blobs(pred_binary[i], min_area)
        return cleaned
    elif pred_binary.ndim == 3:
        cleaned = np.zeros_like(pred_binary)
        for c in range(pred_binary.shape[0]):
            cleaned[c] = remove_small_blobs(pred_binary[c], min_area)
        return cleaned
    else:
        labeled, num_features = ndimage.label(pred_binary > 0.5)
        if num_features == 0:
            return np.zeros_like(pred_binary)
        sizes = ndimage.sum(pred_binary > 0.5, labeled, range(num_features + 1))
        mask = sizes > min_area
        cleaned = mask[labeled]
        return cleaned.astype(np.float32)


def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def evaluate_on_test_set(model, test_loader, device, threshold=0.5, min_area=100):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs)
            
            preds_binary = (probs > threshold).float()
            preds_binary = remove_small_blobs(preds_binary, min_area)
            
            all_preds.append(preds_binary)
            all_targets.append(y_batch.numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    tp = np.sum((all_preds > 0.5) & (all_targets > 0.5))
    fp = np.sum((all_preds > 0.5) & (all_targets <= 0.5))
    fn = np.sum((all_preds <= 0.5) & (all_targets > 0.5))
    
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return f1, precision, recall, tp, fp, fn


def train_model(pos_weight, max_epochs=30):
    logger.info(f"\n{'='*80}")
    logger.info(f"Training with pos_weight={pos_weight}")
    logger.info(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = AttentionUNet(in_channels=12, out_channels=1)
    model = model.to(device)
    
    loss_fn = WBCEDiceLoss(pos_weight=pos_weight, wbce_weight=1.0, dice_weight=1.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_val_f1 = 0
    patience = 6
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch.to(device))
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(y_batch.numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        tp = np.sum((all_preds > 0.5) & (all_targets > 0.5))
        fp = np.sum((all_preds > 0.5) & (all_targets <= 0.5))
        fn = np.sum((all_preds <= 0.5) & (all_targets > 0.5))
        _, _, val_f1 = compute_metrics(tp, fp, fn)
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1:2d}/{max_epochs} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'checkpoints',
                f'apau_net_pos_weight_{pos_weight}.pth'
            )
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"  -> Saved (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    elapsed = time.time() - start_time
    logger.info(f"Training time: {elapsed:.1f}s")
    
    # Load be
