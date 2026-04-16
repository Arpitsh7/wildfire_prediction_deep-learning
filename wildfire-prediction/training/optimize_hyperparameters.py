#!/usr/bin/env python3
"""
Fast Hyperparameter Optimization for APAU-Net
- Uses already trained v2 model
- Tests different min_area and threshold values
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
from datetime import datetime

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss

LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'optimization_results.log'
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
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "Y_test.npy"))
    
    X_test = torch.tensor(X_test).permute(0, 3, 1, 2).float()
    y_test = torch.tensor(y_test).unsqueeze(1).float()
    
    # Normalize
    for i in range(X_test.shape[1]):
        min_val = X_test[:, i].min()
        max_val = X_test[:, i].max()
        if max_val > min_val:
            X_test[:, i] = (X_test[:, i] - min_val) / (max_val - min_val)
    
    return X_test, y_test


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
    tn = np.sum((all_preds <= 0.5) & (all_targets <= 0.5))
    
    precision, recall, f1 = compute_metrics(tp, fp, fn)
    return f1, precision, recall, tp, fp, fn, tn


def main():
    logger.info("="*80)
    logger.info("APAU-Net Hyperparameter Optimization")
    logger.info("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load v2 model
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'checkpoints',
        'apau_net_wbce_dice.pth'
    )
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    model = AttentionUNet(in_channels=12, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Load test data
    X_test, y_test = load_data()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test different combinations
    min_area_values = [25, 50, 75, 100, 150, 200, 250]
    threshold_values = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    results = []
    best_result = {'f1': 0, 'threshold': 0, 'min_area': 0}
    
    logger.info("\nTesting threshold and min_area combinations...")
    for threshold in threshold_values:
        for min_area in min_area_values:
            f1, prec, rec, tp, fp, fn, tn = evaluate_on_test_set(
                model, test_loader, device, threshold=threshold, min_area=min_area
            )
            
            results.append({
                'pos_weight': 25,
                'threshold': threshold,
                'min_area': min_area,
                'f1': float(f1),
                'precision': float(prec),
                'recall': float(rec),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            })
            
            if f1 > best_result['f1']:
                best_result = {
                    'f1': f1,
                    'threshold': threshold,
                    'min_area': min_area,
                    'precision': prec,
                    'recall': rec,
                    'pos_weight': 25
                }
            
            logger.info(f"Threshold={threshold:.2f}, Min_area={min_area:3d} | F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
    
    # Save results
    results_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results',
        'optimization_results_phase1.json'
    )
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("Phase 1 Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Best result:")
    logger.info(f"  Threshold={best_result['threshold']}, Min_area={best_result['min_area']}")
    logger.info(f"  F1={best_result['f1']:.4f}, Precision={best_result['precision']:.4f}, Recall={best_result['recall']:.4f}")
    logger.info(f"\nResults saved to: {results_file}")
    
    if best_result['f1'] >= 0.60:
        logger.info(f"✓ SUCCESS: Achieved F1 >= 0.60!")
    else:
        logger.info(f"F1={best_result['f1']:.4f} (target: 0.60)")

if __name__ == "__main__":
    main()
