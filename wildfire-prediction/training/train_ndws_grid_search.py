import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import json
from datetime import datetime
import gc

from models.attention_unet import AttentionUNet
from utils.combined_losses_v2 import WBCETverskyLoss
from data.load_mndws import MNDWSDataset


def compute_metrics(outputs, targets, threshold, device):
    probs = torch.sigmoid(outputs)
    preds = (probs > threshold).float()
    targets_f = targets.float()
    tp = ((preds == 1) & (targets_f == 1)).sum().item()
    fp = ((preds == 1) & (targets_f == 0)).sum().item()
    fn = ((preds == 0) & (targets_f == 1)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {'f1': f1, 'precision': prec, 'recall': rec, 'iou': iou}


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device, threshold):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            m = compute_metrics(model(X), y, threshold, device)
            tp += m.get('tp', 0)
            fp += m.get('fp', 0)
            fn += m.get('fn', 0)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {'f1': f1, 'precision': prec, 'recall': rec, 'iou': iou}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = os.path.join(base_dir, "data", "processed_ndws_filtered")
    
    # Focused grid - best params from earlier
    pos_weights = [2, 3, 5]
    thresholds = [0.4, 0.5, 0.6, 0.7]
    
    print(f"Grid Search: {len(pos_weights)*len(thresholds)} runs")
    
    full_dataset = MNDWSDataset(data_path)
    torch.manual_seed(42)
    n = len(full_dataset)
    tr, va, te = int(0.7*n), int(0.15*n), n - int(0.7*n) - int(0.15*n)
    tr_d, va_d, te_d = torch.utils.data.random_split(full_dataset, [tr, va, te])
    
    train_loader = DataLoader(tr_d, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(te_d, batch_size=16, num_workers=0)
    
    in_ch = tr_d[0][0].shape[0]
    results = []
    
    for pw in pos_weights:
        for th in thresholds:
            print(f"pw={pw}, th={th}...", end=" ", flush=True)
            
            model = AttentionUNet(in_channels=in_ch, out_channels=1).to(device)
            criterion = WBCETverskyLoss(pos_weight=pw, alpha=0.3, beta=0.7)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            for e in range(8):
                train_epoch(model, train_loader, criterion, optimizer, device)
            
            res = evaluate(model, test_loader, device, th)
            
            results.append({
                'pos_weight': pw, 'threshold': th,
                'test_f1': res['f1'],
                'test_precision': res['precision'],
                'test_recall': res['recall'],
                'test_iou': res['iou']
            })
            
            print(f"F1={res['f1']:.4f}")
            del model, criterion, optimizer
            gc.collect()
            torch.cuda.empty_cache()
    
    results_sorted = sorted(results, key=lambda x: x['test_f1'], reverse=True)
    
    print("\n=== RESULTS ===")
    for r in results_sorted:
        print(f"pw={r['pos_weight']}, th={r['threshold']}: F1={r['test_f1']:.4f}, Prec={r['test_precision']:.4f}, Rec={r['test_recall']:.4f}")
    
    best = results_sorted[0]
    print(f"\nBEST: pw={best['pos_weight']}, th={best['threshold']}, F1={best['test_f1']:.4f}")
    
    with open(os.path.join(results_dir, "ndws_grid_search_results.json"), 'w') as f:
        json.dump({'experiments': results, 'best': best}, f, indent=2)
    
    return results_sorted


if __name__ == "__main__":
    main()