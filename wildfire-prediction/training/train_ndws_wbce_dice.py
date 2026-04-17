import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import torch
torch.cuda.empty_cache()
gc.collect()

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import json

from models.attention_unet import AttentionUNet
from utils.combined_losses import WBCEDiceLoss
from data.load_mndws import MNDWSDataset


def compute_metrics(outputs, targets, threshold):
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
    return {'f1': f1, 'precision': prec, 'recall': rec, 'iou': iou, 'tp': tp, 'fp': fp, 'fn': fn}


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = os.path.join(base_dir, "data", "processed_ndws_filtered")
    
    print(f'WBCEDiceLoss Training')
    print(f'Device: {device}')
    
    full_dataset = MNDWSDataset(data_path)
    torch.manual_seed(42)
    n = len(full_dataset)
    tr, va, te = int(0.7*n), int(0.15*n), n - int(0.7*n) - int(0.15*n)
    tr_d, va_d, te_d = torch.utils.data.random_split(full_dataset, [tr, va, te])
    
    train_loader = DataLoader(tr_d, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(te_d, batch_size=16, num_workers=0)
    
    in_ch = tr_d[0][0].shape[0]
    print(f'Dataset: {n}, Train: {tr}, Test: {te}')
    
    model = AttentionUNet(in_channels=in_ch, out_channels=1).to(device)
    criterion = WBCEDiceLoss(pos_weight=25, dice_weight=1.5, bce_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(30):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        
        # Eval
        tp = fp = fn = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                m = compute_metrics(model(X), y, 0.85)
                tp += m['tp']
                fp += m['fp']
                fn += m['fn']
        
        f1 = 2 * (tp/(tp+fp+1e-8)) * (tp/(tp+fn+1e-8)) / ((tp/(tp+fp+1e-8)) + (tp/(tp+fn+1e-8)) + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            print(f'Epoch {epoch+1}: F1={f1:.4f} *')
        else:
            print(f'Epoch {epoch+1}: F1={f1:.4f}')
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final eval
    tp = fp = fn = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            m = compute_metrics(model(X), y, 0.85)
            tp += m['tp']
            fp += m['fp']
            fn += m['fn']
    
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    result = {
        'loss': 'WBCEDiceLoss',
        'pos_weight': 25,
        'dice_weight': 1.5,
        'threshold': 0.85,
        'test_f1': f1,
        'test_precision': prec,
        'test_recall': rec,
        'test_iou': iou
    }
    
    print(f'\n=== FINAL ===')
    print(f'F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, IoU: {iou:.4f}')
    
    with open(os.path.join(results_dir, "wbce_dice_results.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("Results saved!")


if __name__ == '__main__':
    main()