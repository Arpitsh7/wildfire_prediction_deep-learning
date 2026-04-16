# WBCE + 2×Dice Loss Training Guide for APAU-Net

## Overview
This guide explains the implementation of **WBCE + 2×Dice Loss** with **ReduceLROnPlateau scheduler** and **early stopping** for training APAU-Net on wildfire prediction.

**Baseline**: F1 = 0.46 (BCEWithLogitsLoss)
**Target**: F1 ≥ 0.56 (+22% improvement)

---

## Files Created

### 1. `utils/combined_losses.py` (14 KB)
Comprehensive loss function module containing:

#### Loss Classes
- **DiceLoss**: Direct F1 optimization (mathematically equivalent to F1 score)
- **WBCEDiceLoss**: RECOMMENDED - Loss = 1.0×WBCE + 2.0×DiceLoss
- **FocalDiceLoss**: Focal + Dice for hard example mining
- **WeightedComboLoss**: Fully customizable weighted combination

#### Preset Functions
- `get_wbce_dice_loss()`: Returns configured WBCEDiceLoss instance
- `get_focal_dice_loss()`: Returns configured FocalDiceLoss instance
- `get_balanced_combo_loss()`: Returns balanced BCE+Focal+Dice
- `get_custom_combo_loss()`: Fully customizable configuration

#### Key Features
- Smooth parameter: 1e-7 (tight to true Dice)
- Class weight: 90.33 (handles wildfire imbalance)
- Device-agnostic (automatically handles GPU/CPU)
- Test suite with quick validation

---

### 2. `training/train_apau_net_wbce_dice.py` (19 KB)
Complete training pipeline with advanced features:

#### Core Components
1. **Data Loading**
   - Supports train/val/test split (700/150/150 samples)
   - Min-max normalization per channel
   - Automatic device detection (GPU/CPU)

2. **Loss Function**
   - WBCEDiceLoss: Loss = 1.0×WBCE(pos_weight=90.33) + 2.0×DiceLoss(smooth=1e-7)
   - Weighting breakdown: 33% BCE (imbalance) + 67% Dice (F1 optimization)

3. **Optimizer**
   - AdamW with learning rate 1e-4
   - L2 regularization (weight_decay=1e-4) for small dataset
   - Gradient clipping (max_norm=1.0)

4. **Learning Rate Scheduler**
   - ReduceLROnPlateau
   - Mode: 'max' (maximize validation F1)
   - Factor: 0.5 (reduce LR by 50%)
   - Patience: 3 epochs (wait for improvement)
   - Minimum LR: 1e-6

5. **Early Stopping**
   - Patience: 5 epochs
   - Monitors validation F1 score
   - Saves best model automatically

#### Training Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 25 max | Sufficient for small dataset |
| Batch Size | 32 | Balanced for memory/convergence |
| LR | 1e-4 | Standard for deep networks |
| Weight Decay | 1e-4 | Regularization for 700 samples |
| Gradient Clip | 1.0 | Prevent exploding gradients |
| Scheduler | ReduceLROnPlateau | Adaptive learning |
| Early Stop | patience=5 | Prevent overfitting |

#### Validation & Testing
- Threshold sweep: 0.2-0.8 during training (7 thresholds)
- Threshold sweep: 0.1-0.9 on test set (17 thresholds)
- Metrics tracked: F1, Precision, Recall, IoU, Dice

#### Output
- Checkpoint: `checkpoints/apau_net_wbce_dice.pth`
- Results: `results/apau_net_wbce_dice_results.json`
- Includes:
  - Training history (loss, F1, precision, recall per epoch)
  - Learning rates per epoch
  - Threshold analysis
  - Improvement metrics vs baseline

---

## Loss Function Explanation

### WBCE (Weighted Binary Cross-Entropy)
```
WBCE = -[pos_weight * y * log(σ(x)) + (1-y) * log(1-σ(x))]
```
- Handles class imbalance (fire pixels ~1%)
- pos_weight = 90.33 (computed from training data)
- Penalizes positive class misclassification more

### Dice Loss
```
DiceLoss = 1 - (2*TP) / (2*TP + FP + FN)
         = 1 - Dice Coefficient
         = 1 - F1 Score (for binary classification)
```
- Mathematically equivalent to F1
- Smooth parameter prevents division by zero
- Direct optimization toward F1 metric

### Combined: WBCE + 2×Dice
```
Loss = 1.0 × WBCE(pos_weight=90.33) + 2.0 × DiceLoss(smooth=1e-7)
     = 33% class imbalance handling + 67% F1 optimization
```
- Stable training from two complementary losses
- Heavy focus on Dice for direct F1 improvement
- Maintains class imbalance awareness

---

## Why This Configuration Optimizes F1

### Mathematical Reasoning
1. **Dice Loss = F1**: Training directly optimizes your metric
2. **WBCE handles imbalance**: 90.33 weight prevents "always negative" solution
3. **2.0× multiplier**: 67% gradient contribution to Dice maintains focus

### Empirical Reasoning
1. **Small dataset (700 samples)**: Dice provides better regularization than BCE alone
2. **Class imbalance (1% fire)**: WBCE weight prevents majority class dominance
3. **Semantic segmentation**: Dice standard in medical imaging (same task type)

### Why Better Than Current (BCEWithLogitsLoss)?
- Current: Optimizes cross-entropy (probabilistic), not F1
- New: Optimizes F1 directly + handles imbalance
- Result: +0.10 F1 points expected (0.46 → 0.56)

---

## Running the Training

### Prerequisites
```bash
# Ensure data is loaded
ls data/processed/X_train.npy  # Should exist

# Check APAU-Net model
ls models/attention_unet.py    # Should exist
```

### Basic Usage
```bash
cd /path/to/wildfire-prediction

# Run training
python training/train_apau_net_wbce_dice.py

# Expected output:
# - Real-time training progress
# - Checkpoint saved to checkpoints/apau_net_wbce_dice.pth
# - Results saved to results/apau_net_wbce_dice_results.json
```

### Output Examples
```
Device: cuda
Timestamp: 2026-04-15T23:30:00.000000

================================================================================
APAU-NET TRAINING WITH WBCE + 2×DICE LOSS
================================================================================
Loss Configuration: 1.0 × WBCE(pos_weight=90.33) + 2.0 × DiceLoss(smooth=1e-7)
Expected Improvement: F1 from 0.46 → ≥0.56 (+22%)
================================================================================

Data loaded:
  Train: torch.Size([700, 12, 64, 64]) with 483 fire pixels
  Val:   torch.Size([150, 12, 64, 64]) with 105 fire pixels
  Test:  torch.Size([150, 12, 64, 64]) with 98 fire pixels

Epoch   Train Loss   Val F1     Val Prec   Val Rec    Best Thresh   LR           Status     
-----------------------------
1       0.245123     0.4832     0.4521     0.5234     0.50          1.00e-04     → Best!
2       0.198765     0.5102     0.4876     0.5421     0.55          1.00e-04     → Best!
...
```

### Monitoring Training
- **Train Loss**: Should decrease initially, then stabilize
- **Val F1**: Target should reach 0.55+ by epoch 15-20
- **Best Threshold**: Should converge to 0.5-0.6 range
- **Learning Rate**: Drops when F1 plateaus (ReduceLROnPlateau)
- **Early Stopping**: Triggers if F1 doesn't improve for 5 epochs

---

## Results Format

### Results JSON Structure
```json
{
  "timestamp": "2026-04-15T23:45:00.000000",
  "model": "APAU-Net (Atrous-Pyramid-Attention U-Net)",
  "loss_function": {
    "name": "WBCEDiceLoss",
    "formula": "Loss = 1.0 × WBCE(pos_weight=90.33) + 2.0 × DiceLoss(smooth=1e-7)",
    "components": {
      "wbce_weight": 1.0,
      "wbce_pos_weight": 90.33,
      "dice_weight": 2.0,
      "dice_smooth": 1e-07
    }
  },
  "test_results": {
    "best_threshold": 0.5,
    "f1": 0.56,
    "precision": 0.48,
    "recall": 0.67,
    "iou": 0.38,
    "dice": 0.56
  },
  "improvement_over_baseline": {
    "baseline_f1": 0.46,
    "current_f1": 0.56,
    "absolute_improvement": 0.10,
    "relative_improvement_percent": 21.7
  }
}
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)
**Solution**: Reduce batch_size in train_apau_net_wbce_dice.py
```python
batch_size = 16  # Instead of 32
```

### Issue: F1 Plateaus Early
**Solution**: Check learning rate scheduler
- Reduce factor from 0.5 to 0.7 (gentler decay)
- Increase patience from 3 to 5 (wait longer)

### Issue: Training Oscillates/Diverges
**Solution**: Reduce learning rate
```python
lr=5e-5  # Instead of 1e-4
```

### Issue: Early Stopping Too Aggressive
**Solution**: Increase patience
```python
early_stopping = EarlyStopping(patience=10)  # Instead of 5
```

---

## Comparison: BCEWithLogitsLoss vs WBCE+2×Dice

| Aspect | BCEWithLogitsLoss | WBCE + 2×Dice |
|--------|---|---|
| **Metric optimized** | Cross-entropy | F1 score directly |
| **Class imbalance** | pos_weight=90.33 |
