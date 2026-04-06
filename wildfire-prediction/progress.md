# Progress Report - Wildfire Prediction Project

## Summary
Level 1 implementation complete with basic U-Net and weighted loss to handle class imbalance.

---

## Evolution of Results

### Stage 1: Basic U-Net (No Weighted Loss)
- **F1**: 0.002
- **Issue**: Severe class imbalance (1.09% fire pixels) - model predicted no fire for everything

### Stage 2: Weighted Loss + 5 Epochs
- **F1**: 0.2312
- **Improvement**: ~100x better
- **Class weight**: ~90 (ratio of no-fire to fire pixels)

### Stage 3: Weighted Loss + Threshold Tuning + 5 Epochs
- **Best Threshold**: 0.7
- **Best F1**: 0.2331
- **Best Precision**: 0.1346
- **Best Recall**: 0.8773

---

## Level 1 Final Results

### Threshold Analysis
| Threshold | Precision | Recall | F1 | IoU |
|-----------|-----------|--------|----|----|
| 0.3 | 0.1123 | 0.8963 | 0.1992 | 0.1110 |
| 0.4 | 0.1177 | 0.8907 | 0.2076 | 0.1162 |
| 0.5 | 0.1231 | 0.8857 | 0.2159 | 0.1214 |
| 0.6 | 0.1287 | 0.8816 | 0.2243 | 0.1267 |
| **0.7** | **0.1346** | **0.8773** | **0.2331** | **0.1323** |

**Key Insight**: Higher threshold = better precision & F1 (more conservative predictions)

---

## Why Level 2?

### Current Issues:
1. **Low F1 Score**: 0.2331 - needs significant improvement
2. **Low Precision**: Only 13.46% - too many false positives
3. **Basic Encoder**: No pretrained features being used
4. **No Augmentation**: Limited training data diversity

### Why Now:
- Level 1 baseline established
- Weighted loss and threshold tuning done
- Need better feature extraction to improve precision

---

## Level 2 Plan

### Goals:
1. **Improve Precision**: From 13% → 25%+
2. **Improve F1**: From 0.23 → 0.40+
3. **Better Features**: Use pretrained ResNet18 encoder

### Changes:
| Component | Level 1 | Level 2 |
|-----------|---------|---------|
| Encoder | Basic conv layers | ResNet18 pretrained |
| Decoder | Basic U-Net | Same |
| Augmentation | None | Flip, rotate |
| Epochs | 5 | 10-15 |
| LR Schedule | None | Step decay |

### Expected Improvements:
- Pretrained features from ImageNet → better representation
- Data augmentation → more diverse training samples
- Better precision due to improved features

---

## Changes Made

1. **Model**: Basic U-Net (encoder-decoder with skip connections)
2. **Loss Function**: Changed from BCE to BCEWithLogitsLoss with pos_weight=90
3. **Threshold**: Tested 0.3-0.7, found 0.7 optimal
4. **Batch Size**: 32 (increased from 16 for faster training)

---

## Level 2 Results

| Metric | Value |
|--------|-------|
| Model | U-Net + Augmentation + LR Schedule |
| Epochs | 3 |
| Best Threshold | 0.7 |
| **F1** | **0.0006** |
| Status | ❌ Worse than Level 1 |

### Analysis:
- Level 2 did not improve over Level 1
- The model may need more epochs or different hyperparameters
- Recommendation: Continue with Level 1 or try different approach

---

## Current Status

| Level | Status | F1 Score |
|-------|--------|----------|
| Level 1 | ✅ BEST | 0.2331 |
| Level 2 | 🔄 In Progress | - |

---

## Files Created/Modified

| Step | Description | Status |
|------|-------------|--------|
| 1 | Load TFRecord data | ✅ |
| 2 | Process & split data (70/15/15) | ✅ |
| 3 | Implement Basic U-Net | ✅ |
| 4 | Implement metrics (P/R/F1/IoU) | ✅ |
| 5 | Train with weighted loss | ✅ |
| 6 | Tune thresholds | ✅ |
| 7 | Save checkpoint & results | ✅ |
| 8 | Create context.md | ✅ |
| 9 | Create progress.md | ✅ |
| 10 | Document Level 2 plan | ✅ |

---

## Files Created/Modified
- `models/resnet_unet.py` - U-Net model
- `training/train.py` - Training script
- `utils/metrics.py` - Metrics
- `data/load_data.py` - Data loader
- `checkpoints/level1.pth` - Model checkpoint (Level 1)
- `results/level1_metrics.json` - Results (Level 1)
- `context.md` - Project context
- `progress.md` - This file
