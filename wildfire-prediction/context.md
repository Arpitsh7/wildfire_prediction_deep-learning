# Wildfire Prediction Project - Context

## Project Overview
- **Project**: Wildfire Prediction using Deep Learning
- **Goal**: Predict wildfire presence on 64x64 grid maps using climate/environmental features
- **Dataset**: TFRecord file containing satellite/climate data
- **Data Size**: 1000 samples (700 train, 150 val, 150 test)

## Dataset Features (12 input channels)
1. `tmmx` - Maximum temperature
2. `tmmn` - Minimum temperature
3. `vs` - Wind speed
4. `pr` - Precipitation
5. `sph` - Specific humidity
6. `th` - Temperature humidity
7. `pdsi` - Palmer Drought Severity Index
8. `erc` - Energy Release Component
9. `NDVI` - Normalized Difference Vegetation Index
10. `elevation` - Terrain elevation
11. `population` - Population density
12. `PrevFireMask` - Previous fire mask

## Target
- `FireMask` - Binary fire presence (0 or 1)

## Class Imbalance
- Fire pixels: ~1.09% (highly imbalanced)
- Class weight applied: ~90

---

## Level 1 Results (Completed)

| Metric | Value |
|--------|-------|
| Model | Basic U-Net + Weighted Loss |
| Epochs | 5 |
| Batch Size | 32 |
| Best Threshold | 0.7 |
| **F1** | **0.2331** |
| Precision | 0.1346 |
| Recall | 0.8773 |
| IoU | 0.1323 |

### Key Findings:
- Weighted loss improved F1 from 0.002 → 0.2331 (100x improvement)
- Threshold 0.7 gives best results (conservative predictions)
- Recall is good (87.7%) but precision is low (13.5%)

---

## Why Level 2?

### Current Limitations:
1. **Precision too low**: 13.5% - too many false positives
2. **Basic encoder**: No pretrained features
3. **No augmentation**: Limited training data

### Level 2 Goals:
1. Improve precision from 13% → 25%+
2. Improve F1 from 0.23 → 0.40+
3. Use pretrained ResNet18 for better features

---

## Level 2 Planned Changes

| Component | Level 1 | Level 2 |
|-----------|---------|---------|
| Encoder | Basic conv (64→128→256→512→1024) | ResNet18 pretrained |
| Augmentation | None | Horizontal flip, vertical flip, rotation |
| Epochs | 5 | 10-15 |
| LR Schedule | Constant 1e-4 | Step decay |
| Checkpoint | level1.pth | level2.pth |

---

## Level 2 Results (Completed - Needs Improvement)

| Metric | Value |
|--------|-------|
| Model | U-Net + Augmentation + LR Schedule |
| Epochs | 3 |
| Best Threshold | 0.7 |
| **F1** | **0.0006** |
| Status | Lower than Level 1 |

### Analysis:
- Level 2 did not improve over Level 1
- Need to investigate why the larger model performs worse
- Possible issues: insufficient training time, learning rate, or architecture

---

## Current Status
- **Level 1**: ✅ BEST (F1: 0.2331)
- **Level 2**: 🔄 In Progress

## Files
- `models/resnet_unet.py` - U-Net architecture
- `training/train.py` - Training script
- `utils/metrics.py` - Evaluation metrics
- `data/processed/` - Processed train/val/test data
- `checkpoints/level1.pth` - Level 1 model checkpoint (BEST)
- `checkpoints/level2.pth` - Level 2 model checkpoint
- `results/level1_metrics.json` - Level 1 results
- `results/level2_metrics.json` - Level 2 results
