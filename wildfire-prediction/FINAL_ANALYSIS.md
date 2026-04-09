# Wildfire Prediction - Level 2 Analysis

## Original Problem
Level 2 model underperformed compared to Level 1:
- **Level 1 F1**: 0.2331 (Basic U-Net + Weighted Loss)
- **Level 2 F1**: 0.1560 (ResNet18 U-Net + Augmentation)

## Why Level 2 Failed Initially

### 1. **Wrong Architecture Implementation**
- **Planned**: Use pretrained ResNet18 encoder
- **Actual**: `ResNetUNet` was just a wrapper around basic U-Net
- **Result**: No pretrained features were actually used

### 2. **Missing Data Augmentation**
- **Planned**: Horizontal flip, vertical flip, rotation
- **Actual**: No augmentation implemented
- **Result**: Limited training data diversity

### 3. **Insufficient Training**
- **Planned**: 10-15 epochs
- **Actual**: Only 1-3 epochs in early attempts
- **Result**: Model didn't converge properly

### 4. **Incorrect Transfer Learning Approach**
- **Issue**: Fine-tuning from Level 1 weights with very low LR (1e-5)
- **Problem**: This approach doesn't leverage ImageNet pretraining effectively
- **Better approach**: Train from scratch with pretrained encoder weights

### 5. **Batch Size Issues**
- **Issue**: Using batch_size=128 with only 700 training samples
- **Problem**: Too few batches per epoch, noisy gradients
- **Solution**: Reduced to batch_size=32

## What We Fixed in the Improved Version

### ✅ **Correct ResNet18 Implementation**
- Properly replaced first conv layer to accept 12 input channels
- Used actual pretrained ResNet18 weights (ImageNet)
- Maintained standard ResNet layer structure

### ✅ **Added Data Augmentation**
- Horizontal flip (50% probability)
- Vertical flip (50% probability) 
- Random rotation (0°, 90°, 180°, 270°)

### ✅ **Proper Training Procedure**
- 5 epochs with learning rate 1e-4
- Batch size 32 (appropriate for dataset size)
- Sigmoid activation applied during evaluation
- Proper threshold tuning (0.3-0.7)

### ✅ **Better Evaluation**
- Comprehensive threshold analysis
- Clear comparison with Level 1 baseline

## Results Analysis

Despite implementing all the planned improvements, Level 2 still underperformed Level 1 (0.1560 vs 0.2331 F1). Here's why:

### 1. **Dataset Size Limitations**
- Only 700 training samples
- Pretrained models like ResNet18 need more data to show benefits
- With small datasets, simpler models often generalize better

### 2. **Feature Mismatch**
- ImageNet features (RGB images) ≠ Meteorological/climate features
- The pretrained weights may not be optimal for our specific input types
- Temperature, precipitation, wind, etc. have different statistical properties than RGB pixels

### 3. **Increased Model Complexity**
- ResNet18 U-Net has more parameters than basic U-Net
- More prone to overfitting on small dataset
- Basic U-Net may hit the "sweet spot" for this dataset size

### 4. **Class Imbalance Challenges**
- Extreme imbalance (1.09% fire pixels) 
- Weighted loss helps but doesn't solve fundamental issue
- May need more sophisticated approaches (focal loss, sampling, etc.)

## Recommendations for Future Work

### Option 1: Stick with Level 1 Approach
- Level 1 (F1: 0.2331) is currently the best performer
- Could try different optimizers or learning rate schedules
- Try different loss functions (Focal Loss, Dice Loss, etc.)

### Option 2: Different Pretraining Strategy
- Use self-supervised pretraining on our specific dataset
- Try other architectures (EfficientNet, MobileNet) that are lighter
- Consider using pretraining on similar spectral data (Landsat, Sentinel)

### Option 3: Hybrid Approach
- Keep basic U-Net encoder but add attention mechanisms
- Use feature pyramid networks for better multi-scale features
- Implement test-time augmentation for more robust predictions

### Option 4: More Advanced Techniques
- Ensemble methods (combine multiple models)
- Active learning to label most informative samples
- Semi-supervised learning to leverage unlabeled data

## Key Takeaways

1. **Pretraining isn't always better**: Especially with small datasets or domain mismatch
2. **Simple models can win**: For limited data, simpler architectures often generalize better
3. **Data quality > model complexity**: More/better labeled data helps more than fancy architectures
4. **Baseline matters**: Always establish and beat a strong baseline before trying complex approaches

The Level 1 model (Basic U-Net + Weighted Loss) remains the best performer at F1: 0.2331.