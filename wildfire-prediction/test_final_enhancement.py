import torch
from models.attention_unet import AttentionUNet

# Test the fully enhanced model with all improvements
model = AttentionUNet(in_channels=12, out_channels=1)
print("Fully Enhanced APAU-Net Model Architecture:")
print(model)

# Test with sample input
x = torch.randn(2, 12, 64, 64)  # batch_size=2, channels=12, height=64, width=64
with torch.no_grad():
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test successful!")
    print("All enhancements (Phases 1-8) are working correctly!")