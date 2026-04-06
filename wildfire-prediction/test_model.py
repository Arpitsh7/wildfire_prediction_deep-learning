import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_unet import ResNetUNet

model = ResNetUNet(12, 1)
x = torch.randn(1, 12, 64, 64)
y = model(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
print("Model works!")
