print("Starting simple test...")
import sys
print("Python:", sys.version)
sys.path.insert(0, '.')

import torch
print("Torch:", torch.__version__)

try:
    from models.resnet_unet import ResNetUNet
    print("Model imported OK")
except Exception as e:
    print("Model import failed:", e)

try:
    from datasets.wildfire_dataset import WildfireDataset
    print("Dataset imported OK")
except Exception as e:
    print("Dataset import failed:", e)

print("Test completed")