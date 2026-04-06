import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from models.resnet_unet import ResNetUNet
from utils.metrics import MetricsTracker


def test():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    X = np.load(os.path.join(data_dir, "X_val.npy"))
    y = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    X = torch.tensor(X[:50]).permute(0,3,1,2).float()
    y = torch.tensor(y[:50]).unsqueeze(1).float()
    y = torch.clamp(y, 0, 1)
    
    print(f"Testing on {len(X)} samples...")
    
    model = ResNetUNet(in_channels=12, out_channels=1)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "level1.pth")))
    model.eval()
    
    metrics = MetricsTracker()
    with torch.no_grad():
        pred = model(X)
        metrics.update(pred, y, threshold=0.7)
    
    r = metrics.get_avg()
    print(f"P: {r['precision']:.4f}, R: {r['recall']:.4f}, F1: {r['f1']:.4f}")


if __name__ == "__main__":
    test()
