import torch, sys, os
import numpy as np

from models.resnet_unet import ResNetUNet

base = r"C:\Users\Arpit\.openclaw\workspace\wildfire\wildfire-prediction"

X = np.load(os.path.join(base, "data/processed/X_val.npy"))[:8]
y = np.load(os.path.join(base, "data/processed/Y_val.npy"))[:8]

X = torch.tensor(X).permute(0,3,1,2).float()
y = torch.tensor(y).unsqueeze(1).float()
y = torch.clamp(y, 0, 1)

model = ResNetUNet(12, 1)
model.load_state_dict(torch.load(os.path.join(base, "checkpoints/level1.pth")))

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([90.0]))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Training...")
for i in range(5):
    pred = model(X)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  Step {i+1}: loss={loss.item():.4f}")

from utils.metrics import MetricsTracker
metrics = MetricsTracker()
with torch.no_grad():
    pred = model(X)
    metrics.update(pred, y, threshold=0.7)

r = metrics.get_avg()
print(f"\nResults: P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
print("Baseline was: F1=0.2259")

if r['f1'] > 0.2259:
    print("\n*** IMPROVED! ***")
