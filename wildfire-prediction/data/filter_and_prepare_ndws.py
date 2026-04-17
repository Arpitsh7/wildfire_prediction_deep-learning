import numpy as np
import os

DATA_DIR = "C:/Users/Arpit/.openclaw/workspace/wildfire/data/processed"
OUTPUT_DIR = "C:/Users/Arpit/.openclaw/workspace/wildfire/wildfire-prediction/data/processed_ndws_filtered"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading NDWS data...")
X = np.load(os.path.join(DATA_DIR, "X.npy"))
Y = np.load(os.path.join(DATA_DIR, "Y.npy"))

print(f"Original: {X.shape[0]} patches")
print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"Y shape: {Y.shape}, dtype: {Y.dtype}")

fire_mask = Y.max(axis=(1, 2)) > 0
X_filtered = X[fire_mask]
Y_filtered = Y[fire_mask]

print(f"\nAfter filtering (fire patches only):")
print(f"Filtered: {X_filtered.shape[0]} patches")
print(f"Fire patches: {(Y_filtered.max(axis=(1,2)) > 0).sum()}")
print(f"Non-fire patches: {(Y_filtered.max(axis=(1,2)) == 0).sum()}")

print(f"\nSaving to {OUTPUT_DIR}...")

batch_size = 2000
for i in range(0, len(X_filtered), batch_size):
    end = min(i + batch_size, len(X_filtered))
    np.save(f"{OUTPUT_DIR}/X_batch_{i//batch_size}.npy", X_filtered[i:end].astype(np.float16))
    np.save(f"{OUTPUT_DIR}/Y_batch_{i//batch_size}.npy", (Y_filtered[i:end] > 0).astype(np.float16))
    print(f"Saved batch {i//batch_size}: {end-i} samples")

print("\nFiltering complete!")