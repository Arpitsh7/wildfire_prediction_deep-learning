import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

dataset_path = r"D:\Downloads-D\dataset_ndsw.tfrecord"

dataset = tf.data.TFRecordDataset(dataset_path)

feature_description = {
    "tmmx": tf.io.FixedLenFeature([64, 64], tf.float32),
    "tmmn": tf.io.FixedLenFeature([64, 64], tf.float32),
    "vs": tf.io.FixedLenFeature([64, 64], tf.float32),
    "pr": tf.io.FixedLenFeature([64, 64], tf.float32),
    "sph": tf.io.FixedLenFeature([64, 64], tf.float32),
    "th": tf.io.FixedLenFeature([64, 64], tf.float32),
    "pdsi": tf.io.FixedLenFeature([64, 64], tf.float32),
    "erc": tf.io.FixedLenFeature([64, 64], tf.float32),
    "NDVI": tf.io.FixedLenFeature([64, 64], tf.float32),
    "elevation": tf.io.FixedLenFeature([64, 64], tf.float32),
    "population": tf.io.FixedLenFeature([64, 64], tf.float32),
    "PrevFireMask": tf.io.FixedLenFeature([64, 64], tf.float32),
    "FireMask": tf.io.FixedLenFeature([64, 64], tf.float32)
}

def parse_example(example):
    parsed = tf.io.parse_single_example(example, feature_description)
    
    X = tf.stack([
        parsed["tmmx"],
        parsed["tmmn"],
        parsed["vs"],
        parsed["pr"],
        parsed["sph"],
        parsed["th"],
        parsed["pdsi"],
        parsed["erc"],
        parsed["NDVI"],
        parsed["elevation"],
        parsed["population"],
        parsed["PrevFireMask"]
    ], axis=-1)
    
    Y = parsed["FireMask"]
    
    return X, Y

print("Parsing TFRecord dataset...")
dataset = dataset.map(parse_example)

X = []
Y = []

for i, (x, y) in enumerate(dataset):
    X.append(x.numpy())
    Y.append(y.numpy())
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i + 1} samples...")

X = np.array(X)
Y = np.array(Y)

print(f"Total samples: {len(X)}")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")

indices = np.arange(len(X))
train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/Y_train.npy", Y_train)
np.save("data/processed/X_val.npy", X_val)
np.save("data/processed/Y_val.npy", Y_val)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/Y_test.npy", Y_test)

print("Data saved successfully!")
