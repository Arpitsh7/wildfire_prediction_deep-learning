import tensorflow as tf
import numpy as np
import os 
dataset_path = r"D:\Downloads-D\dataset_ndsw.tfrecord"

dataset = tf.data.TFRecordDataset(dataset_path)

feature_description = {
    "tmmx": tf.io.FixedLenFeature([64,64], tf.float32),
    "tmmn": tf.io.FixedLenFeature([64,64], tf.float32),
    "vs": tf.io.FixedLenFeature([64,64], tf.float32),
    "pr": tf.io.FixedLenFeature([64,64], tf.float32),
    "sph": tf.io.FixedLenFeature([64,64], tf.float32),
    "th": tf.io.FixedLenFeature([64,64], tf.float32),
    "pdsi": tf.io.FixedLenFeature([64,64], tf.float32),
    "erc": tf.io.FixedLenFeature([64,64], tf.float32),
    "NDVI": tf.io.FixedLenFeature([64,64], tf.float32),
    "elevation": tf.io.FixedLenFeature([64,64], tf.float32),
    "population": tf.io.FixedLenFeature([64,64], tf.float32),
    "PrevFireMask": tf.io.FixedLenFeature([64,64], tf.float32),

    "FireMask": tf.io.FixedLenFeature([64,64], tf.float32)
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


dataset = dataset.map(parse_example)

X = []
Y = []

for x,y in dataset:

    X.append(x.numpy())
    Y.append(y.numpy())

X = np.array(X)
Y = np.array(Y)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X.npy", X)
np.save("data/processed/Y.npy", Y)