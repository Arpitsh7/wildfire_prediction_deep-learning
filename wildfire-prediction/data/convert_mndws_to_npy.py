import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

# ================= CONFIG =================
TFRECORD_PATH = "data/raw/mndws_dataset/ndws_western_dataset"
SAVE_PATH = "data/processed_mndws"
BATCH_SIZE = 2000   # reduce if RAM is low

os.makedirs(SAVE_PATH, exist_ok=True)

# ================= GET FILES =================
files = tf.io.gfile.glob(f"{TFRECORD_PATH}/*.tfrecord")

print("Found TFRecord files:", len(files))
print(files[:3])

if len(files) == 0:
    raise ValueError("❌ No TFRecord files found. Check path.")

# ================= FEATURE KEYS =================
feature_keys = [
    'fuel2', 'population', 'wind_75', 'elevation', 'tmp_day', 'erc',
    'wdir_gust', 'bi', 'wdir_wind', 'water', 'NDVI', 'fuel1',
    'viirs_PrevFireMask', 'impervious', 'tmp_75', 'gust_med',
    'chili', 'pdsi', 'wind_avg', 'pr', 'fuel3', 'avg_sph'
]

target_key = 'viirs_FireMask'

# ================= FEATURE DESCRIPTION =================
feature_description = {}

for key in feature_keys:
    feature_description[key] = tf.io.FixedLenFeature([64, 64], tf.float32)

feature_description[target_key] = tf.io.FixedLenFeature([64, 64], tf.float32)

# ================= PARSER =================
def parse_example(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Input (64,64,22)
    x = tf.stack([example[key] for key in feature_keys], axis=-1)

    # Target (64,64)
    y = example[target_key]

    return x.numpy(), y.numpy()

# ================= PROCESS DATA =================
X_batch, Y_batch = [], []
batch_id = 0
total_samples = 0

for file in files:
    print(f"\nProcessing: {os.path.basename(file)}")

    dataset = tf.data.TFRecordDataset(file)

    for raw_record in tqdm(dataset):
        try:
            x, y = parse_example(raw_record)

            # Reduce memory
            x = x.astype(np.float16)
            y = (y > 0).astype(np.float16)

            X_batch.append(x)
            Y_batch.append(y)
            total_samples += 1

            # Save batch
            if len(X_batch) >= BATCH_SIZE:
                np.save(f"{SAVE_PATH}/X_batch_{batch_id}.npy", np.array(X_batch))
                np.save(f"{SAVE_PATH}/Y_batch_{batch_id}.npy", np.array(Y_batch))

                print(f"✅ Saved batch {batch_id} ({len(X_batch)} samples)")

                X_batch, Y_batch = [], []
                batch_id += 1

        except Exception:
            continue  # skip problematic records

# ================= SAVE REMAINING =================
if len(X_batch) > 0:
    np.save(f"{SAVE_PATH}/X_batch_{batch_id}.npy", np.array(X_batch))
    np.save(f"{SAVE_PATH}/Y_batch_{batch_id}.npy", np.array(Y_batch))

    print(f"✅ Saved final batch {batch_id} ({len(X_batch)} samples)")

# ================= DONE =================
print("\n🎉 mNDWS conversion COMPLETE!")
print(f"Total samples processed: {total_samples}")
print(f"Saved in: {SAVE_PATH}")