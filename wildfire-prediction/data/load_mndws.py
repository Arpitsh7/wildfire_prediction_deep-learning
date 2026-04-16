import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MNDWSDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        # Get batch files
        self.x_files = sorted([f for f in os.listdir(data_path) if f.startswith("X_batch")])
        self.y_files = sorted([f for f in os.listdir(data_path) if f.startswith("Y_batch")])

        assert len(self.x_files) == len(self.y_files), "Mismatch in X and Y batches"

        # Compute batch sizes (without loading into RAM)
        self.batch_sizes = []
        for f in self.x_files:
            arr = np.load(os.path.join(data_path, f), mmap_mode='r')
            self.batch_sizes.append(len(arr))

        self.cumulative_sizes = np.cumsum(self.batch_sizes)

        # Cache (only one batch in RAM)
        self.current_batch_idx = -1
        self.batch_data = None
        self.batch_labels = None

    def __len__(self):
        return self.cumulative_sizes[-1]

    def load_batch(self, batch_idx):
        if batch_idx != self.current_batch_idx:
            self.batch_data = np.load(
                os.path.join(self.data_path, self.x_files[batch_idx]),
                mmap_mode='r'
            )
            self.batch_labels = np.load(
                os.path.join(self.data_path, self.y_files[batch_idx]),
                mmap_mode='r'
            )
            self.current_batch_idx = batch_idx

    def __getitem__(self, idx):
        batch_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        
        batch_idx = min(batch_idx, len(self.batch_sizes) - 1)

        if batch_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[batch_idx - 1]
        
        sample_idx = min(sample_idx, self.batch_sizes[batch_idx] - 1)

        self.load_batch(batch_idx)

        x = self.batch_data[sample_idx]   # (64,64,12 or 22)
        y = self.batch_labels[sample_idx] # (64,64)

        # Convert to torch
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  # (C,H,W)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)      # (1,H,W)

        return x, y