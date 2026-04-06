import torch
from torch.utils.data import Dataset
import numpy as np

class WildfireDataset(Dataset):

    def __init__(self, X_path, Y_path):

        X = np.load(X_path)
        Y = np.load(Y_path)

        # convert (N,H,W,C) → (N,C,H,W)
        self.X = torch.tensor(X).permute(0,3,1,2).float()
        self.Y = torch.tensor(Y).unsqueeze(1).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]