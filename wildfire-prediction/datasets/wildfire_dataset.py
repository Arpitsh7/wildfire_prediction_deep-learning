import torch
from torch.utils.data import Dataset
import numpy as np

class WildfireDataset(Dataset):

    def __init__(self, X_path, Y_path, augment=False):
        X = np.load(X_path)
        Y = np.load(Y_path)
        Y = np.clip(Y, 0, 1)
        
        self.X = torch.tensor(X).permute(0,3,1,2).float()
        self.Y = torch.tensor(Y).unsqueeze(1).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.augment:
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[1])
            
            if np.random.random() > 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[0])
            
            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])
        
        return x, y