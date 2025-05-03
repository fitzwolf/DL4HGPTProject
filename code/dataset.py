import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RichSyntheticICUDataset(Dataset):
    """
    Loads synthetic ICU data from a folder containing X.npy (features) and y.npy (labels).
    Converts them to PyTorch tensors.
    """
    def __init__(self, data_folder):
        self.X = np.load(os.path.join(data_folder, 'X.npy'))  # Shape: (N, C, T)
        self.y = np.load(os.path.join(data_folder, 'y.npy'))  # Shape: (N,)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LongevityDataset(Dataset):
    """
    Loads synthetic ICU data using longevity labels.
    """
    def __init__(self, X_path, longevity_labels_path):
        self.X = np.load(X_path)  # Shape: (N, C, T)
        self.y = np.load(longevity_labels_path)  # Shape: (N,)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
