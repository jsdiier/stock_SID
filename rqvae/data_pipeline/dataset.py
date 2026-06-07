import numpy as np
import torch
from torch.utils.data import Dataset


class WeeklyEmbeddingDataset(Dataset):
    """
    Loads a pre-built (N, 35) float32 numpy array and serves samples
    as float32 tensors for RQ-VAE training.
    """

    def __init__(self, npy_path: str):
        self.data = torch.from_numpy(np.load(npy_path)).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
