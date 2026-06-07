"""
PyTorch Dataset for SFT training.
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class SIDDataset(Dataset):
    """
    Each item is (context, target):
      context : (H, K) long  — H past weeks of SID codes (left-padded with PAD_CODE)
      target  : (K,)   long  — next week's SID codes (ground truth)
    """

    def __init__(self, contexts: np.ndarray, targets: np.ndarray):
        if len(contexts) != len(targets):
            raise ValueError("contexts and targets must have the same length")
        self.contexts = torch.from_numpy(contexts).long()   # (N, H, K)
        self.targets  = torch.from_numpy(targets).long()    # (N, K)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]
