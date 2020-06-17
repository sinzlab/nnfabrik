from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nnfabrik.datasets.dataset import Dataset


class ToyDataset(Dataset):
    def __call__(self, seed: int, **config) -> Dict:
        np.random.seed(seed)
        x, y = (
            np.random.rand(1000, 5).astype(np.float32),
            np.random.rand(1000, 1).astype(np.float32),
        )
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

        return {
            "train": DataLoader(dataset, batch_size=config.get("batch_size", 64)),
            "validation": None,
            "test": None,
        }
