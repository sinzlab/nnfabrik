from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def toy_dataset_fn(seed: int, **config) -> Dict:
    """
    Returns data loaders for the given config
    Args:
        seed (int): random seed that will make shuffling and other random operations deterministic
    Returns:
        data_loaders (dict): containing "train", "validation" and "test" data loaders
    """
    np.random.seed(seed)
    x, y = (np.random.rand(1000, 5).astype(np.float32), np.random.rand(1000, 1).astype(np.float32))
    train_dataset = TensorDataset(torch.from_numpy(x[:800]), torch.from_numpy(y[:800]))
    dev_dataset = TensorDataset(torch.from_numpy(x[800:900]), torch.from_numpy(y[800:900]))
    test_dataset = TensorDataset(torch.from_numpy(x[900:]), torch.from_numpy(y[900:]))

    return {
        "train": DataLoader(train_dataset, batch_size=config.get("batch_size", 64)),
        "validation": DataLoader(dev_dataset, batch_size=config.get("batch_size", 64)),
        "test": DataLoader(test_dataset, batch_size=config.get("batch_size", 64)),
    }
