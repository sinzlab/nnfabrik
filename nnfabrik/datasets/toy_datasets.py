import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def toy_dataset_fn(seed, batch_size=64):
    
    np.random.seed(seed)
    x, y = np.random.rand(1000, 5).astype(np.float32), np.random.rand(1000, 1).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    
    return DataLoader(dataset, batch_size=batch_size)