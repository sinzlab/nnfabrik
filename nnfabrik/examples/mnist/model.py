from typing import Dict
import numpy as np
import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h_dim: int = 5) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, out_dim)
        self.nl = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nl(self.fc1(x))
        return self.softmax(self.fc2(x))


def mnist_model_fn(dataloaders: Dict, seed: int, **config) -> torch.nn.Module:
    """
    Builds a model object for the given config
    Args:
        data_loaders (dict): a dictionary of data loaders
        seed (int): random seed (e.g. for model initialization)
    Returns:
        Instance of torch.nn.Module
    """
    # get the input and output dimension for the model
    first_input, first_output = next(iter(dataloaders["train"]))
    in_dim = np.prod(first_input.shape[1:])
    out_dim = 10

    torch.manual_seed(seed)  # for reproducibility (almost)
    model = MNISTModel(in_dim, out_dim, h_dim=config.get("h_dim", 5))

    return model
