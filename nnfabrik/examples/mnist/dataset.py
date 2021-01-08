from typing import Dict

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_dataset_fn(seed: int, **config) -> Dict:
    """
    Returns data loaders for the given config
    Args:
        seed: random seed that will make shuffling and other random operations deterministic
    Returns:
        data_loaders: containing "train", "validation" and "test" data loaders
    """
    np.random.seed(seed)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # (mean,std) of MNIST train set
        ]
    )
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    validation_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )  # for simplicity, we use the test set for validation
    test_dataset = datasets.MNIST("../data", train=False, download=True, transform=transform)
    batch_size = config.get("batch_size", 64)
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size),
        "validation": DataLoader(validation_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }
