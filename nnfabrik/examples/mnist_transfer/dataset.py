"""
Implements the dataset that is used for the knowledge distillation example with our TransferredTrainedModel-table
"""

from typing import Dict, Tuple

import numpy as np
from PIL.Image import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTKnowledgeDistillation(datasets.MNIST):
    def __init__(self, logits: np.array, *args, **kwargs) -> None:
        """
        Simple dataset that provides the logits that correspond to a specific MNIST image
        Args:
            logits: numpy array of logits that should follow the order of MNIST images
            *args: arguments handed to MNIST
            **kwargs: key-word arguments handed to MNIST
        """
        super().__init__(*args, **kwargs)
        self.logits = logits

    def __getitem__(self, index: int) -> Tuple[Image, np.array]:
        img, target = super().__getitem__(index)
        return img, self.logits[index]


def mnist_dataset_fn(seed: int, **config) -> Dict[str, DataLoader]:
    """
    Returns data loaders for the given config
    Args:
        seed (int): random seed that will make shuffling and other random operations deterministic
    Returns:
        data_loaders (dict): containing "train", "validation" and "test" data loaders
    """
    np.random.seed(seed)

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # (mean,std) of MNIST train set
    ]
    if config.get("apply_augmentation"):
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list
    transform = transforms.Compose(transform_list)
    train_dataset = MNISTKnowledgeDistillation(
        config["transfer_data"]["train"], "../data", train=True, download=True, transform=transform
    )
    validation_dataset = datasets.MNIST(
        "../data", train=False, download=True, transform=transform
    )  # for simplicity, we use the test set for validation
    test_dataset = datasets.MNIST("../data", train=False, download=True, transform=transform)
    batch_size = config.get("batch_size", 64)
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=config.get("shuffle", True)),
        "validation": DataLoader(validation_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }
