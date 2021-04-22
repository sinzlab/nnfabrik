"""
Implements the trainer and data generator that is used for the knowledge distillation example
with our TransferredTrainedModel-table
"""

from typing import Dict, Tuple, Callable

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from nnfabrik.examples.mnist.trainer import MNISTTrainer


class MNISTDataGenerator:
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        seed: int,
    ) -> None:
        """
        This is used in the intermediate step to generate the logits from the old model
        """

        self.model = model
        self.trainloader = dataloaders["train"]
        self.testloader = dataloaders["test"]
        self.seed = seed

    def generate(self) -> Tuple[float, Dict, Dict, Dict]:
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()  # To have tqdm output without line-breaks between steps
        torch.manual_seed(self.seed)
        logits_train = []
        for x, y in tqdm(self.trainloader):
            x_flat = x.flatten(1, -1)  # treat the images as flat vectors
            logits_train.append(self.model(x_flat))
        train = {"train": torch.cat(logits_train).detach().to("cpu").numpy()}
        return 0.0, {"transfer_data": train}, self.model.state_dict()


class MNISTKnowledgeDistillationTrainer(MNISTTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        seed: int,
        epochs: int = 5,
    ) -> None:
        """
        This is used to train on logits.
        """
        super().__init__(model, dataloaders, seed, epochs)
        self.loss_fn = nn.MSELoss()

    def train(self):
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()  # To have tqdm output without line-breaks between steps
        torch.manual_seed(self.seed)
        epoch_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0
            epoch_samples = 0
            for x, y in tqdm(self.trainloader):
                # forward:
                self.optimizer.zero_grad()
                x_flat = x.flatten(1, -1)  # treat the images as flat vectors
                logits = self.model(x_flat)
                loss = self.loss_fn(logits, y)
                # backward:
                loss.backward()
                self.optimizer.step()
                # keep track of accuracy:
                epoch_loss += loss.item()
                epoch_samples += y.shape[0]
            epoch_losses.append(epoch_loss / epoch_samples)

        return epoch_losses[-1], (epoch_losses, self.epochs), self.model.state_dict()


def mnist_trainer_fn(
    model: torch.nn.Module, dataloaders: Dict[str, DataLoader], seed: int, uid: Tuple, cb: Callable, **config
) -> Tuple[float, Dict, Dict]:
    """
    Trainer function providing the knwowledge distillation trainer.
    Args:
        model (torch.nn.Module): initialized model to train
        data_loaders (dict): containing "train", "validation" and "test" data loaders
        seed (int): random seed
        uid (tuple): keys that uniquely identify this trainer call
        cb : callback function to ping the database and potentially save the checkpoint
    Returns:
        score: performance score of the model
        output: user specified validation object based on the 'stop function'
        model_state: the full state_dict() of the trained model
    """
    trainer = MNISTKnowledgeDistillationTrainer(model, dataloaders, seed, epochs=config.get("epochs", 2))
    out = trainer.train()

    return out


def mnist_data_gen_fn(
    model: torch.nn.Module, dataloaders: Dict[str,DataLoader], seed: int, uid: Tuple, cb: Callable, **config
) -> Tuple[float, Dict, Dict]:
    """
    Trainer function providing the knwowledge distillation data generator.
    Args:
        model (torch.nn.Module): initialized model to train
        data_loaders (dict): containing "train", "validation" and "test" data loaders
        seed (int): random seed
        uid (tuple): keys that uniquely identify this trainer call
        cb : callback function to ping the database and potentially save the checkpoint
    Returns:
        train: numpy array containing new train set (only targets)
        test: numpy array containing new test set (only targets)
        model_state: the full state_dict() of the trained model
    """
    generator = MNISTDataGenerator(model, dataloaders, seed)
    out = generator.generate()
    return out
