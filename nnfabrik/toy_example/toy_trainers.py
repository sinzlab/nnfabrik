from typing import Dict, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ToyTrainer:
    def __init__(self, model, dataloaders, seed, epochs=5):

        self.model = model
        self.trainloader = dataloaders["train"]
        self.seed = seed
        self.epochs = epochs
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self):
        torch.manual_seed(self.seed)
        losses = []
        for epoch in range(self.epochs):

            _losses = []
            for x, y in self.trainloader:

                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                _losses.append(loss.item())

        losses.append(np.mean(_losses))

        return losses[-1], (losses, self.epochs), self.model.state_dict()


def toy_trainer_fn(
    model: torch.nn.Module, dataloaders: Dict, seed: Tuple, uid: Tuple, cb: Callable, **config
) -> Tuple[float, Dict, Dict]:
    """"
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
    trainer = ToyTrainer(model, dataloaders, seed, epochs=config.get("epochs", 5))
    out = trainer.train()

    return out
