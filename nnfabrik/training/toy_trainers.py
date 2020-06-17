from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nnfabrik.training.trainer import Trainer


class ToyTrainer(Trainer):
    def __init__(self, model, data_loaders, seed, **config):
        super().__init__(model, data_loaders, seed, **config)
        self.train_loader = data_loaders["train"]
        self.epochs = config.get("epochs", 5)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def __call__(self, uid: Tuple, cb: Callable):
        torch.manual_seed(self.seed)
        losses = []
        for epoch in range(self.epochs):

            _losses = []
            for x, y in self.train_loader:

                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                _losses.append(loss.item())

        losses.append(np.mean(_losses))

        return losses[-1], (losses, self.epochs), self.model.state_dict()
