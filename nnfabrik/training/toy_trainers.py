import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ToyTrainer:
    def __init__(self, model, dataloaders, seed, epochs=5):

        self.model = model
        self.trainloader = dataloaders
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


def toy_trainer_fn(model, dataloaders, seed, epochs=5, **kwargs):
    trainer = ToyTrainer(model, dataloaders, seed, epochs=epochs)
    out = trainer.train()
    
    return out
