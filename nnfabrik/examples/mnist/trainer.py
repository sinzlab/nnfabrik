from typing import Dict, Tuple, Callable, List, Any

from tqdm import tqdm
import torch
from torch import nn
from torch import optim


class MNISTTrainer:
    def __init__(
        self, model: nn.Module, dataloaders: Dict, seed: int, epochs: int = 5,
    ) -> None:

        self.model = model
        self.trainloader = dataloaders["train"]
        self.seed = seed
        self.epochs = epochs
        self.loss_fn = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train_loop(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, int]:
        # forward:
        self.optimizer.zero_grad()
        x_flat = x.flatten(1, -1)  # treat the images as flat vectors
        logits = self.model(x_flat)
        loss = self.loss_fn(logits, y)
        # backward:
        loss.backward()
        self.optimizer.step()
        # keep track of accuracy:
        _, predicted = logits.max(1)
        predicted_correct = predicted.eq(y).sum().item()
        total = y.shape[0]
        return predicted_correct, total

    def train(self) -> Tuple[float, Tuple[List[float],int], Dict]:
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()  # To have tqdm output without line-breaks between steps
        torch.manual_seed(self.seed)
        accs = []
        for epoch in range(self.epochs):
            predicted_correct = 0
            total = 0
            for x, y in tqdm(self.trainloader):
                p, t = self.train_loop(x, y)
                predicted_correct += p
                total += t

            accs.append(100.0 * predicted_correct / total)

        return accs[-1], (accs, self.epochs), self.model.state_dict()


def mnist_trainer_fn(
    model: torch.nn.Module,
    dataloaders: Dict,
    seed: int,
    uid: Dict,
    cb: Callable,
    **config
) -> Tuple[float, Any, Dict]:
    """"
    Args:
        model: initialized model to train
        data_loaders: containing "train", "validation" and "test" data loaders
        seed: random seed
        uid: database keys that uniquely identify this trainer call
        cb: callback function to ping the database and potentially save the checkpoint
    Returns:
        score: performance score of the model
        output: user specified validation object based on the 'stop function'
        model_state: the full state_dict() of the trained model
    """
    trainer = MNISTTrainer(model, dataloaders, seed, epochs=config.get("epochs", 2))
    out = trainer.train()

    return out
