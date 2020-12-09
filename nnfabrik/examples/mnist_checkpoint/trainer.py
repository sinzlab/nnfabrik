from typing import Dict, Tuple, Callable, Optional, List, Any

from tqdm import tqdm
import torch
import torch.nn as nn

from nnfabrik.examples.mnist.trainer import MNISTTrainer


class ChkptTrainer(MNISTTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict,
        seed: int,
        uid: Tuple,
        cb: Callable,
        epochs: int = 5,
        chkpt_options: Optional[Dict] = None,
    ) -> None:
        super(ChkptTrainer, self).__init__(model, dataloaders, seed, epochs)
        self.call_back = cb
        self.uid = uid
        self.accs = []
        self.chkpt_options = chkpt_options if chkpt_options is not None else {}

    def save(self, epoch: int, score: float) -> None:
        state = {
            "score": score,
            "maximize_score": True,
            "tracker": self.accs,
            "optimizer": self.optimizer,
            **self.chkpt_options,
        }
        self.call_back(
            uid=self.uid, epoch=epoch, model=self.model, state=state,
        )  # save model

    def restore(self) -> int:
        loaded_state = {
            "state": {"maximize_score": True, "tracker": self.accs},
            "optimizer": self.optimizer.state_dict(),
        }
        self.call_back(
            uid=self.uid, epoch=-1, model=self.model, state=loaded_state
        )  # load the last epoch if existing
        epoch = loaded_state.get("epoch", -1) + 1
        return epoch

    def train(self) -> Tuple[float, Tuple[List[float],int], Dict]:
        if hasattr(tqdm, "_instances"):
            tqdm._instances.clear()  # To have tqdm output without line-breaks between steps
        torch.manual_seed(self.seed)
        start_epoch = self.restore()
        for epoch in range(start_epoch, self.epochs):
            print(f"Epoch {epoch}")
            predicted_correct = 0
            total = 0
            for x, y in tqdm(self.trainloader):
                p, t = self.main_loop(x, y)
                predicted_correct += p
                total += t
            self.accs.append(100.0 * predicted_correct / total)
            self.save(epoch, self.accs[-1])

        return self.accs[-1], (self.accs, self.epochs), self.model.state_dict()


def chkpt_trainer_fn(
    model: torch.nn.Module,
    dataloaders: Dict,
    seed: int,
    uid: Tuple,
    cb: Callable,
    **config,
) -> Tuple[float, Any, Dict]:
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
    trainer = ChkptTrainer(
        model, dataloaders, seed, uid=uid, cb=cb, epochs=config.get("epochs", 2)
    )
    out = trainer.train()

    return out
