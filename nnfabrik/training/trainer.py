from typing import Tuple, Callable, Dict

import torch


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            data_loaders: Dict,
            seed: int,
            **config
    ):
        """"
        Args:
            model (torch.nn.Module): initialized model to train
            data_loaders (dict): containing "train", "validation" and "test" data loaders
            seed (int): random seed
        """
        self.model = model
        self.data_loaders = data_loaders
        self.seed = seed
        self.config = config

    def __call__(
        self,
        uid: Tuple,
        cb: Callable,
    ) -> Tuple[float, Dict, Dict]:
        """"
        Args:
            uid (tuple): keys that uniquely identify this trainer call
            cb : callback function to ping the database and potentially save the checkpoint
        Returns:
            score: performance score of the model
            output: user specified validation object based on the 'stop function'
            model_state: the full state_dict() of the trained model
        """
        raise NotImplementedError

        return score, output, model.state_dict()
