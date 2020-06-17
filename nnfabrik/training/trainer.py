from typing import Tuple, Callable, Dict

import torch


class Trainer:
    def __call__(
        self,
        model: torch.nn.Module,
        data_loaders: Dict,
        seed: int,
        uid: Tuple,
        cb: Callable,
        **config
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
        raise NotImplementedError

        return score, output, model.state_dict()
