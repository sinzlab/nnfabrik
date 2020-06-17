from typing import Dict

import torch


class ModelBuilder:
    def __call__(self, data_loaders: Dict, seed: int, **config) -> torch.nn.Module:
        """
        Builds a model object for the given config

        Args:
            data_loaders (dict): a dictionary of data loaders
            seed (int): random seed (e.g. for model initialization)

        Returns:
            Instance of torch.nn.Module
        """
        raise NotImplementedError
        return model
