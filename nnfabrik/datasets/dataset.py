from typing import Dict


class Dataset:
    def __call__(self, seed: int, **config) -> Dict:
        """
        Returns data loaders for the given config

        Args:
            seed (int): random seed that will make shuffling and other random operations deterministic

        Returns:
            data_loaders (dict): containing "train", "validation" and "test" data loaders
        """
        raise NotImplementedError
        # subsample images
        data_loaders = {"train": None, "validation": None, "test": None}
        return data_loaders
