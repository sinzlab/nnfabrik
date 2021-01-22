from typing import Dict

from nnfabrik.main import *


class Recipe(dj.Lookup):
    definition = """
    transfer_step: int
    data_transfer: tinyint
    -> Model
    -> Model.proj(prev_model_fn='model_fn', prev_model_hash='model_hash')
    """

    @property
    def post_restr(self) -> str:
        """
        Specifies which restrictions should be applied after transfer, e.g. which part should be kept the same
        """
        return ""

    def add_entry(
        self,
        transfer_from: Dict = None,
        transfer_to: Dict = None,
        transfer_step: int = 0,
        data_transfer: bool = False,
    ):
        """
        Insert a recipe into the table.
        Args:
            transfer_from: entry to be transferred from
            transfer_to: entry to be transferred to
            transfer_step: integer that defines which transfer step this is applied in

        Returns:

        """
        entry = dict(
            **{f"prev_{k}": v for k, v in transfer_from.items()},
            **transfer_to,
        )
        entry["transfer_step"] = transfer_step
        entry["data_transfer"] = 1 if data_transfer else 0
        self.insert1(entry, skip_duplicates=True)


@schema
class DatasetTransferRecipe(Recipe):
    definition = """
    transfer_step: int
    data_transfer: tinyint
    -> Dataset
    -> Dataset.proj(prev_dataset_fn='dataset_fn', prev_dataset_hash='dataset_hash')
    """

    @property
    def post_restr(self):
        """
        This restriction clause is used to make sure that aside from switching datasets,
        the utilized trainer is to remain the same.
        """
        return dj.AndList(
            [
                "model_fn = prev_model_fn",
                "model_hash = prev_model_hash",
                "trainer_fn = prev_trainer_fn",
                "trainer_hash = prev_trainer_hash",
            ]
        )


@schema
class ModelTransferRecipe(Recipe):
    definition = """
    transfer_step: int
    data_transfer: tinyint
    -> Model
    -> Model.proj(prev_model_fn='model_fn', prev_model_hash='model_hash')
    """

    @property
    def post_restr(self):
        """
        This restriction clause is used to make sure that aside from switching models,
        the utilized trainer is to remain the same.
        """
        return dj.AndList(
            [
                "trainer_fn = prev_trainer_fn",
                "trainer_hash = prev_trainer_hash",
                "dataset_fn = prev_dataset_fn",
                "dataset_hash = prev_dataset_hash",
            ]
        )


@schema
class TrainerTransferRecipe(Recipe):
    definition = """
    transfer_step: int
    data_transfer: tinyint
    -> Trainer
    -> Trainer.proj(prev_trainer_fn='trainer_fn', prev_trainer_hash='trainer_hash')
    """

    @property
    def post_restr(self):
        """
        This restriction clause is used to make sure that aside from switching trainers,
        the utilized model is to remain the same.
        """
        return dj.AndList(
            [
                "model_fn = prev_model_fn",
                "model_hash = prev_model_hash",
                "dataset_fn = prev_dataset_fn",
                "dataset_hash = prev_dataset_hash",
            ]
        )


@schema
class TrainerDatasetTransferRecipe(Recipe):
    definition = """
     transfer_step: int
     data_transfer: tinyint
     -> Trainer
     -> Trainer.proj(prev_trainer_fn='trainer_fn', prev_trainer_hash='trainer_hash')
     -> Dataset
     -> Dataset.proj(prev_dataset_fn='dataset_fn', prev_dataset_hash='dataset_hash')
     """

    @property
    def post_restr(self):
        """
        This restriction clause is used to make sure that aside from switching trainers,
        the utilized model is to remain the same.
        """
        return dj.AndList(
            [
                "model_fn = prev_model_fn",
                "model_hash = prev_model_hash",
            ]
        )


@schema
class TrainedModelTransferRecipe(Recipe):
    definition = """
     transfer_step: int
     data_transfer: tinyint
     -> Trainer
     -> Trainer.proj(prev_trainer_fn='trainer_fn', prev_trainer_hash='trainer_hash')
     -> Dataset
     -> Dataset.proj(prev_dataset_fn='dataset_fn', prev_dataset_hash='dataset_hash')
     -> Model
     -> Model.proj(prev_model_fn='model_fn', prev_model_hash='model_hash')
     -> Seed
     -> Seed.proj(prev_seed='seed')
     prev_collapsed_history:    varchar(64)
     """

    @property
    def post_restr(self):
        """
        This restriction clause is used to make sure that aside from switching trainers,
        the utilized model is to remain the same.
        """
        return dj.AndList([])
