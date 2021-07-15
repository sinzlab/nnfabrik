# importing the tables here is a trick to get IntelliSense to work
from nnfabrik.main import Fabrikant, Trainer, Dataset, Model, Seed, my_nnfabrik
from nnfabrik.templates.transfer.recipes import (
    TrainedModelTransferRecipe,
    DatasetTransferRecipe,
    ModelTransferRecipe,
    TrainerTransferRecipe,
    TrainerDatasetTransferRecipe,
)


# define nnfabrik tables here
my_nnfabrik(
    "nnfabrik_example",
    context=locals(),
    additional_tables=(
        TrainedModelTransferRecipe,
        DatasetTransferRecipe,
        ModelTransferRecipe,
        TrainerTransferRecipe,
        TrainerDatasetTransferRecipe,
    ),
)
