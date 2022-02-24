# importing the tables here is a trick to get IntelliSense to work
from nnfabrik.main import Dataset, Fabrikant, Model, Seed, Trainer, my_nnfabrik
from nnfabrik.templates.transfer.recipes import (
    DatasetTransferRecipe,
    ModelTransferRecipe,
    TrainedModelTransferRecipe,
    TrainerDatasetTransferRecipe,
    TrainerTransferRecipe,
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
