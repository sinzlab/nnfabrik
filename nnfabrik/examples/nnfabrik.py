# importing the tables here is a trick to get IntelliSense to work
from nnfabrik.main import Fabrikant, Trainer, Dataset, Model, Seed, my_nnfabrik

# define nnfabrik tables here
my_nnfabrik(
    "nnfabrik_example", context=locals(),
)
