# nnfabrik: a generalized model fitting pipeline
nnfabrik is a model fitting pipeline, mainly developed for neural networks, where training results (i.e. scores, and trained models) as well as any data related to models, trainers, and datasets used for training are stored in datajoint tables.

## Why use it?

Training neural network models commonly involves the following steps:
- load dataset
- initialize a model
- train the model using the dataset

While that would fulfill the training procedure, a huge portion of time spent on finding the best model for your application is dedicated to hyper-parameter selection/optimization. Importantly, each of the above-mentioned steps may require their own specifications which effect the resulting model. For instance, whether to standardize the data, whether to use 2 layers or 20 layers, or wether to use Adam or SGD as the optimizer. This is where nnfabrik becomes very handy by keeping track of models trained for every unique combination of hyperparameters.

## :gear: Installation

You can use one of the following ways to install nnfabrik:

#### 1. Using `pip`
```
pip install nnfabrik
```

#### 2. Via GitHub:
```
pip install git+https://github.com/sinzlab/nnfabrik.git
```

## :fire: Usage
As mentioned above, nnfabrik helps with keeping track of different combinations of hyperparameters used for model creation and training. In order to achieve this nnfabrik would need the necessary components to train a model. These components include:
* **model function**: a function that returns the data used for training
* **model function**: a function that return the model to be trained
* **trainer function**: a function that given dataset and a model trains the model and returns the resulting model

However, to ensure a generalized solution nnfabrik makes some minor assumptions about the inputs and the outputs of the above-mentioned functions. Here are the assumptions:

**Dataset function**
* **input**: must have an argument called `seed`. The rest of the arguments are up to the user and we will refer to them as `dataset_config`.
* **output**: a dictionary of dictionaries in the form of 
    ``` python
    {'train': {
        'data_key1': torch.utils.data.DataLoader, 
        'data_key2': torch.utils.data.DataLoader
        },
    'validation': {
        'data_key1': torch.utils.data.DataLoader, 
        'data_key2': torch.utils.data.DataLoader
        },
    'test': {
        'data_key1': torch.utils.data.DataLoader, 
        'data_key2': torch.utils.data.DataLoader
        }
    }
    ```

**Model function**
* **input**: must have two arguments: `dataloaders` and `seed`. The rest of the arguments are up to the user and we will refer to them as `model_config`.
* **output**: a model object of class `torch.nn.Module`

**Trainer function**
* **input**: must have three arguments: `model`, `dataloaders` and `seed`. The rest of the arguments are up to the user and we will refer to them as `trainer_config`.
* **output**: the trainer returns three objects including: 
  * a single value representing some sort of score (e.g. validation correlation) attributed to the trained model
  * a collection (list, tuple, or dictionary) of any other quantity 
  * the `state_dict` of the trained model.

You can take a look at some examples in [toy_dataset](./nnfabrik/datasets/toy_datasets.py), [toy_model](./nnfabrik/models/toy_models.py), and [toy_trainer](./nnfabrik/training/toy_trainers.py).

Once you have these three functions, all is left to do is to define the corresponding tables. Tables are structured similar to the the functions. That is, we have a `Dataset`, `Model`, and `Trainer` table. Each entry of the table corresponds to an specific instance of the corresponding function. For example one entry of the `Dataset` table refers to a specific dataset function and a specific `dataset_config`.

In addition to the tables which store unique combinations of functions and configuration objects, there is another table, called `TrainedModel` to store the trained models. Each entry of the `TrainedModel` table refers to the resulting model from a unique combination of dataset, model, and trainer.

To get familiar with the tables (e.g. how to define them and add entries) take a look at the [example notebook](./notebooks/nnfabrik_example.ipynb).

We have pretty much covered the most important information about nnfabrik, and it is time to use it. Some basics about the Datajoint Python package (which is the backbone of nnfabrik) might come handy (especially about dealing with tables) and you can learn more about Datajoint [here](https://datajoint.io/).

## :book: Documentation

This is a work in progress. We will update this readme with the corresponding links to the documentation once it is ready.

## :bug: Report bugs (or request features)

In case you find a bug or would like to see some new features added to nnfabrik, please create an issue or contact any of the contributors.