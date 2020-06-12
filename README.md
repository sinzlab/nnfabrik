# nnfabrik: generalized neural network model fitting pipeline
nnfabrik is a model fitting pipeline, mainly developed for neural networks, where training results (i.e. scores, and trained models) as well as any data related to models, trainers, and datasets used for training are stored in datajoint tables.

## Why use it?

Training neural network models commonly involves the following steps:
- load dataset
- initialize a model
- train the model using the dataset

While that would fulfill the training procedure, a huge portion of time spent on finding the best model for your application is dedicated to hyper-parameter selection/optimization. Importantly, each of the abovementioned steps may require their own specifications which effects the resulting model. For instance, whether to standardize the input, whether to use 2 layers or 20 layers, or wether use Adam or SGD as the optimizer. This is where nnfabrik becomes very handy by keeping track of models trained for every unique combination of parameters and hyperparameters.

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



## :book: Documentation

This is a work in progress. We will update this readme with the corresponding links to the documentation once it is ready.

## :bug: Report bugs

