# helper functions concerning the ANN architecture

import torch
from mlutils.training import eval_state
import numpy as np
import random


def get_io_dims(data_loader):
    """
    gets the input and output dimensions from the dataloader.
    :Args
        dataloader: is expected to be a pytorch Dataloader object
            each loader should have as first argument the input in the form of
                [batch_size, channels, px_x, px_y, ...]
            each loader should have as second argument the output in the form of
                [batch_size, output_units, ...]
    :return:
        input_dim: input dimensions, expected to be a tuple in the form of input.shape.
                    for example: (batch_size, channels, px_x, px_y, ...)
        output_dim: out dimensions, expected to be a tuple in the form of output.shape.
                    for example: (batch_size, output_units, ...)
    """
    items = next(iter(data_loader))
    return {k: v.shape for k, v in items._asdict().items()}


def get_dims_for_loader_dict(dataloaders):
    """
    gets the input and outpout dimensions for all dictionary entries of the dataloader

    :param dataloaders: dictionary of dataloaders. Each entry corresponds to a session
    :return: a dictionary with the sessionkey and it's corresponding dimensions
    """
    return {k: get_io_dims(v) for k, v in dataloaders.items()}


def get_module_output(model, input_shape):
    """
    Gets the output dimensions of the convolutional core
        by passing an input image through all convolutional layers

    :param core: convolutional core of the DNN, which final dimensions
        need to be passed on to the readout layer
    :param input_shape: the dimensions of the input

    :return: output dimensions of the core
    """
    initial_device = 'cuda' if next(iter(model.parameters())).is_cuda else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:]).to(device)
            output = model.to(device)(input)
    model.to(initial_device)
    return output.shape


def set_random_seed(seed):
    """
    Sets all random seeds
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
