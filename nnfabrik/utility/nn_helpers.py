# helper functions concerning the ANN architecture

import torch
from mlutils.training import eval_state
import numpy as np

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
    input_batch, output_batch, _ = next(iter(data_loader))
    return input_batch.shape, output_batch.shape

def get_module_output(model, input_shape):
    """
    Gets the output dimensions of the convolutional core
        by passing an input image through all convolutional layers

    :param core: convolutional core of the DNN, which final dimensions
        need to be passed on to the readout layer
    :param input_shape: the dimensions of the input

    :return: output dimensions of the core
    """
    with eval_state(model):
        input_tensor = torch.zeros(input_shape)
        tensor_out = model(input_tensor).shape
    return tensor_out

def set_random_seed(seed):
    """
    Sets all random seeds
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)



