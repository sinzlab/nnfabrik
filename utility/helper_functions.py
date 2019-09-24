# helper functions used by the nnfabrik framework

import torch

def get_in_out_dimensions(data_loader):
    """
        gets the input and output dimensions from the dataloader.
    :param
        dataloader: is expected to be a dict in the form of
                        {
                        'train_loader': torch.utils.data.DataLoader,
                         'val_loader': torch.utils.data.DataLoader,
                         'test_loader: torch.utils.data.DataLoader,
                         }
            each loader should have as first argument the input in the form of
                [batch_size, channels, px_x, px_y, ...]
            each loader should have as second argument the out in some form
                [batch_size, output_units, ...]
    :return:
        input_dim: input dimensions, expected to be a tuple in the form of input.shape.
                    for example: (batch_size, channels, px_x, px_y, ...)
        output_dim: out dimensions, expected to be a tuple in the form of output.shape.
                    for example: (batch_size, output_units, ...)
    """
    train_loader = data_loader["train_loader"]
    input_batch, output_batch, _ = next(iter(train_loader))
    return input_batch.shape, output_batch.shape

def get_core_output_shape(core, input_shape):
    """
    Gets the output dimensions of the convolutional core
        by passing an input image through all convolutional layers

    :param core: convolutional core of the DNN, which final dimensions
        need to be passed on to the readout layer
    :param input_shape: the dimensions of the input

    :return: output dimensions of the core
    """
    train_state = core.training
    core.eval()
    input_tensor = torch.ones(input_shape)
    tensor_out = core(input_tensor).shape[1:]
    core.train(train_state)
    return tensor_out
