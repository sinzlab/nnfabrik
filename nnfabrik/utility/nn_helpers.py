# helper functions concerning the ANN architecture

import torch
from torch import nn
from torch.backends import cudnn

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
    initial_device = "cuda" if next(iter(model.parameters())).is_cuda else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with eval_state(model):
        with torch.no_grad():
            input = torch.zeros(1, *input_shape[1:]).to(device)
            output = model.to(device)(input)
    model.to(initial_device)
    return output.shape


def set_random_seed(seed: int, deterministic: bool = True):
    """
    Sets all random seeds

    :param seed: (int) seed to be set
    :param deterministic: (bool) activates cudnn.deterministic, which might slow down things
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        torch.cuda.manual_seed(seed)


def move_to_device(model, gpu=True, multi_gpu=True):
    """
    Moves given model to GPU(s) if they are available
    :param model: (torch.nn.Module) model to move
    :param gpu: (bool) if True attempt to move to GPU
    :param multi_gpu: (bool) if True attempt to use multi-GPU
    :return: torch.nn.Module, str
    """
    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    if multi_gpu and torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    return model, device


def set_state_dict(
    pretrained_dict,
    model,
    ignore_missing=True,
    match_names=True,
    ignore_dim_mismatch=True,
):
    if ignore_missing:
        model_dict = model.state_dict()
        # 0. Try to match names by adding or removing prefix:
        if match_names:
            first_key_pretrained = list(pretrained_dict.keys())[0].split(".")
            first_key_model = list(model_dict.keys())[0].split(".")
            # find prefix in each:
            for i in range(len(first_key_pretrained)):
                is_prefix = True
                breaking_counter = 0
                for j, key in enumerate(pretrained_dict.keys()):
                    if key.split(".")[:i] != first_key_pretrained[:i]:
                        breaking_counter += 1
                        if breaking_counter > 0.33 * len(pretrained_dict):  # 2/3 should agree on prefix
                            is_prefix = False
                            break
                if is_prefix:
                    pretrained_prefix_end = i
            for i in range(len(first_key_model)):
                is_prefix = True
                for key in model_dict.keys():
                    if key.split(".")[:i] != first_key_model[:i]:
                        is_prefix = False
                        break
                if is_prefix:
                    model_prefix = first_key_model[:i]
            # switch prefixes:
            state_dict_ = {}
            for k, v in pretrained_dict.items():
                if k.split(".")[:pretrained_prefix_end] == first_key_pretrained[:pretrained_prefix_end]:
                    stripped_key = k.split(".")[pretrained_prefix_end:]
                else:
                    stripped_key = k.split(".")
                new_key = model_prefix + stripped_key
                state_dict_[".".join(new_key)] = v
            pretrained_dict = state_dict_

        # 1. filter out missing keys
        filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        unused = set(pretrained_dict.keys()) - set(filtered_pretrained_dict.keys())
        if unused:
            print("Ignored unnecessary keys in pretrained dict:")
            for k in unused:
                print(k)
        missing = set(model_dict.keys()) - set(filtered_pretrained_dict.keys())
        if missing:
            print("Ignored missing keys:")
            for k in missing:
                print(k)

        # 2. overwrite entries in the existing state dict
        for k, v in model_dict.items():
            if v.shape != filtered_pretrained_dict[k].shape and ignore_dim_mismatch:
                print("Ignored shape-mismatched parameters:", k)
                continue
            model_dict[k] = filtered_pretrained_dict[k]

        # 3. load the new state dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(pretrained_dict)

