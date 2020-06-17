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


def find_prefix(s_dict: dict, p_agree: float = 0.66) -> (list, int):
    """
    Finds common prefix among state_dict keys
    :param s_dict: state_dict
    :param p_agree: percentage of keys that should agree for a valid prefix
    :return: (prefix, end index of prefix)
    """
    prefix_end = 0
    first_key = list(s_dict.keys())[0].split(".")
    for i in range(len(first_key)):
        is_prefix = True
        breaking_counter = 0
        # compare current prefix of first_key with all other keys
        for key in s_dict.keys():
            if key.split(".")[:i] != first_key[:i]:
                breaking_counter += 1
                # 2/3 should agree on prefix (i.e. prefix is valid even if 1/3 disagrees)
                p_ignore = 1.0 - p_agree
                if breaking_counter > p_ignore * len(s_dict):
                    is_prefix = False
                    break
        if is_prefix:
            prefix_end = i
    return first_key[:prefix_end], prefix_end


def load_state_dict(
    model,
    state_dict: dict,
    ignore_missing: bool = False,
    ignore_unused: bool = False,
    match_names: bool = False,
    ignore_dim_mismatch: bool = False,
):
    """
    Loads given state_dict into model, but allows for some more flexible loading.

    :param model: nn.Module object
    :param state_dict: dictionary containing a whole state of the module (result of `some_model.state_dict()`)
    :param ignore_missing: if True ignores entries present in model but not in `state_dict`
    :param match_names: if True tries to match names in `state_dict` and `model.state_dict()`
                        by finding and removing a common prefix from the keys in each dict
    :param ignore_dim_mismatch: if True ignores parameters in `state_dict` that don't fit the shape in `model`
    """
    model_dict = model.state_dict()
    # 0. Try to match names by adding or removing prefix:
    if match_names:
        # find prefix keys of each state dict:
        s_pref, s_idx = find_prefix(state_dict)
        m_pref, m_idx = find_prefix(model_dict)
        # switch prefixes:
        stripped_state_dict = {}
        for k, v in state_dict.items():
            if k.split(".")[:s_idx] == s_pref:
                stripped_key = k.split(".")[s_idx:]
            else:
                stripped_key = k.split(".")
            new_key = m_pref + stripped_key
            stripped_state_dict[".".join(new_key)] = v
        state_dict = stripped_state_dict

    # 1. filter out missing keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    unused = set(state_dict.keys()) - set(filtered_state_dict.keys())
    if unused and ignore_unused:
        print("Ignored unnecessary keys in pretrained dict:\n" + "\n".join(unused))
    elif ignore_unused:
        raise RuntimeError(
            "Error in loading state_dict: Unused keys:\n" + "\n".join(unused)
        )
    missing = set(model_dict.keys()) - set(filtered_state_dict.keys())
    if missing and ignore_missing:
        print("Ignored Missing keys:\n" + "\n".join(missing))
    elif missing:
        raise RuntimeError(
            "Error in loading state_dict: Missing keys:\n" + "\n".join(missing)
        )

    # 2. overwrite entries in the existing state dict
    updated_model_dict = {}
    for k, v in filtered_state_dict.items():
        if v.shape != model_dict[k].shape:
            if ignore_dim_mismatch:
                print("Ignored shape-mismatched parameter:", k)
                continue
            else:
                raise RuntimeError(
                    "Error in loading state_dict: Shape-mismatch for key {}".format(k)
                )
        updated_model_dict[k] = v

    # 3. load the new state dict
    model.load_state_dict(updated_model_dict, strict=(not ignore_missing))

