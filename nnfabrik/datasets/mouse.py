from collections import OrderedDict
from itertools import zip_longest
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from mlutils.data.datasets import StaticImageSet
from mlutils.data.transforms import Subsample, ToTensor
from mlutils.data.samplers import SubsetSequentialSampler

from ..utility.nn_helpers import set_random_seed


def mouse_static_loader(path, batch_size, img_seed=None, area='V1', layer='L2/3', tier=None, neuron_ids=None, get_key=False, cuda=True):
    """
    returns a single data
    
    Args:
        path (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        img_seed (int, optional): random seed for images. Defaults to None.
        area (str, optional): the visual area. Defaults to 'V1'.
        layer (str, optional): the layer from visual area. Defaults to 'L2/3'.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids (list, optional): select neurons by their ids. neuron_ids and path should be of same length. Defaults to None.
        get_key (bool, optional): whether to retun the data key, along with the dataloaders. Defaults to False.
        cuda (bool, optional): whether to place the data on gpu or not. Defaults to True.

    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'. 
        if get_key is True it also the data_key (as the first output) followed by the dalaoder dictionary.

    """

    dat = StaticImageSet(path, 'images', 'responses')
#     dat = StaticImageSet(path, 'inputs', 'targets')

    # specify condition(s) for sampling neurons. If you want to sample specific neurons define conditions that would effect idx
    neuron_ids = neuron_ids if neuron_ids else dat.neurons.unit_ids
    conds = ((dat.neurons.area == area) & 
             (dat.neurons.layer == layer) &
             (np.isin(dat.neurons.unit_ids, neuron_ids)))
    
    idx = np.where(conds)[0]
    dat.transforms = [Subsample(idx), ToTensor(cuda)]
    
    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ['train', 'validation', 'test']
    for tier in keys:
        
        if img_seed is not None:
            set_random_seed(img_seed)
            # torch.manual_seed(img_seed)

        # sample images
        subset_idx = np.where(dat.tiers == tier)[0]
        sampler = SubsetRandomSampler(subset_idx) if tier == 'train' else SubsetSequentialSampler(subset_idx)
            
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)
    
    # create the data_key for a specific data path 
    data_key = path.split('static')[-1].split('.')[0].replace('preproc', '')
    
    return (data_key, dataloaders) if get_key else dataloaders


def mouse_static_loaders(paths, batch_size, img_seed=None, area='V1', layer='L2/3', tier=None, neuron_ids=None, cuda=True):
    """
    Returns a dictionary of dataloaders (i.e., trainloaders, valloaders, and testloaders) for >= 1 dataset(s).
    
    Args:
        paths (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        img_seed (int, optional): random seed for images. Defaults to None.
        area (str, optional): the visual area. Defaults to 'V1'.
        layer (str, optional): the layer from visual area. Defaults to 'L2/3'.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids ([type], optional): select neurons by their ids. Defaults to None.
        cuda (bool, optional): whether to place the data on gpu or not. Defaults to True.
    
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """ 
    
    neuron_ids = neuron_ids if neuron_ids is None else []

    dls = OrderedDict({})
    keys = [tier] if tier else ['train', 'validation', 'test']
    for key in keys:
        dls[key] = OrderedDict({})

    for path, neuron_id in zip_longest(paths, neuron_ids, fillvalue=None):
        data_key, loaders = mouse_static_loader(path, batch_size, img_seed=img_seed, 
                                                area=area, layer=layer, cuda=cuda,
                                                tier=tier, get_key=True, neuron_ids=neuron_id)
        for k in dls:
            dls[k][data_key] = loaders[k]
            
    return dls
