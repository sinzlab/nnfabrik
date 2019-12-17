from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from mlutils.data.datasets import StaticImageSet
from mlutils.data.transforms import Subsample, ToTensor


def get_dataloader(path, batch_size, img_seed=None, area='V1', layer='L2/3', tier=None, neuron_ids=None, get_key=False, device='cuda'):
    """
    returns a single data
    
    Args:
        path (list): list of path(s) for the dataset(s)
        batch_size (int): batch size.
        img_seed (int, optional): random seed for images. Defaults to None.
        area (str, optional): the visual area. Defaults to 'V1'.
        layer (str, optional): the layer from visual area. Defaults to 'L2/3'.
        tier (str, optional): tier is a placeholder to specify which set of images to pick for train, val, and test loader. Defaults to None.
        neuron_ids ([type], optional): select neurons by their ids. Defaults to None.
        get_key (bool, optional): whether to retun the data key, along with the dataloaders. Defaults to False.
        device (str, optional): device to place the data on. Defaults to 'cuda'.

    Returns:
        if get_key is False returns a dictionary of dataloaders for one dataset, where the keys are 'train', 'validation', and 'test'. 
        if get_key is True it also the data_key (as the first input) followed by the dalaoder dictionary.

    """

    dat = StaticImageSet(path, 'images', 'responses')
#     dat = StaticImageSet(path, 'inputs', 'targets')

    # specify condition(s) for samping neurons. If you want to sample specific neurons define conditions that would effect idx
    neuron_ids = neuron_ids if neuron_ids else dat.neurons.unit_ids
    conds = ((dat.neurons.area == area) & 
             (dat.neurons.layer == layer) &
             (np.isin(dat.neurons.unit_ids, neuron_ids)))
    
    idx = np.where(conds)[0]
    dat.transforms = [Subsample(idx), ToTensor(cuda=True if device=='cuda' else False)]
    
    # subsample images
    dataloaders = {}
    keys = [tier] if tier else ['train', 'validation', 'test']
    for tier in keys:
        
        # sample images
        subset_idx = np.where(dat.tiers == tier)[0]
        sampler = SubsetRandomSampler(subset_idx)

        if img_seed is not None:
            torch.manual_seed(img_seed)
            
        dataloaders[tier] = DataLoader(dat, sampler=sampler, batch_size=batch_size)
    
    data_key = path.split('static')[-1].split('.')[0].replace('preproc', '')
    
    return (data_key, dataloaders) if get_key else dataloaders


def get_dataloaders(paths, batch_size, img_seed=None, area='V1', layer='L2/3', tier=None, neuron_ids=None, device='cuda'):
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
        device (str, optional): device to place the data on. Defaults to 'cuda'.
    
    Returns:
        dict: dictionary of dictionaries where the first level keys are 'train', 'validation', and 'test', and second level keys are data_keys.
    """ 
    
    dls = OrderedDict({})
    keys = [tier] if tier else ['train', 'validation', 'test']
    for key in keys:
        dls[key] = OrderedDict({})

    for ind, path in enumerate(paths):
        data_key, loaders = get_dataloader(path, batch_size, img_seed=img_seed, 
                                           area=area, layer=layer, device=device,
                                           tier=tier, get_key=True, neuron_ids=neuron_ids[ind] if neuron_ids else None)
        for k in dls.keys():
            dls[k][data_key] = loaders[k]
            
    return dls