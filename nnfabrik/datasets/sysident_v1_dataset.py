import torch
import torch.utils.data as utils
import numpy as np
import pickle
#from retina.retina import warp_image
from collections import namedtuple, Iterable
import os
from mlutils.data.samplers import RepeatsBatchSampler


class ImageCache:
    """
    A simple cache which loads images into memory given a path to the directory where the images are stored.
    Images need to be present as 2D .npy arrays
    """

    def __init__(self, path=None, subsample=1, crop=0, img_mean=None, img_std=None, filename_precision=6):
        """

        path: str - pointing to the directory, where the individual .npy files are present
        subsample: int - amount of downsampling
        crop:  the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_down)]
        img_mean: - mean luminance across all images
        img_std: - std of the luminance across all images
        leading_zeros: - amount leading zeros of the files in the specified folder
        """
        self.cache = {}
        self.path = path
        self.subsample = subsample
        self.crop = crop
        self.img_mean = img_mean
        self.img_std = img_std
        self.leading_zeros = filename_precision

    def __len__(self):
        return len([file for file in os.listdir(self.path) if file.endswith('.npy')])

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, item):
        return [self[i] for i in item] if isinstance(item, Iterable) else self.update(item)

    def update(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            filename = os.path.join(self.path, str(key).zfill(self.leading_zeros) + '.npy')
            image = np.load(filename)
            transformed_image = self.transform_image(image)
            self.cache[key] = transformed_image
            return transformed_image

    def transform_image(self, image):
        """
        applies transformations to the image: downsampling and cropping, z-scoring, and dimension expansion.
        """
        h, w = image.shape
        image = image[self.crop[0][0]:h - self.crop[0][1]:self.subsample, self.crop[1][0]:w - self.crop[1][1]:self.subsample]
        image = (image - self.img_mean) / self.img_std
        image = image[None,]
        return torch.tensor(image).to(torch.float)

    @property
    def cache_size(self):
        return len(self.cache)


class CachedTensorDataset(utils.Dataset):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, names=('inputs', 'targets'), image_cache=None):
        if not all(tensors[0].size(0) == tensor.size(0) for tensor in tensors):
            raise ValueError('The tensors of the dataset have unequal lenghts. The first dim of all tensors has to match exactly.')
        if not len(tensors) == len(names):
            raise ValueError('Number of tensors and names provided have to match.  If there are more than two tensors,'
                             'names have to be passed to the TensorDataset')
        self.tensors = tensors
        self.input_position = names.index("inputs")
        self.DataPoint = namedtuple('DataPoint', names)
        self.image_cache = image_cache

    def __getitem__(self, index):
        """
        retrieves the inputs (= tensors[0]) from the image cache. If the image ID is not present in the cache,
            the cache is updated to load the corresponding image into memory.
        """
        if type(index) == int:
            key = self.tensors[0][index].item()
        else:
            key = self.tensors[0][index].numpy().astype(np.int32)

        tensors_expanded = [tensor[index] if pos != self.input_position else torch.stack(list(self.image_cache[key]))
                            for pos, tensor in enumerate(self.tensors)]

        return self.DataPoint(*tensors_expanded)

    def __len__(self):
        return self.tensors[0].size(0)


def get_cached_loader(image_ids, responses, batch_size, shuffle=True, image_cache=None, repeat_condition=None):
    """

    Args:
        image_ids: an array of image IDs
        responses: Numpy Array, Dimensions: N_images x Neurons
        batch_size: int - batch size for the dataloader
        shuffle: Boolean, shuffles image in the dataloader if True
        image_cache: a cache object which stores the images

    Returns: a PyTorch DataLoader object
    """

    image_ids = torch.tensor(image_ids.astype(np.int32))
    responses = torch.tensor(responses).to(torch.float)
    dataset = CachedTensorDataset(image_ids, responses, image_cache=image_cache)
    sampler = RepeatsBatchSampler(torch.tensor(repeat_condition.astype(np.int32))) if repeat_condition is not None else None

    return utils.DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            sampler=sampler)

def monkey_static_loader(dataset,
                         neuronal_data_files,
                         imagepath,
                         cached_images_path,
                         batch_size=64,
                         seed=None,
                         train_frac=0.8,
                         subsample=1,
                         crop=96,
                         time_bins_sum=12,
                         avg=False):
    """
    Function that returns cached dataloaders for the Center Surround Visual Field Experiments.
        Data recorded by George and Kelli at BCM, Houston.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    required inputs is a list of datafiles specified as a full path, together with a full path
        to a file that contains all the actually images

    Args:
        dataset: a string, identifying the Dataset:
            'V1', 'CSRF_V1', 'CSRF_V4'
        datafiles: a list paths that point to pickle files
        imagepath: a path that points to the image files
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_bottom)]
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if imagepath:
        with open(imagepath, "rb") as pkl:
            images = pickle.load(pkl)

    images = images[:, :, :, None]
    _, h, w = images.shape[:3]

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    images_cropped = images[:, crop[0][0]:h - crop[0][1]:subsample, crop[1][0]:w - crop[1][1]:subsample, :]
    img_mean = np.mean(images_cropped)
    img_std = np.std(images_cropped)

    # set up parameters for the different dataset types
    if dataset == 'V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=images.shape[0] * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)

    # Initialize the Image Cache class
    cache = ImageCache(path=cached_images_path, subsample=subsample, crop=crop, img_mean=img_mean, img_std=img_std)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        if dataset != 'V1':
            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size, image_cache=cache)
        val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size, image_cache=cache)
        test_loader = get_cached_loader(testing_image_ids, responses_test, batch_size=1, shuffle=False,
                                                image_cache=cache, repeat_condition=testing_image_ids)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    return dataloaders



class NamedTensorDataset(utils.Dataset):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, *tensors, names=('inputs','targets')):
        if not all(tensors[0].size(0) == tensor.size(0) for tensor in tensors):
            raise ValueError(
                'The tensors of the dataset have unequal lenghts. The first dim of all tensors has to match exactly.')
        if not len(tensors) == len(names):
            raise ValueError('Number of tensors and names provided have to match.  If there are more than two tensors,'
                             'names have to be passed to the TensorDataset')
        self.tensors = tensors
        self.DataPoint = namedtuple('DataPoint', names)

    def __getitem__(self, index):
        return self.DataPoint(*[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.tensors[0].size(0)


def csrf_v1(datafiles, imagepath, batch_size, seed,
            train_frac=0.8, subsample=1, crop=65,
            time_bins_sum=tuple(range(12)), avg=False,
            crop_h=None, crop_w=None):
    """
    Function that returns the dataloaders for the Center Surround Visual Field V1 Experiment.
        Data recorded by George and Kelli at BCM, Houston.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    required inputs is a list of datafiles specified as a full path, together with a full path
        to a file that contains all the actually images

    Args:
        datafiles: a list paths that point to pickle files
        imagepath: a path that points to the image files
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    #
    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if imagepath:
        with open(imagepath, "rb") as pkl:
            images = pickle.load(pkl)

    images = images[:, :, :, None]
    _, h, w = images.shape[:3]

    if crop_h is None and crop_w is None:
        images_cropped = images[:, crop:h - crop:subsample, crop:w - crop:subsample, :]
    else:
        images_cropped = images[:, crop_h[0]:h - crop_h[1]:subsample, crop_w[0]:w - crop_w[1]:subsample, :]

    img_mean = np.mean(images_cropped)
    img_std = np.std(images_cropped)

    all_train_ids, all_validation_ids = get_validation_split(n_images=images.shape[0],
                                                                train_frac=train_frac,
                                                                seed=seed)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(datafiles):

        #Extract Session ID from the pickle filename

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        # additional information related to session and animal. Has to find its way into datajoint
        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        repetitions_test = raw_data["testing_repetitions"]

        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"]
        testing_image_ids = raw_data["testing_image_ids"]

        responses_test = responses_test.transpose((2, 0, 1))
        responses_train = responses_train.transpose((2, 0, 1))

        # images_train = images[training_image_ids, crop:h - crop:subsample, crop:w - crop:subsample]
        # images_test = images[testing_image_ids, crop:h - crop:subsample, crop:w - crop:subsample]

        images_train = images_cropped[training_image_ids]
        images_test = images_cropped[testing_image_ids]

        images_train = (images_train - img_mean) / img_std
        images_test = (images_test - img_mean) / img_std

        if time_bins_sum is not None:  # then average over given time bins
            responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
            responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        images_val = images_train[val_idx]
        images_train = images_train[train_idx]
        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        train_loader = get_loader_csrf_v1(images_train, responses_train, batch_size=batch_size)
        val_loader = get_loader_csrf_v1(images_val, responses_val, batch_size=batch_size)
        test_loader = get_loader_csrf_v1(images_test, responses_test, batch_size=batch_size, shuffle=False)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    return dataloaders


def sysident_v1(datafiles, imagepath, batch_size, seed,
                train_frac=0.8, subsample=2, crop=30):
    """
    Function that returns the dataloaders for the SysIdent V1 Experiment.
        Data recorded by Stantiago Cadena at BCM, Houston.

    creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'val'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    Args:
        datafiles: a list paths that point to pickle files
        imagepath: a path that points to the image files
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80

    Returns: nested dictionary of dataloaders
    """

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if imagepath:
        with open(imagepath, "rb") as pkl:
            images = pickle.load(pkl)

    images = images[:, :, :, None]
    _, h, w = images.shape[:3]
    images_cropped = images[:, crop:h - crop:subsample, crop:w - crop:subsample, :]
    img_mean = np.mean(images_cropped)
    img_std = np.std(images_cropped)

    # hard Coded Parameter used in the amadeus.pickle file
    n_train_images = int(images.shape[0]*0.8)

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_train_images,
                                                                train_frac=train_frac,
                                                                seed=seed)
    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(datafiles):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        # additional information related to session and animal. Has to find its way into datajoint
        subject_ids = raw_data["subject_id"]
        session_id = raw_data["session_id"]

        data_key = str(session_id)

        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)

        # for proper indexing, IDs have to start from zero
        training_image_ids = raw_data["training_image_ids"] - 1
        testing_image_ids = raw_data["testing_image_ids"] - 1

        images_train = images[training_image_ids, crop:h - crop:subsample, crop:w - crop:subsample]
        images_test = images[testing_image_ids, crop:h - crop:subsample, crop:w - crop:subsample]
        images_train = (images_train - img_mean) / img_std
        images_test = (images_test - img_mean) / img_std

        train_idx = np.isin(training_image_ids, all_train_ids)
        val_idx = np.isin(training_image_ids, all_validation_ids)

        images_val = images_train[val_idx]
        images_train = images_train[train_idx]
        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        train_loader = get_loader_csrf_v1(images_train, responses_train, batch_size=batch_size)
        val_loader = get_loader_csrf_v1(images_val, responses_val, batch_size=batch_size)
        test_loader = get_loader_csrf_v1(images_test, responses_test, batch_size=batch_size, shuffle=False)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    return dataloaders


def get_validation_split(n_images, train_frac, seed):
    """
    Splits the total number of images into train and test set.
    This ensures that in every session, the same train and validation images are being used.

    Args:
        n_images: Total number of images. These will be plit into train and validation set
        train_frac: fraction of images used for the training set
        seed: random seed

    Returns: Two arrays, containing image IDs of the whole imageset, split into train and validation

    """
    if seed: np.random.seed(seed)
    train_idx, val_idx = np.split(np.random.permutation(int(n_images)), [int(n_images*train_frac)])
    assert not np.any(np.isin(train_idx, val_idx)), "train_set and val_set are overlapping sets"

    return train_idx, val_idx


def get_loader_csrf_v1(images, responses, batch_size, shuffle=True):
    """
    Args:
        images: Numpy Array of Images, Dimensions: N x C x W x H
        responses: Numpy Array, Dimensions: N_images x Neurons
        batch_size: int - batch size for the dataloader
        shuffle: Boolean, shuffles image in the dataloader if True

    Returns: a PyTorch DataLoader object

    """

    # Expected Dimension of the Image Tensor is Images x Channels x size_x x size_y
    # In some CSRF files, Channels are at Dim4, the image tensor is thus reshaped accordingly
    if images.shape[1] > 3:
        images = images.transpose((0, 3, 1, 2))

    images = torch.tensor(images).to(torch.float)
    responses = torch.tensor(responses).to(torch.float)

    dataset = NamedTensorDataset(images, responses)
    data_loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader