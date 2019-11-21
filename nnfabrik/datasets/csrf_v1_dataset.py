import torch
import torch.utils.data as utils
import numpy as np
import pickle
#from retina.retina import warp_image
from collections import namedtuple


def csrf_v1(datafiles, imagepath, batch_size, seed,
            train_frac=0.8, subsample=1, crop=65, time_bins_sum=tuple(range(12))):
    """
    creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'val'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    required inputs is a list of datafiles specified as a full path, together with a full path
        to a file that contains all the actually images

    :param datapath: a list of sessions
    :param batch_size:
    :param seed:
    :param imagepath:
    :param train_frac:
    :param subsample:
    :param crop:
    :param time_bins_sum:
    :return:
    """

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'val': {}, 'test': {}}

    if imagepath:
        with open(imagepath, "rb") as pkl:
            images = pickle.load(pkl)

    images = images[:,:,:,None]
    _, h, w = images.shape[:3]
    img_mean = np.mean(images)
    img_std = np.std(images)
    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(datafiles):

        #Extract Session ID from the pickle filename
        data_key = datapath[-20:-7]

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        # additional information related to session and animal. Has to find its way into datajoint
        subject_ids = raw_data["subject_id"]
        session_ids = raw_data["session_id"]
        repetitions_test = raw_data["testing_repetitions"]

        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"]
        testing_image_ids = raw_data["testing_image_ids"]

        responses_test = responses_test.transpose((2, 0, 1))
        responses_train = responses_train.transpose((2, 0, 1))

        images_train = images[training_image_ids, crop:h - crop:subsample, crop:w - crop:subsample]
        images_test = images[testing_image_ids, crop:h - crop:subsample, crop:w - crop:subsample]
        images_train = (images_train - img_mean) / img_std
        images_test = (images_test - img_mean) / img_std

        if time_bins_sum is not None:  # then average over given time bins
            responses_train = np.sum(responses_train[:, :, time_bins_sum], axis=-1)
            responses_test = np.sum(responses_test[:, :, time_bins_sum], axis=-1)

        train_idx, val_idx = get_validation_split(responses_train, train_frac=train_frac, seed=seed)
        images_val = images_train[val_idx]
        images_train = images_train[train_idx]
        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        train_loader = get_loader_csrf_v1(images_train, responses_train, batch_size=batch_size)
        val_loader = get_loader_csrf_v1(images_val, responses_val, batch_size=batch_size)
        test_loader = get_loader_csrf_v1(images_test, responses_test, batch_size=batch_size, shuffle=False)

        dataloaders["train"][data_key] = train_loader
        dataloaders["val"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    return dataloaders


def get_validation_split(responses_train, train_frac=0.8, seed=None):
    """
    gets indices to split the full training set into train and validation data

    :param responses_train:
    :param train_fac:
    :param seed:
    :return: indeces of the training_set and validation_set
    """

    if seed:
        np.random.seed(seed)

    n_images = responses_train.shape[0]
    n_train = int(np.round(n_images * train_frac))

    train_idx = np.random.choice(np.arange(n_images), n_train, replace=False)
    val_idx = np.arange(n_images)[np.logical_not(np.isin(np.arange(n_images), train_idx))]

    assert not np.any(np.isin(train_idx, val_idx)), "train_set and val_set are overlapping sets"
    assert sum((len(train_idx), len(val_idx))) == n_images, "not all training images were used for train/val split"
    return train_idx, val_idx


def get_loader_csrf_v1(images, responses, batch_size, shuffle=True, retina_warp=False):
    """
    :param images:
    :param responses:
    :param batch_size:
    :param shuffle:
    :param retina_warp:
    :return:
    """

    # Expected Dimension of the Image Tensor is Images x Channels x size_x x size_y
    # In some CSRF files, Channels are at Dim4, the image tensor is thus reshaped accordingly
    if images.shape[1] > 3:
        images = images.transpose((0, 3, 1, 2))

    if retina_warp:
        images = np.array(list(map(warp_image, images[:, 0])))[:, None]

    images = torch.tensor(images).to(torch.float)
    responses = torch.tensor(responses).to(torch.float)

    dataset = NamedTensorDataset(images, responses)
    data_loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


class NamedTensorDataset(utils.Dataset):
    """
    Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, names=('inputs','targets')):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        assert len(tensors) == len(names)
        self.tensors = tensors
        self.DataPoint = namedtuple('DataPoint', names)

    def __getitem__(self, index):
        return self.DataPoint(*[tensor[index] for tensor in self.tensors])

    def __len__(self):
        return self.tensors[0].size(0)



# legacy functions for dataloader creation

def csrf_v1_legacy(datapath, image_path, batch_size, seed, train_frac=0.8,
                   subsample=1, crop=65, time_bins_sum=tuple(range(12))):
    v1_data = CSRF_V1_Data(raw_data_path=datapath, image_path=image_path, seed=seed,
                           train_frac=train_frac, subsample=subsample, crop=crop,
                           time_bins_sum=time_bins_sum)

    images, responses, valid_responses = v1_data.train()
    train_loader = get_loader_csrf_V1_legacy(images, responses, 1 * valid_responses, batch_size=batch_size)

    images, responses, valid_responses = v1_data.val()
    val_loader = get_loader_csrf_V1_legacy(images, responses, 1 * valid_responses, batch_size=batch_size)

    images, responses, valid_responses = v1_data.test()
    test_loader = get_loader_csrf_V1_legacy(images, responses, 1 * valid_responses, batch_size=batch_size, shuffle=False)

    data_loader = dict(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    return data_loader


# begin of helper functions

def get_loader_csrf_V1_legacy(images, responses, valid_responses, batch_size=None, shuffle=True, retina_warp=False):
    # Expected Dimension of the Image Tensor is Images x Channels x size_x x size_y
    # In some CSRF files, Channels are at Dim4, the image tensor is thus reshaped accordingly
    if images.shape[1] > 3:
        images = images.transpose((0, 3, 1, 2))

    if retina_warp:
        images = np.array(list(map(warp_image, images[:, 0])))[:, None]

    images = torch.tensor(images).to(torch.float).cuda()

    responses = torch.tensor(responses).cuda().to(torch.float)
    valid_responses = torch.tensor(valid_responses).cuda().to(torch.float)
    dataset = utils.TensorDataset(images, responses, valid_responses)
    data_loader = utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


class CSRF_V1_Data:
    """For use with George's and Kelli's csrf data set."""

    def __init__(self, raw_data_path, image_path=None, seed=None, train_frac=0.8,
                 subsample=1, crop=65, time_bins_sum=tuple(range(7))):
        """
        Args:
            raw_data_path: Path pointing to a pickle file that contains the experimental data.
                            Not all pickle files of the CSRF dataset contain the image data.
                            If the images are missing, an image_path argument should be provided.
            image_path: Path pointing to a pickle file which should contain the image data
                        (training and testing images).
            seed: Random seed for train val data set split (does not affect order of stimuli... in train val split themselves)
            train_frac: Fraction of experiments training data used for model training.
                        Remaining data serves as validation set
                    Float Value between 0 and 1

            subsample: Integer value to downsample the input.
                Example usage:  subsample=1 keeps original resolution
                                subsample=2 cuts the resolution in half

            crop: Integer value to crop stimuli from each side (left, right, bottom, top), before subsampling
            time_bins_sum: a tuple which specifies which times bins are included in the analysis.
                        there are 13 bins (0 to 12), which correspond to 10ms bins from 40 to 160 ms
                        after stimulus presentation
                Exmple usage:   (0,1,2,3) will only include the first four time bins into the analysis
        """
        # unpacking pickle data
        with open(raw_data_path, "rb") as pkl:
            raw_data = pickle.load(pkl)

        self._subject_ids = raw_data["subject_ids"]
        self._session_ids = raw_data["session_ids"]
        self._session_unit_response_link = raw_data["session_unit_response_link"]
        self._repetitions_test = raw_data["repetitions_test"]
        responses_train = raw_data["responses_train"].astype(np.float32)
        self._responses_test = raw_data["responses_test"].astype(np.float32)

        real_responses = np.logical_not(np.isnan(responses_train))
        self._real_responses_test = np.logical_not(np.isnan(self.responses_test))

        images_test = raw_data['images_test']
        if 'test_image_locator' in raw_data:
            test_image_locator = raw_data["test_image_locator"]

        # if an image path is provided, load the images from the corresponding pickle file
        if image_path:
            with open(image_path, "rb") as pkl:
                raw_data = pickle.load(pkl)

        _, h, w = raw_data['images_train'].shape[:3]
        images_train = raw_data['images_train'][:, crop:h - crop:subsample, crop:w - crop:subsample]
        images_test = raw_data['images_test'][:, crop:h - crop:subsample, crop:w - crop:subsample]

        # z-score all images by mean, and sigma of all images
        all_images = np.append(images_train, images_test, axis=0)
        img_mean = np.mean(all_images)
        img_std = np.std(all_images)
        images_train = (images_train - img_mean) / img_std
        self._images_test = (images_test - img_mean) / img_std
        if 'test_image_locator' in raw_data:
            self._images_test = self._images_test[test_image_locator - 1, ::]
        # split into train and val set, images randomly assigned
        train_split, val_split = self.get_validation_split(real_responses, train_frac, seed)
        self._images_train = images_train[train_split]
        self._responses_train = responses_train[train_split]
        self._real_responses_train = real_responses[train_split]

        self._images_val = images_train[val_split]
        self._responses_val = responses_train[val_split]
        self._real_responses_val = real_responses[val_split]

        if seed:
            np.random.seed(seed)

        self._train_perm = np.random.permutation(self._images_train.shape[0])
        self._val_perm = np.random.permutation(self._images_val.shape[0])

        if time_bins_sum is not None:  # then average over given time bins
            self._responses_train = np.sum(self._responses_train[:, :, time_bins_sum], axis=-1)
            self._responses_test = np.sum(self._responses_test[:, :, time_bins_sum], axis=-1)
            self._responses_val = np.sum(self._responses_val[:, :, time_bins_sum], axis=-1)

            # In real responses: If an entry for any time is False, real_responses is False for all times.
            self._real_responses_train = np.all(self._real_responses_train[:, :, time_bins_sum], axis=-1)
            self._real_responses_test = np.all(self._real_responses_test[:, :, time_bins_sum], axis=-1)
            self._real_responses_val = np.all(self._real_responses_val[:, :, time_bins_sum], axis=-1)

        # in responses, change nan to zero. Then: Use real responses vector for all valid responses
        self._responses_train[np.isnan(self._responses_train)] = 0
        self._responses_val[np.isnan(self._responses_val)] = 0
        self._responses_test[np.isnan(self._responses_test)] = 0

        self._minibatch_idx = 0

    # getters
    @property
    def images_train(self):
        """
        Returns:
            train images in current order (changes every time a new epoch starts)
        """
        return np.expand_dims(self._images_train[self._train_perm], -1)

    @property
    def responses_train(self):
        """
        Returns:
            train responses in current order (changes every time a new epoch starts)
        """
        return self._responses_train[self._train_perm]

    # legacy property
    @property
    def real_resps_train(self):
        return self._real_responses_train[self._train_perm]

    @property
    def real_responses_train(self):
        return self._real_responses_train[self._train_perm]

    @property
    def images_val(self):
        return np.expand_dims(self._images_val, -1)

    @property
    def responses_val(self):
        return self._responses_val

    @property
    def images_test(self):
        return np.expand_dims(self._images_test, -1)

    @property
    def responses_test(self):
        return self._responses_test

    @property
    def image_dimensions(self):
        return self.images_train.shape[1:3]

    @property
    def num_neurons(self):
        return self.responses_train.shape[1]

    # methods
    def next_epoch(self):
        """
        Gets new random index permutation for train set, reset minibatch index.
        """
        self._minibatch_idx = 0
        self._train_perm = np.random.permutation(self._train_perm)

    def get_validation_split(self, real_responses_train, train_frac=0.8, seed=None):
        """
            Splits the Training Data into the trainset and validation set.
            The Validation set should recruit itself from the images that most neurons have seen.

        :return: returns permuted indeces for the training and validation set
        """
        if seed:
            np.random.seed(seed)

        num_images = real_responses_train.shape[0]
        Neurons_per_image = np.sum(real_responses_train, axis=1)[:, 0]
        Neurons_per_image_sort_idx = np.argsort(Neurons_per_image)

        top_images = Neurons_per_image_sort_idx[-int(np.floor(train_frac / 2 * num_images)):]
        val_images_idx = np.random.choice(top_images, int(len(top_images) / 2), replace=False)

        train_idx_filter = np.logical_not(np.isin(Neurons_per_image_sort_idx, val_images_idx))
        train_images_idx = np.random.permutation(Neurons_per_image_sort_idx[train_idx_filter])

        return train_images_idx, val_images_idx

    # Methods for compatibility with Santiago's code base.
    def train(self):
        """
            For compatibility with Santiago's code base.

            Returns:
                images_train, responses_train, real_respsonses_train
            """

        return self.images_train, self.responses_train, self.real_responses_train

    def val(self):
        """
        For compatibility with Santiago's code base.

        Returns:
            images_val, responses_val, real_respsonses_val
        """

        return self.images_val, self.responses_val, self._real_responses_val

    def test(self):
        """
            For compatibility with Santiago's code base.

            Returns:
                images_test, responses_test, real_responses_test
            """

        return self.images_test, self.responses_test, self._real_responses_test