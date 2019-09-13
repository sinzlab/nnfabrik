import torch
import torch.utils.data as utils
import numpy as np
import pickle
from builtins import property

# datapath = "data/monkeydata/csrf_dataset_one_35ppd.pickle"

def CSRF_V1(datapath, batch_size, seed,
            train_frac=0.8, subsample=1, crop=65, time_bins_sum=np.arange(7)):


    V1_Data = CSRF_V1_Data(raw_data_path=datapath, seed=seed,
                        train_frac = train_frac, subsample=subsample, crop=crop,
                        time_bins_sum=time_bins_sum)

    images, responses, valid_responses = V1_Data.train()
    train_loader = getLoader_CSRF_V1(images, responses, 1*valid_responses, batch_size)

    images, responses, valid_responses = V1_Data.val()
    val_loader = getLoader_CSRF_V1(images, responses, 1*valid_responses, batch_size)

    images, responses, valid_responses = V1_Data.test()
    test_loader = getLoader_CSRF_V1(images, responses, 1*valid_responses, batch_size)

    DataLoader = dict(train_loader=train_loader,val_loader=val_loader,test_loader=test_loader)
    return DataLoader

def getLoader_CSRF_V1(images, responses, valid_responses, batch_size):

    # Expected Dimension of the Image Tensor is Images x Channels x size_x x size_y
    # In some CSRF files, Channels are at Dim4, the image tensor is thus reshaped accordingly
    im_shape = images.shape
    if im_shape[1]>1:
        # images = torch.tensor(images).view(im_shape[0], im_shape[3], im_shape[1], im_shape[2]).cuda().to(torch.float)
        images = torch.tensor(images).view(im_shape[0], im_shape[3], im_shape[1], im_shape[2]).to(torch.float)
    else:
        images = torch.tensor(images).cuda().to(torch.float)
    """
    responses = torch.tensor(responses).cuda().to(torch.float)
    valid_responses = torch.tensor(valid_responses).cuda().to(torch.float)
    dataset = torch.utils.TensorDataset(images, responses, valid_responses)
    DataLoader = torch.utils.DataLoader(dataset, batch_size=batch_size)
    """
    responses = torch.tensor(responses).to(torch.float)
    valid_responses = torch.tensor(valid_responses).to(torch.float)
    dataset = utils.TensorDataset(images, responses, valid_responses)
    DataLoader = utils.DataLoader(dataset, batch_size=batch_size)

    return DataLoader

class CSRF_V1_Data:
    """For use with George's and Kelli's csrf data set."""

    def __init__(self, raw_data_path=None, seed=None, train_frac=0.8, subsample=1, crop=0, time_bins_sum=None):
        """
        Args:
            raw_data_path: Path pointing to the raw data. Defaults to /gpfs01/bethge/share/csrf_data/csrf_dataset_one.pickle
            seed: Seed for train val data set split (does not affect order of stimuli... in train val split themselves)
            train_frac: Fraction of experiments training data used for model training. Remaining data as val set.
            subsample: Integer values to downsample stimuli
            crop: Integer value to crop stimuli from each side (left, right, bottom, top), before subsampling
            time_bins_sum: array-like, values 0[0, ..., 12], time bins to average over. (40ms to 160ms following image onset in steps of 10ms)
        """

        with open(raw_data_path, "rb") as pkl:
            raw_data = pickle.load(pkl)

        # unpack data

        self.__subject_ids = raw_data["subject_ids"]
        self.__session_ids = raw_data["session_ids"]
        self.__session_unit_response_link = raw_data["session_unit_response_link"]
        self.__repetitions_test = raw_data["repetitions_test"]
        responses_train = raw_data["responses_train"].astype(np.float32)
        self.__responses_test = raw_data["responses_test"].astype(np.float32)

        real_responses = np.logical_not(np.isnan(responses_train))
        self.__real_responses_test = np.logical_not(np.isnan(self.responses_test))

        # crop
        if crop == 0:
            images_train = raw_data["images_train"][:, 0::subsample, 0::subsample]
            images_test = raw_data["images_test"][:, 0::subsample, 0::subsample]
        else:
            images_train = raw_data["images_train"][:, crop:-crop:subsample, crop:-crop:subsample]
            images_test = raw_data["images_test"][:, crop:-crop:subsample, crop:-crop:subsample]

        # z-score all images by mean, and sigma of all images
        all_images = np.append(images_train, images_test, axis=0)
        img_mean = np.mean(all_images)
        img_std = np.std(all_images)
        images_train = (images_train - img_mean) / img_std
        self.__images_test = (images_test - img_mean) / img_std

        # split into train and val set, images randomly assigned
        train_split, val_split = self.get_validation_split(real_responses, train_frac, seed)
        self.__images_train = images_train[train_split, :, :]
        self.__responses_train = responses_train[train_split, :, :]
        self.__real_responses_train = real_responses[train_split, :, :]

        self.__images_val = images_train[val_split, :, :]
        self.__responses_val = responses_train[val_split, :, :]
        self.__real_responses_val = real_responses[val_split, :, :]

        self.__train_perm = np.random.permutation(self.__images_train.shape[0])
        self.__val_perm = np.random.permutation(self.__images_val.shape[0])

        if time_bins_sum is not None:  # then average over given time bins
            self.__responses_train = np.sum(self.__responses_train[:, :, time_bins_sum], axis=-1)
            self.__responses_test = np.sum(self.__responses_test[:, :, time_bins_sum], axis=-1)
            self.__responses_val = np.sum(self.__responses_val[:, :, time_bins_sum], axis=-1)

            # In real responses: If an entry for any time is False, real_responses is False for all times.
            self.__real_responses_train = np.min(self.__real_responses_train[:, :, time_bins_sum], axis=-1)
            self.__real_responses_test = np.min(self.__real_responses_test[:, :, time_bins_sum], axis=-1)
            self.__real_responses_val = np.min(self.__real_responses_val[:, :, time_bins_sum], axis=-1)

        # in responses, change nan to zero. Then: Use real responses vector for all valid responses
        nan_mask = np.isnan(self.__responses_train)
        self.__responses_train[nan_mask] = 0.

        nan_mask = np.isnan(self.__responses_val)
        self.__responses_val[nan_mask] = 0.

        nan_mask = np.isnan(self.__responses_test)
        self.__responses_test[nan_mask] = 0.

        self.__minibatch_idx = 0

    # getters
    @property
    def images_train(self):
        """
        Returns:
            train images in current order (changes every time a new epoch starts)
        """
        return np.expand_dims(self.__images_train[self.__train_perm, ...], -1)

    @property
    def responses_train(self):
        """
        Returns:
            train responses in current order (changes every time a new epoch starts)
        """
        return self.__responses_train[self.__train_perm, ...]

    # legacy property
    @property
    def real_resps_train(self):
        return self.__real_responses_train[self.__train_perm, ...]

    @property
    def real_responses_train(self):
        return self.__real_responses_train[self.__train_perm, ...]

    @property
    def images_val(self):
        return np.expand_dims(self.__images_val, -1)

    @property
    def responses_val(self):
        return self.__responses_val

    @property
    def images_test(self):
        return np.expand_dims(self.__images_test, -1)

    @property
    def responses_test(self):
        return self.__responses_test

    @property
    def px_x(self):
        return self.images_train.shape[1]

    @property
    def px_y(self):
        return self.images_train.shape[2]

    @property
    def num_neurons(self):
        return self.responses_train.shape[1]

    # methods
    def next_epoch(self):
        """
        Gets new random index permutation for train set, reset minibatch index.
        """
        self.__minibatch_idx = 0
        self.__train_perm = np.random.permutation(self.__train_perm)

    def get_validation_split(self, real_responses_train, train_frac=0.8, seed=None):
        """
            Splits the Training Data into the trainset and validation set.
            The Validation set should recruit itself from the images that most neurons have seen.

        :return: returns permuted indeces for the training and validation set
        """
        if seed:
            np.random.seed(seed)  # only affects the next call of a random number generator, i.e. np.random.permutation

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

        return self.images_val, self.responses_val, self.__real_responses_val

    def test(self):
        """
            For compatibility with Santiago's code base.

            Returns:
                images_test, responses_test, real_responses_test
            """

        return self.images_test, self.responses_test, self.__real_responses_test
