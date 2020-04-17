import torch
import torch.utils.data as utils
import numpy as np
import pickle

# These function provide compatibility with the previous data loading logic of monkey V1 Data.
# Individual sessions are no longer identified by a session key for different readouts,
# but all sessions will be in a single loader. This provides backwards compatibility for
# the Divisive Normalization model of Max Burg, and allows for direct comparison to the new way of dataloading as
# a proof of principle for these kinds of models.

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