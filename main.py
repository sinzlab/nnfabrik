import datajoint as dj

dj.config['database.host'] = 'datajoint.ninai.org'
schema = dj.schema('kwilleke_generalized_model_fitting')

# placeholder function
def make_hash(config_input):
    """
        hashes the configurator input for the model-, dataset-, training-function-builders

    :returns: a unique hash for each configurator
    """
    return 'a_un1que_h4sh'


@schema
class Model(dj.Manual):
    definition = """
    configurator: varchar(32)   # name of the configuration function
    config_hash: varchar(64)    # hash of the configuration object
    ---
    config_object: longblob     # configuration object to be passed into the function
    """

    def add_entry(self, configurator, config_object):
        """
        configurator -- name of the function/class that's callable
        config_object -- actual Python object
        """

        config_hash = make_hash(config_object)
        key = dict(configurator=configurator, config_hash=config_hash, config_object=config_object)
        self.insert1(key)

    def build_model(self, img_dim, key=None):
        if key is None:
            key = {}

        configurator, config_object = (self & key).fetch1('configurator', 'config_object')
        config_object = {k: config_object[k][0].item() for k in config_object.dtype.fields}
        config_fn = eval(configurator)
        return config_fn(img_dim, **config_object)


@schema
class Dataset(dj.Manual):
    definition = """
    dataset_loader: varchar(32)         # name of the dataset loader function
    dataset_config_hash: varchar(64)    # hash of the configuration object
    ---
    dataset_config: longblob     # dataset configuration object
    """

    def add_entry(self, dataset_loader, dataset_config):
        """
        inserts one new entry into the Dataset Table

        dataset_loader -- name of dataset function/class that's callable
        dataset_config -- actual Python object with which the dataset function is called
        """

        dataset_config_hash = make_hash(dataset_config)
        key = dict(dataset_loader=dataset_loader, dataset_config_hash=dataset_config_hash,
                   dataset_config=dataset_config)
        self.insert1(key)

    def get_dataloader(self, key=None):
        """
        Returns a dataloader for a given dataset loader function and its corresponding configurations

        dataloader: is expected to be a dict in the form of
                            {
                            'train_loader': torch.utils.data.DataLoader,
                             'val_loader': torch.utils.data.DataLoader,
                             'test_loader: torch.utils.data.DataLoader,
                             }
                             or a similar iterable object

                each loader should have as first argument the input such that
                    next(iter(train_loader)): [input, responses, ...]

                the input should have the following form:
                    [batch_size, channels, px_x, px_y, ...]
        """
        if key is None:
            key = {}

        dataset_loader, dataset_config = (self & key).fetch1('dataset_loader', 'dataset_config')
        dataset_config = {k: dataset_config[k][0].item() for k in dataset_config.dtype.fields}
        config_fn = eval(dataset_loader)
        return config_fn(**dataset_config)


@schema
class Trainer(dj.Manual):
    definition = """
    training_function: varchar(32)     # name of the Trainer loader function
    training_config_hash: varchar(64)  # hash of the configuration object
    ---
    training_config: longblob          # training configuration object
    """

    def add_entry(self, training_function, training_config):
        """
        inserts one new entry into the Trainer Table

        training_function -- name of trainer function/class that's callable
        training_config -- actual Python object with which the trainer function is called
        """
        training_config_hash = make_hash(training_config)
        key = dict(training_function=training_function, training_config_hash=training_config_hash,
                   training_config=training_config)
        self.insert1(key)

    def get_trainer(self, key=None):
        """
        Returns the training function for a given training function and its corresponding configurations
        """
        if key is None:
            key = {}

        training_function, training_config = (self & key).fetch1('training_function', 'training_config')
        training_config = {k: training_config[k][0].item() for k in training_config.dtype.fields}
        return eval(training_function), training_config


@schema
class Seed(dj.Manual):
    definition = """
    seed:   int     # Random seed that is passed to the model- and dataset-builder
    """

    def add_entry(self, seed):
        """
            inserts a user specified seed into the Seed Table
        """
        key = dict(seed=seed)
        self.insert1(key)

    def get_seed(self, key=None):
        """
        gets the seed and is passed on to the TrainedModel table
        """
        if key is None:
            key = {}

        seed = (self & key).fetch1('seed')
        return seed


@schema
class TrainedModel(dj.Computed):
    definition = """
    -> Model
    -> Dataset
    -> Trainer
    -> Seed
    ---
    loss:   longblob  # loss
    output: longblob  # trainer object's output
    """
    # model_state: attach@storage has yet to be added

    def make(self, key):
        trainer, trainer_config = (Trainer & key).get_trainer()
        dataloader = (Dataset & key).get_dataloader()

        # gets the input dimensions from the dataloader
        #
        input_dim, output_dim = self.get_input_dimensions(dataloader)
        # passes the input dimensions to the model builder function
        model = (Model & key).build_model(input_dim, output_dim)
        seed = (Seed & key).get_seed()

        # model training
        loss, output, model_state = trainer(model, seed, **trainer_config, **dataloader)

        key['loss'] = loss
        key['output'] = output
        self.insert1(key)

    def get_in_out_dimensions(self, dataloader):
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
        train_loader = dataloader["train_loader"]
        train_batch = next(iter(train_loader))
        input_batch = train_batch[0]
        output_batch = train_batch[1]
        return input_batch.shape, output_batch.shape