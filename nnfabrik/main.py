import datajoint as dj
import torch
import numpy as np
import os
import tempfile

from . import utility
from . import datasets
from . import training
from . import models

from .utility.dj_helpers import make_hash, gitlog
from .utility.nnf_helper import module_dynamic_import


schema = dj.schema(dj.config['schema_name'])  #dj.schema('nnfabrik_core')

@schema
class Fabrikant(dj.Manual):
    definition = """
    architect_name: varchar(32)       # Name of the contributor that added this entry
    ---
    email: varchar(64)      # e-mail address
    affiliation: varchar(32) # conributor's affiliation
    """


@schema
class Model(dj.Manual):
    definition = """
    configurator: varchar(64)   # name of the configuration function
    config_hash: varchar(64)    # hash of the configuration object
    ---
    config_object: longblob     # configuration object to be passed into the function
    -> Fabrikant.proj(model_architect='architect_name')
    model_comment='' : varchar(64)  # short description
    model_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion

    """

    def add_entry(self, configurator, config_object, architect_name, model_comment=''):
        """
        configurator -- name of the function/class that's callable
        config_object -- actual Python object
        """

        config_hash = make_hash(config_object)
        key = dict(configurator=configurator, config_hash=config_hash, config_object=config_object,
                   model_architect=architect_name, model_comment=model_comment)
        self.insert1(key)


    def build_model(self, dataloader, seed, key=None):
        if key is None:
            key = {}

        configurator, config_object = (self & key).fetch1('configurator', 'config_object')
        model_fn = module_dynamic_import(configurator, pre='models.')
        return model_fn(dataloader, seed, **config_object)


@schema
class Dataset(dj.Manual):
    definition = """
    dataset_loader: varchar(64)         # name of the dataset loader function
    dataset_config_hash: varchar(64)    # hash of the configuration object
    ---
    dataset_config: longblob     # dataset configuration object
    -> Fabrikant.proj(dataset_architect='architect_name')
    dataset_comment='' : varchar(64)  # short description
    dataset_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """

    def add_entry(self, dataset_loader, dataset_config, architect_name, dataset_comment=''):
        """
        inserts one new entry into the Dataset Table
        dataset_loader -- name of dataset function/class that's callable
        dataset_config -- actual Python object with which the dataset function is called
        """

        dataset_config_hash = make_hash(dataset_config)
        key = dict(dataset_loader=dataset_loader, dataset_config_hash=dataset_config_hash,
                   dataset_config=dataset_config, dataset_architect=architect_name, dataset_comment=dataset_comment)
        self.insert1(key)

    def get_dataloader(self, seed, key=None):
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
        dataset_fn = module_dynamic_import(dataset_loader, pre='datasets.')
        return dataset_fn(seed=seed, **dataset_config)


@schema
class Trainer(dj.Manual):
    definition = """
    training_function: varchar(64)     # name of the Trainer loader function
    training_config_hash: varchar(64)  # hash of the configuration object
    ---
    training_config: longblob          # training configuration object
    -> Fabrikant.proj(trainer_architect='architect_name')
    trainer_comment='' : varchar(64)  # short description
    trainer_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """

    def add_entry(self, training_function, training_config, architect_name, trainer_comment=''):
        """
        inserts one new entry into the Trainer Table
        training_function -- name of trainer function/class that's callable
        training_config -- actual Python object with which the trainer function is called
        """
        training_config_hash = make_hash(training_config)
        key = dict(training_function=training_function, training_config_hash=training_config_hash,
                   training_config=training_config, trainer_architect=architect_name, trainer_comment=trainer_comment)
        self.insert1(key)

    def get_trainer(self, key=None):
        """
        Returns the training function for a given training function and its corresponding configurations
        """
        if key is None:
            key = {}

        training_function, training_config = (self & key).fetch1('training_function', 'training_config')
        trainer_fn = module_dynamic_import(training_function, pre='training.')
        return trainer_fn, training_config


@schema
class Seed(dj.Manual):
    definition = """
    seed:   int     # Random seed that is passed to the model- and dataset-builder
    """


@schema
# @gitlog
class TrainedModel(dj.Computed):
    definition = """
    -> Model
    -> Dataset
    -> Trainer
    -> Seed
    ---
    score:   float  # loss
    output: longblob  # trainer object's output
    model_state:  attach@minio
    ->Fabrikant
    trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """

    def make(self, key):
        architect_name = (Fabrikant & key).fetch1('architect_name')
        seed = (Seed & key).fetch1('seed')

        dataloader = (Dataset & key).get_dataloader(seed)
        model = (Model & key).build_model(dataloader, seed)
        trainer, trainer_config = (Trainer & key).get_trainer()

        # model training
        score, output, model_state = trainer(model, seed, **dataloader, **trainer_config)

        with tempfile.TemporaryDirectory() as trained_models:
            filename = make_hash(key) + '.pth.tar'
            filepath = os.path.join(trained_models, filename)
            torch.save(model_state, filepath)
            key['score'] = score
            key['output'] = output
            key['model_state'] = filepath
            key['architect_name'] = architect_name
            self.insert1(key)
