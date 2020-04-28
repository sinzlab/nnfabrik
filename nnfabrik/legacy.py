import torch
import tempfile
import warnings
import os

from . import config
from . import utility
from . import datasets
from . import training
from . import models

import datajoint as dj

dj.config["stores"] = {
    "minio": {  #  store in s3
        "protocol": "s3",
        "endpoint": "cantor.mvl6.uni-tuebingen.de:9000",
        "bucket": "nnfabrik",
        "location": "dj-store",
        "access_key": os.environ.get("MINIO_ACCESS_KEY", "FAKEKEY"),
        "secret_key": os.environ.get("MINIO_SECRET_KEY", "FAKEKEY"),
    }
}

from .builder import get_data, get_trainer, get_model, get_all_parts

from .utility.dj_helpers import make_hash, check_repo_commit
from .utility.nnf_helper import split_module_name, dynamic_import, cleanup_numpy_scalar

# check if schema_name defined, otherwise default to nnfabrik_core
schema = dj.schema(dj.config.get("schema_name", "nnfabrik_core"))


@schema
class Fabrikant(dj.Manual):
    definition = """
    architect_name: varchar(32)       # Name of the contributor that added this entry
    ---
    email: varchar(64)      # e-mail address
    affiliation: varchar(32) # conributor's affiliation
    dj_username: varchar(64) # DataJoint username
    """

    @classmethod
    def get_current_user(cls):
        """
        Lookup the architect_name in Fabrikant corresponding to the currently logged in DataJoint user
        Returns: architect_name if match found, else None
        """
        username = cls.connection.get_user().split("@")[0]
        entry = Fabrikant & dict(dj_username=username)
        if entry:
            return entry.fetch1("architect_name")


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

    @property
    def fn_config(self):
        model_fn, model_config = self.fetch1("configurator", "config_object")
        model_config = cleanup_numpy_scalar(model_config)
        return model_fn, model_config

    def add_entry(self, configurator, config_object, model_architect=None, model_comment=""):
        """
        configurator -- name of the function/class that's callable
        config_object -- actual Python object
        """
        module_path, class_name = split_module_name(configurator)
        config_fn = dynamic_import(module_path, class_name) if module_path else eval("models." + configurator)
        try:
            callable(config_fn)
        except NameError:
            warnings.warn("configurator function does not exist. Table entry rejected")
            return

        config_hash = make_hash(config_object)
        if model_architect is None:
            model_architect = Fabrikant.get_current_user()
        key = dict(
            configurator=configurator,
            config_hash=config_hash,
            config_object=config_object,
            model_architect=model_architect,
            model_comment=model_comment,
        )
        self.insert1(key)

    def build_model(self, dataloader, seed=None, key=None):
        print("Loading model...")
        if key is None:
            key = {}
        configurator, config_object = (self & key).fn_config

        return get_model(configurator, config_object, dataloader, seed=seed)


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

    @property
    def fn_config(self):
        dataset_loader, dataset_config = self.fetch1("dataset_loader", "dataset_config")
        dataset_config = cleanup_numpy_scalar(dataset_config)
        return dataset_loader, dataset_config

    def add_entry(self, dataset_loader, dataset_config, dataset_architect=None, dataset_comment=""):
        """
        inserts one new entry into the Dataset Table
        dataset_loader -- name of dataset function/class that's callable
        dataset_config -- actual Python object with which the dataset function is called
        """

        module_path, class_name = split_module_name(dataset_loader)
        dataset_fn = dynamic_import(module_path, class_name) if module_path else eval("datasets." + dataset_loader)
        try:
            callable(dataset_fn)
        except NameError:
            warnings.warn("dataset_loader function does not exist. Table entry rejected")
            return

        if dataset_architect is None:
            dataset_architect = Fabrikant.get_current_user()

        dataset_config_hash = make_hash(dataset_config)
        key = dict(
            dataset_loader=dataset_loader,
            dataset_config_hash=dataset_config_hash,
            dataset_config=dataset_config,
            dataset_architect=dataset_architect,
            dataset_comment=dataset_comment,
        )
        self.insert1(key)

    def get_dataloader(self, seed=None, key=None):
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

        dataset_loader, dataset_config = (self & key).fn_config

        if seed is not None:
            dataset_config["seed"] = seed  # override the seed if passed in

        return get_data(dataset_loader, dataset_config)


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

    @property
    def fn_config(self):
        training_function, training_config = self.fetch1("training_function", "training_config")
        training_config = cleanup_numpy_scalar(training_config)
        return training_function, training_config

    def add_entry(self, training_function, training_config, trainer_architect=None, trainer_comment=""):
        """
        inserts one new entry into the Trainer Table
        training_function -- name of trainer function/class that's callable
        training_config -- actual Python object with which the trainer function is called
        """

        module_path, class_name = split_module_name(training_function)
        trainer_fn = dynamic_import(module_path, class_name) if module_path else eval("training." + training_function)
        try:
            callable(trainer_fn)
        except NameError:
            warnings.warn("dataset_loader function does not exist. Table entry rejected")
            return

        training_config_hash = make_hash(training_config)

        if trainer_architect is None:
            trainer_architect = Fabrikant.get_current_user()

        key = dict(
            training_function=training_function,
            training_config_hash=training_config_hash,
            training_config=training_config,
            trainer_architect=trainer_architect,
            trainer_comment=trainer_comment,
        )
        self.insert1(key)

    def get_trainer(self, key=None, build_partial=True):
        """
        Returns the training function for a given training function and its corresponding configurations
        """
        if key is None:
            key = {}
        training_function, training_config = (self & key).fn_config

        if build_partial:
            # build the configuration into the function
            return get_trainer(training_function, training_config)
        else:
            # return them separately
            return get_trainer(training_function), training_config


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
    ->[nullable] Fabrikant
    trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """

    def get_full_config(self, key=None, include_state_dict=True):
        if key is None:
            key = self.fetch1("KEY")

        model_fn, model_config = (Model & key).fn_config
        dataset_fn, dataset_config = (Dataset & key).fn_config
        trainer_fn, trainer_config = (Trainer & key).fn_config

        ret = dict(
            model_fn=model_fn,
            model_config=model_config,
            dataset_fn=dataset_fn,
            dataset_config=dataset_config,
            trainer_fn=trainer_fn,
            trainer_config=trainer_config,
        )

        # if trained model exist and include_state_dict is True
        if include_state_dict and (self & key):
            ret["state_dict"] = (self.ModelStorage & key).fetch1("model_state")

        return ret

    class ModelStorage(dj.Part):
        definition = """
        # Contains the paths to the stored models
        -> master
        ---
        model_state:            attach@minio
        """

    class GitLog(dj.Part):
        definition = """
        ->master
        ---
        info :              longblob
        """

    def get_entry(self, key):
        (Dataset & key).fetch()

    def make(self, key):

        commits_info = {name: info for name, info in [check_repo_commit(repo) for repo in config["repos"]]}
        assert len(commits_info) == len(config["repos"])

        if any(["error_msg" in name for name in commits_info.keys()]):
            err_msgs = ["You have uncommited changes."]
            err_msgs.extend([info for name, info in commits_info.items() if "error_msg" in name])
            err_msgs.append("\nPlease commit the changes before running populate.\n")
            raise RuntimeError("\n".join(err_msgs))

        else:

            # by default try to lookup the architect corresponding to the current DJ user
            architect_name = Fabrikant.get_current_user()
            seed = (Seed & key).fetch1("seed")

            config_dict = self.get_full_config(key)
            dataloaders, model, trainer = get_all_parts(**config_dict, seed=seed)

            # model training
            score, output, model_state = trainer(model, seed, **dataloaders)

            with tempfile.TemporaryDirectory() as trained_models:
                filename = make_hash(key) + ".pth.tar"
                filepath = os.path.join(trained_models, filename)
                torch.save(model_state, filepath)

                key["score"] = score
                key["output"] = output
                key["architect_name"] = architect_name
                self.insert1(key)

                key["model_state"] = filepath
                self.ModelStorage.insert1(key, ignore_extra_fields=True)

                # add the git info to the part table
                if commits_info:
                    key["info"] = commits_info
                    self.GitLog().insert1(key, skip_duplicates=True, ignore_extra_fields=True)
