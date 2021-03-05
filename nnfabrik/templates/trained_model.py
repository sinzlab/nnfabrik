import tempfile
import torch
import os
import warnings
import datajoint as dj
from datajoint.fetch import DataJointError
from ..builder import get_all_parts, get_model, get_trainer
from ..utility.dj_helpers import make_hash
from .utility import find_object


class TrainedModelBase(dj.Computed):
    """
    Base class for defining TrainedModel table used to tigger training of models in nnfabrik.

    To use this class, define a new class inheriting from this base class, and decorate with your own
    schema. Furthermore, you have to do one of the following for the class to be functional:
    * Set the class property `nnfabrik` to point to a module or a dictionary context that contains classes
        for tables corresponding to `Fabrikant`, `Seed`, `Dataset`, `Model`, and `Trainer`. Most commonly, you
        would want to simply pass the resulting module object from `my_nnfabrik` output.
    * Set the class property `nnfabrik` to "core" -- this will then make this table refer to
        `Fabrikant`, `Seed`, `Dataset`, `Model`, and `Trainer` as found inside `main` module directly. Note that
        this will therefore depend on the shared "core" tables of nnfabrik.
    * Set the values of the following class properties to individually specify the DataJoint table to use:
        `user_table`, `seed_table`, `dataset_table`, `model_table` and `trainer_table` to specify equivalent
        of `Fabrikant`, `Seed`, `Dataset`, `Model`, and `Trainer`, respectively. You could also set the
        value of `nnfabrik` to a module or "core" as stated above, and specifically override a target table
        via setting one of the table class property as well.
    * The TrainedModel table also needs a storage defined within the dj.config.
        By default, this storage is called "minio", and is set by this template accordingly.
            >>> storage = "minio"
        The best practice is to include this block of code where the TrainedModel table is instantiated.
            >>> if not 'stores' in dj.config:
            >>>     dj.config['stores'] = {}
            >>> dj.config['stores']['minio'] = {  # store in s3
            >>>    'protocol': 's3',
            >>>    'endpoint': os.environ.get('MINIO_ENDPOINT', 'DUMMY_ENDPOINT'),
            >>>    'bucket': 'nnfabrik',
            >>>    'location': 'dj-store',
            >>>    'access_key': os.environ.get('MINIO_ACCESS_KEY', 'FAKEKEY'),
            >>>    'secret_key': os.environ.get('MINIO_SECRET_KEY', 'FAKEKEY')
            >>> }
        The .env file that is used when the docker/singularity container is created is required to have these entries:
            >>> MINIO_ENDPOINT=...
            >>> MINIO_ACCESS_KEY=...
            >>> MINIO_SECRET_KEY=...

    * Example instantiation of a TrainedModel table with the "my_nnfabrik" object, which is the best practice:
        >>> from nnfabrik.main import my_nnfabrik
        >>> from nnfabrik.templates import TrainedModelBase
        >>> my_nnfabrik_module = my_nnfabrik('nnfabrik_schema_name')
        >>> @my_nnfabrik_module.schema
        >>> TrainedModel(TrainedModelBase)
        >>>     nnfabrik = my_nnfabrik_module
        Be sure to have the code block defining the 'store' in the dj.config available here
    """

    database = ""  # hack to suppress DJ error

    nnfabrik = None

    @property
    def model_table(self):
        return find_object(self.nnfabrik, "Model")

    @property
    def dataset_table(self):
        return find_object(self.nnfabrik, "Dataset")

    @property
    def trainer_table(self):
        return find_object(self.nnfabrik, "Trainer")

    @property
    def seed_table(self):
        return find_object(self.nnfabrik, "Seed")

    @property
    def user_table(self):
        return find_object(self.nnfabrik, "Fabrikant", "user_table")

    @property
    def data_info_table(self):
        return find_object(self.nnfabrik, "DataInfo", "data_info_table")

    # storage for the ModelStorage table
    storage = "minio"

    # delimitter to use when concatenating comments from model, dataset, and trainer tables
    comment_delimitter = "."

    # table level comment
    table_comment = "Trained models"

    @property
    def definition(self):
        definition = """
        # {table_comment}
        -> self().model_table
        -> self().dataset_table
        -> self().trainer_table
        -> self().seed_table
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self().user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(
            table_comment=self.table_comment
        )
        return definition

    class ModelStorage(dj.Part):
        @property
        def definition(self):
            definition = """
            # Contains the models state dict, stored externally.
            -> master
            ---
            model_state:            attach@{storage}
            """.format(
                storage=self._master.storage
            )
            return definition

    def get_full_config(self, key=None, include_state_dict=True, include_trainer=True):
        """
        Returns the full configuration dictionary needed to build all components of the network
        training including dataset, model and trainer. The returned dictionary is designed to be
        passed (with dictionary expansion) into the get_all_parts function provided in builder.py.

        Args:
            key - specific key against which to retrieve all configuration. The key must restrict all component
                  tables into a single entry. If None, will assume that this table is already restricted and
                  will obtain an existing single entry.
            include_state_dict (bool) : If True, and if key refers to a model already trained with a corresponding entry in self.ModelStorage,
                  the state_dict of the trained model is retrieved and returned
            include_trainer (bool): If False, then trainer configuration is skipped. Usually desirable when you want to simply retrieve trained model.
        """
        if key is None:
            key = self.fetch1("KEY")

        model_fn, model_config = (self.model_table & key).fn_config
        dataset_fn, dataset_config = (self.dataset_table & key).fn_config

        ret = dict(
            model_fn=model_fn,
            model_config=model_config,
            dataset_fn=dataset_fn,
            dataset_config=dataset_config,
        )

        if include_trainer:
            trainer_fn, trainer_config = (self.trainer_table & key).fn_config
            ret["trainer_fn"] = trainer_fn
            ret["trainer_config"] = trainer_config

        # if trained model exist and include_state_dict is True
        if include_state_dict and (self.ModelStorage & key):
            with tempfile.TemporaryDirectory() as temp_dir:
                state_dict_path = (self.ModelStorage & key).fetch1("model_state", download_path=temp_dir)
                ret["state_dict"] = torch.load(state_dict_path)

        return ret

    def load_model(
        self,
        key=None,
        include_dataloader=True,
        include_trainer=False,
        include_state_dict=True,
        seed: int = None,
    ):
        """
        Load a single entry of the model. If state_dict is available, the model will be loaded with state_dict as well.
        By default the trainer is skipped. Set `include_trainer=True` to also retrieve the trainer function
        as the third return argument.

        Args:
            key - specific key against which to retrieve the model. The key must restrict all component
                  tables into a single entry. If None, will assume that this table is already restricted and
                  will obtain an existing single entry.
            include_dataloader - if True, builds the dataloaer and the model, and returns both.
                                 if False, tries to build the model without requiring dataloader.
                                    Returns the model only when set to False.
            include_trainer - If False (default), will not load or return the trainer.
            include_state_dict - If True, the model is loaded with state_dict if key corresponds to a trained entry.
            seed - Optional seed. If not given and a corresponding entry exists in self.seed_table, seed is taken from there

        Returns
            dataloaders - Loaded dictionary (train, test, validation) of dictionary (data_key) of dataloaders
            model - Loaded model. If key corresponded to an existing entry, it would have also loaded the
                    state_dict unless load_state_dict=False
            trainer - Loaded trainer function. This is not returned if include_trainer=False.
        """
        if key is None:
            key = self.fetch1("KEY")

        # if no explicit seed is provided and there is already a corresponding entry in the seed_table
        # use that seed value
        if seed is None and len(self.seed_table & key) == 1:
            seed = (self.seed_table & key).fetch1("seed")

        config_dict = self.get_full_config(key, include_trainer=include_trainer, include_state_dict=include_state_dict)

        if not include_dataloader:
            try:
                data_info = (self.data_info_table & key).fetch1("data_info")
                model_config_dict = dict(
                    model_fn=config_dict["model_fn"],
                    model_config=config_dict["model_config"],
                    data_info=data_info,
                    seed=seed,
                    state_dict=config_dict.get("state_dict", None),
                    strict=False,
                )

                net = get_model(**model_config_dict)
                return (
                    (
                        net,
                        get_trainer(config_dict["trainer_fn"], config_dict["trainer_config"]),
                    )
                    if include_trainer
                    else net
                )

            except (TypeError, AttributeError, DataJointError):
                warnings.warn(
                    "Model could not be built without the dataloader. Dataloader will be built in order to create the model. "
                    "Make sure to have an The 'model_fn' also has to be able to"
                    "accept 'data_info' as an input arg, and use that over the dataloader to build the model."
                )

            ret = get_all_parts(**config_dict, seed=seed)
            return ret[1:] if include_trainer else ret[1]

        return get_all_parts(**config_dict, seed=seed)

    def call_back(self, uid=None, epoch=None, model=None, info=None):
        """
        Override this implementation to get periodic calls during the training
        by the trainer.

        Args:
            uid - Unique identifier for the trained model entry
            epoch - the iteration count
            model - current model under training
            info - Additional information provided by the trainer
        """
        pass

    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """
        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = self.user_table.get_current_user()
        seed = (self.seed_table & key).fetch1("seed")

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False, seed=seed)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)

        # save resulting model_state into a temporary file to be attached
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + ".pth.tar"
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key["score"] = score
            key["output"] = output
            key["fabrikant_name"] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key["comment"] = self.comment_delimitter.join(comments)
            self.insert1(key)

            key["model_state"] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)


class DataInfoBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    DataInfo table.

    Furthermore, you have to do one of the following for the class to be functional:
    * Set the class property `nnfabrik` to point to a module or a dictionary context that contains classes
        for tables corresponding to `Fabrikant` and `Dataset`. Most commonly, you
        would want to simply pass the resulting module object from `my_nnfabrik` output.
    * Set the class property `nnfabrik` to "core" -- this will then make this table refer to
        `Fabrikant` and `Dataset` as found inside `main` module directly. Note that
        this will therefore depend on the shared "core" tables of nnfabrik.
    * Set the values of the following class properties to individually specify the DataJoint table to use:
        `user_table` and `dataset_table` to specify equivalent
        of `Fabrikant` and `Dataset` respectively. You could also set the
        value of `nnfabrik` to a module or "core" as stated above, and specifically override a target table
        via setting one of the table class property as well.
    """

    nnfabrik = None

    @property
    def dataset_table(self):
        return find_object(self.nnfabrik, "Dataset")

    @property
    def user_table(self):
        return find_object(self.nnfabrik, "Fabrikant", "user_table")

    # table level comment
    table_comment = "Table containing information about i/o dimensions and statistics, per data_key in dataset"

    @property
    def definition(self):
        definition = """
            # {table_comment}
            -> self().dataset_table
            ---
            data_info:                     longblob     # Dictionary of data_keys and i/o information

            ->[nullable] self().user_table
            datainfo_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
            """.format(
            table_comment=self.table_comment
        )
        return definition

    def make(self, key):
        """
        Given a dataset from nnfabrik, extracts the necessary information for building a model in nnfabrik.
        'data_info' is expected to be a dictionary of dictionaries, similar to the dataloaders object.
        For example:
            data_info = {
                        'data_key_0': dict(input_dimensions=[N,c,h,w, ...],
                                           input_channels=[c],
                                           output_dimension=[o, ...],
                                           img_mean=mean_train_images,
                                           img_std=std_train_images),
                        'data_key_1':  ...
                        }
        """
        dataset_fn, dataset_config = (self.dataset_table & key).fn_config
        data_info = dataset_fn(**dataset_config, return_data_info=True)

        fabrikant_name = self.user_table.get_current_user()

        key["fabrikant_name"] = fabrikant_name
        key["data_info"] = data_info
        self.insert1(key)
