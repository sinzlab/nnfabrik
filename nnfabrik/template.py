import datajoint as dj
import tempfile
import torch
import os
from .main import Model, Dataset, Trainer, Seed, Fabrikant
from .builder import get_all_parts, get_model, get_trainer
from .utility.nnf_helper import cleanup_numpy_scalar
from .utility.dj_helpers import gitlog, make_hash
from .builder import resolve_data
from datajoint.fetch import DataJointError


class DataInfoBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    DatasetInfo table. By default, this will inherit from the Dataset as found in nnfabrik.main.
    To change this behavior, overwrite the dataset_table` property.
    """

    dataset_table = Dataset
    user_table = Fabrikant

    # table level comment
    table_comment = "Table containing information about i/o dimensions and statistics, per data_key in dataset"

    @property
    def definition(self):
        definition = """
            # {table_comment}
            -> self.dataset_table
            ---
            data_info:                     longblob     # Dictionary of data_keys and i/o information

            ->[nullable] self.user_table
            datainfo_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
            """.format(table_comment=self.table_comment)
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

        dataset_fn = resolve_data(key["dataset_fn"])
        dataset_config = (self.dataset_table & key).fetch1("dataset_config")
        data_info = dataset_fn(**dataset_config, return_data_info=True)

        fabrikant_name = self.user_table.get_current_user()

        key['fabrikant_name'] = fabrikant_name
        key['data_info'] = data_info
        self.insert1(key, ignore_extra_fields=True)


class TrainedModelBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    TrainedModel table. By default, this will inherit from following 
    Model, Dataset, Trainer and Seed as found in nnfabrik.main, and also point to . To change this behavior,
    overwrite the `model_table`, `dataset_table`, `trainer_table` and `seed_table` class
    properties.
    """

    model_table = Model
    dataset_table = Dataset
    trainer_table = Trainer
    seed_table = Seed
    user_table = Fabrikant
    data_info_table = DataInfoBase

    # delimitter to use when concatenating comments from model, dataset, and trainer tables
    comment_delimitter = '.'

    # table level comment
    table_comment = "Trained models"

    @property
    def definition(self):           
        definition = """
        # {table_comment}
        -> self.model_table
        -> self.dataset_table
        -> self.trainer_table
        -> self.seed_table
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self.user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        """.format(table_comment=self.table_comment)
        return definition

    class ModelStorage(dj.Part):
        storage = 'minio'
        
        @property
        def definition(self):
            definition = """
            # Contains the paths to the stored models
            -> master
            ---
            model_state:            attach@{storage}
            """.format(storage=self.storage)
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
            key = self.fetch1('KEY')

        model_fn, model_config = (self.model_table & key).fn_config
        dataset_fn, dataset_config = (self.dataset_table & key).fn_config


        ret = dict(model_fn=model_fn, model_config=model_config,
                   dataset_fn=dataset_fn, dataset_config=dataset_config)

        if include_trainer:
            trainer_fn, trainer_config = (self.trainer_table & key).fn_config
            ret['trainer_fn'] = trainer_fn
            ret['trainer_config'] = trainer_config

        # if trained model exist and include_state_dict is True
        if include_state_dict and (self.ModelStorage & key):
            with tempfile.TemporaryDirectory() as temp_dir:
                state_dict_path = (self.ModelStorage & key).fetch1('model_state', download_path=temp_dir)
                ret['state_dict'] = torch.load(state_dict_path)

        return ret

    def load_model(self, key=None, include_dataloader=True, include_trainer=False, include_state_dict=True, seed:int=None):
        """
        Load a single entry of the model. If state_dict is available, the model will be loaded with state_dict as well.
        By default the trainer is skipped. Set `include_trainer=True` to also retrieve the trainer function
        as the third return argument.

        Args:
            key - specific key against which to retrieve the model. The key must restrict all component
                  tables into a single entry. If None, will assume that this table is already restricted and
                  will obtain an existing single entry.
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
            key = self.fetch1('KEY')

        if seed is None and len(self.seed_table & key) == 1:
            seed = (self.seed_table & key).fetch1('seed')

        config_dict = self.get_full_config(key, include_trainer=include_trainer, include_state_dict=include_state_dict)

        if not include_dataloader:
            try:
                data_info = (self.data_info_table & key).fetch1('data_info')
                model_config_dict = dict(model_fn=config_dict["model_fn"],
                                         model_config=config_dict["model_config"],
                                         data_info=data_info,
                                         seed=seed,
                                         state_dict=config_dict.get("state_dict", None),
                                         strict=False)

                if include_trainer:
                    trainer_config_dict = dict(trainer_fn=config_dict["trainer_fn"],
                                               trainer_config=config_dict["trainer_config"])

                net = get_model(**model_config_dict)
                return (net, get_trainer(**trainer_config_dict)) if include_trainer else net
            except DataJointError:
                print("The table `DataInfo` needs to be filled in order to load the model without"
                      "the dataloaders. Resorting to the standard way of model loading.")
            except (TypeError, AttributeError):
                print("Model could not be built without the dataloader. Dataloader will be built in order to create the model. "
                      "Make sure to have an The 'model_fn' also has to be able to"
                      "accept 'data_info' as an input arg, and use that over the dataloader to build the model.")

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
        seed = (Seed & key).fetch1('seed')

        # load everything
        dataloaders, model, trainer = self.load_model(key, include_trainer=True, include_state_dict=False, seed=seed)

        # define callback with pinging
        def call_back(**kwargs):
            self.connection.ping()
            self.call_back(**kwargs)

        # model training
        score, output, model_state = trainer(model=model, dataloaders=dataloaders, seed=seed, uid=key, cb=call_back)

        with tempfile.TemporaryDirectory() as temp_dir:
            filename = make_hash(key) + '.pth.tar'
            filepath = os.path.join(temp_dir, filename)
            torch.save(model_state, filepath)

            key['score'] = score
            key['output'] = output
            key['fabrikant_name'] = fabrikant_name
            comments = []
            comments.append((self.trainer_table & key).fetch1("trainer_comment"))
            comments.append((self.model_table & key).fetch1("model_comment"))
            comments.append((self.dataset_table & key).fetch1("dataset_comment"))
            key['comment'] = self.comment_delimitter.join(comments)
            self.insert1(key)

            key['model_state'] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)


class UnitIDsBase(dj.Computed):
    definition = """
        # Template table for Unit IDs

        unit_id:     int            # Neuron Identifier
        ---
        unit_position: int          # position index of the neuron in the dataset
        """


def scoring_function_base(dataloaders, model, per_Neuron=False):
    raise NotImplementedError("Scoring Function has to be implemented")


class ScoringBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    Scores table. This serves as a template for scores than can be computed with a
    trained_model and a dataloader. The dataloader is expected to return batches of the same condition,
    such as image reapeats.
    """

    user_table = Fabrikant
    trainedmodel_table = TrainedModelBase
    unit_table = UnitIDsBase
    scoring_function = scoring_function_base

    # table level comment
    table_comment = "A template table for storing results/scores of a TrainedModel"

    @property
    def definition(self):
        definition = """
            # {table_comment}
            -> self.trainedmodel_table
            ---
            score:      float     # A template for a computed score of a trained model

            ->[nullable] self.user_table
            score_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
            """.format(table_comment=self.table_comment)
        return definition

    class UnitScore(dj.Part):

        @property
        def definition(self):
            definition = """
            # Scores for Individual Neurons
            -> master
            -> unit_table
            ---
            unit_score:     float   # A template for a computed unit score        
            """
            return definition

    def get_repeats_dataloader(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        dataloaders = self.trainedmodel_table().dataset_table.get_dataloader(key=key)
        return dataloaders["test"]

    def make(self, key):
        dataloaders = self.get_repeats_dataloader(key=key)
        model = self.trainedmodel_table().load_model(key=key, include_state_dict=True, include_dataloader=False, include_trainer=False)

        fabrikant_name = self.user_table.get_current_user()
        key['fabrikant_name'] = fabrikant_name
        key['score'] = self.scoring_function(dataloaders=dataloaders, model=model, per_Neuron=False)
        self.insert1(key, ignore_extra_fields=True)

        unit_scores = self.scoring_function(dataloaders=dataloaders, model=model, per_Neuron=True)

        for unit_pos, unit_score in enumerate(unit_scores):
            unit_id = (self.unit_table & key & "unit_position=".format(unit_pos)).fetch1("unit_id")
            key["unit_id"] = unit_id
            key["unit_score"] = unit_score
            self.UnitScore.insert1(key, ignore_extra_fields=True)