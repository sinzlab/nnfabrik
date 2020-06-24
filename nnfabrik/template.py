import datajoint as dj
import tempfile
import numpy as np
import torch
import os
from .main import Model, Dataset, Trainer, Seed, Fabrikant
from .builder import get_all_parts, get_model, get_trainer
from .utility.dj_helpers import gitlog, make_hash
from .builder import resolve_data
from datajoint.fetch import DataJointError


class DataInfoBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    DataInfo table. By default, this will inherit from the Dataset as found in nnfabrik.main.
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
        self.insert1(key)


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

                net = get_model(**model_config_dict)
                return (net, get_trainer(config_dict["trainer_fn"], config_dict["trainer_config"])) if include_trainer else net

            except (TypeError, AttributeError, DataJointError):
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
        seed = (self.seed_table & key).fetch1('seed')

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

        unit_id:        int             # Neuron Identifier
        data_key:       varchar(64)     # Identifier of the dataset which the neuron is a part of
        ---
        unit_index:     int             # position index of the neuron in the dataset and model
        """

    def make(self):
        raise NotImplementedError("Scoring Function has to be implemented")


class ScoringBase(dj.Computed):
    """
    Inherit from this class and decorate with your own schema to create a functional
    Score table. This serves as a template for all scores than can be computed with a
    TrainedModel and a dataloader.
    Each master table has an attribute that stores the grand average score across all Neurons.
    The `UnitScore` part table stores the score for all units.
    In order to instantiate a functional table, the default table attriubutes need to be changed.

    This template table is implementing the following logic:
    1) Loading the a trained model from the TrainedModel table. This table needs to have the 'load_model' method.
    2) gettng a dataloader. Minimally, the dataloader returns batches of inputs and targets.
        The dataloader will be built by the Datast table of nnfabrik as default.
        This table needs to have the 'get_dataloader' method
    3) passing the model and the dataloader to a scoring function, defined in the class attribute. The function is expected to
        return either the grand average score or the score per unit.
    4) The average scores and unit scores will be stored in the maser/part tables of this template.

    Attributes:
        trainedmodel_table (Datajoint Table) - an instantiation of the TrainedModelBase
        unit_table (Datajoint Table) - an instantiation of the UnitIDsBase
        scoring_function (function object) - a function that computes the average score (for the master table),
            as well as the unit scores. An example function can be found in this module under 'scoring_function_base'
        scoring_dataset (str) - following nnfabrik convention, this string specifies the key for the 'dataloaders'
            object. The dataloaders object has to contain at least ['train', 'validation', 'test'].
            This string determines, on what data tier the score is computed on. Defaults to the test set.
        scoring_attribute (str) - name of the non-primary attribute of the master and part tables for the score.
        cache (object) - Stores
    """
    trainedmodel_table = TrainedModelBase
    dataset_table = trainedmodel_table.dataset_table
    unit_table = UnitIDsBase
    function_kwargs = {}
    measure_dataset = "test"
    measure_attribute = "score"
    model_cache = None
    data_cache = None

    @staticmethod
    def measure_function(dataloaders, model, device='cuda', as_dict=True, per_neuron=True):
        raise NotImplementedError("Scoring Function has to be implemented")

    # table level comment
    table_comment = "A template table for storing results/scores of a TrainedModel"

    @property
    def definition(self):
        definition = """
                # {table_comment}
                -> self.trainedmodel_table
                ---
                {measure_attribute}:      float     # A template for a computed score of a trained model
                {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                """.format(table_comment=self.table_comment, measure_attribute=self.measure_attribute)
        return definition

    class Units(dj.Part):
        @property
        def definition(self):
            definition = """
                # Scores for Individual Neurons
                -> master
                -> master.unit_table
                ---
                unit_{measure_attribute}:     float   # A template for a computed unit score        
                """.format(measure_attribute=self._master.measure_attribute)
            return definition

    def get_model(self, key=None):
        if self.model_cache is None:
            model = self.trainedmodel_table().load_model(key=key,
                                                         include_state_dict=True,
                                                         include_dataloader=False)
        else:
            model = self.model_cache.load(key=key,
                                          include_state_dict=True,
                                          include_dataloader=False)
        return model


    def get_dataloaders(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        dataloaders = self.dataset_table().get_dataloader(key=key) if self.data_cache is None else self.data_cache.load(key=key)
        return dataloaders[self.measure_dataset]

    def get_repeats_dataloaders(self, key=None):
        raise NotImplementedError("Function to return the repeats-dataloader has to be implemented")

    def get_avg_of_unit_dict(self, unit_scores_dict):
        return np.mean(np.hstack([v for v in unit_scores_dict.values()]))

    def insert_unit_measures(self, key, unit_measures_dict):
        key = key.copy()
        for data_key, unit_scores in unit_measures_dict.items():
            for unit_index, unit_score in enumerate(unit_scores):
                if "unit_id" in key: key.pop("unit_id")
                if "data_key" in key: key.pop("data_key")
                neuron_key = dict(unit_index=unit_index, data_key=data_key)
                unit_id = ((self.unit_table & key) & neuron_key).fetch1("unit_id")
                key["unit_id"] = unit_id
                key["unit_{}".format(self.measure_attribute)] = unit_score
                key["data_key"] = data_key
                self.Units.insert1(key, ignore_extra_fields=True)

    def make(self, key):

        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(
            key=key)
        model = self.get_model(key=key)
        unit_measures_dict = self.measure_function(model=model,
                                                 dataloaders=dataloaders,
                                                 device='cuda',
                                                 as_dict=True,
                                                 per_neuron=True,
                                                 **self.function_kwargs)

        key[self.measure_attribute] = self.get_avg_of_unit_dict(unit_measures_dict)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_measures(key=key, unit_measures_dict=unit_measures_dict)


class SummaryScoringBase(ScoringBase):
    """
    A template scoring table with the same logic as ScoringBase, but for scores that do not have unit scores, but
    an overall score per model only.
    """
    unit_table = None
    Units = None

    def make(self, key):

        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(
            key=key)
        model = self.get_model(key=key)
        key[self.measure_attribute] = self.measure_function(model=model,
                                                            dataloaders=dataloaders,
                                                            device='cuda',
                                                            **self.function_kwargs)
        self.insert1(key, ignore_extra_fields=True)


class MeasuresBase(ScoringBase):
    trainedmodel_table = None
    dataset_table = Dataset

    # table level comment
    table_comment = "A template table for storing measures / descriptive statistics of the Dataset"

    @property
    def definition(self):
        definition = """
                    # {table_comment}
                    -> self.dataset_table
                    ---
                    {measure_attribute}:      float     # A template for a computed score of a trained model
                    {measure_attribute}_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
                    """.format(table_comment=self.table_comment, measure_attribute=self.measure_attribute)
        return definition

    class Units(dj.Part):
        @property
        def definition(self):
            definition = """
                # Scores for Individual Neurons
                -> master
                -> master.unit_table
                ---
                unit_{measure_attribute}:     float   # A template for a computed unit score        
                """.format(measure_attribute=self._master.measure_attribute)
            return definition

    def make(self, key):

        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(key=key)
        unit_measures_dict = self.measure_function(dataloaders=dataloaders,
                                                   as_dict=True,
                                                   per_neuron=True,
                                                   **self.function_kwargs)

        key[self.measure_attribute] = self.get_avg_of_unit_dict(unit_measures_dict)
        self.insert1(key, ignore_extra_fields=True)
        self.insert_unit_measures(key=key, unit_measures_dict=unit_measures_dict)


class SummaryMeasuresBase(MeasuresBase):
    unit_table = None
    Units = None

    # table level comment
    table_comment = "A template table for storing measures / descriptive statistics of the Dataset"

    def make(self, key):
        dataloaders = self.get_repeats_dataloaders(key=key) if self.measure_dataset == 'test' else self.get_dataloaders(key=key)
        key[self.measure_attribute] = self.measure_function(dataloaders=dataloaders,
                                                            **self.function_kwargs)
        self.insert1(key, ignore_extra_fields=True)


class TransferredTrainedModelBase(TrainedModelBase):
    """
    A modified version of TrainedModel table which enables step-wise table population given the
    step specification. Refer to the corresponding example notebook for a demo.
    """

    table_comment = "Transferred trained models"

    @property
    def definition(self):
        definition = """
        # {table_comment}
        transfer_step:                     int          # transfer step
        -> self.model_table
        -> self.dataset_table
        -> self.trainer_table
        -> self.seed_table
        prev_model_fn:                     varchar(64)
        prev_model_hash:                   varchar(64)
        prev_dataset_fn:                   varchar(64)
        prev_dataset_hash:                 varchar(64)
        prev_trainer_fn:                   varchar(64)
        prev_trainer_hash:                 varchar(64)
        ---
        comment='':                        varchar(768) # short description 
        score:                             float        # loss
        output:                            longblob     # trainer object's output
        ->[nullable] self.user_table
        trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
        current_model_fn:                  varchar(64)
        current_model_hash:                varchar(64)
        current_dataset_fn:                varchar(64)
        current_dataset_hash:              varchar(64)
        current_trainer_fn:                varchar(64)
        current_trainer_hash:              varchar(64)
        """.format(table_comment=self.table_comment)
        return definition

    class ModelStorage(TrainedModelBase.ModelStorage):
        pass

    def _transfer_recipe(self, transfer_step):
        """
        Combines multiple transfer recipes and their resitrictions as specified by post_restr attribute.
        The combination is transfer-step-specific, meaning only the recipes relevant for a specific transfer step would be combined.

        Combining recipes are pretty easy and the user does not need to interact with this method directly. Below is an example:
        Let us assume you have two recipe tables: TrainerRecipe and ModelRecipe, the you can attach all this recipes to your
        TransferTrainedModel table as follow:

        ``` Python
            TransferTrainedModel.transfer_recipe = [TrainerRecipe, ModelRecipe]
        ```

        The rest (combining the recipes and their restrictions) is taken care of by this method.

        Args:
            transfer_step (int): table population trasnfer step.

        Returns:
            string or datajoint AndList: A single or combined restriction of one or multiple recipes, respectively.
        """

        if isinstance(self.transfer_recipe, list):

            # get the recipes that have a entry for a specific transfer step
            # transfer_recipe = [tr & f"transfer_step = {transfer_step}" for tr in self.transfer_recipe if tr & f"transfer_step = {transfer_step}"]
            transfer_recipe = []
            # loop over the transfer recipes
            for tr in self.transfer_recipe:
                # check if an entry exists for a specific transfer step in the recipe
                if tr & f"transfer_step = {transfer_step}":
                    # if it exists add that entry to the list of recipes (relevant for a specific transfer step)
                    transfer_recipe.append(tr & f"transfer_step = {transfer_step}")

            # join all the recipes (and their post_restr)
            joined = transfer_recipe[0]

            if len(transfer_recipe) > 1:
                for t in transfer_recipe[1:]:
                    joined *= t

                joined.post_restr = dj.AndList([recipe.post_restr for recipe in self.transfer_recipe])

            return joined

        else:
            return self.transfer_recipe

    @property
    def key_source(self):

        if hasattr(self, "transfer_recipe"):

            # project (rename) attributes of the existing transfereedmodel table to the same name but with prefix "prev"
            prev_transferredmodel = self.proj(prev_model_fn='current_model_fn', prev_model_hash='current_model_hash',
                                               prev_dataset_fn='current_dataset_fn', prev_dataset_hash='current_dataset_hash',
                                               prev_trainer_fn='current_trainer_fn', prev_trainer_hash='current_trainer_hash',
                                               prev_step='transfer_step', transfer_step='transfer_step + 1') * dj.U('transfer_step', # make these attributes primary keys
                                                                                                                    'prev_model_fn', 'prev_model_hash',
                                                                                                                    'prev_dataset_fn', 'prev_dataset_hash',
                                                                                                                    'prev_trainer_fn', 'prev_trainer_hash')

            # get the current transfer step
            transfer_step = prev_transferredmodel.fetch('transfer_step').max() if prev_transferredmodel else 0

            if transfer_step:

                # get the necessay attributes to filter the prev_transferredmodel with the transfer recipe
                prev_transferredmodel = dj.U('transfer_step', 'prev_model_fn', 'prev_model_hash', 'prev_dataset_fn', 'prev_dataset_hash', 'prev_trainer_fn', 'prev_trainer_hash') & prev_transferredmodel

                # get the entries that match the one in TransferRecipe (for specification of previous)
                transfer_from = prev_transferredmodel * self._transfer_recipe(transfer_step)

                transfers = dj.U("transfer_step",
                                "model_fn", "model_hash",
                                "dataset_fn", "dataset_hash",
                                "trainer_fn", "trainer_hash",
                                "seed",
                                "prev_model_fn", "prev_model_hash",
                                "prev_dataset_fn", "prev_dataset_hash",
                                "prev_trainer_fn", "prev_trainer_hash") & Model * Dataset * Trainer * Seed * transfer_from & self._transfer_recipe(transfer_step).post_restr

                return transfers.proj()

            else:

                # set of models to be trained from scratch (I arbitrary chose to only train if trainer_id + model_id < 3)
                step_0 = Model * Dataset * Trainer * Seed

                # add transfer_step and prev_hash as prim keys
                base = dj.U('transfer_step',
                            'prev_model_fn', 'prev_model_hash',
                            'prev_dataset_fn', 'prev_dataset_hash',
                            'prev_trainer_fn', 'prev_trainer_hash') * step_0.proj(transfer_step='0',
                                                                                prev_model_fn='""', prev_model_hash='""',
                                                                                prev_dataset_fn='""', prev_dataset_hash='""',
                                                                                prev_trainer_fn='""', prev_trainer_hash='""')
                return base.proj()

        else:
            # normal entries as a combinatio of Dataset, Model, Trainer, and Seed tables
            step_0 = Model * Dataset * Trainer * Seed

            # add transfer_step and prev_hash as prim keys
            base = dj.U('transfer_step',
                        'prev_model_fn', 'prev_model_hash',
                        'prev_dataset_fn', 'prev_dataset_hash',
                        'prev_trainer_fn', 'prev_trainer_hash') * step_0.proj(transfer_step='0',
                                                                            prev_model_fn='""', prev_model_hash='""',
                                                                            prev_dataset_fn='""', prev_dataset_hash='""',
                                                                            prev_trainer_fn='""', prev_trainer_hash='""')
            return base.proj()


    def make(self, key):
        """
        Given key specifying configuration for dataloaders, model and trainer,
        trains the model and saves the trained model.
        """

        # lookup the fabrikant corresponding to the current DJ user
        fabrikant_name = Fabrikant.get_current_user()
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

            key['current_model_fn'], key['current_model_hash'] = (Model & key).fetch1('model_fn', 'model_hash')
            key['current_dataset_fn'], key['current_dataset_hash'] = (Dataset & key).fetch1('dataset_fn', 'dataset_hash')
            key['current_trainer_fn'], key['current_trainer_hash'] = (Trainer & key).fetch1('trainer_fn', 'trainer_hash')

            self.insert1(key)

            key['model_state'] = filepath

            self.ModelStorage.insert1(key, ignore_extra_fields=True)
