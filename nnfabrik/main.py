import warnings
import types
from typing import Union, Optional, MutableMapping, Tuple

import datajoint as dj

from .builder import (
    resolve_model,
    resolve_data,
    resolve_trainer,
    get_data,
    get_model,
    get_trainer,
)
from .utility.dj_helpers import make_hash, CustomSchema, Schema
from .utility.nnf_helper import cleanup_numpy_scalar

if "nnfabrik.schema_name" in dj.config:
    raise DeprecationWarning(
        "use of 'nnfabrik.schema_name' in dj.config is deprecated, use nnfabrik.main.my_nnfabrik function instead",
    )


class Fabrikant(dj.Manual):
    definition = """
    fabrikant_name: varchar(32)       # Name of the contributor that added this entry
    ---
    full_name="": varchar(128) # full name of the person
    email: varchar(64)         # e-mail address
    affiliation: varchar(32)   # conributor's affiliation (e.g. Sinz Lab)
    dj_username: varchar(64)   # DataJoint username
    """

    def add_entry(
        self,
        name,
        email,
        affiliation,
        full_name="",
        dj_username=None,
        skip_duplicates=False,
        return_pk_only=True,
    ):
        """
        Add a new entry into Fabrikant table. If `dj_username` is omitted, then the current
        database connection user is used.
        Args:
            name (str): A short name to identify yourself.
            email (str): Email address.
            affiliation (str): Lab or institutional affiliation.
            full_name (str, optional): Full name. Defaults to "".
            dj_username (str, optional): DataJoint username. Defaults to None, in which case
                the username of the current connection is used.
            skip_duplicates (bool, optional): If True, no error is thrown when a duplicate entry (i.e. entry with same model_fn and model_config) is found. Defaults to False.
            return_pk_only (bool, optional): If True, only the primary key attribute for the new entry or corresponding existing entry is returned. Otherwise, the entire
                entry is returned. Defaults to True.
        Returns:
            dict: the entry in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """
        if dj_username is None:
            dj_username = self.connection.get_user().split("@")[0]

        key = dict(
            fabrikant_name=name,
            full_name=full_name,
            email=email,
            affiliation=affiliation,
            dj_username=dj_username,
        )

        # overlap in DJ username is not allowed either
        existing = self.proj() & key or (self & dict(dj_username=dj_username)).proj()
        if existing:
            key = (self & (existing)).fetch1()
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
            else:
                raise ValueError("Corresponding entry already exists: {}".format(key))
        else:
            self.insert1(key, ignore_extra_fields=True)

        if return_pk_only:
            key = {k: key[k] for k in self.heading.primary_key}

        return key

    @classmethod
    def get_current_user(cls):
        """
        Lookup the fabrikant_name in Fabrikant corresponding to the currently logged in DataJoint user
        Returns: fabrikant_name if match found, else None
        """
        username = cls.connection.get_user().split("@")[0]
        entry = Fabrikant & dict(dj_username=username)
        if entry:
            return entry.fetch1("fabrikant_name")


class Model(dj.Manual):
    definition = """
    model_fn:                   varchar(64)   # name of the model function
    model_hash:                 varchar(64)   # hash of the model configuration
    ---
    model_config:               longblob      # model configuration to be passed into the function
    -> Fabrikant.proj(model_fabrikant='fabrikant_name')
    model_comment='' :          varchar(256)   # short description
    model_ts=CURRENT_TIMESTAMP: timestamp     # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        model_fn, model_config = self.fetch1("model_fn", "model_config")
        model_config = cleanup_numpy_scalar(model_config)
        return model_fn, model_config

    @staticmethod
    def resolve_fn(fn_name):
        return resolve_model(fn_name)

    def add_entry(
        self,
        model_fn,
        model_config,
        model_comment="",
        model_fabrikant=None,
        skip_duplicates=False,
        return_pk_only=True,
    ):
        """
        Add a new entry to the model.
        Args:
            model_fn (str, Callable): name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `models` subpackage.
            model_config (dict): Python dictionary containing keyword arguments for the model_fn
            model_comment - Optional comment for the entry.
            model_fabrikant (str): The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based on the database user name for the existing connection.
            skip_duplicates (bool, optional): If True, no error is thrown when a duplicate entry (i.e. entry with same model_fn and model_config) is found. Defaults to False.
            return_pk_only (bool, optional): If True, only the primary key attribute for the new entry or corresponding existing entry is returned. Otherwise, the entire
                entry is returned. Defaults to True.
        Returns:
            dict: the entry in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """
        if not isinstance(model_fn, str):
            # infer the full path to the callable
            model_fn = model_fn.__module__ + "." + model_fn.__name__

        try:
            resolve_model(model_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        if model_fabrikant is None:
            model_fabrikant = Fabrikant.get_current_user()

        model_hash = make_hash(model_config)
        key = dict(
            model_fn=model_fn,
            model_hash=model_hash,
            model_config=model_config,
            model_fabrikant=model_fabrikant,
            model_comment=model_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        if return_pk_only:
            key = {k: key[k] for k in self.heading.primary_key}

        return key

    def build_model(self, dataloaders=None, seed=None, key=None, data_info=None):
        """
        Builds a Pytorch module by calling the model_fn with the corresponding model_config. The table has to be
        restricted to one entry in order for this method to return a model.
        Either the dataloaders or data_info have to be specified to determine the size of the input and thus the
        appropriate model settings.
        Args:
            dataloaders (dict) -  a dictionary of dataloaders. The model will infer its shape from these dataloaders
            seed (int) -  random seed
            key (dict) - datajoint key
            data_info (dict) - contains all necessary information about the input in order to build the model.
        Returns:
            A PyTorch module.
        """
        if dataloaders is None and data_info is None:
            raise ValueError(
                "dataloaders and data_info can not both be None. To build the model, either dataloaders or"
                "data_info have to be passed."
            )

        print("Building model...")
        if key is None:
            key = {}
        model_fn, model_config = (self & key).fn_config

        return get_model(
            model_fn,
            model_config,
            dataloaders=dataloaders,
            seed=seed,
            data_info=data_info,
        )


class Dataset(dj.Manual):
    definition = """
    dataset_fn:                     varchar(64)    # name of the dataset loader function
    dataset_hash:                   varchar(64)    # hash of the configuration object
    ---
    dataset_config:                 longblob       # dataset configuration object
    -> Fabrikant.proj(dataset_fabrikant='fabrikant_name')
    dataset_comment='' :            varchar(256)    # short description
    dataset_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        dataset_fn, dataset_config = self.fetch1("dataset_fn", "dataset_config")
        dataset_config = cleanup_numpy_scalar(dataset_config)
        return dataset_fn, dataset_config

    @staticmethod
    def resolve_fn(fn_name):
        return resolve_data(fn_name)

    def add_entry(
        self,
        dataset_fn,
        dataset_config,
        dataset_comment="",
        dataset_fabrikant=None,
        skip_duplicates=False,
        return_pk_only=True,
    ):
        """
        Add a new entry to the dataset.
        Args:
            dataset_fn (string, Callable): name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `models` subpackage.
            dataset_config (dict): Python dictionary containing keyword arguments for the dataset_fn
            dataset_comment (str, optional):  Comment for the entry. Defaults to "" (emptry string)
            dataset_fabrikant (string): The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based
                on the database user name for the existing connection.
            skip_duplicates (bool, optional): If True, no error is thrown when a duplicate entry (i.e. entry with same model_fn and model_config) is found. Defaults to False.
            return_pk_only (bool, optional): If True, only the primary key attribute for the new entry or corresponding existing entry is returned. Otherwise, the entire
                entry is returned. Defaults to True.
        Returns:
            dict: the entry in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """
        if not isinstance(dataset_fn, str):
            # infer the full path to the callable
            dataset_fn = dataset_fn.__module__ + "." + dataset_fn.__name__

        try:
            resolve_data(dataset_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        if dataset_fabrikant is None:
            dataset_fabrikant = Fabrikant.get_current_user()

        dataset_hash = make_hash(dataset_config)
        key = dict(
            dataset_fn=dataset_fn,
            dataset_hash=dataset_hash,
            dataset_config=dataset_config,
            dataset_fabrikant=dataset_fabrikant,
            dataset_comment=dataset_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        if return_pk_only:
            key = {k: key[k] for k in self.heading.primary_key}

        return key

    def get_dataloader(self, seed=None, key=None):
        """
        Returns a dataloader for a given dataset loader function and its corresponding configurations
        dataloader: is expected to be a dict in the form of
                            {
                            'train': torch.utils.data.DataLoader,
                            'val': torch.utils.data.DataLoader,
                            'test: torch.utils.data.DataLoader,
                             }
                             or a similar iterable object
                each loader should have as first argument the input such that
                    next(iter(train_loader)): [input, responses, ...]
                the input should have the following form:
                    [batch_size, channels, px_x, px_y, ...]
        """
        # TODO: update the docstring

        if key is None:
            key = {}

        dataset_fn, dataset_config = (self & key).fn_config

        if seed is not None:
            dataset_config["seed"] = seed  # override the seed if passed in

        return get_data(dataset_fn, dataset_config)


class Trainer(dj.Manual):
    definition = """
    trainer_fn:                     varchar(64)     # name of the Trainer loader function
    trainer_hash:                   varchar(64)     # hash of the configuration object
    ---
    trainer_config:                 longblob        # training configuration object
    -> Fabrikant.proj(trainer_fabrikant='fabrikant_name')
    trainer_comment='' :            varchar(256)     # short description
    trainer_ts=CURRENT_TIMESTAMP:   timestamp       # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        trainer_fn, trainer_config = self.fetch1("trainer_fn", "trainer_config")
        trainer_config = cleanup_numpy_scalar(trainer_config)
        return trainer_fn, trainer_config

    @staticmethod
    def resolve_fn(fn_name):
        return resolve_trainer(fn_name)

    def add_entry(
        self,
        trainer_fn,
        trainer_config,
        trainer_comment="",
        trainer_fabrikant=None,
        skip_duplicates=False,
        return_pk_only=True,
    ):
        """
        Add a new entry to the trainer.
        Args:
            trainer_fn (str): name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
                dynamic name resolution will be attempted. Other wise, the name will be checked inside `models` subpackage.
            trainer_config (dict): Python dictionary containing keyword arguments for the trainer_fn.
            trainer_comment (str, optional): Optional comment for the entry. Defaults to "" (empty string).
            trainer_fabrikant (str): The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based
                on the database user name for the existing connection.
            skip_duplicates (bool, optional): If True, no error is thrown when a duplicate entry (i.e. entry with same model_fn and model_config) is found. Defaults to False.
            return_pk_only (bool, optional): If True, only the primary key attribute for the new entry or corresponding existing entry is returned. Otherwise, the entire
                entry is returned. Defaults to True.
        Returns:
            dict: the entry in the table corresponding to the new (or possibly existing, if skip_duplicates=True) entry.
        """
        if not isinstance(trainer_fn, str):
            # infer the full path to the callable
            trainer_fn = trainer_fn.__module__ + "." + trainer_fn.__name__

        try:
            resolve_trainer(trainer_fn)
        except (NameError, TypeError) as e:
            warnings.warn(str(e) + "\nTable entry rejected")
            return

        if trainer_fabrikant is None:
            trainer_fabrikant = Fabrikant.get_current_user()

        trainer_hash = make_hash(trainer_config)
        key = dict(
            trainer_fn=trainer_fn,
            trainer_hash=trainer_hash,
            trainer_config=trainer_config,
            trainer_fabrikant=trainer_fabrikant,
            trainer_comment=trainer_comment,
        )

        existing = self.proj() & key
        if existing:
            if skip_duplicates:
                warnings.warn("Corresponding entry found. Skipping...")
                key = (self & (existing)).fetch1()
            else:
                raise ValueError("Corresponding entry already exists")
        else:
            self.insert1(key)

        if return_pk_only:
            key = {k: key[k] for k in self.heading.primary_key}

        return key

    def get_trainer(self, key=None, build_partial=True):
        """
        Returns the trainer function and its corresponding configurations. If build_partial=True (default), then it constructs
        a partial function with configuration object already passed in, thus returning only a single function.
        """
        if key is None:
            key = {}
        trainer_fn, trainer_config = (self & key).fn_config

        if build_partial:
            # build the configuration into the function
            return get_trainer(trainer_fn, trainer_config)
        else:
            # return them separately
            return get_trainer(trainer_fn), trainer_config


class Seed(dj.Manual):
    definition = """
    seed:   int     # Random seed that is passed to the model- and dataset-builder
    """


class Experiments(dj.Manual):
    # Table to keep track of collections of trained networks that form an experiment.
    # Instructions:
    # 1) Make an entry in Experiments with an experiment name and description
    # 2) Insert all combinations of dataset, model and trainer for this experiment name in Experiments.Restrictions
    # 2) Populate the TrainedModel table by restricting it with Experiments.Restrictions and the experiment name
    # 3) After training, join this table with TrainedModel and restrict by experiment name to get your results
    # 4) An example notebook can be found here: https://github.com/sinzlab/nnfabrik/tree/master/nnfabrik/examples/notebooks
    definition = """
    # This table contains the experiments and their descriptions
    experiment_name: varchar(100)                     # name of experiment
    ---
    -> Fabrikant.proj(experiment_fabrikant='fabrikant_name')
    experiment_comment='': varchar(2000)              # short description
    experiment_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    class Restrictions(dj.Part):
        definition = """
        # This table contains the corresponding hashes to filter out models which form the respective experiment
        -> master
        -> Dataset
        -> Trainer
        -> Model
        ---
        experiment_restriction_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
        """

    def add_entry(
        self,
        experiment_name,
        experiment_fabrikant,
        experiment_comment,
        restrictions,
        skip_duplicates=False,
    ):
        self.insert1(
            dict(
                experiment_name=experiment_name,
                experiment_fabrikant=experiment_fabrikant,
                experiment_comment=experiment_comment,
            ),
            skip_duplicates=skip_duplicates,
        )

        restrictions = [{**{"experiment_name": experiment_name}, **res} for res in restrictions]
        self.Restrictions.insert(restrictions, skip_duplicates=skip_duplicates)


def my_nnfabrik(
    schema: Union[str, Schema],
    additional_tables: Tuple = (),
    module_name: Optional[str] = None,
    context: Optional[MutableMapping] = None,
    spawn_existing_tables: bool = False,
) -> Optional[types.ModuleType]:
    """
    Create a custom nnfabrik module under specified DataJoint schema,
    instantitaing Model, Dataset, and Trainer tables. If `use_common_fabrikant`
    is set to True, the new tables will depend on the common Fabrikant table.
    Otherwise, a separate copy of Fabrikant table will also be prepared.
    Examples:
        Use of this function should replace any existing use of `nnfabrik` tables done via modifying the
        `nnfabrik.schema_name` property in `dj.config`.
        As an example, if you previously had a code like this:
        >>> dj.config['nfabrik.schema_name'] = 'my_schema'
        >>> from nnfabrik import main # importing nnfabrik tables
        do this instead:
        >>> from nnfabrik.main import my_nnfabrik
        >>> main = my_nnfabrik('my_schema')    # this has the same effect as defining nnfabrik tables in schema `my_schema`
        Also, you can achieve the equivalent of:
        >>> dj.config['nfabrik.schema_name'] = 'my_schema'
        >>> from nnfabrik.main import *
        by doing
        >>> from nnfabrik.main import my_nnfabrik
        >>> my_nnfabrik('my_schema', context=locals())
    Args:
        schema (str or dj.Schema): Name of schema or dj.Schema object
        use_common_fabrikant (bool, optional): If True, new tables will depend on the
           common Fabrikant table. If False, new copy of Fabrikant will be created and used.
           Defaults to True.
        use_common_seed (bool, optional): If True, new tables will depend on the
           common Seed table. If False, new copy of Seed will be created and used.
           Defaults to False.
        module_name (str, optional): Name property of the returned Python module object.
            Defaults to None, in which case the name of the schema will be used.
        context (dict, optional): If non None value is provided, then a module is not created and
            instead the tables are defined inside the context.
        spawn_existing_tables (bool, optional): If True, perform `spawn_missing_tables` operation
            onto the newly created table. Defaults to False.
    Raises:
        ValueError: If `use_common_fabrikant` is True but the target `schema` already contains its own
            copy of `Fabrikant` table, or if `use_common_seed` is True but the target `schema` already
            contains its own copy of `Seed` table.
    Returns:
        Python Module object or None: If `context` was None, a new Python module containing
            nnfabrik tables defined under the schema. The module's schema property points
            to the schema object as well. Otherwise, nothing is returned.
    """
    if isinstance(schema, str):
        schema = CustomSchema(schema)

    tables = [Seed, Fabrikant, Model, Dataset, Trainer] + list(additional_tables)

    module = None
    if context is None:
        module_name = schema.database if module_name is None else module_name
        module = types.ModuleType(module_name)
        context = module.__dict__

    context["schema"] = schema

    # spawn all existing tables into the module
    # TODO: replace with a cheaper check operation
    temp_context = context if spawn_existing_tables else {}
    schema.spawn_missing_classes(temp_context)

    if use_common_fabrikant:
        if "Fabrikant" in temp_context:
            raise ValueError(
                "The schema already contains a Fabrikant table despite setting use_common_fabrikant=True. "
                "Either rerun with use_common_fabrikant=False or remove the Fabrikant table in the schema"
            )
        context["Fabrikant"] = Fabrikant
        # skip creating Fabrikant table
        tables.remove(Fabrikant)

    if use_common_seed:
        if "Seed" in temp_context:
            raise ValueError(
                "The schema already contains a Seed table despite setting use_common_seed=True. "
                "Either rerun with use_common_seed=False or remove the Seed table in the schema"
            )
        context["Seed"] = Seed
        # skip creating Seed table
        tables.remove(Seed)

    for table in tables:
        new_table = type(table.__name__, (table,), dict(__doc__=table.__doc__))
        context[table.__name__] = schema(new_table, context=context)

    # this returns None if context was set
    return (
        module
        if not return_main_tables
        else [module_name, map(module.__dict__.get, ["Fabrikant", "Dataset", "Model", "Trainer", "Seed"])]
    )
