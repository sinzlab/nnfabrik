import torch
import tempfile
import warnings
import os
import datajoint as dj

from .builder import resolve_model, resolve_data, resolve_trainer, get_data, get_model, get_trainer, get_all_parts
from .utility.dj_helpers import make_hash, check_repo_commit
from .utility.nnf_helper import split_module_name, dynamic_import, cleanup_numpy_scalar


# set external store based on env vars
dj.config['stores'] = {
    'minio': {    #  store in s3
        'protocol': 's3',
        'endpoint': 'cantor.mvl6.uni-tuebingen.de:9000',
        'bucket': 'nnfabrik',
        'location': 'dj-store',
        'access_key': os.environ.get('MINIO_ACCESS_KEY', 'FAKEKEY'),
        'secret_key': os.environ.get('MINIO_SECRET_KEY', 'FAKEKEY')
    }
}


# check if schema_name defined, otherwise default to nnfabrik_core
schema = dj.schema(dj.config.get('schema_name', 'nnfabrik_core'))


@schema
class Fabrikant(dj.Manual):
    definition = """
    fabrikant_name: varchar(32)       # Name of the contributor that added this entry
    ---
    email: varchar(64)      # e-mail address
    affiliation: varchar(32) # conributor's affiliation
    dj_username: varchar(64) # DataJoint username
    """

    @classmethod
    def get_current_user(cls):
        """
        Lookup the fabrikant_name in Fabrikant corresponding to the currently logged in DataJoint user
        Returns: fabrikant_name if match found, else None
        """
        username = cls.connection.get_user().split('@')[0]
        entry = (Fabrikant & dict(dj_username=username))
        if entry:
            return entry.fetch1('fabrikant_name')


@schema
class Model(dj.Manual):
    definition = """
    model_fn:                   varchar(64)   # name of the model function
    model_hash:                 varchar(64)   # hash of the model configuration
    ---
    model_config:               longblob      # model configuration to be passed into the function
    -> Fabrikant.proj(model_fabrikant='fabrikant_name')
    model_comment='' :          varchar(64)   # short description
    model_ts=CURRENT_TIMESTAMP: timestamp     # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        model_fn, model_config = self.fetch1('model_fn', 'model_config')
        model_config = cleanup_numpy_scalar(model_config)
        return model_fn, model_config

    def add_entry(self, model_fn, model_config, model_fabrikant=None, model_comment=''):
        """
        Add a new entry to the model.
        model_fn (string) - name of a callable object. If name contains multiple parts separated by `.`, this is assumed to be found in a another module and
            dynamic name resolution will be attempted. Other wise, the name will be checked inside `models` subpackage.
        model_config (dict) - Python dictionary containing keyword arguments for the model_fn
        model_fabrikant (string) - The fabrikant name. Must match an existing entry in Fabrikant table. If ignored, will attempt to resolve Fabrikant based on the database user name for the existing connection.
        model_comment - Optional comment for the entry.
        """
        try:
            resolve_model(model_fn)
        except NameError, TypeError as e:
            warnings.warn(str(e) + '\nTable entry rejected')
            return

        if model_fabrikant is None:
            model_fabrikant = Fabrikant.get_current_user()

        model_hash = make_hash(model_config)
        key = dict(model_fn=model_fn, model_hash=model_hash, model_config=model_config,
                   model_fabrikant=model_fabrikant, model_comment=model_comment)
        self.insert1(key)

    def build_model(self, dataloaders, seed=None, key=None):
        print('Loading model...')
        if key is None:
            key = {}
        model_fn, model_config = (self & key).fn_config

        return get_model(model_fn, model_config, dataloaders, seed=seed)


@schema
class Dataset(dj.Manual):
    definition = """
    dataset_fn:                     varchar(64)    # name of the dataset loader function
    dataset_hash:                   varchar(64)    # hash of the configuration object
    ---
    dataset_config:                 longblob       # dataset configuration object
    -> Fabrikant.proj(dataset_fabrikant='fabrikant_name')
    dataset_comment='' :            varchar(64)    # short description
    dataset_ts=CURRENT_TIMESTAMP:   timestamp      # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        dataset_fn, dataset_config = self.fetch1('dataset_fn', 'dataset_config')
        dataset_config = cleanup_numpy_scalar(dataset_config)
        return dataset_fn, dataset_config

    def add_entry(self, dataset_fn, dataset_config, dataset_fabrikant=None, dataset_comment=''):
        """
        inserts one new entry into the Dataset Table
        dataset_fn -- name of dataset function/class that's callable
        dataset_config -- actual Python object with which the dataset function is called
        """

        try:
            resolve_data(dataset_fn)
        except NameError, TypeError as e:
            warnings.warn(str(e) + '\nTable entry rejected')
            return

        if dataset_fabrikant is None:
            dataset_fabrikant = Fabrikant.get_current_user()

        dataset_hash = make_hash(dataset_config)
        key = dict(dataset_fn=dataset_fn, dataset_hash=dataset_hash,
                   dataset_config=dataset_config, dataset_fabrikant=dataset_fabrikant, dataset_comment=dataset_comment)
        self.insert1(key)

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
        #TODO: update the docstring

        if key is None:
            key = {}

        dataset_fn, dataset_config = (self & key).fn_config

        if seed is not None:
            dataset_config['seed'] = seed # override the seed if passed in

        return get_data(dataset_fn, dataset_config)


@schema
class Trainer(dj.Manual):
    definition = """
    trainer_fn:                     varchar(64)     # name of the Trainer loader function
    trainer_hash:                   varchar(64)     # hash of the configuration object
    ---
    trainer_config:                 longblob        # training configuration object
    -> Fabrikant.proj(trainer_fabrikant='fabrikant_name')
    trainer_comment='' :            varchar(64)     # short description
    trainer_ts=CURRENT_TIMESTAMP:   timestamp       # UTZ timestamp at time of insertion
    """

    @property
    def fn_config(self):
        trainer_fn, trainer_config = self.fetch1('trainer_fn', 'trainer_config')
        trainer_config = cleanup_numpy_scalar(trainer_config)
        return trainer_fn, trainer_config

    def add_entry(self, trainer_fn, trainer_config, trainer_fabrikant=None, trainer_comment=''):
        """
        inserts one new entry into the Trainer Table
        trainer_fn -- name of trainer function/class that's callable
        trainer_config -- actual Python object with which the trainer function is called
        """
        try:
            resolve_trainer(trainer_fn)
        except NameError, TypeError as e:
            warnings.warn(str(e) + '\nTable entry rejected')
            return

        if trainer_fabrikant is None:
            trainer_fabrikant = Fabrikant.get_current_user()

        trainer_hash = make_hash(trainer_config)
        key = dict(trainer_fn=trainer_fn, trainer_hash=trainer_hash,
                   trainer_config=trainer_config, trainer_fabrikant=trainer_fabrikant,
                   trainer_comment=trainer_comment)
        self.insert1(key)

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
    score:                             float        # loss
    output:                            longblob     # trainer object's output
    ->[nullable] Fabrikant
    trainedmodel_ts=CURRENT_TIMESTAMP: timestamp    # UTZ timestamp at time of insertion
    """

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

    
    def get_full_config(self, key=None, include_state_dict=True, skip_trainer=False):
        if key is None:
            key = self.fetch1('KEY')

        model_fn, model_config = (Model & key).fn_config
        dataset_fn, dataset_config = (Dataset & key).fn_config
        

        ret = dict(model_fn=model_fn, model_config=model_config,
                   dataset_fn=dataset_fn, dataset_config=dataset_config)

        if not skip_trainer:
            trainer_fn, trainer_config = (Trainer & key).fn_config
            ret['trainer_fn'] = trainer_fn
            ret['trainer_config'] = trainer_config

        # if trained model exist and include_state_dict is True
        if include_state_dict and (self.ModelStorage & key):
            with tempfile.TemporaryDirectory() as temp_dir:
                state_dict_path = (self.ModelStorage & key).fetch1('model_state', download_path=temp_dir)
                ret['state_dict'] = torch.load(state_dict_path)

        return ret

    def load_model(self, key=None):
        """
        Load a single entry of the model. If state_dict is available, the model will be loaded with state_dict as well.
        """
        if key is None:
            key = self.fetch1('KEY')

        seed = (Seed & key).fetch1('seed')
        config_dict = self.get_full_config(key, skip_trainer=True)
        dataloaders, model = get_all_parts(**config_dict, seed=seed)
        return dataloaders, model


    def make(self, key):

        commits_info = {name: info for name, info in [check_repo_commit(repo) for repo in config['repos']]}
        assert len(commits_info) == len(config['repos'])

        if any(['error_msg' in name for name in commits_info.keys()]):
            err_msgs = ["You have uncommited changes."]
            err_msgs.extend([info for name, info in commits_info.items() if 'error_msg' in name])
            err_msgs.append("\nPlease commit the changes before running populate.\n")
            raise RuntimeError('\n'.join(err_msgs))
        else:

            # lookup the fabrikant corresponding to the current DJ user
            fabrikant_name = Fabrikant.get_current_user()
            seed = (Seed & key).fetch1('seed')

            config_dict = self.get_full_config(key, include_state_dict=False)
            dataloaders, model, trainer = get_all_parts(**config_dict, seed=seed)

            # model training
            score, output, model_state = trainer(model, seed, dataloaders)

            with tempfile.TemporaryDirectory() as temp_dir:
                filename = make_hash(key) + '.pth.tar'
                filepath = os.path.join(temp_dir, filename)
                torch.save(model_state, filepath)

                key['score'] = score
                key['output'] = output
                key['fabrikant_name'] = fabrikant_name
                self.insert1(key)

                key['model_state'] = filepath
                self.ModelStorage.insert1(key, ignore_extra_fields=True)

                # add the git info to the part table
                if commits_info:
                    key['info'] = commits_info
                    self.GitLog().insert1(key, skip_duplicates=True, ignore_extra_fields=True)