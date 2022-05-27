import numpy as np
from scipy.stats import loguniform
from ax.service.managed_loop import optimize
from .nnf_helper import split_module_name, dynamic_import
from nnfabrik.main import *
import datajoint as dj


class Bayesian:
    """
    A hyperparameter optimization tool based on Facebook Ax (https://ax.dev/), integrated with nnfabrik.
    This tool, iteratively, optimizes for hyperparameters that improve a specific score representing
    model performance (the same score used in TrainedModel table). In every iteration (after every training),
    it automatically adds an entry to the corresponding tables, and populated the trained model table (i.e.
    trains the model) for that specific entry.

    Args:
        dataset_fn (str): name of the dataset function
        dataset_config (dict): dictionary of arguments for dataset function that are fixed
        dataset_config_auto (dict): dictionary of arguments for dataset function that are to be optimized
        model_fn (str): name of the model function
        model_config (dict): dictionary of arguments for model function that are fixed
        model_config_auto (dict): dictionary of arguments for model function that are to be optimized
        trainer_fn (str): name of the trainer function
        trainer_config (dict): dictionary of arguments for trainer function that are fixed
        trainer_config_auto (dict): dictionary of arguments for trainer function that are to be optimized
        architect (str): Name of the contributor that added this entry
        trained_model_table (str): name (importable) of the trained_model_table
        total_trials (int, optional): Number of experiments (i.e. training) to run. Defaults to 5.
        arms_per_trial (int, optional): Number of different configurations used for training (for more details check https://ax.dev/docs/glossary.html#trial). Defaults to 1.
        comment (str, optional): Comments about this optimization round. It will be used to fill up the comment entry of dataset, model, and trainer table. Defaults to "Bayesian optimization of Hyper params.".
    """

    def __init__(
        self,
        dataset_fn,
        dataset_config,
        dataset_config_auto,
        model_fn,
        model_config,
        model_config_auto,
        trainer_fn,
        trainer_config,
        trainer_config_auto,
        architect,
        trained_model_table,
        total_trials=5,
        arms_per_trial=1,
        comment="Bayesian optimization of Hyper params.",
    ):

        self.fns = dict(dataset=dataset_fn, model=model_fn, trainer=trainer_fn)
        self.fixed_params = self.get_fixed_params(dataset_config, model_config, trainer_config)
        self.auto_params = self.get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto)
        self.architect = architect
        self.total_trials = total_trials
        self.arms_per_trial = arms_per_trial
        self.comment = comment

        # import TrainedModel definition
        module_path, class_name = split_module_name(trained_model_table)
        self.trained_model_table = dynamic_import(module_path, class_name)

    @staticmethod
    def get_fixed_params(dataset_config, model_config, trainer_config):
        """
        Returns a single dictionary including the fixed parameters for dataset, model, and trainer.

        Args:
            dataset_config (dict): dictionary of arguments for dataset function that are fixed
            model_config (dict): dictionary of arguments for model function that are fixed
            trainer_config (dict): dictionary of arguments for trainer function that are fixed

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values
            are the corresponding dictionary of fixed arguments.
        """
        return dict(dataset=dataset_config, model=model_config, trainer=trainer_config)

    @staticmethod
    def get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto):
        """
        Changes the parameters to be optimized to a ax-friendly format, i.e. a list of dictionaries where
        each entry of the list corresponds to a single parameter.

        Ax requires a list of parameters (to be optimized) including the name and other specifications.
        Here we provide that list while keeping the arguments still separated by adding "dataset",
        "model", or "trainer" to the beginning of the parameter name.

        Args:
            dataset_config_auto (dict): dictionary of arguments for dataset function that are to be optimized
            model_config_auto (dict): dictionary of arguments for model function that are to be optimized
            trainer_config_auto (dict): dictionary of arguments for trainer function that are to be optimized

        Returns:
            list: list of dictionaries where each dictionary specifies a single parameter to be optimized.
        """
        dataset_params = []
        for k, v in dataset_config_auto.items():
            dd = {"name": "dataset.{}".format(k)}
            dd.update(v)
            dataset_params.append(dd)

        model_params = []
        for k, v in model_config_auto.items():
            dd = {"name": "model.{}".format(k)}
            dd.update(v)
            model_params.append(dd)

        trainer_params = []
        for k, v in trainer_config_auto.items():
            dd = {"name": "trainer.{}".format(k)}
            dd.update(v)
            trainer_params.append(dd)

        return dataset_params + model_params + trainer_params

    @staticmethod
    def _combine_params(auto_params, fixed_params):
        """
        Combining the auto (to-be-optimized) and fixed parameters (to have a single object representing all the arguments
        used for a specific function)

        Args:
            auto_params (dict): dictionary of to-be-optimized parameters, i.e. A dictionary of dictionaries where keys are
            dataset, model, and trainer and the values are the corresponding dictionary of to-be-optimized arguments.
            fixed_params (dict): dictionary of fixed parameters, i.e. A dictionary of dictionaries where keys are dataset,
            model, and trainer and the values are the corresponding dictionary of fixed arguments.

        Returns:
            dict: dictionary of parameters (fixed and to-be-optimized), i.e. A dictionary of dictionaries where keys are
            dataset, model, and trainer and the values are the corresponding dictionary of arguments.
        """
        keys = ["dataset", "model", "trainer"]
        params = {}
        for key in keys:
            params[key] = fixed_params[key]
            params[key].update(auto_params[key])

        return {key: params[key] for key in keys}

    @staticmethod
    def _split_config(params):
        """
        Reverses the operation of `get_auto_params` (from ax-friendly format to a dictionary of dictionaries where keys are
        dataset, model, and trainer and the values are a dictionary of the corresponding arguments)

        Args:
            params (dict): dictionary of dictionaries where each dictionary specifies a single parameter to be optimized.

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values are the
            corresponding dictionary of to-be-optimized arguments.
        """
        config = dict(dataset={}, model={}, trainer={}, others={})
        for k, v in params.items():
            config[k.split(".")[0]][k.split(".")[1]] = v

        return config

    def train_evaluate(self, auto_params):
        """
        For a given set of parameters, add an entry to the corresponding tables, and populate the trained model
        table for that specific entry.

        Args:
            auto_params (dict): dictionary of dictionaries where each dictionary specifies a single parameter to be optimized.

        Returns:
            float: the score of the trained model for the specific entry in trained model table
        """
        config = self._combine_params(self._split_config(auto_params), self.fixed_params)

        # insert the stuff into their corresponding tables
        dataset_hash = make_hash(config["dataset"])
        entry_exists = {
            "dataset_fn": "{}".format(self.fns["dataset"])
        } in self.trained_model_table().dataset_table() and {
            "dataset_hash": "{}".format(dataset_hash)
        } in self.trained_model_table().dataset_table()
        if not entry_exists:
            self.trained_model_table().dataset_table().add_entry(
                self.fns["dataset"],
                config["dataset"],
                dataset_fabrikant=self.architect,
                dataset_comment=self.comment,
            )

        model_hash = make_hash(config["model"])
        entry_exists = {"model_fn": "{}".format(self.fns["model"])} in self.trained_model_table().model_table() and {
            "model_hash": "{}".format(model_hash)
        } in self.trained_model_table().model_table()
        if not entry_exists:
            self.trained_model_table().model_table().add_entry(
                self.fns["model"],
                config["model"],
                model_fabrikant=self.architect,
                model_comment=self.comment,
            )

        trainer_hash = make_hash(config["trainer"])
        entry_exists = {
            "trainer_fn": "{}".format(self.fns["trainer"])
        } in self.trained_model_table().trainer_table() and {
            "trainer_hash": "{}".format(trainer_hash)
        } in self.trained_model_table().trainer_table()
        if not entry_exists:
            self.trained_model_table().trainer_table().add_entry(
                self.fns["trainer"],
                config["trainer"],
                trainer_fabrikant=self.architect,
                trainer_comment=self.comment,
            )

        # get the primary key values for all those entries
        restriction = (
            'dataset_fn in ("{}")'.format(self.fns["dataset"]),
            'dataset_hash in ("{}")'.format(dataset_hash),
            'model_fn in ("{}")'.format(self.fns["model"]),
            'model_hash in ("{}")'.format(model_hash),
            'trainer_fn in ("{}")'.format(self.fns["trainer"]),
            'trainer_hash in ("{}")'.format(trainer_hash),
        )

        # populate the table for those primary keys
        self.trained_model_table().populate(*restriction)

        # get the score of the model for this specific set of hyperparameters
        score = (self.trained_model_table() & dj.AndList(restriction)).fetch("score")[0]

        return score

    def run(self):
        """
        Runs Bayesian optimization.

        Returns:
            tuple: The returned values are similar to that of Ax (refer to https://ax.dev/docs/api.html)
        """
        best_parameters, values, experiment, model = optimize(
            parameters=self.auto_params,
            evaluation_function=self.train_evaluate,
            objective_name="val_corr",
            minimize=False,
            total_trials=self.total_trials,
            arms_per_trial=self.arms_per_trial,
        )

        return self._split_config(best_parameters), values, experiment, model


class Random:
    """
    Random hyperparameter search, integrated with nnfabrik.
    Similar to Bayesian optimization tool, but instead of optimizing for hyperparameters to maximize a score,
    in every iteration (after every training), it randomly samples new value for the specified parameters, adds an
    entry to the corresponding tables, and populated the trained model table (i.e. trains the model) for that specific entry.

    Args:
        dataset_fn (str): name of the dataset function
        dataset_config (dict): dictionary of arguments for dataset function that are fixed
        dataset_config_auto (dict): dictionary of arguments for dataset function that are to be randomly sampled
        model_fn (str): name of the model function
        model_config (dict): dictionary of arguments for model function that are fixed
        model_config_auto (dict): dictionary of arguments for model function that are to be randomly sampled
        trainer_fn (str): name of the trainer function
        trainer_config (dict): dictionary of arguments for trainer function that are fixed
        trainer_config_auto (dict): dictionary of arguments for trainer function that are to be randomly sampled
        architect (str): Name of the contributor that added this entry
        trained_model_table (str): name (importable) of the trained_model_table
        total_trials (int, optional): Number of experiments (i.e. training) to run. Defaults to 5.
        comment (str, optional): Comments about this optimization round. It will be used to fill up the comment entry of dataset, model, and trainer table. Defaults to "Bayesian optimization of Hyper params.".
    """

    def __init__(
        self,
        dataset_fn,
        dataset_config,
        dataset_config_auto,
        model_fn,
        model_config,
        model_config_auto,
        trainer_fn,
        trainer_config,
        trainer_config_auto,
        seed_config_auto,
        architect,
        trained_model_table,
        total_trials=5,
        comment="Random search for hyper params.",
    ):

        self.fns = dict(dataset=dataset_fn, model=model_fn, trainer=trainer_fn)
        self.fixed_params = self.get_fixed_params(dataset_config, model_config, trainer_config)
        self.auto_params = self.get_auto_params(
            dataset_config_auto, model_config_auto, trainer_config_auto, seed_config_auto
        )
        self.architect = architect
        self.total_trials = total_trials
        self.comment = comment

        # import TrainedModel definition
        module_path, class_name = split_module_name(trained_model_table)
        self.trained_model_table = dynamic_import(module_path, class_name)

    @staticmethod
    def get_fixed_params(dataset_config, model_config, trainer_config):
        """
        Returs a single dictionary including the fixed parameters for dataset, model, and trainer.

        Args:
            dataset_config (dict): dictionary of arguments for dataset function that are fixed
            model_config (dict): dictionary of arguments for model function that are fixed
            trainer_config (dict): dictionary of arguments for trainer function that are fixed

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values are the corresponding
            dictionary of fixed arguments.
        """
        return dict(dataset=dataset_config, model=model_config, trainer=trainer_config)

    @staticmethod
    def get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto, seed_config_auto):
        """
        Returns the parameters, which are to be randomly sampled, in a list.
        Here we followed the same convention as in the Bayesian class, to have the API as similar as possible.

        Args:
            dataset_config_auto (dict): dictionary of arguments for dataset function that are to be randomly sampled
            model_config_auto (dict): dictionary of arguments for model function that are to be randomly sampled
            trainer_config_auto (dict): dictionary of arguments for trainer function that are to be randomly sampled

        Returns:
            list: list of dictionaries where each dictionary specifies a single parameter to be randomly sampled.
        """
        dataset_params = []
        for k, v in dataset_config_auto.items():
            dd = {"name": "dataset.{}".format(k)}
            dd.update(v)
            dataset_params.append(dd)

        model_params = []
        for k, v in model_config_auto.items():
            dd = {"name": "model.{}".format(k)}
            dd.update(v)
            model_params.append(dd)

        trainer_params = []
        for k, v in trainer_config_auto.items():
            dd = {"name": "trainer.{}".format(k)}
            dd.update(v)
            trainer_params.append(dd)

        seed_params = []
        for k, v in seed_config_auto.items():
            dd = {"name": "seed.{}".format(k)}
            dd.update(v)
            seed_params.append(dd)

        return dataset_params + model_params + trainer_params + seed_params

    @staticmethod
    def _combine_params(auto_params, fixed_params):
        """
        Combining the auto and fixed parameters (to have a single object representing all the arguments used for a specific function)

        Args:
            auto_params (dict): dictionary of to-be-sampled parameters, i.e. A dictionary of dictionaries where keys are dataset,
            model, and trainer and the values are the corresponding dictionary of to-be-sampled arguments.
            fixed_params (dict): dictionary of fixed parameters, i.e. A dictionary of dictionaries where keys are dataset, model,
            and trainer and the values are the corresponding dictionary of fixed arguments.

        Returns:
            dict: dictionary of parameters (fixed and to-be-sampled), i.e. A dictionary of dictionaries where keys are dataset,
            model, and trainer and the values are the corresponding dictionary of arguments.
        """
        keys = ["dataset", "model", "trainer", "seed"]
        params = {}
        for key in keys:
            params[key] = fixed_params[key] if key in fixed_params else {}
            params[key].update(auto_params[key])

        return {key: params[key] for key in keys}

    @staticmethod
    def _split_config(params):
        """
        Reverses the operation of `get_auto_params` (from a list of parameters (ax-friendly format) to a dictionary of
        dictionaries where keys are dataset, model, and trainer and the values are a dictionary of the corresponding arguments)

        Args:
            params (dict): list of dictionaries where each dictionary specifies a single parameter to be sampled.

        Returns:
            dict: A dictionary of dictionaries where keys are dataset, model, and trainer and the values are the corresponding
            dictionary of to-be-sampled arguments.
        """
        config = dict(dataset={}, model={}, trainer={}, seed={}, others={})
        for k, v in params.items():
            config[k.split(".")[0]][k.split(".")[1]] = v

        return config

    def train_evaluate(self, auto_params):
        """
        For a given set of parameters, add an entry to the corresponding tables, and populated the trained model
        table for that specific entry.

        Args:
            auto_params (dict): list of dictionaries where each dictionary specifies a single parameter to be sampled.

        """
        config = self._combine_params(self._split_config(auto_params), self.fixed_params)

        # insert the stuff into their corresponding tables
        seed = config["seed"]["seed"]
        if not dict(seed=seed) in self.trained_model_table().seed_table():
            self.trained_model_table().seed_table().insert1(dict(seed=seed))

        dataset_hash = make_hash(config["dataset"])
        entry_exists = {
            "dataset_fn": "{}".format(self.fns["dataset"])
        } in self.trained_model_table().dataset_table() and {
            "dataset_hash": "{}".format(dataset_hash)
        } in self.trained_model_table().dataset_table()
        if not entry_exists:
            self.trained_model_table().dataset_table().add_entry(
                self.fns["dataset"],
                config["dataset"],
                dataset_fabrikant=self.architect,
                dataset_comment=self.comment,
            )

        model_hash = make_hash(config["model"])
        entry_exists = {"model_fn": "{}".format(self.fns["model"])} in self.trained_model_table().model_table() and {
            "model_hash": "{}".format(model_hash)
        } in self.trained_model_table().model_table()
        if not entry_exists:
            self.trained_model_table().model_table().add_entry(
                self.fns["model"],
                config["model"],
                model_fabrikant=self.architect,
                model_comment=self.comment,
            )

        trainer_hash = make_hash(config["trainer"])
        entry_exists = {
            "trainer_fn": "{}".format(self.fns["trainer"])
        } in self.trained_model_table().trainer_table() and {
            "trainer_hash": "{}".format(trainer_hash)
        } in self.trained_model_table().trainer_table()
        if not entry_exists:
            self.trained_model_table().trainer_table().add_entry(
                self.fns["trainer"],
                config["trainer"],
                trainer_fabrikant=self.architect,
                trainer_comment=self.comment,
            )

        # get the primary key values for all those entries
        restriction = (
            f'seed in ("{seed}")',
            'dataset_fn in ("{}")'.format(self.fns["dataset"]),
            'dataset_hash in ("{}")'.format(dataset_hash),
            'model_fn in ("{}")'.format(self.fns["model"]),
            'model_hash in ("{}")'.format(model_hash),
            'trainer_fn in ("{}")'.format(self.fns["trainer"]),
            'trainer_hash in ("{}")'.format(trainer_hash),
        )

        # populate the table for those primary keys
        self.trained_model_table().populate(*restriction)

    def gen_params_value(self):
        """
        Generates new values (samples randomly) for each parameter.

        Returns:
            dict: A dictionary containing the parameters whose values should be sampled.
        """
        np.random.seed(None)
        auto_params_val = {}
        for param in self.auto_params:
            if param["type"] == "fixed":
                auto_params_val.update({param["name"]: param["value"]})
            elif param["type"] == "choice":
                auto_params_val.update({param["name"]: np.random.choice(param["values"])})
            elif param["type"] == "range":
                if "log_scale" in param and param["log_scale"]:
                    auto_params_val.update({param["name"]: loguniform.rvs(*param["bounds"])})
                else:
                    auto_params_val.update({param["name"]: np.random.uniform(*param["bounds"])})
            elif param["type"] == "int":
                auto_params_val.update({param["name"]: np.random.randint(np.iinfo(np.int32).max)})

        return auto_params_val

    def run(self):
        """
        Runs the random hyperparameter search, for as many trials as specified.
        """
        # n_trials = len(self.trained_model_table().seed_table()) * self.total_trials
        # init_len = len(self.trained_model_table())
        for _ in range(self.total_trials):
            self.train_evaluate(self.gen_params_value())
