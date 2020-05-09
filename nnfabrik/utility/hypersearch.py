import warnings
import numpy as np
import torch
from ax.service.managed_loop import optimize
from .dj_helpers import make_hash
from .nnf_helper import split_module_name, dynamic_import
from nnfabrik.main import *


class Bayesian():

    def __init__(self,
                 dataset_fn, dataset_config, dataset_config_auto,
                 model_fn, model_config, model_config_auto,
                 trainer_fn, trainer_config, trainer_config_auto,
                 architect,
                 trained_model_table,
                 total_trials=5,
                 arms_per_trial=1,
                 comment="Bayesian optimization of Hyper params."):

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
    def get_fixed_params(datase_config, model_config, trainer_config):
        return dict(dataset=datase_config, model=model_config, trainer=trainer_config)

    @staticmethod
    def get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto):
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
        keys = ['dataset', 'model', 'trainer']
        params = {}
        for key in keys:
            params[key] = fixed_params[key]
            params[key].update(auto_params[key])

        return {key:params[key] for key in keys}


    @staticmethod
    def _split_config(params):
        config = dict(dataset={}, model={}, trainer={}, others={})
        for k, v in params.items():
            config[k.split(".")[0]][k.split(".")[1]] = v

        return config

    @staticmethod
    def _add_etery(Table, fn, config):
        entry_hash = make_hash(config)
        entry_exists = {"configurator": "{}".format(fn)} in Table() and {"config_hash": "{}".format(entry_hash)} in Table()
        if not entry_exists:
            Table().add_entry(fn, config,
                              model_fabrikant=self.architect,
                              model_comment="AutoMLing")
        return fn, entry_hash

    def train_evaluate(self, auto_params):
        config = self._combine_params(self._split_config(auto_params), self.fixed_params)

        # insert the stuff into their corresponding tables
        dataset_hash = make_hash(config['dataset'])
        entry_exists = {"dataset_fn": "{}".format(self.fns['dataset'])} in self.trained_model_table.dataset_table() and {"dataset_hash": "{}".format(dataset_hash)} in self.trained_model_table.dataset_table()
        if not entry_exists:
            self.trained_model_table.dataset_table().add_entry(self.fns['dataset'], config['dataset'],
                                dataset_fabrikant=self.architect,
                                dataset_comment=self.comment)

        model_hash = make_hash(config['model'])
        entry_exists = {"model_fn": "{}".format(self.fns['model'])} in self.trained_model_table.model_table() and {"model_hash": "{}".format(model_hash)} in self.trained_model_table.model_table()
        if not entry_exists:
            self.trained_model_table.model_table().add_entry(self.fns['model'], config['model'],
                              model_fabrikant=self.architect,
                              model_comment=self.comment)

        trainer_hash = make_hash(config['trainer'])
        entry_exists = {"trainer_fn": "{}".format(self.fns['trainer'])} in self.trained_model_table.trainer_table() and {"trainer_hash": "{}".format(trainer_hash)} in self.trained_model_table.trainer_table()
        if not entry_exists:
            self.trained_model_table.trainer_table().add_entry(self.fns['trainer'], config['trainer'],
                                trainer_fabrikant=self.architect,
                                trainer_comment=self.comment)

        # get the primary key values for all those entries
        restriction = ('dataset_fn in ("{}")'.format(self.fns['dataset']), 'dataset_hash in ("{}")'.format(dataset_hash),
                       'model_fn in ("{}")'.format(self.fns['model']), 'model_hash in ("{}")'.format(model_hash),
                       'trainer_fn in ("{}")'.format(self.fns['trainer']), 'trainer_hash in ("{}")'.format(trainer_hash),)

        # populate the table for those primary keys
        self.trained_model_table().populate(*restriction)

        # get the score of the model for this specific set of hyperparameters
        score = (self.trained_model_table() & dj.AndList(restriction)).fetch("score")[0]

        return score

    def run(self):
        best_parameters, values, experiment, model = optimize(
            parameters=self.auto_params,
            evaluation_function=self.train_evaluate,
            objective_name='val_corr',
            minimize=False,
            total_trials=self.total_trials,
            arms_per_trial=self.arms_per_trial,
        )

        return self._split_config(best_parameters), values, experiment, model


class Random():

    def __init__(self,
                 dataset_fn, dataset_config, dataset_config_auto,
                 model_fn, model_config, model_config_auto,
                 trainer_fn, trainer_config, trainer_config_auto,
                 architect,
                 trained_model_table,
                 total_trials=5,
                 arms_per_trial=1,
                 comment="Random search for hyper params."):

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
    def get_fixed_params(datase_config, model_config, trainer_config):
        return dict(dataset=datase_config, model=model_config, trainer=trainer_config)

    @staticmethod
    def get_auto_params(dataset_config_auto, model_config_auto, trainer_config_auto):
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
        keys = ['dataset', 'model', 'trainer']
        params = {}
        for key in keys:
            params[key] = fixed_params[key]
            params[key].update(auto_params[key])

        return {key:params[key] for key in keys}


    @staticmethod
    def _split_config(params):
        config = dict(dataset={}, model={}, trainer={}, others={})
        for k, v in params.items():
            config[k.split(".")[0]][k.split(".")[1]] = v

        return config


    def train_evaluate(self, auto_params):
        config = self._combine_params(self._split_config(auto_params), self.fixed_params)

        # insert the stuff into their corresponding tables
        dataset_hash = make_hash(config['dataset'])
        entry_exists = {"dataset_fn": "{}".format(self.fns['dataset'])} in self.trained_model_table.dataset_table() and {"dataset_hash": "{}".format(dataset_hash)} in self.trained_model_table.dataset_table()
        if not entry_exists:
            self.trained_model_table.dataset_table().add_entry(self.fns['dataset'], config['dataset'],
                                dataset_fabrikant=self.architect,
                                dataset_comment=self.comment)

        model_hash = make_hash(config['model'])
        entry_exists = {"model_fn": "{}".format(self.fns['model'])} in self.trained_model_table.model_table() and {"model_hash": "{}".format(model_hash)} in self.trained_model_table.model_table()
        if not entry_exists:
            self.trained_model_table.model_table().add_entry(self.fns['model'], config['model'],
                              model_fabrikant=self.architect,
                              model_comment=self.comment)

        trainer_hash = make_hash(config['trainer'])
        entry_exists = {"trainer_fn": "{}".format(self.fns['trainer'])} in self.trained_model_table.trainer_table() and {"trainer_hash": "{}".format(trainer_hash)} in self.trained_model_table.trainer_table()
        if not entry_exists:
            self.trained_model_table.trainer_table().add_entry(self.fns['trainer'], config['trainer'],
                                trainer_fabrikant=self.architect,
                                trainer_comment=self.comment)

        # get the primary key values for all those entries
        restriction = ('dataset_fn in ("{}")'.format(self.fns['dataset']), 'dataset_hash in ("{}")'.format(dataset_hash),
                       'model_fn in ("{}")'.format(self.fns['model']), 'model_hash in ("{}")'.format(model_hash),
                       'trainer_fn in ("{}")'.format(self.fns['trainer']), 'trainer_hash in ("{}")'.format(trainer_hash),)

        # populate the table for those primary keys
        self.trained_model_table().populate(*restriction)


    def gen_params_value(self):
        np.random.seed(None)
        auto_params_val = {}
        for param in self.auto_params:
            if param['type'] == 'fixed':
                auto_params_val.update({param['name']: param['value']})
            elif param['type'] == 'choice':
                auto_params_val.update({param['name']: np.random.choice(param['values'])})
            elif param['type'] == 'range':
                auto_params_val.update({param['name']: np.random.uniform(*param['bounds'])})

        return auto_params_val


    def run(self):
        n_trials = len(self.trained_model_table.seed_table()) * self.total_trials
        init_len = len(self.trained_model_table())
        while len(self.trained_model_table()) - init_len < n_trials:
            self.train_evaluate(self.gen_params_value())
