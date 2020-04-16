from . import utility
from . import datasets
from . import training
from . import models
from functools import partial

from .utility.nnf_helper import split_module_name, dynamic_import



def resolve_fn(fn_name, default_base):
    """
    Given a string `fn_name`, resolves the name into a callable object. If the name has multiple `.` separated parts, treat all but the last
    as module names to trace down to the final name. If just the name is given, tries to resolve the name in the `default_base` module name context
    with direct eval of `{default_base}.{fn_name}` in this function's context.

    Raises `NameError` if no object matching the name is found and `TypeError` if the resolved object is not callabe.

    When successful, returns the resolved, callable object.
    """
    module_path, class_name = split_module_name(fn_name)
    
    try:
        fn_obj = dynamic_import(module_path, class_name) if module_path else eval('{}.{}'.format(default_base, class_name))
    except NameError:
        raise NameError("Function `{}` does not exist".format(class_name))

    if not callable(fn_obj):
        raise TypeError("The object named {} is not callable.".format(class_name))
    
    return fn_obj
        

# provide convenience alias for resolving model, dataset, and trainer
resolve_model = partial(resolve_fn, default_base='models')
resolve_data = partial(resolve_fn, default_base='datasets')
resolve_trainer = partial(resolve_fn, default_base='training')


def get_model(model_fn, model_config, dataloaders=None, seed=None, state_dict=None, strict=True, data_info=None):
    """
    Resolves `model_fn` and invokes the resolved function with `model_config` keyword arguments as well as the `dataloader` and `seed`.
    Note that the resolved `model_fn` is expected to accept the `dataloader` as the first positional argument and `seed` as a keyword argument.
    If you pass in `state_dict`, the resulting nn.Module instance will be loaded with the state_dict, using appropriate `strict` mode for loading.

    Args:
        model_fn: string name of the model builder function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        model_config: a dictionary containing keyword arguments to be passed into the resolved `model_fn`
        dataloaders: (a dictionary of) dataloaders to be passed into the resolved `model_fn` as the first positional argument
        seed: randomization seed to be passed in to as a keyword argument into the resolved `model_fn`
        state_dict: If provided, the resulting nn.Module object will be loaded with the state_dict before being returned
        strict: Controls the `strict` mode of nn.Module.load_state_dict

    Returns:
        Resulting nn.Module object.
    """

    if isinstance(model_fn, str):
        model_fn = resolve_model(model_fn)


    net = model_fn(dataloaders, seed=seed, **model_config) if data_info is None else model_fn(dataloaders, data_info=data_info, seed=seed, **model_config)

    if state_dict is not None:
        net.load_state_dict(state_dict, strict=strict)

    return net


def get_data(dataset_fn, dataset_config):
    """
    Resolves `dataset_fn` and invokes the resolved function onto the `dataset_config` configuration dictionary. The resulting
    dataloader will be returned.

    Args:
        dataset_fn: string name of the dataloader function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        dataset_config: a dictionary containing keyword arguments to be passed into the resolved `dataset_fn`

    Returns:
        Result of invoking the resolved `dataset_fn` with `dataset_config` as keyword arguments.
    """
    if isinstance(dataset_fn, str):
        dataset_fn = resolve_data(dataset_fn)

    return dataset_fn(**dataset_config)


def get_trainer(trainer_fn, trainer_config=None):
    """
    If `trainer_fn` string is passed, resolves and returns the corresponding function. If `trainer_config` is passed in,
    a partial function is created with the configuration object expanded.

    Args:
        trainer_fn: string name of the function path to be resolved. Alternatively, you can pass in a callable object and no name resolution will be performed.
        trainer_config: If passed in, a partial function will be created expanding `trainer_config` as the keyword arguments into the resolved trainer_fn

    Returns:
        Resolved trainer function
    """

    if isinstance(trainer_fn, str):
        trainer_fn = resolve_trainer(trainer_fn)

    if trainer_config is not None:
        trainer_fn = partial(trainer_fn, **trainer_config)

    return trainer_fn


def get_all_parts(dataset_fn, dataset_config, model_fn, model_config, seed=None, state_dict=None, strict=True, trainer_fn=None, trainer_config=None):

    if seed is not None and 'seed' not in dataset_config:
        dataset_config['seed'] = seed  # override the seed if passed in

    dataloaders = get_data(dataset_fn, dataset_config)

    model = get_model(model_fn, model_config, dataloaders, seed=seed, state_dict=state_dict, strict=strict)

    if trainer_fn is not None:
        trainer = get_trainer(trainer_fn, trainer_config)
        return dataloaders, model, trainer
    else:
        return dataloaders, model
