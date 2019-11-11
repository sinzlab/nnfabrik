from importlib import import_module
import numpy as np

def cleanup_numpy_scalar(data):
    """
    Recursively cleanups up a (potentially nested data structure of) 
    objects, replacing any scalar numpy instance with the corresponding
    Python native datatype.
    """
    if isinstance(data, np.generic):
        if data.shape == ():
            data = data.item()
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = cleanup_numpy_scalar(v)
    elif isinstance(data, (list, tuple)):
        data = [cleanup_numpy_scalar(e) for e in data]
    return data

def split_module_name(abs_class_name):
    abs_module_path = '.'.join(abs_class_name.split('.')[:-1])
    class_name = abs_class_name.split('.')[-1]
    return (abs_module_path, class_name)


def dynamic_import(abs_module_path, class_name):
    module_object = import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class


