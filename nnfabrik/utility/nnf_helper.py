from importlib import import_module
import numpy as np
from ..utility.dj_helpers import make_hash, cleanup_numpy_scalar


def split_module_name(abs_class_name):
    abs_module_path = ".".join(abs_class_name.split(".")[:-1])
    class_name = abs_class_name.split(".")[-1]
    return (abs_module_path, class_name)


def dynamic_import(abs_module_path, class_name):
    module_object = import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class


class FabrikCache:
    def __init__(self, base_table, cache_size_limit=10):
        self.base_table = base_table
        self.cache_size_limit = cache_size_limit
        self.cache = dict()
        if hasattr(self.base_table, 'load_model'):
            self.load_function = self.base_table().load_model
        elif hasattr(self.base_table, 'get_dataloader'):
            self.load_function = self.base_table().get_dataloader
        elif hasattr(self.base_table, 'build_model'):
            self.load_function = self.base_table().build_model
        else:
            raise ValueError("Base table needs to have a 'load_model', 'get_dataloader', or 'build_model' method")

    def load(self, key, **kwargs):
        if self.cache_size_limit == 0:
            return self._load_model(key, **kwargs)
        if not self._is_cached(key):
            self._cache_model(key, **kwargs)
        return self._get_cached_model(key)

    def _load_model(self, key, **kwargs):
        return self.load_function(key=key, **kwargs)

    def _is_cached(self, key):
        if self._hash_trained_model_key(key) in self.cache:
            return True
        return False

    def _cache_model(self, key, **kwargs):
        """Caches a model and makes sure the cache is not bigger than the specified limit."""
        self.cache[self._hash_trained_model_key(key)] = self._load_model(key, **kwargs)
        if len(self.cache) > self.cache_size_limit:
            del self.cache[list(self.cache)[0]]

    def _get_cached_model(self, key):
        return self.cache[self._hash_trained_model_key(key)]

    def _hash_trained_model_key(self, key):
        """Creates a hash from the part of the key corresponding to the primary key of the trained model table."""
        return make_hash({k: key[k] for k in self.base_table().primary_key})