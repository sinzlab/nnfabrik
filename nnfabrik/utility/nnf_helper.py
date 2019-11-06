from importlib import import_module


def split_module_name(abs_class_name):
    abs_module_path = '.'.join(abs_class_name.split('.')[:-1])
    class_name = abs_class_name.split('.')[-1]
    return (abs_module_path, class_name)


def dynamic_import(abs_module_path, class_name):
    module_object = import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class


def module_dynamic_import(abs_class_name, pre=''):
    module_path, class_name = split_module_name(abs_class_name)
    return dynamic_import(module_path, class_name) if module_path else eval(pre + configurator)
