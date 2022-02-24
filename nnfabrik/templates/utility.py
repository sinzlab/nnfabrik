from types import ModuleType
from typing import Union

from .. import main


def find_object(context: Union[ModuleType, dict], attribute: str, prop_name: str = None):
    """
    Helper function to resolve an object matching the name attribute
    inside the context. If it's not found, throws ValueError suggesting
    the user to override the `nnfabrik` class property or to a specific
    class property for the table.

    Args:
        context (Union[ModuleType, dict]): A context object in which the name attribute would be checked.
            Can either be a module object or a dictionary.
        attribute (str): Name of object being sought.
        prop_name (str, optional): The property name under which this object is being sought. Defaults to None,
            in which case the name is infered to be lower(attribute) + '_table'. E.g. `model_table` for attribute
            'Model'.

    Raises:
        ValueError: if an object with name `attribute` is not found inside the context.

    Returns:
        Any: the object with name `attribute` found inside the context.
    """
    # if context of string "core" given, then use the core main module as the context
    if context == "core":
        context = main

    if prop_name is None:
        prop_name = attribute.lower() + "_table"

    if context is None:
        raise ValueError("Please specify either `nnfabrik` or `{}` property for the class".format(prop_name))

    if isinstance(context, ModuleType):
        context = context.__dict__

    return context[attribute]
