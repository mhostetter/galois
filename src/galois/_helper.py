"""
A module containing various helper functions for the library.
"""

import builtins
import inspect
import sys

SPHINX_BUILD = hasattr(builtins, "__sphinx_build__")


def verify_isinstance(argument, types, optional=False):
    """
    Verifies that the argument is an instance of the type(s).
    """
    if optional and argument is None:
        return

    if not isinstance(argument, types):
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = string[string.find("(") + 1 : -1].split(",")
        argument_name = args[0]
        raise TypeError(f"Argument {argument_name!r} must be an instance of {types}, not {type(argument)}.")


def verify_issubclass(argument, types, optional=False):
    """
    Verifies that the argument is a subclass of the type(s).
    """
    if optional and argument is None:
        return

    # Need this try/except because issubclass(instance, (classes,)) will itself raise a TypeError.
    # Instead, we'd like to raise our own TypeError.
    try:
        valid = issubclass(argument, types)
    except TypeError:
        valid = False

    if not valid:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = string[string.find("(") + 1 : -1].split(",")
        argument_name = args[0]
        raise TypeError(f"Argument {argument_name!r} must be a subclass of {types}, not {type(type(argument))}.")


def verify_literal(argument, literals):
    if not argument in literals:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = string[string.find("(") + 1 : -1].split(",")
        argument_name = args[0]
        raise ValueError(f"Argument {argument_name!r} must be one of {literals}, not {argument!r}.")


def export(obj):
    """
    Marks an object for exporting into the public API.

    This decorator appends the object's name to the private module's __all__ list. The private module should
    then be imported in galois/__init__.py using from ._private_module import *. It also modifies the object's
    __module__ to "galois".
    """
    # Determine the private module that defined the object
    module = sys.modules[obj.__module__]

    if not SPHINX_BUILD:
        # Set the object's module to the package name. This way the REPL will display the object
        # as galois.obj and not galois._private_module.obj
        obj.__module__ = "galois"

    # Append this object to the private module's "all" list
    public_members = getattr(module, "__all__", [])
    public_members.append(obj.__name__)
    module.__all__ = public_members

    return obj


def method_of(class_):
    """
    Monkey-patches the decorated function into the class as a method. The class should already have a stub method
    that raises `NotImplementedError`. The docstring of the stub method is replaced with the docstring of the
    decorated function.

    This is used to separate code into multiple files while still keeping the methods in the same class.
    """

    def decorator(func):
        setattr(class_, func.__name__, func)
        getattr(class_, func.__name__).__doc__ = func.__doc__

        return func

    return decorator


def extend_docstring(method, replace=None, docstring=""):
    """
    A decorator to extend the docstring of `method` with the provided docstring. The decorator also finds
    and replaces and key-value pair in `replace`.
    """
    replace = {} if replace is None else replace

    def decorator(obj):
        parent_docstring = getattr(method, "__doc__", "")
        if parent_docstring is None:
            return obj
        for from_str, to_str in replace.items():
            parent_docstring = parent_docstring.replace(from_str, to_str)
        obj.__doc__ = parent_docstring + "\n" + docstring

        return obj

    return decorator
