"""
A module containing various helper functions for the library.
"""

from __future__ import annotations

import builtins
import inspect
import sys
import textwrap
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

SPHINX_BUILD = hasattr(builtins, "__sphinx_build__")


def _argument_names():
    """
    Finds the source code argument names from the function that called a verification function.
    """
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[2]  # function() -> verify() -> _argument_name()
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find("(") + 1 : -1].split(",")
    args = [arg.strip() for arg in args]  # Strip leading/trailing whitespace
    # args = [arg.split("=")[0].strip() for arg in args]  # Remove default values and strip whitespace
    return tuple(args)


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


def verify_arraylike(
    x: npt.ArrayLike | None,
    dtype: npt.DTypeLike | None = None,
    # Data types
    optional: bool = False,
    int: bool = False,
    float: bool = False,
    complex: bool = False,
    # Value constraints
    real: bool = False,
    imaginary: bool = False,
    negative: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    inclusive_min: float | None = None,
    inclusive_max: float | None = None,
    exclusive_min: float | None = None,
    exclusive_max: float | None = None,
    # Dimension and size constraints
    atleast_1d: bool = False,
    atleast_2d: bool = False,
    atleast_3d: bool = False,
    ndim: int | None = None,
    size: int | None = None,
    sizes: tuple | list | None = None,
    size_multiple: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> npt.NDArray:
    """
    Converts the argument to a NumPy array and verifies the conditions.
    """
    if optional and x is None:
        return x

    x = np.asarray(x, dtype=dtype)

    if int:
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int, not {x.dtype}.")
    if float:
        if not (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating)):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float, not {x.dtype}.")
    if complex:
        if not (
            np.issubdtype(x.dtype, np.integer)
            or np.issubdtype(x.dtype, np.floating)
            or np.issubdtype(x.dtype, np.complexfloating)
        ):
            raise TypeError(f"Argument {_argument_names()[0]!r} must be an int or float or complex, not {x.dtype}.")

    if real:
        if not np.isrealobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be real, not complex.")
    if imaginary:
        if not np.iscomplexobj(x):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be complex, not real.")
    if negative:
        if np.any(x >= 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be negative, not {x}.")
    if non_negative:
        if np.any(x < 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be non-negative, not {x}.")
    if positive:
        if np.any(x <= 0):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be positive, not {x}.")

    if inclusive_min is not None:
        if np.any(x < inclusive_min):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at least {inclusive_min}, not {x}.")
    if inclusive_max is not None:
        if np.any(x > inclusive_max):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be at most {inclusive_max}, not {x}.")
    if exclusive_min is not None:
        if np.any(x <= exclusive_min):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be greater than {exclusive_min}, not {x}.")
    if exclusive_max is not None:
        if np.any(x >= exclusive_max):
            raise ValueError(f"Argument {_argument_names()[0]!r} must be less than {exclusive_max}, not {x}.")

    if atleast_1d:
        x = np.atleast_1d(x)
    if atleast_2d:
        x = np.atleast_2d(x)
    if atleast_3d:
        x = np.atleast_3d(x)
    if ndim is not None:
        if not x.ndim == ndim:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have {ndim} dimensions, not {x.ndim}.")
    if size is not None:
        if not x.size == size:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have {size} elements, not {x.size}.")
    if sizes is not None:
        if not x.size in sizes:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have on of {sizes} elements, not {x.size}.")
    if size_multiple is not None:
        if not x.size % size_multiple == 0:
            raise ValueError(
                f"Argument {_argument_names()[0]!r} must have a size that is a multiple of {size_multiple}, not {x.size}."
            )
    if shape is not None:
        if not x.shape == shape:
            raise ValueError(f"Argument {_argument_names()[0]!r} must have shape {shape}, not {x.shape}.")

    return x


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


def extend_docstring(method: Any, replace: dict[str, str] | None = None, extra: str = "") -> Callable:
    """
    Decorator to extend the docstring of `method` with `extra`.

    - Reads the parent docstring (cleaned/dedented).
    - Applies string replacements.
    - Appends a dedented version of `extra`.
    """
    replace = {} if replace is None else replace

    # Get a nicely-cleaned parent docstring (works fine for functions *and* properties)
    parent_doc = inspect.getdoc(method) or ""
    for from_str, to_str in replace.items():
        parent_doc = parent_doc.replace(from_str, to_str)

    # Clean up the extra text so section headers are at column 0
    extra_doc = textwrap.dedent(extra).strip("\n")

    if parent_doc and extra_doc:
        combined = parent_doc.rstrip() + "\n\n" + extra_doc + "\n"
    elif extra_doc:
        combined = extra_doc + "\n"
    else:
        combined = parent_doc

    def decorator(obj: Any) -> Any:
        obj.__doc__ = combined
        return obj

    return decorator
