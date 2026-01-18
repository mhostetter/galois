"""
A module containing various helper functions for the library.
"""

from __future__ import annotations

import builtins
import inspect
import sys
import textwrap
from typing import Any, Callable

SPHINX_BUILD = hasattr(builtins, "__sphinx_build__")


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
