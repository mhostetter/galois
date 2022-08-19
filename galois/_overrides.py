import builtins

SPHINX_BUILD = hasattr(builtins, "__sphinx_build__")


def set_module(module):
    """
    A decorator to update the __module__ variable as is done in NumPy.

    References
    ----------
    * https://numpy.org/devdocs/release/1.16.0-notes.html#module-attribute-now-points-to-public-modules
    * https://github.com/numpy/numpy/blob/544094aed5fdca536b300d0820fe41f22729ec66/numpy/core/overrides.py#L94-L109
    """
    def decorator(obj):
        if not SPHINX_BUILD:
            # Sphinx gets confused when parsing overloaded functions when the module is modified using this decorator.
            # We set the __sphinx_build__ variable in conf.py and avoid modifying the module when building the docs.
            if module is not None:
                obj.__module__ = module

        return obj

    return decorator


def extend_docstring(method, replace={}, docstring=""):  # pylint: disable=dangerous-default-value
    """
    A decorator to append the docstring of a `method` with the docstring the the decorated method.
    """
    def decorator(obj):
        parent_docstring = getattr(method, "__doc__", "")
        for from_str, to_str in replace.items():
            parent_docstring = parent_docstring.replace(from_str, to_str)
        obj.__doc__ = parent_docstring + "\n" + docstring

        return obj

    return decorator
