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


def extend_docstring(method, replace={}):  # pylint: disable=dangerous-default-value
    """
    A decorator to append the docstring of a `method` with the docstring the the decorated method.
    """
    def decorator(obj):
        doc_1 = getattr(method, "__doc__", "")
        doc_2 = getattr(obj, "__doc__", "")
        for from_str, to_str in replace.items():
            doc_1 = doc_1.replace(from_str, to_str)
        obj.__doc__ = doc_1 + "\n" + doc_2

        return obj

    return decorator


def classproperty(obj):
    """
    In Python 3.9, decorating class properties is possible using:

        @classmethod
        @property
        def foo(cls):
            return cls._foo

    In Python 3.8 and lower, the class property must be specified in a metaclass. However, Sphinx cannot document
    the metaclass's properties. We add this function inside a Sphinx-build if statement to wrap the metaclass's
    properties using Python 3.9 syntax (if if building with lower versions). This will document correctly. And when
    not building the docs, the metaclass property is referenced normally.

        class MetaFoo(type):
            @property
            def foo(cls):
                return cls._foo

        class Foo(metaclass=MetaFoo):
            if hasattr(builtins, "__sphinx_build__"):
                foo = classproperty(MetaFoo.foo)
    """
    ret = classmethod(property(obj.fget, obj.fset, obj.fdel, obj.__doc__))
    ret.__doc__ = obj.__doc__
    # ret.__annotations__ = obj.__annotations__

    return ret
