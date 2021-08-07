import types

from .._modular import primitive_root, is_primitive_root

from ._array import FieldArray
from ._gf2 import GF2
from ._meta_gfp import GFpMeta

# pylint: disable=redefined-builtin


def GF_prime(characteristic, primitive_element=None, verify=True, compile=None, display=None):
    """
    Class factory for prime fields GF(p).
    """
    degree = 1
    order = characteristic**degree
    name = f"GF{characteristic}"

    # Get default primitive element
    if primitive_element is None:
        primitive_element = primitive_root(characteristic)

    # Check primitive element range
    if not 0 < primitive_element < order:
        raise ValueError(f"Argument `primitive_element` must be non-zero in the field 0 < x < {order}, not {primitive_element}.")

    # If the requested field has already been constructed, return it
    key = (order, primitive_element)
    if key in GF_prime._classes:
        cls = GF_prime._classes[key]
        if compile is not None:
            cls.compile(compile)
        if display is not None:
            cls.display(display)
        return cls

    # Since this is a new class, set `compile` and `display` to their default values
    if compile is None:
        compile = "auto"
    if display is None:
        display = "int"

    if verify and not is_primitive_root(primitive_element, characteristic):
        raise ValueError(f"Argument `primitive_element` must be a primitive root modulo {characteristic}, {primitive_element} is not.")

    if characteristic == 2:
        cls = GF2
        cls.compile(compile)
    else:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GFpMeta,
            "characteristic": characteristic,
            "degree": degree,
            "order": order,
            "primitive_element": primitive_element,
            "is_primitive_poly": True,
            "compile": compile
        })

    cls.__module__ = "galois"
    cls.display(display)

    # Add class to dictionary of flyweights
    GF_prime._classes[key] = cls

    return cls

GF_prime._classes = {}
