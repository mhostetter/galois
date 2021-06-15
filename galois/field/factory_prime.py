import types

from ..modular import is_primitive_root
from ..prime import is_prime

from .array import FieldArray
from .gf2 import GF2
from .meta_gfp import GFpMeta

# pylint: disable=protected-access


def GF_prime(characteristic, primitive_element=None, verify=True, mode="auto"):
    """
    Class factory for prime fields GF(p).
    """
    if not isinstance(characteristic, int):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not isinstance(primitive_element, (type(None), int)):
        raise TypeError(f"Argument `primitive_element` must be a int, not {type(primitive_element)}.")
    degree = 1
    order = characteristic**degree

    # If the requested field has already been constructed, return it
    key = (order, primitive_element)
    if key in GF_prime._classes:
        cls = GF_prime._classes[key]
        cls.compile(mode)
        return cls

    name = f"GF{characteristic}_{degree}" if degree > 1 else f"GF{characteristic}"

    if primitive_element is not None:
        if not 0 < primitive_element < order:
            raise ValueError(f"Argument `primitive_element` must be non-zero in the field 0 < x < {order}, not {primitive_element}.")
        if verify and not is_primitive_root(primitive_element, characteristic):
            raise ValueError(f"Argument `primitive_element` must be a primitive root modulo {characteristic}, {primitive_element} is not.")

    if characteristic == 2:
        cls = GF2
        cls.compile(mode)
    else:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GFpMeta,
            "characteristic": characteristic,
            "degree": 1,
            "order": characteristic**1,
            "primitive_element": primitive_element,
            "is_primitive_poly": True,
            "mode": mode
        })

    cls.__module__ = "galois"

    # Add class to dictionary of flyweights
    GF_prime._classes[key] = cls

    return cls

GF_prime._classes = {}
