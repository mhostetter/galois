import types

from .array import GFArray
from .gf2 import GF2
from .meta_gfp import GFpMeta
from .modular import primitive_root, is_primitive_root
from .prime import is_prime


def GF_prime(characteristic, primitive_element=None, verify_primitive=True, mode="auto", target="cpu"):
    if not isinstance(characteristic, int):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    degree = 1
    order = characteristic**degree

    if not isinstance(primitive_element, (type(None), int)):
        raise TypeError(f"Argument `primitive_element` must be a int, not {type(primitive_element)}.")
    if primitive_element is None:
        primitive_element = primitive_root(characteristic)
        verify_primitive = False
    if not 0 < primitive_element < order:
        raise ValueError(f"Argument `primitive_element` must be non-zero in the field 0 < x < {order}, not {primitive_element}.")

    key = (order, primitive_element)

    # If the requested field has already been constructed, return it
    if key in GF_prime.classes:
        cls = GF_prime.classes[key]
        cls.compile(mode, target)
        return cls

    name = f"GF{characteristic}_{degree}" if degree > 1 else f"GF{characteristic}"

    if verify_primitive and not is_primitive_root(primitive_element, characteristic):
        raise ValueError(f"Argument `primitive_element` must be a primitive root modulo {characteristic}, {primitive_element} is not.")

    if characteristic == 2:
        GF2.compile(mode, target)
        cls = GF2
    else:
        cls = types.new_class(name, bases=(GFArray,), kwds={
            "metaclass": GFpMeta,
            "characteristic": characteristic,
            "primitive_element": primitive_element,
            "target": target,
            "mode": mode
        })

    # Add class to dictionary of flyweights
    GF_prime.classes[key] = cls

    return cls

GF_prime.classes = {}
