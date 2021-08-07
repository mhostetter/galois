import types

import numpy as np

from ._array import FieldArray
from ._factory_prime import GF_prime
from ._meta_gf2m import GF2mMeta
from ._meta_gfpm import GFpmMeta
from ._poly import Poly
from ._poly_functions import conway_poly, is_irreducible, is_primitive_element
from ._poly_functions import primitive_element as _primitive_element  # To avoid name conflict with GF_extension() arguments

# pylint: disable=redefined-builtin


def GF_extension(characteristic, degree, irreducible_poly=None, primitive_element=None, verify=True, compile=None, display=None):
    """
    Class factory for extension fields GF(p^m).
    """
    # pylint: disable=too-many-statements
    order = characteristic**degree
    name = f"GF{characteristic}_{degree}"
    prime_subfield = GF_prime(characteristic)
    is_primitive_poly = None
    verify_poly = verify
    verify_element = verify

    # Get default irreducible polynomial
    if irreducible_poly is None:
        irreducible_poly = conway_poly(characteristic, degree)
        is_primitive_poly = True
        verify_poly = False  # We don't need to verify Conway polynomials are irreducible
        if primitive_element is None:
            primitive_element = Poly.Identity(prime_subfield)
            verify_element = False  # We know `g(x) = x` is a primitive element of the Conway polynomial because Conway polynomials are primitive polynomials
    elif isinstance(irreducible_poly, int):
        irreducible_poly = Poly.Integer(irreducible_poly, field=prime_subfield)
    elif isinstance(irreducible_poly, str):
        irreducible_poly = Poly.String(irreducible_poly, field=prime_subfield)
    elif isinstance(irreducible_poly, (tuple, list, np.ndarray)):
        irreducible_poly = Poly(irreducible_poly, field=prime_subfield)
    elif not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be an int, tuple, list, np.ndarray, or galois.Poly, not {type(irreducible_poly)}.")

    # Get default primitive element
    if primitive_element is None:
        primitive_element = _primitive_element(irreducible_poly)
        verify_element = False
    elif isinstance(primitive_element, int):
        primitive_element = Poly.Integer(primitive_element, field=prime_subfield)
    elif isinstance(primitive_element, str):
        primitive_element = Poly.String(primitive_element, field=prime_subfield)
    elif isinstance(primitive_element, (tuple, list, np.ndarray)):
        primitive_element = Poly(primitive_element, field=prime_subfield)
    elif not isinstance(primitive_element, Poly):
        raise TypeError(f"Argument `primitive_element` must be an int, tuple, list, np.ndarray, or galois.Poly, not {type(primitive_element)}.")

    # Check polynomial fields and degrees
    if not irreducible_poly.field.order == characteristic:
        raise ValueError(f"Argument `irreducible_poly` must be over {prime_subfield.name}, not {irreducible_poly.field.name}.")
    if not irreducible_poly.degree == degree:
        raise ValueError(f"Argument `irreducible_poly` must have degree equal to {degree}, not {irreducible_poly.degree}.")
    if not primitive_element.field.order == characteristic:
        raise ValueError(f"Argument `primitive_element` must be a polynomial over {prime_subfield.name}, not {primitive_element.field.name}.")
    if not primitive_element.degree < degree:
        raise ValueError(f"Argument `primitive_element` must have degree strictly less than {degree}, not {primitive_element.degree}.")

    # If the requested field has already been constructed, return it
    key = (order, primitive_element.integer, irreducible_poly.integer)
    if key in GF_extension._classes:
        cls = GF_extension._classes[key]
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

    if verify_poly and not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is not.")
    if verify_element and not is_primitive_element(primitive_element, irreducible_poly):
        raise ValueError(f"Argument `primitive_element` must be a multiplicative generator of GF({characteristic}^{degree}), {primitive_element} is not.")

    if characteristic == 2:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GF2mMeta,
            "characteristic": characteristic,
            "degree": degree,
            "order": order,
            "irreducible_poly": irreducible_poly,
            "is_primitive_poly": is_primitive_poly,
            "primitive_element": primitive_element.integer,
            "prime_subfield": prime_subfield,
            "compile": compile
        })
    else:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GFpmMeta,
            "characteristic": characteristic,
            "degree": degree,
            "order": order,
            "irreducible_poly": irreducible_poly,
            "is_primitive_poly": is_primitive_poly,
            "primitive_element": primitive_element.integer,
            "prime_subfield": prime_subfield,
            "compile": compile
        })

    cls.__module__ = "galois"
    cls.display(display)

    # Add class to dictionary of flyweights
    GF_extension._classes[key] = cls

    return cls

GF_extension._classes = {}
