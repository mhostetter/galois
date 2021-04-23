import types

from .array import GFArray
from .conway import conway_poly
from .gf_prime import GF_prime
from .meta_gf2m import GF2mMeta
# from .meta_gfpm import GFpmMeta
from .poly import Poly
from .poly_functions import is_irreducible, is_primitive_element
from .poly_functions import primitive_element as _primitive_element  # To avoid name conflict with GF() arguments
from .prime import is_prime


def GF_extension(characteristic, degree, irreducible_poly=None, primitive_element=None, verify_irreducible=True, verify_primitive=True, mode="auto", target="cpu"):
    # pylint: disable=too-many-branches,too-many-statements
    if not isinstance(characteristic, int):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, int):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree > 1:
        raise ValueError(f"Argument `degree` must be greater than 1, not {degree}.")
    order = characteristic**degree
    ground_field = GF_prime(characteristic)

    if irreducible_poly is None and primitive_element is None:
        irreducible_poly = conway_poly(characteristic, degree)
        primitive_element = Poly.Identity(ground_field)
        verify_irreducible = False  # We don't need to verify Conway polynomials are irreducible
        verify_primitive = False  # We know `g(x) = x` is a primitive root of the Conway polynomial because Conway polynomials are primitive polynomials

    # Get default irreducible polynomial
    if irreducible_poly is None:
        irreducible_poly = conway_poly(characteristic, degree)
        verify_irreducible = False  # We don't need to verify Conway polynomials are irreducible
    elif isinstance(irreducible_poly, int):
        irreducible_poly = Poly.Integer(irreducible_poly, field=ground_field)
    elif not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be an integer or galois.Poly, not {type(irreducible_poly)}.")

    # Get default primitive element
    if primitive_element is None:
        primitive_element = _primitive_element(irreducible_poly)
        verify_primitive = False
    elif isinstance(primitive_element, int):
        primitive_element = Poly.Integer(primitive_element, field=ground_field)
    elif not isinstance(primitive_element, Poly):
        raise TypeError(f"Argument `primitive_element` must be an integer or galois.Poly, not {type(primitive_element)}.")

    # Check polynomial fields and degrees
    if not irreducible_poly.field.order == characteristic:
        raise ValueError(f"Argument `irreducible_poly` must be over GF({characteristic}), not {irreducible_poly.field.name}.")
    if not irreducible_poly.degree == degree:
        raise ValueError(f"Argument `irreducible_poly` must have degree equal to {degree}, not {irreducible_poly.degree}.")
    if not primitive_element.field.order == characteristic:
        raise ValueError(f"Argument `primitive_element` must be a polynomial over GF({characteristic}), not {primitive_element.field.name}.")
    if not primitive_element.degree < degree:
        raise ValueError(f"Argument `primitive_element` must have degree strictly less than {degree}, not {primitive_element.degree}.")

    key = (order, primitive_element.integer, irreducible_poly.integer)

    # If the requested field has already been constructed, return it
    if key in GF_extension.classes:
        cls = GF_extension.classes[key]
        cls.compile(mode, target)
        return cls

    name = f"GF{characteristic}_{degree}" if degree > 1 else f"GF{characteristic}"

    # For extension fields
    if verify_irreducible and not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is not.")
    if verify_primitive and not is_primitive_element(primitive_element, irreducible_poly):
        raise ValueError(f"Argument `primitive_element` must be a multiplicative generator of GF({characteristic}^{degree}), {primitive_element} is not.")

    if characteristic == 2:
        cls = types.new_class(name, bases=(GFArray,), kwds={
            "metaclass": GF2mMeta,
            "degree": degree,
            "irreducible_poly": irreducible_poly,
            "primitive_element": primitive_element,
            "ground_field": ground_field,
            "target": target,
            "mode": mode
        })

    else:
        # cls = types.new_class(name, bases=(GFArray,), kwds={
        #     "metaclass": GFpmMeta,
        #     "characteristic": characteristic,
        #     "degree": degree,
        #     "irreducible_poly": irreducible_poly,
        #     "primitive_element": primitive_element,
        #     "ground_field": ground_field,
        #     "target": target,
        #     "mode": mode
        # })
        raise NotImplementedError

    # Add class to dictionary of flyweights
    GF_extension.classes[key] = cls

    return cls

GF_extension.classes = {}
