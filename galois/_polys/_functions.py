"""
A module with functions to generate minimal polynomials.
"""
import numpy as np

from .._fields import FieldArray
from .._overrides import set_module

from ._poly import Poly

__all__ = ["is_monic", "minimal_poly"]


@set_module("galois")
def is_monic(poly):
    r"""
    Determines whether the polynomial is monic, i.e. having leading coefficient equal to 1.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial over a Galois field.

    Returns
    -------
    bool
        `True` if the polynomial is monic.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(7)
        p = galois.Poly([1,0,4,5], field=GF); p
        galois.is_monic(p)

    .. ipython:: python

        p = galois.Poly([3,0,4,5], field=GF); p
        galois.is_monic(p)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    return poly.nonzero_coeffs[0] == 1


@set_module("galois")
def minimal_poly(element):
    r"""
    Computes the minimal polynomial :math:`m_e(x) \in \mathrm{GF}(p)[x]` of a Galois field
    element :math:`e \in \mathrm{GF}(p^m)`.

    The *minimal polynomial* of a Galois field element :math:`e \in \mathrm{GF}(p^m)` is the polynomial of
    minimal degree over :math:`\mathrm{GF}(p)` for which :math:`e` is a root when evaluated in :math:`\mathrm{GF}(p^m)`.
    Namely, :math:`m_e(x) \in \mathrm{GF}(p)[x] \in \mathrm{GF}(p^m)[x]` and :math:`m_e(e) = 0` over :math:`\mathrm{GF}(p^m)`.

    Parameters
    ----------
    element : galois.FieldArray
        Any element :math:`e` of the Galois field :math:`\mathrm{GF}(p^m)`. This must be a 0-D array.

    Returns
    -------
    galois.Poly
        The minimal polynomial :math:`m_e(x)` over :math:`\mathrm{GF}(p)` of the element :math:`e`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(2**4)
        e = GF.primitive_element; e
        m_e = galois.minimal_poly(e); m_e
        # Evaluate m_e(e) in GF(2^4)
        m_e(e, field=GF)

    For a given element :math:`e`, the minimal polynomials of :math:`e` and all its conjugates are the same.

    .. ipython:: python

        # The conjugates of e
        conjugates = np.unique(e**(2**np.arange(0, 4))); conjugates
        for conjugate in conjugates:
            print(galois.minimal_poly(conjugate))

    Not all elements of :math:`\mathrm{GF}(2^4)` have minimal polynomials with degree-:math:`4`.

    .. ipython:: python

        e = GF.primitive_element**5; e
        # The conjugates of e
        conjugates = np.unique(e**(2**np.arange(0, 4))); conjugates
        for conjugate in conjugates:
            print(galois.minimal_poly(conjugate))

    In prime fields, the minimal polynomial of :math:`e` is simply :math:`m_e(x) = x - e`.

    .. ipython:: python

        GF = galois.GF(7)
        e = GF(3); e
        m_e = galois.minimal_poly(e); m_e
        m_e(e)
    """
    if not isinstance(element, FieldArray):
        raise TypeError(f"Argument `element` must be an element of a Galois field, not {type(element)}.")
    if not element.ndim == 0:
        raise ValueError(f"Argument `element` must be a single array element with dimension 0, not {element.ndim}-D.")

    field = type(element)
    x = Poly.Identity(field=field)

    if field.is_prime_field:
        return x - element
    else:
        conjugates = np.unique(element**(field.characteristic**np.arange(0, field.degree)))
        poly = Poly.Roots(conjugates, field=field)
        poly = Poly(poly.coeffs, field=field.prime_subfield)
        return poly
