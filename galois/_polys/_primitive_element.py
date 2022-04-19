"""
A module containing functions to generate and test primitive elements of finite fields.
"""
from __future__ import annotations

import random
from typing import List
from typing_extensions import Literal

from .._modular import totatives
from .._overrides import set_module
from .._prime import factors

from ._irreducible import is_irreducible
from ._poly import Poly, PolyLike

__all__ = ["is_primitive_element", "primitive_element", "primitive_elements"]


@set_module("galois")
def is_primitive_element(element: PolyLike, irreducible_poly: Poly) -> bool:
    r"""
    Determines if :math:`g` is a primitive element of the Galois field :math:`\mathrm{GF}(q^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Parameters
    ----------
    element
        An element :math:`g` of :math:`\mathrm{GF}(q^m)` is a polynomial over :math:`\mathrm{GF}(q)` with degree
        less than :math:`m`.
    irreducible_poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` that defines the extension
        field :math:`\mathrm{GF}(q^m)`.

    Returns
    -------
    :
        `True` if :math:`g` is a primitive element of :math:`\mathrm{GF}(q^m)`.

    See Also
    --------
    primitive_element, FieldArray.primitive_element

    Examples
    --------
    Find all primitive elements for the degree :math:`4` extension of :math:`\mathrm{GF}(3)`.

    .. ipython:: python

        f = galois.conway_poly(3, 4); f
        g = galois.primitive_elements(f); g

    Note from the list above that :math:`x + 2` is a primitive element, but :math:`x + 1` is not.

    .. ipython:: python

        galois.is_primitive_element("x + 2", f)
        # x + 1 over GF(3) has integer equivalent of 4
        galois.is_primitive_element(4, f)
    """
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    field = irreducible_poly.field

    # Convert element into a Poly object
    element = Poly._PolyLike(element, field=field)

    if not element.field == irreducible_poly.field:
        raise ValueError(f"Arguments `element` and `irreducible_poly` must be over the same field, not {element.field.name} and {irreducible_poly.field.name}.")
    if not element.degree < irreducible_poly.degree:
        raise ValueError(f"Argument `element` must have degree less than `irreducible_poly`, not {element.degree} and {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")

    return _is_primitive_element(element, irreducible_poly)


def _is_primitive_element(element: Poly, irreducible_poly: Poly) -> bool:
    """
    A private version of `is_primitive_element()` without type checking/conversion for internal use.
    """
    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    order = q**m - 1  # Multiplicative order of GF(q^m)
    primes, _ = factors(order)

    for k in sorted([order // pi for pi in primes]):
        g = pow(element, k, irreducible_poly)
        if g == 1:
            return False

    g = pow(element, order, irreducible_poly)
    if g != 1:
        return False

    return True


@set_module("galois")
def primitive_element(irreducible_poly: Poly, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Finds a primitive element :math:`g` of the Galois field :math:`\mathrm{GF}(q^m)` with degree-:math:`m` irreducible polynomial
    :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Parameters
    ----------
    irreducible_poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` that defines the extension field :math:`\mathrm{GF}(q^m)`.
    method
        The search method for finding the primitive element.

    Returns
    -------
    :
        A primitive element :math:`g` of :math:`\mathrm{GF}(q^m)` with irreducible polynomial :math:`f(x)`. The primitive element :math:`g` is
        a polynomial over :math:`\mathrm{GF}(q)` with degree less than :math:`m`.

    See Also
    --------
    is_primitive_element, FieldArray.primitive_element

    Examples
    --------
    .. tab-set::

        .. tab-item:: Min

            Find the smallest primitive element for the degree :math:`5` extension of :math:`\mathrm{GF}(7)`.

            .. ipython:: python

                f = galois.conway_poly(7, 5); f
                g = galois.primitive_element(f); g

            Construct the extension field :math:`\mathrm{GF}(7^5)`. Note, by default, :func:`~galois.GF` uses a Conway polynomial
            as its irreducible polynomial.

            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF)
                int(g) == GF.primitive_element

        .. tab-item:: Max

            Find the largest primitive element for the degree :math:`5` extension of :math:`\mathrm{GF}(7)`.

            .. ipython:: python

                f = galois.conway_poly(7, 5); f
                g = galois.primitive_element(f, method="max"); g

            Construct the extension field :math:`\mathrm{GF}(7^5)`. Note, by default, :func:`~galois.GF` uses a Conway polynomial
            as its irreducible polynomial.

            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF)
                int(g) in GF.primitive_elements

        .. tab-item:: Random

            Find a random primitive element for the degree :math:`5` extension of :math:`\mathrm{GF}(7)`.

            .. ipython:: python

                f = galois.conway_poly(7, 5); f
                g = galois.primitive_element(f, method="random"); g

            Construct the extension field :math:`\mathrm{GF}(7^5)`. Note, by default, :func:`~galois.GF` uses a Conway polynomial
            as its irreducible polynomial.

            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF)
                int(g) in GF.primitive_elements
    """
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument `irreducible_poly` must have degree greater than 1, not {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    field = irreducible_poly.field
    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    start = q
    stop = q**m

    if method == "min":
        for integer in range(start, stop):
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                break
    elif method == "max":
        for integer in range(stop - 1, start - 1, -1):
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                break
    else:
        while True:
            integer = random.randint(start, stop - 1)
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                break

    return element


@set_module("galois")
def primitive_elements(irreducible_poly: Poly) -> List[Poly]:
    r"""
    Finds all primitive elements :math:`g` of the Galois field :math:`\mathrm{GF}(q^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Parameters
    ----------
    irreducible_poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` that defines the extension
        field :math:`\mathrm{GF}(q^m)`.

    Returns
    -------
    :
        List of all primitive elements of :math:`\mathrm{GF}(q^m)` with irreducible polynomial :math:`f(x)`. Each primitive
        element :math:`g` is a polynomial over :math:`\mathrm{GF}(q)` with degree less than :math:`m`.

    See Also
    --------
    is_primitive_element, FieldArray.primitive_elements

    Notes
    -----
    The number of primitive elements of :math:`\mathrm{GF}(q^m)` is :math:`\phi(q^m - 1)`, where
    :math:`\phi(n)` is the Euler totient function. See :obj:`~galois.euler_phi`.

    Examples
    --------
    Find all primitive elements for the degree :math:`4` extension of :math:`\mathrm{GF}(3)`.

    .. ipython:: python

        f = galois.conway_poly(3, 4); f
        g = galois.primitive_elements(f); g

    Construct the extension field :math:`\mathrm{GF}(3^4)`. Note, by default, :func:`~galois.GF` uses a Conway polynomial
    as its irreducible polynomial.

    .. ipython:: python

        GF = galois.GF(3**4)
        print(GF)
        np.array_equal([int(gi) for gi in g], GF.primitive_elements)

    The number of primitive elements is given by :math:`\phi(q^m - 1)`.

    .. ipython:: python

        phi = galois.euler_phi(3**4 - 1); phi
        len(g) == phi
    """
    # Find one primitive element first
    element = primitive_element(irreducible_poly)

    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    elements = []
    for totative in totatives(q**m - 1):
        h = pow(element, totative, irreducible_poly)
        elements.append(h)

    elements = sorted(elements, key=int)  # Sort element lexicographically

    return elements
