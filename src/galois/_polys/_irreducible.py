"""
A module containing functions to generate and test irreducible polynomials.
"""
from __future__ import annotations

import functools
import random
from typing import Iterator
from typing_extensions import Literal

from .._domains import _factory
from .._helper import export, verify_isinstance
from .._prime import is_prime_power

from ._poly import Poly


@export
def irreducible_poly(order: int, degree: int, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    :group: polys-irreducible

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired irreducible polynomial.
    method
        The search method for finding the irreducible polynomial.

        - `"min"` (default): Returns the lexicographically-minimal monic irreducible polynomial.
        - `"max"`: Returns the lexicographically-maximal monic irreducible polynomial.
        - `"random"`: Returns a randomly generated degree-:math:`m` monic irreducible polynomial.

    Returns
    -------
    :
        The degree-:math:`m` monic irreducible polynomial over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    Poly.is_irreducible, primitive_poly, conway_poly

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    Find the lexicographically minimal and maximal monic irreducible polynomial. Also find a random monic irreducible
    polynomial.

    .. ipython:: python

        galois.irreducible_poly(7, 3)
        galois.irreducible_poly(7, 3, method="max")
        galois.irreducible_poly(7, 3, method="random")

    Monic irreducible polynomials scaled by non-zero field elements (now non-monic) are also irreducible.

    .. ipython:: python

        GF = galois.GF(7)
        f = galois.irreducible_poly(7, 5, method="random"); f
        f.is_irreducible()
        g = f * GF(3); g
        g.is_irreducible()
    """
    verify_isinstance(order, int)
    verify_isinstance(degree, int)
    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument 'degree' must be at least 1, not {degree}. There are no irreducible polynomials with degree 0.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    if method == "min":
        return next(irreducible_polys(order, degree))
    elif method == "max":
        return next(irreducible_polys(order, degree, reverse=True))
    else:
        return _random_search(order, degree)


@export
def irreducible_polys(order: int, degree: int, reverse: bool = False) -> Iterator[Poly]:
    r"""
    Iterates through all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    :group: polys-irreducible

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired irreducible polynomial.
    reverse
        Indicates to return the irreducible polynomials from lexicographically maximal to minimal. The default is `False`.

    Returns
    -------
    :
        An iterator over all degree-:math:`m` monic irreducible polynomials over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    Poly.is_irreducible, primitive_polys

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    Find all monic irreducible polynomials over :math:`\mathrm{GF}(3)` with degree 4. You may also use `tuple()` on
    the returned generator.

    .. ipython:: python

        list(galois.irreducible_polys(3, 4))

    Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the polynomials that would
    have been found after the `break` condition is never incurred.

    .. ipython:: python

        for poly in galois.irreducible_polys(3, 4, reverse=True):
            if poly.coeffs[1] < 2:  # Early exit condition
                break
            print(poly)

    Or, manually iterate over the generator.

    .. ipython:: python

        generator = galois.irreducible_polys(3, 4, reverse=True); generator
        next(generator)
        next(generator)
        next(generator)
    """
    verify_isinstance(order, int)
    verify_isinstance(degree, int)
    verify_isinstance(reverse, bool)
    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 0:
        raise ValueError(f"Argument 'degree' must be at least 0, not {degree}.")

    field = _factory.FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(q)
    start = order**degree
    stop = 2*order**degree
    step = 1

    if reverse:
        start, stop, step = stop - 1, start - 1, -1

    while True:
        poly = _deterministic_search(field, start, stop, step)
        if poly is not None:
            start = int(poly) + step
            yield poly
        else:
            break


@functools.lru_cache(maxsize=4096)
def _deterministic_search(field, start, stop, step) -> Poly | None:
    """
    Searches for an irreducible polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if poly.is_irreducible():
            return poly

    return None


def _random_search(order, degree) -> Poly:
    """
    Searches for a random irreducible polynomial.
    """
    field = _factory.FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2*order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if poly.is_irreducible():
            return poly
