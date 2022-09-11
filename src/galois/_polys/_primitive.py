"""
A module containing functions to generate and test primitive polynomials.
"""
from __future__ import annotations

import functools
import random
from typing import Iterator
from typing_extensions import Literal

from .._domains import _factory
from .._databases import ConwayPolyDatabase
from .._helper import export, verify_isinstance
from .._prime import is_prime, is_prime_power

from ._poly import Poly


@export
def primitive_poly(order: int, degree: int, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Returns a monic primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    :group: polys-primitive

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired primitive polynomial.
    method
        The search method for finding the primitive polynomial.

        - `"min"` (default): Returns the lexicographically-minimal monic primitive polynomial.
        - `"max"`: Returns the lexicographically-maximal monic primitive polynomial.
        - `"random"`: Returns a randomly generated degree-:math:`m` monic primitive polynomial.

    Returns
    -------
    :
        The degree-:math:`m` monic primitive polynomial over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    Poly.is_primitive, matlab_primitive_poly, conway_poly

    Notes
    -----
    If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also primitive.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    Find the lexicographically minimal and maximal monic primitive polynomial. Also find a random monic primitive
    polynomial.

    .. ipython:: python

        galois.primitive_poly(7, 3)
        galois.primitive_poly(7, 3, method="max")
        galois.primitive_poly(7, 3, method="random")

    Notice :func:`~galois.primitive_poly` returns the lexicographically-minimal primitive polynomial but
    :func:`~galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials. This is sometimes the same polynomial.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

    However, it is not always.

    .. ipython:: python

        galois.primitive_poly(7, 10)
        galois.conway_poly(7, 10)

    Monic primitive polynomials scaled by non-zero field elements (now non-monic) are also primitive.

    .. ipython:: python

        GF = galois.GF(7)
        f = galois.primitive_poly(7, 5, method="random"); f
        f.is_primitive()
        g = f * GF(3); g
        g.is_primitive()
    """
    verify_isinstance(order, int)
    verify_isinstance(degree, int)

    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    if method == "min":
        return next(primitive_polys(order, degree))
    elif method == "max":
        return next(primitive_polys(order, degree, reverse=True))
    else:
        return _random_search(order, degree)


@export
def primitive_polys(order: int, degree: int, reverse: bool = False) -> Iterator[Poly]:
    r"""
    Iterates through all monic primitive polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    :group: polys-primitive

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired primitive polynomial.
    reverse
        Indicates to return the primitive polynomials from lexicographically maximal to minimal. The default is `False`.

    Returns
    -------
    :
        An iterator over all degree-:math:`m` monic primitive polynomials over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    Poly.is_primitive, irreducible_polys

    Notes
    -----
    If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also primitive.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    Find all monic primitive polynomials over :math:`\mathrm{GF}(3)` with degree 4. You may also use `tuple()` on
    the returned generator.

    .. ipython:: python

        list(galois.primitive_polys(3, 4))

    Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the polynomials that would
    have been found after the `break` condition is never incurred.

    .. ipython:: python

        for poly in galois.primitive_polys(3, 4, reverse=True):
            if poly.coeffs[1] < 2:  # Early exit condition
                break
            print(poly)

    Or, manually iterate over the generator.

    .. ipython:: python

        generator = galois.primitive_polys(3, 4, reverse=True); generator
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
    Searches for a primitive polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if poly.is_primitive():
            return poly

    return None


def _random_search(order, degree) -> Poly:
    """
    Searches for a random primitive polynomial.
    """
    field = _factory.FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2*order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if poly.is_primitive():
            return poly


@export
def conway_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    :group: polys-primitive

    Parameters
    ----------
    characteristic
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree
        The degree :math:`m` of the Conway polynomial.

    Returns
    -------
    :
        The degree-:math:`m` Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)`.

    See Also
    --------
    Poly.is_primitive, primitive_poly, matlab_primitive_poly

    Raises
    ------
    LookupError
        If the Conway polynomial :math:`C_{p,m}(x)` is not found in Frank Luebeck's database.

    Notes
    -----
    A Conway polynomial is an irreducible and primitive polynomial over :math:`\mathrm{GF}(p)` that provides a standard
    representation of :math:`\mathrm{GF}(p^m)` as a splitting field of :math:`C_{p,m}(x)`. Conway polynomials
    provide compatability between fields and their subfields and, hence, are the common way to represent extension
    fields.

    The Conway polynomial :math:`C_{p,m}(x)` is defined as the lexicographically-minimal monic primitive polynomial
    of degree :math:`m` over :math:`\mathrm{GF}(p)` that is compatible with all :math:`C_{p,n}(x)` for :math:`n` dividing
    :math:`m`.

    This function uses `Frank Luebeck's Conway polynomial database <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_
    for fast lookup, not construction.

    Examples
    --------
    Notice :func:`~galois.primitive_poly` returns the lexicographically-minimal primitive polynomial but
    :func:`~galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials. This is sometimes the same polynomial.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

    However, it is not always.

    .. ipython:: python

        galois.primitive_poly(7, 10)
        galois.conway_poly(7, 10)
    """
    verify_isinstance(characteristic, int)
    verify_isinstance(degree, int)

    if not is_prime(characteristic):
        raise ValueError(f"Argument 'characteristic' must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")

    coeffs = ConwayPolyDatabase().fetch(characteristic, degree)
    field = _factory.FIELD_FACTORY(characteristic)
    poly = Poly(coeffs, field=field)

    return poly


@export
def matlab_primitive_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns Matlab's default primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    :group: polys-primitive

    Parameters
    ----------
    characteristic
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired primitive polynomial.

    Returns
    -------
    :
        Matlab's default degree-:math:`m` primitive polynomial over :math:`\mathrm{GF}(p)`.

    See Also
    --------
    Poly.is_primitive, primitive_poly, conway_poly

    Notes
    -----
    This function returns the same result as Matlab's `gfprimdf(m, p)`. Matlab uses the primitive polynomial with minimum terms
    (equivalent to `galois.primitive_poly(p, m, method="min-terms")`) as the default... *mostly*. There are three
    notable exceptions:

    1. :math:`\mathrm{GF}(2^7)` uses :math:`x^7 + x^3 + 1`, not :math:`x^7 + x + 1`.
    2. :math:`\mathrm{GF}(2^{14})` uses :math:`x^{14} + x^{10} + x^6 + x + 1`, not :math:`x^{14} + x^5 + x^3 + x + 1`.
    3. :math:`\mathrm{GF}(2^{16})` uses :math:`x^{16} + x^{12} + x^3 + x + 1`, not :math:`x^{16} + x^5 + x^3 + x^2 + 1`.

    Warning
    -------
    This has been tested for all the :math:`\mathrm{GF}(2^m)` fields for :math:`2 \le m \le 16` (Matlab doesn't support
    larger than 16). And it has been spot-checked for :math:`\mathrm{GF}(p^m)`. There may exist other exceptions. Please
    submit a GitHub issue if you discover one.

    References
    ----------
    * Lin, S. and Costello, D. Error Control Coding. Table 2.7.

    Examples
    --------
    .. ipython:: python

        galois.primitive_poly(2, 6)
        galois.matlab_primitive_poly(2, 6)

    .. ipython:: python

        galois.primitive_poly(2, 7)
        galois.matlab_primitive_poly(2, 7)
    """
    verify_isinstance(characteristic, int)
    verify_isinstance(degree, int)

    if not is_prime(characteristic):
        raise ValueError(f"Argument 'characteristic' must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")

    # Textbooks and Matlab use the lexicographically-minimal primitive polynomial for the default. But for some
    # reason, there are three exceptions. I can't determine why.
    if characteristic == 2 and degree == 7:
        # Not the lexicographically-minimal of `x^7 + x + 1`
        return Poly.Degrees([7, 3, 0])
    elif characteristic == 2 and degree == 14:
        # Not the lexicographically-minimal of `x^14 + x^5 + x^3 + x + 1`
        return Poly.Degrees([14, 10, 6, 1, 0])
    elif characteristic == 2 and degree == 16:
        # Not the lexicographically-minimal of `x^16 + x^5 + x^3 + x^2 + 1`
        return Poly.Degrees([16, 12, 3, 1, 0])
    else:
        return primitive_poly(characteristic, degree)
