"""
A module containing functions to generate and test primitive polynomials.
"""
import functools
import random
from typing import Iterator, Optional
from typing_extensions import Literal

import numpy as np

from .._domains._array import FIELD_FACTORY
from .._databases import ConwayPolyDatabase
from .._overrides import set_module
from .._prime import factors, is_prime, is_prime_power

from ._poly import Poly
from ._irreducible import is_irreducible

__all__ = ["is_primitive", "primitive_poly", "primitive_polys", "conway_poly", "matlab_primitive_poly"]


@set_module("galois")
def is_primitive(poly: Poly) -> bool:
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is primitive.

    Parameters
    ----------
    poly
        A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Returns
    -------
    :
        `True` if the polynomial is primitive.

    See Also
    --------
    is_irreducible, primitive_poly, matlab_primitive_poly, primitive_polys

    Notes
    -----
    A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is *primitive* if it is irreducible and
    :math:`f(x)\ |\ (x^k - 1)` for :math:`k = q^m - 1` and no :math:`k` less than :math:`q^m - 1`.

    References
    ----------
    * Algorithm 4.77 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf

    Examples
    --------
    All Conway polynomials are primitive.

    .. ipython:: python

        f = galois.conway_poly(2, 8); f
        galois.is_primitive(f)

        f = galois.conway_poly(3, 5); f
        galois.is_primitive(f)

    The irreducible polynomial of :math:`\mathrm{GF}(2^8)` for AES is not primitive.

    .. ipython:: python

        f = galois.Poly.Degrees([8, 4, 3, 1, 0]); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")

    if poly.degree == 0:
        # Over fields, f(x) = 0 is the zero element of GF(p^m)[x] and f(x) = c are the units of GF(p^m)[x]. Both the
        # zero element and the units are not irreducible over the polynomial ring GF(p^m)[x], and therefore cannot
        # be primitive.
        return False

    if poly.field.order == 2 and poly.degree == 1:
        # There is only one primitive polynomial in GF(2)
        return poly == Poly([1, 1])

    if poly.coeffs[-1] == 0:
        # A primitive polynomial cannot have zero constant term
        # TODO: Why isn't f(x) = x primitive? It's irreducible and passes the primitivity tests.
        return False

    if not is_irreducible(poly):
        # A polynomial must be irreducible to be primitive
        return False

    field = poly.field
    q = field.order
    m = poly.degree
    one = Poly.One(field)

    primes, _ = factors(q**m - 1)
    x = Poly.Identity(field)
    for ki in sorted([(q**m - 1) // pi for pi in primes]):
        # f(x) must not divide (x^((q^m - 1)/pi) - 1) for f(x) to be primitive, where pi are the prime factors of q**m - 1
        h = pow(x, ki, poly)
        g = (h - one) % poly
        if g == 0:
            return False

    return True


@set_module("galois")
def primitive_poly(order: int, degree: int, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Returns a monic primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

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
    is_primitive, matlab_primitive_poly, conway_poly

    Notes
    -----
    If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also primitive.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Search methods

            Find the lexicographically-minimal monic primitive polynomial.

            .. ipython:: python

                galois.primitive_poly(7, 3)

            Find the lexicographically-maximal monic primitive polynomial.

            .. ipython:: python

                galois.primitive_poly(7, 3, method="max")

            Find a random monic primitive polynomial.

            .. ipython:: python

                galois.primitive_poly(7, 3, method="random")

        .. tab-item:: Primitive vs. Conway

            Notice :func:`~galois.primitive_poly` returns the lexicographically-minimal primitive polynomial but
            :func:`~galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
            with smaller Conway polynomials.

            This is sometimes the same polynomial.

            .. ipython:: python

                galois.primitive_poly(2, 4)
                galois.conway_poly(2, 4)

            However, it is not always.

            .. ipython:: python

                galois.primitive_poly(7, 10)
                galois.conway_poly(7, 10)

        .. tab-item:: Properties

            Find a random monic primitive polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

            .. ipython:: python

                f = galois.primitive_poly(7, 5, method="random"); f
                galois.is_primitive(f)

            Monic primitive polynomials scaled by non-zero field elements (now non-monic) are also primitive.

            .. ipython:: python

                GF = galois.GF(7)
                g = f * GF(3); g
                galois.is_primitive(g)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    if method == "min":
        return next(primitive_polys(order, degree))
    elif method == "max":
        return next(primitive_polys(order, degree, reverse=True))
    else:
        return _random_search(order, degree)


@set_module("galois")
def primitive_polys(order: int, degree: int, reverse: bool = False) -> Iterator[Poly]:
    r"""
    Iterates through all monic primitive polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

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
    is_primitive, irreducible_polys

    Notes
    -----
    If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also primitive.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Return full list

            All monic primitive polynomials over :math:`\mathrm{GF}(3)` with degree :math:`4`. You may also use :func:`tuple` on
            the returned generator.

            .. ipython:: python

                list(galois.primitive_polys(3, 4))

        .. tab-item:: For loop

            Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the polynomials that would
            have been found after the `break` condition is never incurred.

            .. ipython:: python

                for poly in galois.primitive_polys(3, 4, reverse=True):
                    if poly.coeffs[1] < 2:  # Early exit condition
                        break
                    print(poly)

        .. tab-item:: Manual iteration

            Or, manually iterate over the generator.

            .. ipython:: python

                generator = galois.primitive_polys(3, 4, reverse=True); generator
                next(generator)
                next(generator)
                next(generator)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 0:
        raise ValueError(f"Argument `degree` must be at least 0, not {degree}.")

    field = FIELD_FACTORY(order)

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
def _deterministic_search(field, start, stop, step) -> Optional[Poly]:
    """
    Searches for an primitive polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if is_primitive(poly):
            return poly

    return None


def _random_search(order, degree) -> Poly:
    """
    Searches for a random primitive polynomial.
    """
    field = FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2*order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if is_primitive(poly):
            return poly


@set_module("galois")
def conway_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

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
    is_primitive, primitive_poly, matlab_primitive_poly

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
    with smaller Conway polynomials.

    This is sometimes the same polynomial.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

    However, it is not always.

    .. ipython:: python

        galois.primitive_poly(7, 10)
        galois.conway_poly(7, 10)
    """
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")

    coeffs = ConwayPolyDatabase().fetch(characteristic, degree)
    field = FIELD_FACTORY(characteristic)
    poly = Poly(coeffs, field=field)

    return poly


@set_module("galois")
def matlab_primitive_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns Matlab's default primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

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
    is_primitive, primitive_poly, conway_poly

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
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")

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
