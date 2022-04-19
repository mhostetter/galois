"""
A module containing functions to generate and test irreducible polynomials.
"""
import functools
import random
from typing import Iterator, Optional
from typing_extensions import Literal

import numpy as np

from .._domains._array import FIELD_FACTORY
from .._overrides import set_module
from .._prime import factors, is_prime_power

from ._functions import gcd
from ._poly import Poly

__all__ = ["is_irreducible", "irreducible_poly", "irreducible_polys"]


@set_module("galois")
def is_irreducible(poly: Poly) -> bool:
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)` is irreducible.

    Parameters
    ----------
    poly
        A polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    :
        `True` if the polynomial is irreducible.

    See Also
    --------
    is_primitive, irreducible_poly, irreducible_polys

    Notes
    -----
    A polynomial :math:`f(x) \in \mathrm{GF}(p^m)[x]` is *reducible* over :math:`\mathrm{GF}(p^m)` if it can
    be represented as :math:`f(x) = g(x) h(x)` for some :math:`g(x), h(x) \in \mathrm{GF}(p^m)[x]` of strictly
    lower degree. If :math:`f(x)` is not reducible, it is said to be *irreducible*. Since Galois fields are not algebraically
    closed, such irreducible polynomials exist.

    This function implements Rabin's irreducibility test. It says a degree-:math:`m` polynomial :math:`f(x)`
    over :math:`\mathrm{GF}(q)` for prime power :math:`q` is irreducible if and only if :math:`f(x)\ |\ (x^{q^m} - x)`
    and :math:`\textrm{gcd}(f(x),\ x^{q^{m_i}} - x) = 1` for :math:`1 \le i \le k`, where :math:`m_i = m/p_i` for
    the :math:`k` prime divisors :math:`p_i` of :math:`m`.

    References
    ----------
    * Rabin, M. Probabilistic algorithms in finite fields. SIAM Journal on Computing (1980), 273–280. https://apps.dtic.mil/sti/pdfs/ADA078416.pdf
    * Gao, S. and Panarino, D. Tests and constructions of irreducible polynomials over finite fields. https://www.math.clemson.edu/~sgao/papers/GP97a.pdf
    * Section 4.5.1 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf
    * https://en.wikipedia.org/wiki/Factorization_of_polynomials_over_finite_fields

    Examples
    --------
    .. ipython:: python

        # Conway polynomials are always irreducible (and primitive)
        f = galois.conway_poly(2, 5); f

        # f(x) has no roots in GF(2), a necessary but not sufficient condition of being irreducible
        f.roots()

        galois.is_irreducible(f)

    .. ipython:: python

        g = galois.irreducible_poly(2**4, 2, method="random"); g
        h = galois.irreducible_poly(2**4, 3, method="random"); h
        f = g * h; f

        galois.is_irreducible(f)
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")

    if poly.degree == 0:
        # Over fields, f(x) = 0 is the zero element of GF(p^m)[x] and f(x) = c are the units of GF(p^m)[x]. Both the
        # zero element and the units are not irreducible over the polynomial ring GF(p^m)[x].
        return False

    if poly.degree == 1:
        # f(x) = x + a (even a = 0) in any Galois field is irreducible
        return True

    if poly.coeffs[-1] == 0:
        # g(x) = x can be factored, therefore it is not irreducible
        return False

    if poly.field.order == 2 and poly.nonzero_coeffs.size % 2 == 0:
        # Polynomials over GF(2) with degree at least 2 and an even number of terms satisfy f(1) = 0, hence
        # g(x) = x + 1 can be factored. Section 4.5.2 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf.
        return False

    field = poly.field
    q = field.order
    m = poly.degree
    x = Poly.Identity(field)

    primes, _ = factors(m)
    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([m // pi for pi in primes]):
        # The GCD of f(x) and (x^(q^(m/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of m
        hi = pow(h0, q**(ni - n0), poly)
        g = gcd(poly, hi - x)
        if g != 1:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(q^m) - x) to be irreducible
    h = pow(h0, q**(m - n0), poly)
    g = (h - x) % poly
    if g != 0:
        return False

    return True


@set_module("galois")
def irreducible_poly(order: int, degree: int, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

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
    is_irreducible, primitive_poly, conway_poly

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Search methods

            Find the lexicographically-minimal monic irreducible polynomial.

            .. ipython:: python

                galois.irreducible_poly(7, 3)

            Find the lexicographically-maximal monic irreducible polynomial.

            .. ipython:: python

                galois.irreducible_poly(7, 3, method="max")

            Find a random monic irreducible polynomial.

            .. ipython:: python

                galois.irreducible_poly(7, 3, method="random")

        .. tab-item:: Properties

            Find a random monic irreducible polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

            .. ipython:: python

                f = galois.irreducible_poly(7, 5, method="random"); f
                galois.is_irreducible(f)

            Monic irreducible polynomials scaled by non-zero field elements (now non-monic) are also irreducible.

            .. ipython:: python

                GF = galois.GF(7)
                g = f * GF(3); g
                galois.is_irreducible(g)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no irreducible polynomials with degree 0.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    if method == "min":
        return next(irreducible_polys(order, degree))
    elif method == "max":
        return next(irreducible_polys(order, degree, reverse=True))
    else:
        return _random_search(order, degree)


@set_module("galois")
def irreducible_polys(order: int, degree: int, reverse: bool = False) -> Iterator[Poly]:
    r"""
    Iterates through all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

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
    is_irreducible, primitive_polys

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Return full list

            All monic irreducible polynomials over :math:`\mathrm{GF}(3)` with degree :math:`4`. You may also use :func:`tuple` on
            the returned generator.

            .. ipython:: python

                list(galois.irreducible_polys(3, 4))

        .. tab-item:: For loop

            Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the polynomials that would
            have been found after the `break` condition is never incurred.

            .. ipython:: python

                for poly in galois.irreducible_polys(3, 4, reverse=True):
                    if poly.coeffs[1] < 2:  # Early exit condition
                        break
                    print(poly)

        .. tab-item:: Manual iteration

            Or, manually iterate over the generator.

            .. ipython:: python

                generator = galois.irreducible_polys(3, 4, reverse=True); generator
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
    Searches for an irreducible polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if is_irreducible(poly):
            return poly

    return None


def _random_search(order, degree) -> Poly:
    """
    Searches for a random irreducible polynomial.
    """
    field = FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2*order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if is_irreducible(poly):
            return poly
