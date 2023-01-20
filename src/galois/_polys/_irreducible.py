"""
A module containing functions to generate and test irreducible polynomials.
"""
from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Type

from typing_extensions import Literal

from .._domains import Array, _factory
from .._helper import export, verify_isinstance
from .._prime import is_prime_power
from ._poly import Poly

if TYPE_CHECKING:
    from .._fields import FieldArray


@export
def irreducible_poly(
    order: int,
    degree: int,
    terms: int | None = None,
    method: Literal["min", "max", "random"] = "min",
) -> Poly:
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Arguments:
        order: The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
        degree: The degree :math:`m` of the desired irreducible polynomial.
        terms: The desired number of non-zero terms :math:`t` in the polynomial. The default is `None` which disregards
            the number of terms while searching for the polynomial.
        method: The search method for finding the irreducible polynomial.

            - `"min"` (default): Returns the lexicographically-first polynomial.
            - `"max"`: Returns the lexicographically-last polynomial.
            - `"random"`: Returns a random polynomial.

    Returns:
        The degree-:math:`m` monic irreducible polynomial over :math:`\mathrm{GF}(q)`.

    Raises:
        RuntimeError: If no monic irreducible polynomial of degree :math:`m` over :math:`\mathrm{GF}(q)` with
            :math:`t` terms exists. If `terms=None`, this should never be raised.

    See Also:
        Poly.is_irreducible, primitive_poly, conway_poly

    Notes:
        If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and
        :math:`a \in \mathrm{GF}(q) \backslash \{0\}`, then :math:`a \cdot f(x)` is also irreducible.

        In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of
        :math:`\mathrm{GF}(q)`.

    Examples:
        Find the lexicographically first and last monic irreducible polynomial. Also find a random monic
        irreducible polynomial.

        .. ipython:: python

            galois.irreducible_poly(7, 3)
            galois.irreducible_poly(7, 3, method="max")
            galois.irreducible_poly(7, 3, method="random")

        Find an irreducible polynomial with three terms.

        .. ipython:: python

            galois.irreducible_poly(7, 3, terms=3)

        Monic irreducible polynomials scaled by non-zero field elements (now non-monic) are also irreducible.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.irreducible_poly(7, 5, method="random"); f
            f.is_irreducible()
            g = f * GF(3); g
            g.is_irreducible()

    Group:
        polys-irreducible
    """
    verify_isinstance(order, int)
    verify_isinstance(degree, int)
    verify_isinstance(terms, int, optional=True)

    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no irreducible polynomials with degree 0."
        )
    if terms is not None and not 1 <= terms <= degree + 1:
        raise ValueError(f"Argument 'terms' must be at least 1 and at most {degree + 1}, not {terms}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    try:
        if method == "min":
            poly = next(irreducible_polys(order, degree, terms))
        elif method == "max":
            poly = next(irreducible_polys(order, degree, terms, reverse=True))
        else:
            poly = _random_search(order, degree, terms)
    except StopIteration as e:
        terms_str = "any" if terms is None else str(terms)
        raise RuntimeError(
            f"No monic irreducible polynomial of degree {degree} over GF({order}) with {terms_str} terms exists."
        ) from e

    return poly


@export
def irreducible_polys(
    order: int,
    degree: int,
    terms: int | None = None,
    reverse: bool = False,
) -> Iterator[Poly]:
    r"""
    Iterates through all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Arguments:
        order: The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
        degree: The degree :math:`m` of the desired irreducible polynomial.
        terms: The desired number of non-zero terms :math:`t` in the polynomials. The default is `None` which
            disregards the number of terms while searching for the polynomials.
        reverse: Indicates to return the irreducible polynomials from lexicographically last to first.
            The default is `False`.

    Returns:
        An iterator over all degree-:math:`m` monic irreducible polynomials over :math:`\mathrm{GF}(q)`.

    See Also:
        Poly.is_irreducible, primitive_polys

    Notes:
        If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and
        :math:`a \in \mathrm{GF}(q) \backslash \{0\}`, then :math:`a \cdot f(x)` is also irreducible.

        In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of
        :math:`\mathrm{GF}(q)`.

    Examples:
        Find all monic irreducible polynomials over :math:`\mathrm{GF}(3)` with degree 4. You may also use `tuple()` on
        the returned generator.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4))

        Find all monic irreducible polynomials with three terms.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4, terms=3))

        Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the
        polynomials that would have been found after the `break` condition is never incurred.

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

    Group:
        polys-irreducible
    """
    verify_isinstance(order, int)
    verify_isinstance(degree, int)
    verify_isinstance(terms, int, optional=True)
    verify_isinstance(reverse, bool)

    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 0:
        raise ValueError(f"Argument 'degree' must be at least 0, not {degree}.")
    if terms is not None and not 1 <= terms <= degree + 1:
        raise ValueError(f"Argument 'terms' must be at least 1 and at most {degree + 1}, not {terms}.")

    if terms is None:
        # Iterate over and test all monic polynomials of degree m over GF(q).
        start = order**degree
        stop = 2 * order**degree
        step = 1
        if reverse:
            start, stop, step = stop - 1, start - 1, -1
        field = _factory.FIELD_FACTORY(order)

        while True:
            poly = _deterministic_search(field, start, stop, step)
            if poly is not None:
                start = int(poly) + step
                yield poly
            else:
                break
    else:
        # Iterate over and test monic polynomials of degree m over GF(q) with `terms` non-zero terms.
        yield from _deterministic_search_fixed_terms(order, degree, terms, "is_irreducible", reverse)


@functools.lru_cache(maxsize=4096)
def _deterministic_search(
    field: Type[FieldArray],
    start: int,
    stop: int,
    step: int,
) -> Poly | None:
    """
    Searches for an irreducible polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if poly.is_irreducible():
            return poly

    return None


def _deterministic_search_fixed_terms(
    order: int,
    degree: int,
    terms: int,
    test: str,
    reverse: bool = False,
) -> Iterator[Poly]:
    """
    Iterates over all polynomials of the given degree and number of non-zero terms in lexicographical
    order, only yielding those that pass the specified test (either 'is_irreducible()' or 'is_primitive()').
    """
    assert test in ["is_irreducible", "is_primitive"]
    field = _factory.FIELD_FACTORY(order)

    # A wrapper function around range to iterate forwards or backwards.
    def direction(x):
        if reverse:
            return reversed(x)
        return x

    # Initialize the search by setting the first term to x^m with coefficient 1. This function will
    # recursively add the remaining terms, with the last term being x^0.
    yield from _deterministic_search_fixed_terms_recursive([degree], [1], terms - 1, test, field, direction)


def _deterministic_search_fixed_terms_recursive(
    degrees: Iterable[int],
    coeffs: Iterable[int],
    terms: int,
    test: str,
    field: Type[Array],
    direction: Callable[[Iterable[int]], Iterable[int]],
) -> Iterator[Poly]:
    """
    Recursively finds all polynomials having non-zero coefficients `coeffs` with degree `degrees` with `terms`
    additional non-zero terms. The polynomials are found in lexicographical order, only yielding those that pass
    the specified test (either 'is_irreducible()' or 'is_primitive()').
    """
    if terms == 0:
        # There are no more terms, yield the polynomial.
        poly = Poly.Degrees(degrees, coeffs, field=field)
        if getattr(poly, test)():
            yield poly
    elif terms == 1:
        # The last term must be the x^0 term, so we don't need to loop over possible degrees.
        for coeff in direction(range(1, field.order)):
            next_degrees = (*degrees, 0)
            next_coeffs = (*coeffs, coeff)
            yield from _deterministic_search_fixed_terms_recursive(
                next_degrees, next_coeffs, terms - 1, test, field, direction
            )
    else:
        # Find the next term's degree. It must be at least terms - 1 so that the polynomial can have the specified
        # number of terms of lesser degree. It must also be less than the degree of the previous term.
        for degree in direction(range(terms - 1, degrees[-1])):
            for coeff in direction(range(1, field.order)):
                next_degrees = (*degrees, degree)
                next_coeffs = (*coeffs, coeff)
                yield from _deterministic_search_fixed_terms_recursive(
                    next_degrees, next_coeffs, terms - 1, test, field, direction
                )


def _random_search(order: int, degree: int, terms: int | None) -> Poly:
    """
    Searches for a random irreducible polynomial.
    """
    field = _factory.FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2 * order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if terms is not None and poly.nonzero_coeffs.size != terms:
            continue
        if poly.is_irreducible():
            return poly
