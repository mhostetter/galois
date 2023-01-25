"""
A module containing functions that search for monic irreducible or primitive polynomials.

Searches are performed in lexicographic, reverse lexicographic, or random order. Additionally, a fixed number
of non-zero terms may be specified. Memoization is heavily used to prevent repeated, expensive searches.
"""
from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Type

import numpy as np

from .._domains import Array, _factory
from . import _constructors

if TYPE_CHECKING:
    from ._poly import Poly


@functools.lru_cache(maxsize=8192)
def _deterministic_search(
    field: Type[Array],
    start: int,
    stop: int,
    step: int,
    test: str,
) -> Poly | None:
    """
    Searches for a monic polynomial in the specified range, returning the first one that passes the specified test
    (either 'is_irreducible()' or 'is_primitive()'). This function returns `None` if no such polynomial exists.
    """
    for element in range(start, stop, step):
        poly = _constructors.POLY_INT(element, field=field)
        if getattr(poly, test)():
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
    Iterates over all monic polynomials of the given degree and number of non-zero terms in lexicographical
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
    Recursively finds all monic polynomials having non-zero coefficients `coeffs` with degree `degrees` with `terms`
    additional non-zero terms. The polynomials are found in lexicographical order, only yielding those that pass
    the specified test (either 'is_irreducible()' or 'is_primitive()').
    """
    if terms == 0:
        # There are no more terms, yield the polynomial.
        poly = _constructors.POLY_DEGREES(degrees, coeffs, field=field)
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


def _random_search(order: int, degree: int, test: str) -> Iterator[Poly]:
    """
    Searches for a random monic polynomial of specified degree, only yielding those that pass the specified test
    (either 'is_irreducible()' or 'is_primitive()').
    """
    assert test in ["is_irreducible", "is_primitive"]
    field = _factory.FIELD_FACTORY(order)

    # Only search monic polynomials of degree m over GF(q)
    start = order**degree
    stop = 2 * order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = _constructors.POLY_INT(integer, field=field)
        if getattr(poly, test)():
            yield poly


def _random_search_fixed_terms(
    order: int,
    degree: int,
    terms: int,
    test: str,
) -> Iterator[Poly]:
    """
    Searches for a random monic polynomial of specified degree and number of non-zero terms, only yielding those that
    pass the specified test (either 'is_irreducible()' or 'is_primitive()').
    """
    assert test in ["is_irreducible", "is_primitive"]
    field = _factory.FIELD_FACTORY(order)

    if terms == 1:
        # The x^m term is always 1. If there's only one term, then the x^m is the polynomial.
        poly = _constructors.POLY_DEGREES([degree], [1], field=field)
        if getattr(poly, test)():
            yield poly
    else:
        while True:
            # The x^m term is always 1 and the x^0 term is always non-zero.
            mid_degrees = random.sample(range(1, degree), terms - 2)
            mid_coeffs = np.random.randint(1, field.order, terms - 2)
            x0_coeff = np.random.randint(1, field.order)
            degrees = (degree, *mid_degrees, 0)
            coeffs = (1, *mid_coeffs, x0_coeff)
            poly = _constructors.POLY_DEGREES(degrees, coeffs, field=field)
            if getattr(poly, test)():
                yield poly


@functools.lru_cache(maxsize=8192)
def _minimum_terms(order: int, degree: int, test: str) -> int:
    """
    Finds the minimum number of terms of an irreducible or primitive polynomial of specified degree over the
    finite field of specified order.
    """
    assert test in ["is_irreducible", "is_primitive"]

    if order == 2 and degree > 1:
        # In GF(2), polynomials with even terms are always reducible. The only exception is x + 1.
        start, stop, step = 1, degree + 2, 2
    else:
        start, stop, step = 1, degree + 2, 1

    for terms in range(start, stop, step):
        try:
            # If a polynomial with the specified number of terms exists, then the current number of terms is
            # the minimum number of terms.
            next(_deterministic_search_fixed_terms(order, degree, terms, test))
            return terms
        except StopIteration:
            # Continue to the next number of terms.
            pass

    poly_type = "irreducible" if test == "is_irreducible" else "primitive"
    raise RuntimeError(
        f"Could not find the minimum number of terms for a degree-{degree} {poly_type} polynomial over GF({order}). "
        "This should never happen. Please open a GitHub issue."
    )
