"""
A module containing functions that search for monic irreducible or primitive polynomials.

Searches are performed in lexicographic, reverse lexicographic, or random order. Additionally, a fixed number
of non-zero terms may be specified. Memoization is heavily used to prevent repeated, expensive searches.
"""

from __future__ import annotations

import functools
import random
from typing import Iterator, Sequence, Type

import numpy as np

from .._domains import Array, _factory
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
        poly = Poly.Int(element, field=field)
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
    Iterates over all monic polynomials of the given degree and number of non-zero terms in lexicographic
    order, only yielding those that pass the specified test (either 'is_irreducible()' or 'is_primitive()').

    One of the non-zero degrees is the x^m term and the other is the x^0 term. The x^0 term is required so that
    the polynomial is irreducible.
    """
    assert test in ["is_irreducible", "is_primitive"]
    field = _factory.FIELD_FACTORY(order)

    # A wrapper function around range to iterate forwards or backwards.
    def direction(x):
        if reverse:
            return reversed(x)
        return x

    def recursive(
        degrees: Sequence[int],
        coeffs: Sequence[int],
        terms: int,
    ) -> Iterator[Poly]:
        if terms == 0:
            # There are no more terms, yield the polynomial.
            poly = Poly.Degrees(degrees, coeffs, field=field)
            if getattr(poly, test)():
                yield poly
        elif terms == 1:
            # The last term must be the x^0 term, so we don't need to loop over possible degrees.
            for c in direction(range(1, field.order)):
                next_degrees = (*degrees, 0)
                next_coeffs = (*coeffs, c)
                yield from recursive(next_degrees, next_coeffs, terms - 1)
        else:
            # Find the next term's degree. It must be at least terms - 1 so that the polynomial can have the specified
            # number of terms of lesser degree. It must also be less than the degree of the previous term.
            for d in direction(range(terms - 1, degrees[-1])):
                for c in direction(range(1, field.order)):
                    next_degrees = (*degrees, d)
                    next_coeffs = (*coeffs, c)
                    yield from recursive(next_degrees, next_coeffs, terms - 1)

    # Initialize the search by setting the first term to x^m with coefficient 1. This function will
    # recursively add the remaining terms, with the last term being x^0.
    yield from recursive([degree], [1], terms - 1)


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
        poly = Poly.Int(integer, field=field)
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
        poly = Poly.Degrees([degree], [1], field=field)
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
            poly = Poly.Degrees(degrees, coeffs, field=field)
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
