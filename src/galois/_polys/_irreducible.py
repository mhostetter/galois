"""
A module containing functions to generate and test irreducible polynomials.
"""
from __future__ import annotations

from typing import Iterator

from typing_extensions import Literal

from .._domains import _factory
from .._helper import export, verify_isinstance
from .._prime import is_prime_power
from ._poly import (
    Poly,
    _deterministic_search,
    _deterministic_search_fixed_terms,
    _minimum_terms,
    _random_search,
    _random_search_fixed_terms,
)


@export
def irreducible_poly(
    order: int,
    degree: int,
    terms: int | str | None = None,
    method: Literal["min", "max", "random"] = "min",
) -> Poly:
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Arguments:
        order: The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
        degree: The degree :math:`m` of the desired irreducible polynomial.
        terms: The desired number of non-zero terms :math:`t` in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        method: The search method for finding the irreducible polynomial.

            - `"min"` (default): Returns the lexicographically-first polynomial.
            - `"max"`: Returns the lexicographically-last polynomial.
            - `"random"`: Returns a random polynomial.

    Returns:
        The degree-:math:`m` monic irreducible polynomial over :math:`\mathrm{GF}(q)`.

    Raises:
        RuntimeError: If no monic irreducible polynomial of degree :math:`m` over :math:`\mathrm{GF}(q)` with
            :math:`t` terms exists. If `terms` is `None` or `"min"`, this should never be raised.

    See Also:
        Poly.is_irreducible, primitive_poly, conway_poly

    Notes:
        If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and
        :math:`a \in \mathrm{GF}(q) \backslash \{0\}`, then :math:`a \cdot f(x)` is also irreducible.

        In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of
        :math:`\mathrm{GF}(q)`.

    Examples:
        Find the lexicographically-first, lexicographically-last, and a random monic irreducible polynomial.

        .. ipython:: python

            galois.irreducible_poly(7, 3)
            galois.irreducible_poly(7, 3, method="max")
            galois.irreducible_poly(7, 3, method="random")

        Find the lexicographically-first monic irreducible polynomial with four terms.

        .. ipython:: python

            galois.irreducible_poly(7, 3, terms=4)

        Find the lexicographically-first monic irreducible polynomial with the minimum number of non-zero terms.

        .. ipython:: python

            galois.irreducible_poly(7, 3, terms="min")

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
    verify_isinstance(terms, (int, str), optional=True)

    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no irreducible polynomials with degree 0."
        )
    if isinstance(terms, int) and not 1 <= terms <= degree + 1:
        raise ValueError(f"Argument 'terms' must be at least 1 and at most {degree + 1}, not {terms}.")
    if isinstance(terms, str) and not terms in ["min"]:
        raise ValueError(f"Argument 'terms' must be 'min', not {terms!r}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    try:
        if method == "min":
            return next(irreducible_polys(order, degree, terms))
        if method == "max":
            return next(irreducible_polys(order, degree, terms, reverse=True))

        # Random search
        if terms is None:
            return next(_random_search(order, degree, "is_irreducible"))
        if terms == "min":
            terms = _minimum_terms(order, degree, "is_irreducible")
        return next(_random_search_fixed_terms(order, degree, terms, "is_irreducible"))

    except StopIteration as e:
        terms_str = "any" if terms is None else str(terms)
        raise RuntimeError(
            f"No monic irreducible polynomial of degree {degree} over GF({order}) with {terms_str} terms exists."
        ) from e


@export
def irreducible_polys(
    order: int,
    degree: int,
    terms: int | str | None = None,
    reverse: bool = False,
) -> Iterator[Poly]:
    r"""
    Iterates through all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Arguments:
        order: The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
        degree: The degree :math:`m` of the desired irreducible polynomial.
        terms: The desired number of non-zero terms :math:`t` in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

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

        Find all monic irreducible polynomials with four terms.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4, terms=4))

        Find all monic irreducible polynomials with the minimum number of non-zero terms.

        .. ipython:: python

            list(galois.irreducible_polys(3, 4, terms="min"))

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
    verify_isinstance(terms, (int, str), optional=True)
    verify_isinstance(reverse, bool)

    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 0:
        raise ValueError(f"Argument 'degree' must be at least 0, not {degree}.")
    if isinstance(terms, int) and not 1 <= terms <= degree + 1:
        raise ValueError(f"Argument 'terms' must be at least 1 and at most {degree + 1}, not {terms}.")
    if isinstance(terms, str) and not terms in ["min"]:
        raise ValueError(f"Argument 'terms' must be 'min', not {terms!r}.")

    if terms == "min":
        # Find the minimum number of terms required to produce an irreducible polynomial of degree m over GF(q).
        # Then yield all monic irreducible polynomials of with that number of terms.
        min_terms = _minimum_terms(order, degree, "is_irreducible")
        yield from _deterministic_search_fixed_terms(order, degree, min_terms, "is_irreducible", reverse)
    elif isinstance(terms, int):
        # Iterate over and test monic polynomials of degree m over GF(q) with `terms` non-zero terms.
        yield from _deterministic_search_fixed_terms(order, degree, terms, "is_irreducible", reverse)
    else:
        # Iterate over and test all monic polynomials of degree m over GF(q).
        start = order**degree
        stop = 2 * order**degree
        step = 1
        if reverse:
            start, stop, step = stop - 1, start - 1, -1
        field = _factory.FIELD_FACTORY(order)

        while True:
            poly = _deterministic_search(field, start, stop, step, "is_irreducible")
            if poly is not None:
                start = int(poly) + step
                yield poly
            else:
                break
