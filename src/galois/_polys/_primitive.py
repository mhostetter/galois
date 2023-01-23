"""
A module containing functions to generate and test primitive polynomials.
"""
from __future__ import annotations

from typing import Iterator

from typing_extensions import Literal

from .._databases import ConwayPolyDatabase, PrimitivePolyDatabase
from .._domains import _factory
from .._helper import export, verify_isinstance
from .._prime import is_prime, is_prime_power
from ._poly import (
    Poly,
    _deterministic_search,
    _deterministic_search_fixed_terms,
    _minimum_terms,
    _random_search,
    _random_search_fixed_terms,
)


@export
def primitive_poly(
    order: int,
    degree: int,
    terms: int | str | None = None,
    method: Literal["min", "max", "random"] = "min",
    use_database: bool = True,
) -> Poly:
    r"""
    Returns a monic primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Arguments:
        order: The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
        degree: The degree :math:`m` of the desired primitive polynomial.
        terms: The desired number of non-zero terms :math:`t` in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        method: The search method for finding the primitive polynomial.

            - `"min"` (default): Returns the lexicographically-first polynomial.
            - `"max"`: Returns the lexicographically-last polynomial.
            - `"random"`: Returns a random polynomial.

    Returns:
        The degree-:math:`m` monic primitive polynomial over :math:`\mathrm{GF}(q)`.

    Raises:
        RuntimeError: If no monic primitive polynomial of degree :math:`m` over :math:`\mathrm{GF}(q)` with
            :math:`t` terms exists. If `terms` is `None` or `"min"`, this should never be raised.

    See Also:
        Poly.is_primitive, matlab_primitive_poly, conway_poly

    Notes:
        If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and
        :math:`a \in \mathrm{GF}(q) \backslash \{0\}`, then :math:`a \cdot f(x)` is also primitive.

        In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
        of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
        of :math:`\mathrm{GF}(q^m)` such that
        :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples:
        Find the lexicographically-first, lexicographically-last, and a random monic primitive polynomial.

        .. ipython:: python

            galois.primitive_poly(7, 3)
            galois.primitive_poly(7, 3, method="max")
            galois.primitive_poly(7, 3, method="random")

        Find the lexicographically-first monic primitive polynomial with four terms.

        .. ipython:: python

            galois.primitive_poly(7, 3, terms=4)

        Find the lexicographically-first monic irreducible polynomial with the minimum number of non-zero terms.

        .. ipython:: python

            galois.primitive_poly(7, 3, terms="min")

        Notice :func:`~galois.primitive_poly` returns the lexicographically-first primitive polynomial but
        :func:`~galois.conway_poly` returns the lexicographically-first primitive polynomial that is *consistent*
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

    Group:
        polys-primitive
    """
    verify_isinstance(order, int)
    verify_isinstance(degree, int)
    verify_isinstance(terms, (int, str), optional=True)

    if not is_prime_power(order):
        raise ValueError(f"Argument 'order' must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0."
        )
    if isinstance(terms, int) and not 1 <= terms <= degree + 1:
        raise ValueError(f"Argument 'terms' must be at least 1 and at most {degree + 1}, not {terms}.")
    if isinstance(terms, str) and not terms in ["min"]:
        raise ValueError(f"Argument 'terms' must be 'min', not {terms!r}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    if terms == "min" and method == "min":
        try:
            if not use_database:
                raise LookupError
            db = PrimitivePolyDatabase()
            degrees, coeffs = db.fetch(order, degree)
            field = _factory.FIELD_FACTORY(order)
            poly = Poly.Degrees(degrees, coeffs, field=field)
            return poly
        except LookupError:
            terms = _minimum_terms(order, degree, "is_primitive")

    try:
        if method == "min":
            return next(primitive_polys(order, degree, terms))
        if method == "max":
            return next(primitive_polys(order, degree, terms, reverse=True))

        # Random search
        if terms is None:
            return next(_random_search(order, degree, "is_primitive"))

        return next(_random_search_fixed_terms(order, degree, terms, "is_primitive"))

    except StopIteration as e:
        terms_str = "any" if terms is None else str(terms)
        raise RuntimeError(
            f"No monic primitive polynomial of degree {degree} over GF({order}) with {terms_str} terms exists."
        ) from e


@export
def primitive_polys(
    order: int,
    degree: int,
    terms: int | str | None = None,
    reverse: bool = False,
) -> Iterator[Poly]:
    r"""
    Iterates through all monic primitive polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Arguments:
        order: The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
        degree: The degree :math:`m` of the desired primitive polynomial.
        terms: The desired number of non-zero terms :math:`t` in the polynomial.

            - `None` (default): Disregards the number of terms while searching for the polynomial.
            - `int`: The exact number of non-zero terms in the polynomial.
            - `"min"`: The minimum possible number of non-zero terms.

        reverse: Indicates to return the primitive polynomials from lexicographically last to first.
            The default is `False`.

    Returns:
        An iterator over all degree-:math:`m` monic primitive polynomials over :math:`\mathrm{GF}(q)`.

    See Also:
        Poly.is_primitive, irreducible_polys

    Notes:
        If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and
        :math:`a \in \mathrm{GF}(q) \backslash \{0\}`, then :math:`a \cdot f(x)` is also primitive.

        In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
        of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
        of :math:`\mathrm{GF}(q^m)` such that
        :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples:
        Find all monic primitive polynomials over :math:`\mathrm{GF}(3)` with degree 4. You may also use `tuple()` on
        the returned generator.

        .. ipython:: python

            list(galois.primitive_polys(3, 4))

        Find all monic primitive polynomials with five terms.

        .. ipython:: python

            list(galois.primitive_polys(3, 4, terms=5))

        Find all monic primitive polynomials with the minimum number of non-zero terms.

        .. ipython:: python

            list(galois.primitive_polys(3, 4, terms="min"))

        Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the
        polynomials that would have been found after the `break` condition is never incurred.

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

    Group:
        polys-primitive
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
        # Find the minimum number of terms required to produce an primitive polynomial of degree m over GF(q).
        # Then yield all monic primitive polynomials of with that number of terms.
        min_terms = _minimum_terms(order, degree, "is_primitive")
        yield from _deterministic_search_fixed_terms(order, degree, min_terms, "is_primitive", reverse)
    elif isinstance(terms, int):
        # Iterate over and test monic polynomials of degree m over GF(q) with `terms` non-zero terms.
        yield from _deterministic_search_fixed_terms(order, degree, terms, "is_primitive", reverse)
    else:
        # Iterate over and test all monic polynomials of degree m over GF(q).
        start = order**degree
        stop = 2 * order**degree
        step = 1
        if reverse:
            start, stop, step = stop - 1, start - 1, -1
        field = _factory.FIELD_FACTORY(order)

        while True:
            poly = _deterministic_search(field, start, stop, step, "is_primitive")
            if poly is not None:
                start = int(poly) + step
                yield poly
            else:
                break


@export
def conway_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Arguments:
        characteristic: The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial
            is over.
        degree: The degree :math:`m` of the Conway polynomial.

    Returns:
        The degree-:math:`m` Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)`.

    See Also:
        Poly.is_primitive, primitive_poly, matlab_primitive_poly

    Raises:
        LookupError: If the Conway polynomial :math:`C_{p,m}(x)` is not found in Frank Luebeck's database.

    Notes:
        A Conway polynomial is an irreducible and primitive polynomial over :math:`\mathrm{GF}(p)` that provides a
        standard representation of :math:`\mathrm{GF}(p^m)` as a splitting field of :math:`C_{p,m}(x)`.
        Conway polynomials provide compatibility between fields and their subfields and, hence, are the common way to
        represent extension fields.

        The Conway polynomial :math:`C_{p,m}(x)` is defined as the lexicographically-first monic primitive polynomial
        of degree :math:`m` over :math:`\mathrm{GF}(p)` that is compatible with all :math:`C_{p,n}(x)` for :math:`n`
        dividing :math:`m`.

        This function uses `Frank Luebeck's Conway polynomial database
        <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_ for fast lookup, not construction.

    Examples:
        Notice :func:`~galois.primitive_poly` returns the lexicographically-first primitive polynomial but
        :func:`~galois.conway_poly` returns the lexicographically-first primitive polynomial that is *consistent*
        with smaller Conway polynomials. This is sometimes the same polynomial.

        .. ipython:: python

            galois.primitive_poly(2, 4)
            galois.conway_poly(2, 4)

        However, it is not always.

        .. ipython:: python

            galois.primitive_poly(7, 10)
            galois.conway_poly(7, 10)

    Group:
        polys-primitive
    """
    verify_isinstance(characteristic, int)
    verify_isinstance(degree, int)

    if not is_prime(characteristic):
        raise ValueError(f"Argument 'characteristic' must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0."
        )

    coeffs = ConwayPolyDatabase().fetch(characteristic, degree)
    field = _factory.FIELD_FACTORY(characteristic)
    poly = Poly(coeffs, field=field)

    return poly


@export
def matlab_primitive_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns Matlab's default primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Arguments:
        characteristic: The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial
            is over.
        degree: The degree :math:`m` of the desired primitive polynomial.

    Returns:
        Matlab's default degree-:math:`m` primitive polynomial over :math:`\mathrm{GF}(p)`.

    See Also:
        Poly.is_primitive, primitive_poly, conway_poly

    Notes:
        This function returns the same result as Matlab's `gfprimdf(m, p)`. Matlab uses the lexicographically-first
        primitive polynomial with minimum terms, which is equivalent to `galois.primitive_poly(p, m, terms="min")`.
        There are three notable exceptions, however:

        1. In :math:`\mathrm{GF}(2^7)`, Matlab uses :math:`x^7 + x^3 + 1`,
           not :math:`x^7 + x + 1`.
        2. In :math:`\mathrm{GF}(2^{14})`, Matlab uses :math:`x^{14} + x^{10} + x^6 + x + 1`,
           not :math:`x^{14} + x^5 + x^3 + x + 1`.
        3. In :math:`\mathrm{GF}(2^{16})`, Matlab uses :math:`x^{16} + x^{12} + x^3 + x + 1`,
           not :math:`x^{16} + x^5 + x^3 + x^2 + 1`.

    Warning:
        This has been tested for all the :math:`\mathrm{GF}(2^m)` fields for :math:`2 \le m \le 16` (Matlab doesn't
        support larger than 16). And it has been spot-checked for :math:`\mathrm{GF}(p^m)`. There may exist other
        exceptions. Please submit a `GitHub issue <https://github.com/mhostetter/galois/issues>`_ if you discover one.

    References:
        - Lin, S. and Costello, D. Error Control Coding. Table 2.7.

    Examples:
        .. ipython:: python

            galois.primitive_poly(2, 6, terms="min")
            galois.matlab_primitive_poly(2, 6)

        Below is one of the exceptions.

        .. ipython:: python

            galois.primitive_poly(2, 7, terms="min")
            galois.matlab_primitive_poly(2, 7)

    Group:
        polys-primitive
    """
    verify_isinstance(characteristic, int)
    verify_isinstance(degree, int)

    if not is_prime(characteristic):
        raise ValueError(f"Argument 'characteristic' must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0."
        )

    # Textbooks and Matlab use the lexicographically-first primitive polynomial with minimal terms for the default.
    # But for some reason, there are three exceptions. I can't determine why.
    if characteristic == 2 and degree == 7:
        # Not the lexicographically-first of x^7 + x + 1.
        return Poly.Degrees([7, 3, 0])

    if characteristic == 2 and degree == 14:
        # Not the lexicographically-first of x^14 + x^5 + x^3 + x + 1.
        return Poly.Degrees([14, 10, 6, 1, 0])

    if characteristic == 2 and degree == 16:
        # Not the lexicographically-first of x^16 + x^5 + x^3 + x^2 + 1.
        return Poly.Degrees([16, 12, 3, 1, 0])

    return primitive_poly(characteristic, degree)
