"""
A module containing functions to find and test Conway polynomials.
"""

from __future__ import annotations

import functools
from typing import Iterator, Sequence

from .._databases import ConwayPolyDatabase
from .._domains import _factory
from .._helper import export, method_of, verify_isinstance
from .._prime import divisors, is_prime
from ._poly import Poly
from ._primitive import is_primitive


@method_of(Poly)
def is_conway(f: Poly, search: bool = False) -> bool:
    r"""
    Checks whether the degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is the
    Conway polynomial $C_{p,m}(x)$.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Arguments:
        search: Manually search for Conway polynomials if they are not included in `Frank Luebeck's database
            <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_. The default is `False`.

            .. slow-performance::

                Manually searching for a Conway polynomial is *very* computationally expensive.

    Returns:
        `True` if the polynomial $f(x)$ is the Conway polynomial $C_{p,m}(x)$.

    Raises:
        LookupError: If `search=False` and the Conway polynomial $C_{p,m}$ is not found in Frank Luebeck's
            database.

    See Also:
        conway_poly, Poly.is_conway_consistent, Poly.is_primitive

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is the *Conway polynomial*
        $C_{p,m}(x)$ if it is monic, primitive, compatible with Conway polynomials $C_{p,n}(x)$ for all
        $n \mid m$, and is lexicographically first according to a special ordering.

        A Conway polynomial $C_{p,m}(x)$ is *compatible* with Conway polynomials $C_{p,n}(x)$ for
        $n \mid m$ if $C_{p,n}(x^r)$ divides $C_{p,m}(x)$, where $r = \frac{p^m - 1}{p^n - 1}$.

        The Conway lexicographic ordering is defined as follows. Given two degree-$m$ polynomials
        $g(x) = \sum_{i=0}^m g_i x^i$ and $h(x) = \sum_{i=0}^m h_i x^i$, then $g < h$ if and only if
        there exists $i$ such that $g_j = h_j$ for all $j > i$ and
        $(-1)^{m-i} g_i < (-1)^{m-i} h_i$.

    References:
        - http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP7.html
        - Lenwood S. Heath, Nicholas A. Loehr, New algorithms for generating Conway polynomials over finite fields,
          Journal of Symbolic Computation, Volume 38, Issue 2, 2004, Pages 1003-1024,
          https://www.sciencedirect.com/science/article/pii/S0747717104000331.

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([1, 1, 2, 4], field=GF); f
            g = galois.Poly([1, 6, 0, 4], field=GF); g
            f.is_primitive()
            g.is_primitive()

        They are also consistent with all smaller Conway polynomials.

        .. ipython:: python

            f.is_conway_consistent()
            g.is_conway_consistent()

        Among the multiple candidate Conway polynomials, the lexicographically first (accordingly to a special
        lexicographical order) is the Conway polynomial.

        .. ipython:: python

            f.is_conway()
            g.is_conway()
            galois.conway_poly(7, 3)
    """
    verify_isinstance(search, bool)
    if not is_prime(f.field.order):
        raise ValueError(f"Conway polynomials are only defined over prime fields, not order {f.field.order}.")

    p = f.field.order
    m = f.degree
    C_pm = conway_poly(p, m, search=search)

    return f == C_pm


@method_of(Poly)
@functools.lru_cache()
def is_conway_consistent(f: Poly, search: bool = False) -> bool:
    r"""
    Determines whether the degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is consistent
    with smaller Conway polynomials $C_{p,n}(x)$ for all $n \mid m$.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Arguments:
        search: Manually search for Conway polynomials if they are not included in `Frank Luebeck's database
            <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_. The default is `False`.

            .. slow-performance::

                Manually searching for a Conway polynomial is *very* computationally expensive.

    Returns:
        `True` if the polynomial $f(x)$ is primitive and consistent with smaller Conway polynomials
        $C_{p,n}(x)$ for all $n \mid m$.

    Raises:
        LookupError: If `search=False` and a smaller Conway polynomial $C_{p,n}$ is not found in Frank Luebeck's
            database.

    See Also:
        conway_poly, Poly.is_conway, Poly.is_primitive

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is *compatible* with Conway polynomials
        $C_{p,n}(x)$ for $n \mid m$ if $C_{p,n}(x^r)$ divides $f(x)$, where
        $r = \frac{p^m - 1}{p^n - 1}$.

        A Conway-consistent polynomial has all the properties of a Conway polynomial except that it is not
        necessarily lexicographically first (according to a special ordering).

    References:
        - http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP7.html
        - Lenwood S. Heath, Nicholas A. Loehr, New algorithms for generating Conway polynomials over finite fields,
          Journal of Symbolic Computation, Volume 38, Issue 2, 2004, Pages 1003-1024,
          https://www.sciencedirect.com/science/article/pii/S0747717104000331.

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([1, 1, 2, 4], field=GF); f
            g = galois.Poly([1, 6, 0, 4], field=GF); g
            f.is_primitive()
            g.is_primitive()

        They are also consistent with all smaller Conway polynomials.

        .. ipython:: python

            f.is_conway_consistent()
            g.is_conway_consistent()

        Among the multiple candidate Conway polynomials, the lexicographically first (accordingly to a special
        lexicographical order) is the Conway polynomial.

        .. ipython:: python

            f.is_conway()
            g.is_conway()
            galois.conway_poly(7, 3)
    """
    verify_isinstance(search, bool)
    if not is_prime(f.field.order):
        raise ValueError(f"Conway polynomials are only defined over prime fields, not order {f.field.order}.")

    field = f.field
    p = field.order
    m = f.degree
    x = Poly.Identity(field)

    if not is_primitive(f):
        # A Conway polynomial must be primitive.
        return False

    # For f_m(x) to be a Conway polynomial, f_n(x^r) must divide f_m(x) for all n | m,
    # where r = (p^m - 1) // (p^n - 1).
    proper_divisors = divisors(m)[:-1]  # Exclude m itself

    for n in proper_divisors:
        f_n = conway_poly(p, n, search=search)
        r = (p**m - 1) // (p**n - 1)
        x_r = pow(x, r, f)
        g = f_n(x_r) % f
        if g != 0:
            return False

    return True


@export
def conway_poly(characteristic: int, degree: int, search: bool = False) -> Poly:
    r"""
    Returns the Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$ with degree $m$.

    Arguments:
        characteristic: The prime characteristic $p$ of the field $\mathrm{GF}(p)$ that the polynomial
            is over.
        degree: The degree $m$ of the Conway polynomial.
        search: Manually search for Conway polynomials if they are not included in `Frank Luebeck's database
            <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_. The default is `False`.

            .. slow-performance::

                Manually searching for a Conway polynomial is *very* computationally expensive.

    Returns:
        The degree-$m$ Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$.

    See Also:
        Poly.is_conway, Poly.is_conway_consistent, Poly.is_primitive, primitive_poly

    Raises:
        LookupError: If `search=False` and the Conway polynomial $C_{p,m}$ is not found in Frank Luebeck's
            database.

    Notes:
        A degree-$m$ polynomial $f(x)$ over $\mathrm{GF}(p)$ is the *Conway polynomial*
        $C_{p,m}(x)$ if it is monic, primitive, compatible with Conway polynomials $C_{p,n}(x)$ for all
        $n \mid m$, and is lexicographically first according to a special ordering.

        A Conway polynomial $C_{p,m}(x)$ is *compatible* with Conway polynomials $C_{p,n}(x)$ for
        $n \mid m$ if $C_{p,n}(x^r)$ divides $C_{p,m}(x)$, where $r = \frac{p^m - 1}{p^n - 1}$.

        The Conway lexicographic ordering is defined as follows. Given two degree-$m$ polynomials
        $g(x) = \sum_{i=0}^m g_i x^i$ and $h(x) = \sum_{i=0}^m h_i x^i$, then $g < h$ if and only if
        there exists $i$ such that $g_j = h_j$ for all $j > i$ and
        $(-1)^{m-i} g_i < (-1)^{m-i} h_i$.

        The Conway polynomial $C_{p,m}(x)$ provides a standard representation of $\mathrm{GF}(p^m)$ as a
        splitting field of $C_{p,m}(x)$. Conway polynomials provide compatibility between fields and their
        subfields and, hence, are the common way to represent extension fields.

    References:
        - http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/CP7.html
        - Lenwood S. Heath, Nicholas A. Loehr, New algorithms for generating Conway polynomials over finite fields,
          Journal of Symbolic Computation, Volume 38, Issue 2, 2004, Pages 1003-1024,
          https://www.sciencedirect.com/science/article/pii/S0747717104000331.

    Examples:
        All Conway polynomials are primitive.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([1, 1, 2, 4], field=GF); f
            g = galois.Poly([1, 6, 0, 4], field=GF); g
            f.is_primitive()
            g.is_primitive()

        They are also consistent with all smaller Conway polynomials.

        .. ipython:: python

            f.is_conway_consistent()
            g.is_conway_consistent()

        Among the multiple candidate Conway polynomials, the lexicographically first (accordingly to a special
        lexicographical order) is the Conway polynomial.

        .. ipython:: python

            f.is_conway()
            g.is_conway()
            galois.conway_poly(7, 3)

    Group:
        polys-primitive
    """
    verify_isinstance(characteristic, int)
    verify_isinstance(degree, int)
    verify_isinstance(search, bool)

    if not is_prime(characteristic):
        raise ValueError(f"Argument 'characteristic' must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(
            f"Argument 'degree' must be at least 1, not {degree}. There are no primitive polynomials with degree 0."
        )

    try:
        return _conway_poly_database(characteristic, degree)
    except LookupError as e:
        if search:
            return _conway_poly_search(characteristic, degree)
        raise e


def _conway_poly_database(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$ with degree $m$
    from Frank Luebeck's database.

    Raises:
        LookupError: If the Conway polynomial $C_{p,m}(x)$ is not found in Frank Luebeck's database.
    """
    db = ConwayPolyDatabase()
    degrees, coeffs = db.fetch(characteristic, degree)
    field = _factory.FIELD_FACTORY(characteristic)
    poly = Poly.Degrees(degrees, coeffs, field=field)
    return poly


@functools.lru_cache()
def _conway_poly_search(characteristic: int, degree: int) -> Poly:
    r"""
    Manually searches for the Conway polynomial $C_{p,m}(x)$ over $\mathrm{GF}(p)$ with degree $m$.
    """
    for poly in _conway_lexicographic_order(characteristic, degree):
        if is_conway_consistent(poly):
            return poly

    raise RuntimeError(
        f"The Conway polynomial C_{{{characteristic},{degree}}}(x) could not be found. "
        "This should never happen. Please open a GitHub issue."
    )


def _conway_lexicographic_order(
    characteristic: int,
    degree: int,
) -> Iterator[Poly]:
    r"""
    Yields all monic polynomials of degree $m$ over $\mathrm{GF}(p)$ in the lexicographic order
    defined for Conway polynomials.
    """
    field = _factory.FIELD_FACTORY(characteristic)

    def recursive(
        degrees: Sequence[int],
        coeffs: Sequence[int],
    ) -> Iterator[Poly]:
        if len(degrees) == degree + 1:
            # print(characteristic, degree, degrees, coeffs)
            yield Poly.Degrees(degrees, coeffs, field=field)
        else:
            d = degrees[-1] - 1  # The new degree

            if (degree - d) % 2 == 1:
                all_coeffs = [0, *reversed(range(1, characteristic))]
            else:
                all_coeffs = [0, *range(1, characteristic)]
            # print(d, all_coeffs)

            for x in all_coeffs:
                next_degrees = (*degrees, d)
                next_coeffs = (*coeffs, x)
                yield from recursive(next_degrees, next_coeffs)

    yield from recursive([degree], [1])
