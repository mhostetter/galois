"""
A module with functions for polynomials over Galois fields.
"""
from typing import Tuple, List

import numpy as np

from .._domains import Array
from .._overrides import set_module

from ._poly import Poly

__all__ = [
    "lagrange_poly",
    "square_free_factorization", "distinct_degree_factorization", "equal_degree_factorization",
    "is_monic",
]


###############################################################################
# Divisibility
###############################################################################

def gcd(a, b):
    """
    This function is wrapped and documented in `_polymorphic.gcd()`.
    """
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    r2, r1 = a, b
    while r1 != 0:
        r2, r1 = r1, r2 % r1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 //= c

    return r2


def egcd(a, b):
    """
    This function is wrapped and documented in `_polymorphic.egcd()`.
    """
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    field = a.field
    zero = Poly.Zero(field)
    one = Poly.One(field)

    r2, r1 = a, b
    s2, s1 = one, zero
    t2, t1 = zero, one

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q*r1
        s2, s1 = s1, s2 - q*s1
        t2, t1 = t1, t2 - q*t1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 //= c
        s2 //= c
        t2 //= c

    return r2, s2, t2


def lcm(*args):
    """
    This function is wrapped and documented in `_polymorphic.lcm()`.
    """
    field = args[0].field

    lcm_  = Poly.One(field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}.")
        lcm_ = (lcm_ * arg) // gcd(lcm_, arg)

    # Make the LCM monic
    lcm_ //= lcm_.coeffs[0]

    return lcm_


def prod(*args):
    """
    This function is wrapped and documented in `_polymorphic.prod()`.
    """
    field = args[0].field

    prod_  = Poly.One(field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}.")
        prod_ *= arg

    return prod_


###############################################################################
# Special polynomials
###############################################################################

@set_module("galois")
def lagrange_poly(x: Array, y: Array) -> Poly:
    r"""
    Computes the Lagrange interpolating polynomial :math:`L(x)` such that :math:`L(x_i) = y_i`.

    Parameters
    ----------
    x
        An array of :math:`x_i` values for the coordinates :math:`(x_i, y_i)`. Must be 1-D. Must have no
        duplicate entries.
    y
        An array of :math:`y_i` values for the coordinates :math:`(x_i, y_i)`. Must be 1-D. Must be the same
        size as :math:`x`.

    Returns
    -------
    :
        The Lagrange polynomial :math:`L(x)`.

    Notes
    -----
    The Lagrange interpolating polynomial is defined as

    .. math::
        L(x) = \sum_{j=0}^{k-1} y_j \ell_j(x)

    .. math::
        \ell_j(x) = \prod_{\substack{0 \le m < k \\ m \ne j}} \frac{x - x_m}{x_j - x_m} .

    It is the polynomial of minimal degree that satisfies :math:`L(x_i) = y_i`.

    References
    ----------
    * https://en.wikipedia.org/wiki/Lagrange_polynomial

    Examples
    --------
    Create random :math:`(x, y)` pairs in :math:`\mathrm{GF}(3^2)`.

    .. ipython:: python

        GF = galois.GF(3**2)
        x = GF.Elements(); x
        y = GF.Random(x.size); y

    Find the Lagrange polynomial that interpolates the coordinates.

    .. ipython:: python

        L = galois.lagrange_poly(x, y); L

    Show that the polynomial evaluated at :math:`x` is :math:`y`.

    .. ipython:: python

        np.array_equal(L(x), y)
    """
    if not isinstance(x, Array):
        raise TypeError(f"Argument `x` must be a FieldArray, not {type(x)}.")
    if not isinstance(y, Array):
        raise TypeError(f"Argument `y` must be a FieldArray, not {type(y)}.")
    if not type(x) == type(y):  # pylint: disable=unidiomatic-typecheck
        raise TypeError(f"Arguments `x` and `y` must be over the same Galois field, not {type(x)} and {type(y)}.")
    if not x.ndim == 1:
        raise ValueError(f"Argument `x` must be 1-D, not have shape {x.shape}.")
    if not y.ndim == 1:
        raise ValueError(f"Argument `y` must be 1-D, not have shape {y.shape}.")
    if not x.size == y.size:
        raise ValueError(f"Arguments `x` and `y` must be the same size, not {x.size} and {y.size}.")
    if not x.size == np.unique(x).size:
        raise ValueError(f"Argument `x` must have unique entries, not {x}.")

    field = type(x)
    L = Poly.Zero(field)  # The Lagrange polynomial L(x)
    k = x.size  # The number of coordinates

    for j in range(k):
        lj = Poly.One(field)
        for m in range(k):
            if m == j:
                continue
            lj *= Poly([1, -x[m]], field=field) // (x[j] - x[m])
        L += y[j] * lj

    return L


###############################################################################
# Polynomial factorization
###############################################################################

def factors(poly):
    """
    This function is wrapped and documented in `_polymorphic.factors()`.
    """
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must be non-constant, not {poly}.")
    if not is_monic(poly):
        raise ValueError(f"Argument `poly` must be monic, not {poly}.")

    factors_, multiplicities = [], []

    # Step 1: Find all the square-free factors
    sf_factors, sf_multiplicities = square_free_factorization(poly)

    # Step 2: Find all the factors with distinct degree
    for sf_factor, sf_multiplicity in zip(sf_factors, sf_multiplicities):
        df_factors, df_degrees = distinct_degree_factorization(sf_factor)

        # Step 3: Find all the irreducible factors with degree d
        for df_factor, df_degree in zip(df_factors, df_degrees):
            f = equal_degree_factorization(df_factor, df_degree)
            factors_.extend(f)
            multiplicities.extend([sf_multiplicity,]*len(f))

    # Sort the factors in increasing-multiplicity order
    factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: int(item[0])))

    return list(factors_), list(multiplicities)


@set_module("galois")
def square_free_factorization(poly: Poly) -> Tuple[List[Poly], List[int]]:
    r"""
    Factors the monic polynomial :math:`f(x)` into a product of square-free polynomials.

    Parameters
    ----------
    poly
        A non-constant, monic polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    :
        The list of non-constant, square-free polynomials :math:`h_i(x)` in the factorization.
    :
        The list of corresponding multiplicities :math:`i`.

    See Also
    --------
    factors, distinct_degree_factorization, equal_degree_factorization

    Notes
    -----
    The Square-Free Factorization algorithm factors :math:`f(x)` into a product of :math:`m` square-free polynomials :math:`h_j(x)`
    with multiplicity :math:`j`.

    .. math::
        f(x) = \prod_{j=1}^{m} h_j(x)^j

    Some :math:`h_j(x) = 1`, but those polynomials are not returned by this function.

    A complete polynomial factorization is implemented in :func:`~galois.factors`.

    References
    ----------
    * Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.1.7.
    * Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples
    --------
    Suppose :math:`f(x) = x(x^3 + 2x + 4)(x^2 + 4x + 1)^3` over :math:`\mathrm{GF}(5)`. Each polynomial :math:`x`, :math:`x^3 + 2x + 4`,
    and :math:`x^2 + 4x + 1` are all irreducible over :math:`\mathrm{GF}(5)`.

    .. ipython:: python

        GF = galois.GF(5)
        a = galois.Poly([1,0], field=GF); a, galois.is_irreducible(a)
        b = galois.Poly([1,0,2,4], field=GF); b, galois.is_irreducible(b)
        c = galois.Poly([1,4,1], field=GF); c, galois.is_irreducible(c)
        f = a * b * c**3; f

    The square-free factorization is :math:`\{x(x^3 + 2x + 4), x^2 + 4x + 1\}` with multiplicities :math:`\{1, 3\}`.

    .. ipython:: python

        galois.square_free_factorization(f)
        [a*b, c], [1, 3]
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must be non-constant, not {poly}.")
    if not is_monic(poly):
        raise ValueError(f"Argument `poly` must be monic, not {poly}.")

    field = poly.field
    p = field.characteristic
    one = Poly.One(field=field)

    factors_ = []
    multiplicities = []

    # w is the product (without multiplicity) of all factors of f that have multiplicity not divisible by p
    d = gcd(poly, poly.derivative())
    w = poly // d

    # Step 1: Find all factors in w
    i = 1
    while w != one:
        y = gcd(w, d)
        z = w // y
        if z != one and i % p != 0:
            factors_.append(z)
            multiplicities.append(i)
        w = y
        d = d // y
        i = i + 1
    # d is now the product (with multiplicity) of the remaining factors of f

    # Step 2: Find all remaining factors (their multiplicities are divisible by p)
    if d != one:
        degrees = [degree // p for degree in d.nonzero_degrees]
        coeffs = d.nonzero_coeffs ** (field.characteristic**(field.degree - 1))  # The inverse Frobenius automorphism of the coefficients
        delta = Poly.Degrees(degrees, coeffs=coeffs, field=field)  # The p-th root of d(x)
        f, m = square_free_factorization(delta)
        factors_.extend(f)
        multiplicities.extend([mi*p for mi in m])

    # Sort the factors in increasing-multiplicity order
    factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: item[1]))

    return list(factors_), list(multiplicities)


@set_module("galois")
def distinct_degree_factorization(poly: Poly) -> Tuple[List[Poly], List[int]]:
    r"""
    Factors the monic, square-free polynomial :math:`f(x)` into a product of polynomials whose irreducible factors all have
    the same degree.

    Parameters
    ----------
    poly
        A monic, square-free polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    :
        The list of polynomials :math:`f_i(x)` whose irreducible factors all have degree :math:`i`.
    :
        The list of corresponding distinct degrees :math:`i`.

    See Also
    --------
    factors, square_free_factorization, equal_degree_factorization

    Notes
    -----
    The Distinct-Degree Factorization algorithm factors a square-free polynomial :math:`f(x)` with degree :math:`d` into a product of :math:`d` polynomials
    :math:`f_i(x)`, where :math:`f_i(x)` is the product of all irreducible factors of :math:`f(x)` with degree :math:`i`.

    .. math::
        f(x) = \prod_{i=1}^{d} f_i(x)

    For example, suppose :math:`f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)` over :math:`\mathrm{GF}(2)`, then the distinct-degree
    factorization is

    .. math::
        f_1(x) &= x(x + 1) = x^2 + x \\
        f_2(x) &= x^2 + x + 1 \\
        f_3(x) &= (x^3 + x + 1)(x^3 + x^2 + 1) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 \\
        f_i(x) &= 1\ \textrm{for}\ i = 4, \dots, 10.

    Some :math:`f_i(x) = 1`, but those polynomials are not returned by this function. In this example, the function returns
    :math:`\{f_1(x), f_2(x), f_3(x)\}` and :math:`\{1, 2, 3\}`.

    The Distinct-Degree Factorization algorithm is often applied after the Square-Free Factorization algorithm, see :func:`~galois.square_free_factorization`.
    A complete polynomial factorization is implemented in :func:`~galois.factors`.

    References
    ----------
    * Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.2.2.
    * Section 2.2 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples
    --------
    From the example in the notes, suppose :math:`f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)` over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        a = galois.Poly([1, 0]); a, galois.is_irreducible(a)
        b = galois.Poly([1, 1]); b, galois.is_irreducible(b)
        c = galois.Poly([1, 1, 1]); c, galois.is_irreducible(c)
        d = galois.Poly([1, 0, 1, 1]); d, galois.is_irreducible(d)
        e = galois.Poly([1, 1, 0, 1]); e, galois.is_irreducible(e)
        f = a * b * c * d * e; f

    The distinct-degree factorization is :math:`\{x(x + 1), x^2 + x + 1, (x^3 + x + 1)(x^3 + x^2 + 1)\}` whose irreducible factors
    have degrees :math:`\{1, 2, 3\}`.

    .. ipython:: python

        galois.distinct_degree_factorization(f)
        [a*b, c, d*e], [1, 2, 3]
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must be non-constant, not {poly}.")
    if not is_monic(poly):
        raise ValueError(f"Argument `poly` must be monic, not {poly}.")
    # TODO: Add check if the polynomial is square-free

    field = poly.field
    q = field.order
    n = poly.degree
    one = Poly.One(field=field)
    x = Poly.Identity(field=field)

    factors_ = []
    degrees = []

    a = poly
    h = x

    l = 1
    while l <= n // 2 and a != one:
        h = pow(h, q, a)
        z = gcd(a, h - x)
        if z != one:
            factors_.append(z)
            degrees.append(l)
            a = a // z
            h = h % a
        l += 1

    if a != one:
        factors_.append(a)
        degrees.append(a.degree)

    return factors_, degrees


@set_module("galois")
def equal_degree_factorization(poly: Poly, degree: int) -> List[Poly]:
    r"""
    Factors the monic, square-free polynomial :math:`f(x)` of degree :math:`rd` into a product of :math:`r` irreducible factors with
    degree :math:`d`.

    Parameters
    ----------
    poly
        A monic, square-free polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.
    degree
        The degree :math:`d` of each irreducible factor of :math:`f(x)`.

    Returns
    -------
    :
        The list of :math:`r` irreducible factors :math:`\{g_1(x), \dots, g_r(x)\}` in lexicographically-increasing order.

    See Also
    --------
    factors, square_free_factorization, distinct_degree_factorization

    Notes
    -----
    The Equal-Degree Factorization algorithm factors a square-free polynomial :math:`f(x)` with degree :math:`rd` into a product of :math:`r`
    irreducible polynomials each with degree :math:`d`. This function implements the Cantor-Zassenhaus algorithm, which is probabilistic.

    The Equal-Degree Factorization algorithm is often applied after the Distinct-Degree Factorization algorithm, see :func:`~galois.distinct_degree_factorization`.
    A complete polynomial factorization is implemented in :func:`~galois.factors`.

    References
    ----------
    * Section 2.3 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf
    * Section 1 from https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec8.pdf

    Examples
    --------
    Factor a product of degree-:math:`1` irreducible polynomials over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        a = galois.Poly([1, 0]); a, galois.is_irreducible(a)
        b = galois.Poly([1, 1]); b, galois.is_irreducible(b)
        f = a * b; f
        galois.equal_degree_factorization(f, 1)

    Factor a product of degree-:math:`3` irreducible polynomials over :math:`\mathrm{GF}(5)`.

    .. ipython:: python

        GF = galois.GF(5)
        a = galois.Poly([1, 0, 2, 1], field=GF); a, galois.is_irreducible(a)
        b = galois.Poly([1, 4, 4, 4], field=GF); b, galois.is_irreducible(b)
        f = a * b; f
        galois.equal_degree_factorization(f, 3)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must be non-constant, not {poly}.")
    if not is_monic(poly):
        raise ValueError(f"Argument `poly` must be monic, not {poly}.")
    if not poly.degree % degree == 0:
        raise ValueError(f"Argument `degree` must be divide the degree of the polynomial, {degree} does not divide {poly.degree}.")
    # TODO: Add check if the polynomial is square-free

    field = poly.field
    q = field.order
    r = poly.degree // degree
    one = Poly.One(field)

    factors_ = [poly]
    while len(factors_) < r:
        h = Poly.Random(degree, field=field)
        g = gcd(poly, h)
        if g == one:
            g = pow(h, (q**degree - 1)//2, poly) - one
        i = 0
        for u in list(factors_):
            if u.degree <= degree:
                continue
            d = gcd(g, u)
            if d not in [one, u]:
                factors_.remove(u)
                factors_.append(d)
                factors_.append(u // d)
            i += 1

    # Sort the factors in lexicographically-increasing order
    factors_ = sorted(factors_, key=int)

    return factors_


###############################################################################
# Polynomial tests
###############################################################################

@set_module("galois")
def is_monic(poly: Poly) -> bool:
    r"""
    Determines whether the polynomial is monic.

    Parameters
    ----------
    poly
        A polynomial over a Galois field.

    Returns
    -------
    :
        `True` if the polynomial is monic.

    See Also
    --------
    is_irreducible, is_primitive

    Notes
    -----
    A monic polynomial has its highest-degree, non-zero coefficient with value 1.

    Examples
    --------
    A monic polynomial over :math:`\mathrm{GF}(7)`.

    .. ipython:: python

        GF = galois.GF(7)
        p = galois.Poly([1, 0, 4, 5], field=GF); p
        galois.is_monic(p)

    A non-monic polynomial over :math:`\mathrm{GF}(7)`.

    .. ipython:: python

        GF = galois.GF(7)
        p = galois.Poly([3, 0, 4, 5], field=GF); p
        galois.is_monic(p)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")

    return poly.nonzero_coeffs[0] == 1


def is_square_free(poly: Poly) -> bool:
    """
    This function is wrapped and documented in `_polymorphic.is_square_free()`.
    """
    if not is_monic(poly):
        poly //= poly.coeffs[0]

    # Constant polynomials are square-free
    if poly.degree == 0:
        return True

    _, multiplicities = square_free_factorization(poly)

    return multiplicities == [1,]
