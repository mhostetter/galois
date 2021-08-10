import random

import numpy as np

from .._database import ConwayPolyDatabase
from .._factor import factors
from .._modular import totatives
from .._overrides import set_module
from .._prime import is_prime

from ._array import FieldArray
from ._factory_prime import GF_prime
from ._poly import Poly

__all__ = [
    "poly_gcd", "poly_egcd", "poly_pow",
    "poly_factors", "square_free_factorization", "distinct_degree_factorization", "equal_degree_factorization",
    "irreducible_poly", "irreducible_polys", "primitive_poly", "primitive_polys",
    "matlab_primitive_poly", "conway_poly", "minimal_poly",
    "is_monic", "is_irreducible", "is_primitive",
    "is_primitive_element", "primitive_element", "primitive_elements"
]


@set_module("galois")
def poly_gcd(a, b):
    r"""
    Finds the greatest common divisor of two polynomials :math:`a(x)` and :math:`b(x)`
    over :math:`\mathrm{GF}(p^m)`.

    This function implements the Euclidean Algorithm.

    Parameters
    ----------
    a : galois.Poly
        A polynomial :math:`a(x)` over :math:`\mathrm{GF}(p^m)`.
    b : galois.Poly
        A polynomial :math:`b(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    galois.Poly
        Polynomial greatest common divisor of :math:`a(x)` and :math:`b(x)`.

    References
    ----------
    * Algorithm 2.218 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Roots([2,2,2,3,6], field=GF); a
        b = galois.Poly.Roots([1,2], field=GF); b

        # a(x) and b(x) only share the root 2 in common, therefore their GCD is d(x) = x - 2 = x + 5
        gcd = galois.poly_gcd(a, b); gcd

        # The GCD has only 2 as a root with multiplicity 1
        gcd.roots(multiplicity=True)

    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Roots([2,2,2,3,6], field=GF); a
        b = galois.Poly.Roots([1,2,2], field=GF); b

        # a(x) and b(x) only share the root 2 in common (with multiplicity 2), therefore their GCD is d(x) = (x - 2)^2 = x^2 + 3x + 4
        gcd = galois.poly_gcd(a, b); gcd

        # The GCD has only 2 as a root with multiplicity 2
        gcd.roots(multiplicity=True)
    """
    if not isinstance(a, Poly):
        raise TypeError(f"Argument `a` must be a galois.Poly, not {type(a)}.")
    if not isinstance(b, Poly):
        raise TypeError(f"Argument `b` must be a galois.Poly, not {type(b)}.")
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    field = a.field
    zero = Poly.Zero(field)

    r2, r1 = a, b

    while r1 != zero:
        r2, r1 = r1, r2 % r1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 /= c

    return r2


@set_module("galois")
def poly_egcd(a, b):
    r"""
    Finds the polynomial multiplicands of :math:`a(x)` and :math:`b(x)` such that :math:`a(x)s(x) + b(x)t(x) = \mathrm{gcd}(a(x), b(x))`.

    This function implements the Extended Euclidean Algorithm.

    Parameters
    ----------
    a : galois.Poly
        A polynomial :math:`a(x)` over :math:`\mathrm{GF}(p^m)`.
    b : galois.Poly
        A polynomial :math:`b(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    galois.Poly
        Polynomial greatest common divisor of :math:`a(x)` and :math:`b(x)`.
    galois.Poly
        Polynomial :math:`s(x)`, such that :math:`a(x)s(x) + b(x)t(x) = \mathrm{gcd}(a(x), b(x))`.
    galois.Poly
        Polynomial :math:`t(x)`, such that :math:`a(x)s(x) + b(x)t(x) = \mathrm{gcd}(a(x), b(x))`.

    References
    ----------
    * Algorithm 2.221 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Roots([2,2,2,3,6], field=GF); a
        b = galois.Poly.Roots([1,2], field=GF); b

        # a(x) and b(x) only share the root 2 in common, therefore their GCD is d(x) = x - 2 = x + 5
        gcd, s, t = galois.poly_egcd(a, b); gcd, s, t

        # The GCD has only 2 as a root with multiplicity 1
        gcd.roots(multiplicity=True)

        a*s + b*t == gcd

    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Roots([2,2,2,3,6], field=GF); a
        b = galois.Poly.Roots([1,2,2], field=GF); b

        # a(x) and b(x) only share the root 2 in common (with multiplicity 2), therefore their GCD is d(x) = (x - 2)^2 = x^2 + 3x + 4
        gcd, s, t = galois.poly_egcd(a, b); gcd, s, t

        # The GCD has only 2 as a root with multiplicity 2
        gcd.roots(multiplicity=True)

        a*s + b*t == gcd
    """
    if not isinstance(a, Poly):
        raise TypeError(f"Argument `a` must be a galois.Poly, not {type(a)}.")
    if not isinstance(b, Poly):
        raise TypeError(f"Argument `b` must be a galois.Poly, not {type(b)}.")
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    field = a.field
    zero = Poly.Zero(field)
    one = Poly.One(field)

    r2, r1 = a, b
    s2, s1 = one, zero
    t2, t1 = zero, one

    while r1 != zero:
        q = r2 / r1
        r2, r1 = r1, r2 - q*r1
        s2, s1 = s1, s2 - q*s1
        t2, t1 = t1, t2 - q*t1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 /= c
        s2 /= c
        t2 /= c

    return r2, s2, t2


@set_module("galois")
def poly_pow(base_poly, exponent, modulus_poly):
    r"""
    Efficiently exponentiates a polynomial :math:`f(x)` to the power :math:`k` reducing by modulo :math:`g(x)`,
    :math:`f(x)^k\ \textrm{mod}\ g(x)`.

    Parameters
    ----------
    base_poly : galois.Poly
        The polynomial to be exponentiated :math:`f(x)`.
    exponent : int
        The non-negative exponent :math:`k`.
    modulus_poly : galois.Poly
        The reducing polynomial :math:`g(x)`.

    Returns
    -------
    galois.Poly
        The resulting polynomial :math:`h(x) = f(x)^k\ \textrm{mod}\ g(x)`.

    Notes
    -----
    This function implements the Square-and-Multiply Algorithm for polynomials. The algorithm is more efficient than exponentiating
    first and then reducing modulo :math:`g(x)`, especially for very large exponents. Instead, this algorithm repeatedly squares :math:`f(x)`,
    reducing modulo :math:`g(x)` at each step. This function is the polynomial equivalent of :func:`galois.pow`.

    References
    ----------
    * Algorithm 2.227 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        f = galois.Poly.Random(10, field=GF); f
        g = galois.Poly.Random(7, field=GF); g

        # %timeit f**10_000 % g
        # 2.61 s ± 339 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        f**10_000 % g

        # %timeit galois.poly_pow(f, 10_000, g)
        # 9.88 ms ± 778 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        galois.poly_pow(f, 10_000, g)
    """
    if not isinstance(base_poly, Poly):
        raise TypeError(f"Argument `base_poly` must be a galois.Poly, not {type(base_poly)}.")
    if not isinstance(exponent, (int, np.integer)):
        raise TypeError(f"Argument `exponent` must be an integer, not {type(exponent)}.")
    if not isinstance(modulus_poly, Poly):
        raise TypeError(f"Argument `modulus_poly` must be a galois.Poly, not {type(modulus_poly)}.")
    if not exponent >= 0:
        raise ValueError(f"Argument `exponent` must be non-negative, not {exponent}.")

    if exponent == 0:
        return Poly.One(base_poly.field)

    result_s = base_poly  # The "squaring" part
    result_m = Poly.One(base_poly.field)  # The "multiplicative" part

    while exponent > 1:
        if exponent % 2 == 0:
            result_s = (result_s * result_s) % modulus_poly
            exponent //= 2
        else:
            result_m = (result_m * result_s) % modulus_poly
            exponent -= 1

    result = (result_s * result_m) % modulus_poly

    return result


@set_module("galois")
def poly_factors(poly):
    r"""
    Factors the polynomial :math:`f(x)` into a product of irreducible factors :math:`f(x) = g_1(x)^{e_1} g_2(x)^{e_2} \dots g_k(x)^{e_k}`.

    This function implements the Square-Free Factorization algorithm.

    Parameters
    ----------
    poly : galois.Poly
        The polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)` to be factored.

    Returns
    -------
    list
        The list of :math:`k` polynomial factors :math:`\{g_1(x), g_2(x), \dots, g_k(x)\}` sorted in increasing lexicographic order.
    list
        The list of corresponding multiplicities :math:`\{e_1, e_2, \dots, e_k\}`.

    References
    ----------
    * D. Hachenberger, D. Jungnickel. Topics in Galois Fields. Algorithm 6.1.7.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF2
        # Ensure the factors are irreducible by using Conway polynomials
        g1, g2, g3 = galois.conway_poly(2, 3), galois.conway_poly(2, 4), galois.conway_poly(2, 5)
        g1, g2, g3
        e1, e2, e3 = 4, 3, 2
        # Construct the composite polynomial
        f = g1**e1 * g2**e2 * g3**e3
        galois.poly_factors(f)

    .. ipython:: python

        GF = galois.GF(3)
        # Ensure the factors are irreducible by using Conway polynomials
        g1, g2, g3 = galois.conway_poly(3, 3), galois.conway_poly(3, 4), galois.conway_poly(3, 5)
        g1, g2, g3
        e1, e2, e3 = 5, 4, 3
        # Construct the composite polynomial
        f = g1**e1 * g2**e2 * g3**e3
        galois.poly_factors(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")

    field = poly.field
    p = field.characteristic
    one = Poly.One(field=field)

    L = Poly.One(field=field)
    r = 0
    factors_ = []
    multiplicities = []

    if not is_monic(poly):
        factors_.append(Poly(poly.coeffs[0], field=field))
        multiplicities.append(1)
        poly /= poly.coeffs[0]

    def square_free_factorization(c, r):
        nonlocal L, factors_, multiplicities
        i = 1
        a = c.copy()
        b = c.derivative()
        d = poly_gcd(a, b)
        w = a / d

        while w != one:
            y = poly_gcd(w, d)
            z = w / y
            if z != one and i % p != 0:
                L *= z**(i * p**r)
                factors_.append(z)
                multiplicities.append(i * p**r)
            i = i + 1
            w = y
            d = d / y

        return d

    d = square_free_factorization(poly, r)

    while d != one:
        degrees = [degree // p for degree in d.degrees]
        coeffs = d.coeffs
        delta = Poly.Degrees(degrees, coeffs=coeffs, field=field)  # The p-th root of d(x)
        r += 1
        d = square_free_factorization(delta, r)

    # Sort the factor in lexicographically-increasing order
    factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: item[0].integer))

    return list(factors_), list(multiplicities)


@set_module("galois")
def square_free_factorization(poly):
    r"""
    Factors the monic polynomial :math:`f(x)` into a product of square-free polynomials.

    Parameters
    ----------
    poly : galois.Poly
        A non-constant, monic polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    list
        The list of non-constant, square-free polynomials :math:`h_i(x)` in the factorization.
    list
        The list of corresponding multiplicities :math:`i`.

    Notes
    -----
    The Square-Free Factorization algorithm factors :math:`f(x)` into a product of :math:`m` square-free polynomials :math:`h_j(x)`
    with multiplicity :math:`j`.

    .. math::
        f(x) = \prod_{j=1}^{m} h_j(x)^j

    Some :math:`h_j(x) = 1`, but those polynomials are not returned by this function.

    A complete polynomial factorization is implemented in :func:`galois.poly_factors`.

    References
    ----------
    * D. Hachenberger, D. Jungnickel. Topics in Galois Fields. Algorithm 6.1.7.
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
    d = poly_gcd(poly, poly.derivative())
    w = poly / d

    # Step 1: Find all factors in w
    i = 1
    while w != one:
        y = poly_gcd(w, d)
        z = w / y
        if z != one and i % p != 0:
            factors_.append(z)
            multiplicities.append(i)
        w = y
        d = d / y
        i = i + 1
    # d is now the product (with multiplicity) of the remaining factors of f

    # Step 2: Find all remaining factors (their multiplicities are divisible by p)
    if d != one:
        degrees = [degree // p for degree in d.degrees]
        coeffs = d.coeffs
        delta = Poly.Degrees(degrees, coeffs=coeffs, field=field)  # The p-th root of d(x)
        f, m = square_free_factorization(delta)
        factors_.extend(f)
        multiplicities.extend([mi*p for mi in m])

    # Sort the factors in increasing-multiplicity order
    factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: item[1]))

    return list(factors_), list(multiplicities)


@set_module("galois")
def distinct_degree_factorization(poly):
    r"""
    Factors the monic, square-free polynomial :math:`f(x)` into a product of polynomials whose irreducible factors all have
    the same degree.

    Parameters
    ----------
    poly : galois.Poly
        A monic, square-free polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    list
        The list of polynomials :math:`f_i(x)` whose irreducible factors all have degree :math:`i`.
    list
        The list of corresponding distinct degrees :math:`i`.

    Notes
    -----
    The Distinct-Degree Factorization algorithm factors a square-free polynomial :math:`f(x)` with degree :math:`d` into a product of :math:`d` polynomials
    :math:`f_i(x)`, where :math:`f_i(x)` is the product of all irreducible factors of :math:`f(x)` with degree :math:`i`.

    .. math::
        f(x) = \prod_{i=1}^{d} f_i(x)

    For example, suppose :math:`f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)` over :math:`\mathrm{GF}(2)`. Then the distinct-degree
    factorization is

    .. math::
        f_1(x) &= x(x + 1) = x^2 + x

        f_2(x) &= x^2 + x + 1

        f_3(x) &= (x^3 + x + 1)(x^3 + x^2 + 1) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

        f_i(x) &= 1\ \textrm{for}\ i = 4, \dots, 10.

    Some :math:`f_i(x) = 1`, but those polynomials are not returned by this function. In this example, the function returns
    :math:`\{f_1(x), f_2(x), f_3(x)\}` and :math:`\{1, 2, 3\}`.

    The Distinct-Degree Factorization algorithm is often applied after the Square-Free Factorization algorithm, see :func:`galois.square_free_factorization`.
    A complete polynomial factorization is implemented in :func:`galois.poly_factors`.

    References
    ----------
    * D. Hachenberger, D. Jungnickel. Topics in Galois Fields. Algorithm 6.2.2.
    * Section 2.2 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples
    --------
    From the example in the notes, suppose :math:`f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)` over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        a = galois.Poly([1,0]); a, galois.is_irreducible(a)
        b = galois.Poly([1,1]); b, galois.is_irreducible(b)
        c = galois.Poly([1,1,1]); c, galois.is_irreducible(c)
        d = galois.Poly([1,0,1,1]); d, galois.is_irreducible(d)
        e = galois.Poly([1,1,0,1]); e, galois.is_irreducible(e)
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

    a = poly.copy()
    h = x.copy()

    l = 1
    while l <= n // 2 and a != one:
        h = poly_pow(h, q, a)
        z = poly_gcd(a, h - x)
        if z != one:
            factors_.append(z)
            degrees.append(l)
            a = a / z
            h = h % a
        l += 1

    if a != one:
        factors_.append(a)
        degrees.append(a.degree)

    return factors_, degrees


@set_module("galois")
def equal_degree_factorization(poly, d):
    r"""
    Factors the monic, square-free polynomial :math:`f(x)` of degree :math:`rd` into a product of :math:`r` irreducible factors with
    degree :math:`d`.

    Parameters
    ----------
    poly : galois.Poly
        A monic, square-free polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.
    d : int
        The degree :math:`d` of each irreducible factor of :math:`f(x)`.

    Returns
    -------
    list
        The list of :math:`r` irreducible factors :math:`\{g_1(x), \dots, g_r(x)\}` in lexicographically-increasing order.

    Notes
    -----
    The Equal-Degree Factorization algorithm factors a square-free polynomial :math:`f(x)` with degree :math:`rd` into a product of :math:`r`
    irreducible polynomials each with degree :math:`d`. This function implements the Cantor-Zassenhaus algorithm, which is probabilistic.

    The Equal-Degree Factorization algorithm is often applied after the Distinct-Degree Factorization algorithm, see :func:`galois.distinct_degree_factorization`.
    A complete polynomial factorization is implemented in :func:`galois.poly_factors`.

    References
    ----------
    * Section 2.3 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples
    --------
    Factor a product of degree-:math:`1` irreducible polynomials over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        a = galois.Poly([1,0]); a, galois.is_irreducible(a)
        b = galois.Poly([1,1]); b, galois.is_irreducible(b)
        f = a * b; f
        galois.equal_degree_factorization(f, 1)

    Factor a product of degree-:math:`3` irreducible polynomials over :math:`\mathrm{GF}(5)`.

    .. ipython:: python

        GF = galois.GF(5)
        a = galois.Poly([1,0,2,1], field=GF); a, galois.is_irreducible(a)
        b = galois.Poly([1,4,4,4], field=GF); b, galois.is_irreducible(b)
        f = a * b; f
        galois.equal_degree_factorization(f, 3)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not isinstance(d, (int, np.integer)):
        raise TypeError(f"Argument `d` must be an integer, not {type(d)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must be non-constant, not {poly}.")
    if not is_monic(poly):
        raise ValueError(f"Argument `poly` must be monic, not {poly}.")
    if not poly.degree % d == 0:
        raise ValueError(f"Argument `d` must be divide the degree of the polynomial, {d} does not divide {poly.degree}.")
    # TODO: Add check if the polynomial is square-free

    field = poly.field
    q = field.order
    n = poly.degree
    r = poly.degree // d
    one = Poly.One(field=field)

    factors_ = [poly]
    while len(factors_) < r:
        h = Poly.Random(random.randint(1, n - 1), field=field)
        g = poly_gcd(poly, h)
        if g == one:
            g = poly_pow(h, (q**d - 1)//2, poly) - one
        for u in factors_:
            gcd = poly_gcd(g, u)
            if gcd not in [one, u]:
                factors_.remove(u)
                factors_.append(gcd)
                factors_.append(u / gcd)

    # Sort the factors in lexicographically-increasing order
    factors_ = sorted(factors_, key=lambda item: item.integer)

    return factors_


@set_module("galois")
def irreducible_poly(characteristic, degree, method="min"):
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired irreducible polynomial.
    method : str, optional
        The search method for finding the irreducible polynomial.

        * `"min"` (default): Returns the lexicographically-minimal monic irreducible polynomial.
        * `"max"`: Returns the lexicographically-maximal monic irreducible polynomial.
        * `"random"`: Returns a randomly generated degree-:math:`m` monic irreducible polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` monic irreducible polynomial over :math:`\mathrm{GF}(p)`.

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(p)` and :math:`a \in \mathrm{GF}(p) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible. In addition to other applications, :math:`f(x)` produces the field extension
    :math:`\mathrm{GF}(p^m)` of :math:`\mathrm{GF}(p)`.

    Examples
    --------
    The lexicographically-minimal monic irreducible polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

    .. ipython:: python

        p = galois.irreducible_poly(7, 5); p
        galois.is_irreducible(p)

    The lexicographically-maximal monic irreducible polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

    .. ipython:: python

        p = galois.irreducible_poly(7, 5, method="max"); p
        galois.is_irreducible(p)

    A random monic irreducible polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

    .. ipython:: python

        p = galois.irreducible_poly(7, 5, method="random"); p
        galois.is_irreducible(p)

    Irreducible polynomials scaled by non-zero field elements are also irreducible.

    .. ipython:: python

        GF = galois.GF(7)
        p = galois.irreducible_poly(7, 5); p
        galois.is_irreducible(p)
        galois.is_irreducible(p * GF(3))
    """
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    GF = GF_prime(characteristic)

    # Only search monic polynomials of degree m over GF(p)
    min_ = characteristic**degree
    max_ = 2*characteristic**degree

    if method == "random":
        while True:
            integer = random.randint(min_, max_ - 1)
            poly = Poly.Integer(integer, field=GF)
            if is_irreducible(poly):
                break
    else:
        elements = range(min_, max_) if method == "min" else range(max_ - 1, min_ - 1, -1)
        for element in elements:
            poly = Poly.Integer(element, field=GF)
            if is_irreducible(poly):
                break

    return poly


@set_module("galois")
def irreducible_polys(characteristic, degree):
    r"""
    Returns all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired irreducible polynomial.

    Returns
    -------
    list
        All degree-:math:`m` monic irreducible polynomials over :math:`\mathrm{GF}(p)`.

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(p)` and :math:`a \in \mathrm{GF}(p) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible. In addition to other applications, :math:`f(x)` produces the field extension
    :math:`\mathrm{GF}(p^m)` of :math:`\mathrm{GF}(p)`.

    Examples
    --------
    All monic irreducible polynomials over :math:`\mathrm{GF}(2)` with degree :math:`5`.

    .. ipython:: python

        galois.irreducible_polys(2, 5)
    """
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

    GF = GF_prime(characteristic)

    # Only search monic polynomials of degree m over GF(p)
    min_ = characteristic**degree
    max_ = 2*characteristic**degree

    polys = []
    for element in range(min_, max_):
        poly = Poly.Integer(element, field=GF)
        if is_irreducible(poly):
            polys.append(poly)

    return polys


@set_module("galois")
def primitive_poly(characteristic, degree, method="min"):
    r"""
    Returns a monic primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired primitive polynomial.
    method : str, optional
        The search method for finding the primitive polynomial.

        * `"min"` (default): Returns the lexicographically-minimal monic primitive polynomial.
        * `"max"`: Returns the lexicographically-maximal monic primitive polynomial.
        * `"random"`: Returns a randomly generated degree-:math:`m` monic primitive polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` monic primitive polynomial over :math:`\mathrm{GF}(p)`.

    Notes
    -----
    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(p^m)`
    of :math:`\mathrm{GF}(p)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(p^m)` such that :math:`\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m-2}\}`.

    Examples
    --------
    Notice :func:`galois.primitive_poly` returns the lexicographically-minimal primitive polynomial, where
    :func:`galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials, which is not *necessarily* the same.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

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
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    GF = GF_prime(characteristic)

    # Only search monic polynomials of degree m over GF(p)
    min_ = characteristic**degree
    max_ = 2*characteristic**degree

    if method == "random":
        while True:
            integer = random.randint(min_, max_ - 1)
            poly = Poly.Integer(integer, field=GF)
            if is_primitive(poly):
                break
    else:
        elements = range(min_, max_) if method == "min" else range(max_ - 1, min_ - 1, -1)
        for element in elements:
            poly = Poly.Integer(element, field=GF)
            if is_primitive(poly):
                break

    return poly


@set_module("galois")
def primitive_polys(characteristic, degree):
    r"""
    Returns all monic primitive polynomials :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired primitive polynomial.

    Returns
    -------
    list
        All degree-:math:`m` monic primitive polynomials over :math:`\mathrm{GF}(p)`.

    Notes
    -----
    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(p^m)`
    of :math:`\mathrm{GF}(p)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(p^m)` such that :math:`\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m-2}\}`.

    Examples
    --------
    All monic primitive polynomials over :math:`\mathrm{GF}(2)` with degree :math:`5`.

    .. ipython:: python

        galois.primitive_polys(2, 5)

    All monic primitive polynomials over :math:`\mathrm{GF}(3)` with degree :math:`4`.

    .. ipython:: python

        galois.primitive_polys(3, 4)
    """
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

    GF = GF_prime(characteristic)

    # Only search monic polynomials of degree m over GF(p)
    min_ = characteristic**degree
    max_ = 2*characteristic**degree

    polys = []
    for element in range(min_, max_):
        poly = Poly.Integer(element, field=GF)
        if is_primitive(poly):
            polys.append(poly)

    return polys


@set_module("galois")
def conway_poly(characteristic, degree):
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the Conway polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)`.

    Raises
    ------
    LookupError
        If the Conway polynomial :math:`C_{p,m}(x)` is not found in Frank Luebeck's database.

    Notes
    -----
    A Conway polynomial is a an irreducible and primitive polynomial over :math:`\mathrm{GF}(p)` that provides a standard
    representation of :math:`\mathrm{GF}(p^m)` as a splitting field of :math:`C_{p,m}(x)`. Conway polynomials
    provide compatability between fields and their subfields, and hence are the common way to represent extension
    fields.

    The Conway polynomial :math:`C_{p,m}(x)` is defined as the lexicographically-minimal monic primitive polynomial
    of degree :math:`m` over :math:`\mathrm{GF}(p)` that is compatible with all :math:`C_{p,n}(x)` for :math:`n` dividing
    :math:`m`.

    This function uses `Frank Luebeck's Conway polynomial database <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_
    for fast lookup, not construction.

    Examples
    --------
    Notice :func:`galois.primitive_poly` returns the lexicographically-minimal primitive polynomial, where
    :func:`galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials, which is not *necessarily* the same.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

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
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

    coeffs = ConwayPolyDatabase().fetch(characteristic, degree)
    GF = GF_prime(characteristic)
    poly = Poly(coeffs, field=GF)

    return poly


@set_module("galois")
def matlab_primitive_poly(characteristic, degree):
    r"""
    Returns Matlab's default primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired primitive polynomial.

    Returns
    -------
    galois.Poly
        Matlab's default degree-:math:`m` primitive polynomial over :math:`\mathrm{GF}(p)`.

    Notes
    -----
    This function returns the same result as Matlab's `gfprimdf(m, p)`. Matlab uses the lexicographically-minimal
    primitive polynomial (equivalent to `galois.primitive_poly(p, m)`) as the default... *mostly*. There are three
    notable exceptions:

    1. :math:`\mathrm{GF}(2^7)` uses :math:`x^7 + x^3 + 1`, not :math:`x^7 + x + 1`.
    2. :math:`\mathrm{GF}(2^{14})` uses :math:`x^{14} + x^{10} + x^6 + x + 1`, not :math:`x^{14} + x^5 + x^3 + x + 1`.
    3. :math:`\mathrm{GF}(2^{16})` uses :math:`x^{16} + x^{12} + x^3 + x + 1`, not :math:`x^{16} + x^5 + x^3 + x^2 + 1`.

    Warning
    -------
    This has been tested for all the :math:`\mathrm{GF}(2^m)` fields for :math:`2 \le m \le 16` (Matlab doesn't support
    larger than 16). And it has been spot-checked for :math:`\mathrm{GF}(p^m)`. There may exist other exceptions. Please
    submit a GitHub issue if you discover one.

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
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

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


@set_module("galois")
def minimal_poly(element):
    r"""
    Computes the minimal polynomial :math:`m_e(x) \in \mathrm{GF}(p)[x]` of a Galois field
    element :math:`e \in \mathrm{GF}(p^m)`.

    The *minimal polynomial* of a Galois field element :math:`e \in \mathrm{GF}(p^m)` is the polynomial of
    minimal degree over :math:`\mathrm{GF}(p)` for which :math:`e` is a root when evaluated in :math:`\mathrm{GF}(p^m)`.
    Namely, :math:`m_e(x) \in \mathrm{GF}(p)[x] \in \mathrm{GF}(p^m)[x]` and :math:`m_e(e) = 0` over :math:`\mathrm{GF}(p^m)`.

    Parameters
    ----------
    element : galois.FieldArray
        Any element :math:`e` of the Galois field :math:`\mathrm{GF}(p^m)`. This must be a 0-D array.

    Returns
    -------
    galois.Poly
        The minimal polynomial :math:`m_e(x)` over :math:`\mathrm{GF}(p)` of the element :math:`e`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(2**4)
        e = GF.primitive_element; e
        m_e = galois.minimal_poly(e); m_e
        # Evaluate m_e(e) in GF(2^4)
        m_e(e, field=GF)

    For a given element :math:`e`, the minimal polynomials of :math:`e` and all its conjugates are the same.

    .. ipython:: python

        # The conjugates of e
        conjugates = np.unique(e**(2**np.arange(0, 4))); conjugates
        for conjugate in conjugates:
            print(galois.minimal_poly(conjugate))

    Not all elements of :math:`\mathrm{GF}(2^4)` have minimal polynomials with degree-:math:`4`.

    .. ipython:: python

        e = GF.primitive_element**5; e
        # The conjugates of e
        conjugates = np.unique(e**(2**np.arange(0, 4))); conjugates
        for conjugate in conjugates:
            print(galois.minimal_poly(conjugate))

    In prime fields, the minimal polynomial of :math:`e` is simply :math:`m_e(x) = x - e`.

    .. ipython:: python

        GF = galois.GF(7)
        e = GF(3); e
        m_e = galois.minimal_poly(e); m_e
        m_e(e)
    """
    if not isinstance(element, FieldArray):
        raise TypeError(f"Argument `element` must be an element of a Galois field, not {type(element)}.")
    if not element.ndim == 0:
        raise ValueError(f"Argument `element` must be a single array element with dimension 0, not {element.ndim}-D.")

    field = type(element)
    x = Poly.Identity(field=field)

    if field.is_prime_field:
        return x - element
    else:
        conjugates = np.unique(element**(field.characteristic**np.arange(0, field.degree)))
        poly = Poly.Roots(conjugates, field=field)
        poly = Poly(poly.coeffs, field=field.prime_subfield)
        return poly


@set_module("galois")
def is_monic(poly):
    r"""
    Determines whether the polynomial is monic, i.e. having leading coefficient equal to 1.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial over a Galois field.

    Returns
    -------
    bool
        `True` if the polynomial is monic.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(7)
        p = galois.Poly([1,0,4,5], field=GF); p
        galois.is_monic(p)

    .. ipython:: python

        p = galois.Poly([3,0,4,5], field=GF); p
        galois.is_monic(p)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    return poly.nonzero_coeffs[0] == 1


@set_module("galois")
def is_irreducible(poly):
    r"""
    Checks whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` is irreducible.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    Returns
    -------
    bool
        `True` if the polynomial is irreducible.

    Notes
    -----
    A polynomial :math:`f(x) \in \mathrm{GF}(p)[x]` is *reducible* over :math:`\mathrm{GF}(p)` if it can
    be represented as :math:`f(x) = g(x) h(x)` for some :math:`g(x), h(x) \in \mathrm{GF}(p)[x]` of strictly
    lower degree. If :math:`f(x)` is not reducible, it is said to be *irreducible*. Since Galois fields are not algebraically
    closed, such irreducible polynomials exist.

    This function implements Rabin's irreducibility test. It says a degree-:math:`m` polynomial :math:`f(x)`
    over :math:`\mathrm{GF}(p)` for prime :math:`p` is irreducible if and only if :math:`f(x)\ |\ (x^{p^m} - x)`
    and :math:`\textrm{gcd}(f(x),\ x^{p^{m_i}} - x) = 1` for :math:`1 \le i \le k`, where :math:`m_i = m/p_i` for
    the :math:`k` prime divisors :math:`p_i` of :math:`m`.

    References
    ----------
    * M. O. Rabin. Probabilistic algorithms in finite fields. SIAM Journal on Computing (1980), 273–280. https://apps.dtic.mil/sti/pdfs/ADA078416.pdf
    * S. Gao and D. Panarino. Tests and constructions of irreducible polynomials over finite fields. https://www.math.clemson.edu/~sgao/papers/GP97a.pdf
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

        g = galois.conway_poly(2, 4); g
        h = galois.conway_poly(2, 5); h
        f = g * h; f

        # Even though f(x) has no roots in GF(2), it is still reducible
        f.roots()

        galois.is_irreducible(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")
    if not poly.field.is_prime_field:
        raise ValueError(f"We can only check irreducibility of polynomials over prime fields GF(p), not {poly.field.name}.")

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
    p = field.order
    m = poly.degree
    zero = Poly.Zero(field)
    one = Poly.One(field)
    x = Poly.Identity(field)

    primes, _ = factors(m)
    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([m // pi for pi in primes]):
        # The GCD of f(x) and (x^(p^(m/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of m
        hi = poly_pow(h0, p**(ni - n0), poly)
        g = poly_gcd(poly, hi - x)
        if g != one:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(p^m) - x) to be irreducible
    h = poly_pow(h0, p**(m - n0), poly)
    g = (h - x) % poly
    if g != zero:
        return False

    return True


@set_module("galois")
def is_primitive(poly):
    r"""
    Checks whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` is primitive.

    A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` is *primitive* if it is irreducible and
    :math:`f(x)\ |\ (x^k - 1)` for :math:`k = p^m - 1` and no :math:`k` less than :math:`p^m - 1`.

    Parameters
    ----------
    poly : galois.Poly
        A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    Returns
    -------
    bool
        `True` if the polynomial is primitive.

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

        f = galois.Poly.Degrees([8,4,3,1,0]); f
        galois.is_primitive(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")
    if not poly.field.is_prime_field:
        raise ValueError(f"We can only check irreducibility of polynomials over prime fields GF(p), not {poly.field.name}.")

    if poly.field.order == 2 and poly.degree == 1:
        # There is only one primitive polynomial in GF(2)
        return poly == Poly([1,1])

    if poly.coeffs[-1] == 0:
        # A primitive polynomial cannot have zero constant term
        # TODO: Why isn't f(x) = x primitive? It's irreducible and passes the primitivity tests.
        return False

    if not is_irreducible(poly):
        # A polynomial must be irreducible to be primitive
        return False

    field = poly.field
    p = field.order
    m = poly.degree
    zero = Poly.Zero(field)
    one = Poly.One(field)

    primes, _ = factors(p**m - 1)
    x = Poly.Identity(field)
    for ki in sorted([(p**m - 1) // pi for pi in primes]):
        # f(x) must not divide (x^((p^m - 1)/pi) - 1) for f(x) to be primitive, where pi are the prime factors of p**m - 1
        h = poly_pow(x, ki, poly)
        g = (h - one) % poly
        if g == zero:
            return False

    return True


@set_module("galois")
def is_primitive_element(element, irreducible_poly):  # pylint: disable=redefined-outer-name
    r"""
    Determines if :math:`g(x)` is a primitive element of the Galois field :math:`\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    Parameters
    ----------
    element : galois.Poly
        An element :math:`g(x)` of :math:`\mathrm{GF}(p^m)` as a polynomial over :math:`\mathrm{GF}(p)` with degree
        less than :math:`m`.
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` that defines the extension field :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    bool
        `True` if :math:`g(x)` is a primitive element of :math:`\mathrm{GF}(p^m)` with irreducible polynomial
        :math:`f(x)`.

    Notes
    -----
    The number of primitive elements of :math:`\mathrm{GF}(p^m)` is :math:`\phi(p^m - 1)`, where :math:`\phi(n)` is the Euler totient function,
    see :func:`galois.euler_phi`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)

        g = galois.Poly.Identity(GF); g
        galois.is_primitive_element(g, f)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)

        g = galois.Poly.Identity(GF); g
        galois.is_primitive_element(g, f)
    """
    if not isinstance(element, Poly):
        raise TypeError(f"Argument `element` must be a galois.Poly, not {type(element)}.")
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    if not element.field == irreducible_poly.field:
        raise ValueError(f"Arguments `element` and `irreducible_poly` must be over the same field, not {element.field} and {irreducible_poly.field}.")
    if not element.degree < irreducible_poly.degree:
        raise ValueError(f"Argument `element` must have degree less than `irreducible_poly`, not {element.degree} and {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")

    field = irreducible_poly.field
    p = field.order
    m = irreducible_poly.degree
    one = Poly.One(field)

    order = p**m - 1  # Multiplicative order of GF(p^m)
    primes, _ = factors(order)

    for k in sorted([order // pi for pi in primes]):
        g = poly_pow(element, k, irreducible_poly)
        if g == one:
            return False

    g = poly_pow(element, order, irreducible_poly)
    if g != one:
        return False

    return True


@set_module("galois")
def primitive_element(irreducible_poly, start=None, stop=None, reverse=False):  # pylint: disable=redefined-outer-name
    r"""
    Finds the smallest primitive element :math:`g(x)` of the Galois field :math:`\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    Parameters
    ----------
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` that defines the extension field :math:`\mathrm{GF}(p^m)`.
    start : int, optional
        Starting value (inclusive, integer representation of the polynomial) in the search for a primitive element :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p`, which corresponds to :math:`g(x) = x` over :math:`\mathrm{GF}(p)`.
    stop : int, optional
        Stopping value (exclusive, integer representation of the polynomial) in the search for a primitive element :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p^m`, which corresponds to :math:`g(x) = x^m` over :math:`\mathrm{GF}(p)`.
    reverse : bool, optional
        Search for a primitive element in reverse order, i.e. find the largest primitive element first. Default is `False`.

    Returns
    -------
    galois.Poly
        A primitive element of :math:`\mathrm{GF}(p^m)` with irreducible polynomial :math:`f(x)`. The primitive element :math:`g(x)` is
        a polynomial over :math:`\mathrm{GF}(p)` with degree less than :math:`m`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        galois.primitive_element(f)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        galois.primitive_element(f)
    """
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    if not isinstance(start, (type(None), int, np.integer)):
        raise TypeError(f"Argument `start` must be an integer, not {type(start)}.")
    if not isinstance(stop, (type(None), int, np.integer)):
        raise TypeError(f"Argument `stop` must be an integer, not {type(stop)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")
    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument `irreducible_poly` must have degree greater than 1, not {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")

    p = irreducible_poly.field.order
    m = irreducible_poly.degree
    start = p if start is None else start
    stop = p**m if stop is None else stop
    if not 1 <= start < stop <= p**m:
        raise ValueError(f"Arguments must satisfy `1 <= start < stop <= p^m`, `1 <= {start} < {stop} <= {p**m}` doesn't.")

    field = GF_prime(p)

    possible_elements = range(start, stop)
    if reverse:
        possible_elements = reversed(possible_elements)

    for integer in possible_elements:
        element = Poly.Integer(integer, field=field)
        if is_primitive_element(element, irreducible_poly):
            return element

    return None


@set_module("galois")
def primitive_elements(irreducible_poly, start=None, stop=None, reverse=False):  # pylint: disable=redefined-outer-name
    r"""
    Finds all primitive elements :math:`g(x)` of the Galois field :math:`\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    The number of primitive elements of :math:`\mathrm{GF}(p^m)` is :math:`\phi(p^m - 1)`, where
    :math:`\phi(n)` is the Euler totient function. See :obj:galois.euler_phi`.

    Parameters
    ----------
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` that defines the extension field :math:`\mathrm{GF}(p^m)`.
    start : int, optional
        Starting value (inclusive, integer representation of the polynomial) in the search for primitive elements :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p`, which corresponds to :math:`g(x) = x` over :math:`\mathrm{GF}(p)`.
    stop : int, optional
        Stopping value (exclusive, integer representation of the polynomial) in the search for primitive elements :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p^m`, which corresponds to :math:`g(x) = x^m` over :math:`\mathrm{GF}(p)`.
    reverse : bool, optional
        Search for primitive elements in reverse order, i.e. largest to smallest. Default is `False`.

    Returns
    -------
    list
        List of all primitive elements of :math:`\mathrm{GF}(p^m)` with irreducible polynomial :math:`f(x)`. Each primitive element :math:`g(x)` is
        a polynomial over :math:`\mathrm{GF}(p)` with degree less than :math:`m`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        g = galois.primitive_elements(f); g
        len(g) == galois.euler_phi(3**2 - 1)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        g = galois.primitive_elements(f); g
        len(g) == galois.euler_phi(3**2 - 1)
    """
    # NOTE: `irreducible_poly` will be verified in the call to `primitive_element()`
    if not isinstance(start, (type(None), int, np.integer)):
        raise TypeError(f"Argument `start` must be an integer, not {type(start)}.")
    if not isinstance(stop, (type(None), int, np.integer)):
        raise TypeError(f"Argument `stop` must be an integer, not {type(stop)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")

    element = primitive_element(irreducible_poly)

    p = irreducible_poly.field.order
    m = irreducible_poly.degree
    start = p if start is None else start
    stop = p**m if stop is None else stop
    if not 1 <= start < stop <= p**m:
        raise ValueError(f"Arguments must satisfy `1 <= start < stop <= p^m`, `1 <= {start} < {stop} <= {p**m}` doesn't.")

    elements = []
    for totative in totatives(p**m - 1):
        h = poly_pow(element, totative, irreducible_poly)
        elements.append(h)

    elements = [e for e in elements if start <= e.integer < stop]  # Only return elements in the search range
    elements = sorted(elements, key=lambda e: e.integer, reverse=reverse)  # Sort element lexicographically

    return elements
