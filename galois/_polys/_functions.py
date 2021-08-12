"""
A module of various functions on polynomials, such as GCD and modular exponentiation.
"""
import numpy as np

from .._overrides import set_module

from ._poly import Poly

__all__ = ["poly_gcd", "poly_egcd", "poly_pow"]


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
