import random

import numpy as np

from ..factor import prime_factors
from ..modular import totatives
from ..overrides import set_module
from ..structure import is_field

from .factory_prime import GF_prime
from .poly import Poly

__all__ = [
    "poly_gcd", "poly_pow", "poly_factors",
    "irreducible_poly", "irreducible_polys", "primitive_poly", "primitive_polys", "minimal_poly",
    "is_monic", "is_irreducible", "is_primitive",
    "is_primitive_element", "primitive_element", "primitive_elements"
]


@set_module("galois")
def poly_gcd(a, b):
    """
    Finds the greatest common divisor of two polynomials :math:`a(x)` and :math:`b(x)`
    over :math:`\\mathrm{GF}(q)`.

    This implementation uses the Extended Euclidean Algorithm.

    Parameters
    ----------
    a : galois.Poly
        A polynomial :math:`a(x)` over :math:`\\mathrm{GF}(q)`.
    b : galois.Poly
        A polynomial :math:`b(x)` over :math:`\\mathrm{GF}(q)`.

    Returns
    -------
    galois.Poly
        Polynomial greatest common divisor of :math:`a(x)` and :math:`b(x)`.
    galois.Poly
        Polynomial :math:`x(x)`, such that :math:`a x + b y = \\textrm{gcd}(a, b)`.
    galois.Poly
        Polynomial :math:`y(x)`, such that :math:`a x + b y = \\textrm{gcd}(a, b)`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Roots([2,2,2,3,6], field=GF); a

        # a(x) and b(x) only share the root 2 in common
        b = galois.Poly.Roots([1,2], field=GF); b

        gcd, x, y = galois.poly_gcd(a, b)

        # The GCD has only 2 as a root with multiplicity 1
        gcd.roots(multiplicity=True)

        a*x + b*y == gcd
    """
    if not isinstance(a, Poly):
        raise TypeError(f"Argument `a` must be of type galois.Poly, not {type(a)}.")
    if not isinstance(b, Poly):
        raise TypeError(f"Argument `b` must be of type galois.Poly, not {type(b)}.")
    if not a.field == b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {str(a.field)} and {str(b.field)}.")

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

    # Make the gcd polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 /= c
        s2 /= c
        t2 /= c

    return r2, s2, t2


@set_module("galois")
def poly_pow(poly, power, modulus):
    """
    Efficiently exponentiates a polynomial :math:`f(x)` to the power :math:`k` reducing by modulo :math:`g(x)`,
    :math:`f(x)^k\\ \\textrm{mod}\\ g(x)`.

    The algorithm is more efficient than exponentiating first and then reducing modulo :math:`g(x)`. Instead,
    this algorithm repeatedly squares :math:`f(x)`, reducing modulo :math:`g(x)` at each step. This is the polynomial
    equivalent of :func:`galois.pow`.

    Parameters
    ----------
    poly : galois.Poly
        The polynomial to be exponentiated :math:`f(x)`.
    power : int
        The non-negative exponent :math:`k`.
    modulus : galois.Poly
        The reducing polynomial :math:`g(x)`.

    Returns
    -------
    galois.Poly
        The resulting polynomial :math:`h(x) = f^k\\ \\textrm{mod}\\ g`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        f = galois.Poly.Random(10, field=GF); f
        g = galois.Poly.Random(7, field=GF); g

        # %timeit f**200 % g
        # 1.23 s ± 41.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        f**200 % g

        # %timeit galois.poly_pow(f, 200, g)
        # 41.7 ms ± 468 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        galois.poly_pow(f, 200, g)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not isinstance(power, (int, np.integer)):
        raise TypeError(f"Argument `power` must be an integer, not {type(power)}.")
    if not isinstance(modulus, Poly):
        raise TypeError(f"Argument `modulus` must be a galois.Poly, not {type(modulus)}.")
    if not power >= 0:
        raise ValueError(f"Argument `power` must be non-negative, not {power}.")

    if power == 0:
        return Poly.One(poly.field)

    result_s = poly  # The "squaring" part
    result_m = Poly.One(poly.field)  # The "multiplicative" part

    while power > 1:
        if power % 2 == 0:
            result_s = (result_s * result_s) % modulus
            power //= 2
        else:
            result_m = (result_m * result_s) % modulus
            power -= 1

    result = (result_s * result_m) % modulus

    return result


@set_module("galois")
def poly_factors(poly):
    """
    Factors the polynomial :math:`f(x)` into a product of :math:`n` irreducible factors :math:`f(x) = g_0(x)^{k_0} g_1(x)^{k_1} \\dots g_{n-1}(x)^{k_{n-1}}`
    with :math:`k_0 \\le k_1 \\le \\dots \\le k_{n-1}`.

    This function implements the Square-Free Factorization algorithm.

    Parameters
    ----------
    poly : galois.Poly
        The polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p^m)` to be factored.

    Returns
    -------
    list
        The list of :math:`n` polynomial factors :math:`\\{g_0(x), g_1(x), \\dots, g_{n-1}(x)\\}`.
    list
        The list of :math:`n` polynomial multiplicities :math:`\\{k_0, k_1, \\dots, k_{n-1}\\}`.

    References
    ----------
    * D. Hachenberger, D. Jungnickel. Topics in Galois Fields. Algorithm 6.1.7.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF2
        # Ensure the factors are irreducible by using Conway polynomials
        g0, g1, g2 = galois.conway_poly(2, 3), galois.conway_poly(2, 4), galois.conway_poly(2, 5)
        g0, g1, g2
        k0, k1, k2 = 2, 3, 4
        # Construct the composite polynomial
        f = g0**k0 * g1**k1 * g2**k2
        galois.poly_factors(f)

    .. ipython:: python

        GF = galois.GF(3)
        # Ensure the factors are irreducible by using Conway polynomials
        g0, g1, g2 = galois.conway_poly(3, 3), galois.conway_poly(3, 4), galois.conway_poly(3, 5)
        g0, g1, g2
        k0, k1, k2 = 3, 4, 6
        # Construct the composite polynomial
        f = g0**k0 * g1**k1 * g2**k2
        galois.poly_factors(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not is_monic(poly):
        raise ValueError(f"Argument `poly` must be monic (otherwise there's a trivial 0-degree factor), not a leading coefficient of {poly.coeffs[0]}.")

    field = poly.field
    p = field.characteristic
    one = Poly.One(field=field)

    L = Poly.One(field=field)
    r = 0
    factors = []
    multiplicities = []

    def square_free_factorization(c, r):
        nonlocal L, factors, multiplicities
        i = 1
        a = c.copy()
        b = c.derivative()
        d = poly_gcd(a, b)[0]
        w = a / d

        while w != one:
            y = poly_gcd(w, d)[0]
            z = w / y
            if z != one and i % p != 0:
                L *= z**(i * p**r)
                factors.append(z)
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

    factors, multiplicities = zip(*sorted(zip(factors, multiplicities), key=lambda item: item[1]))

    return list(factors), list(multiplicities)


@set_module("galois")
def irreducible_poly(characteristic, degree, method="random"):
    """
    Returns a degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired polynomial that produces the field extension :math:`\\mathrm{GF}(p^m)`
        of :math:`\\mathrm{GF}(p)`.
    method : str, optional
        The search method for finding the irreducible polynomial, either `"random"` (default), `"smallest"`, or `"largest"`. The random
        search method will randomly generate degree-:math:`m` polynomials and test for irreducibility. The smallest/largest search
        method will produce polynomials in increasing/decreasing lexicographical order and test for irreducibility.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` irreducible polynomial over :math:`\\mathrm{GF}(p)`.

    Examples
    --------
    .. ipython:: python

        p = galois.irreducible_poly(7, 5); p
        galois.is_irreducible(p)
        p = galois.irreducible_poly(7, 5, method="smallest"); p
        galois.is_irreducible(p)
        p = galois.irreducible_poly(7, 5, method="largest"); p
        galois.is_irreducible(p)

    For the extension field :math:`\\mathrm{GF}(2^8)`, notice the lexicographically-smallest irreducible polynomial
    is not primitive. The Conway polynomial :math:`C_{2,8}` is the lexicographically-smallest irreducible *and primitive*
    polynomial.

    .. ipython:: python

        p = galois.irreducible_poly(2, 8, method="smallest"); p
        galois.is_irreducible(p), galois.is_primitive(p)

        p = galois.conway_poly(2, 8); p
        galois.is_irreducible(p), galois.is_primitive(p)
    """
    if not method in ["random", "smallest", "largest"]:
        raise ValueError(f"Argument `method` must be in ['random', 'smallest', 'largest'], not {method}.")
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
        if method == "smallest":
            elements = range(min_, max_)
        else:
            elements = range(max_ - 1, min_ - 1, -1)

        for element in elements:
            poly = Poly.Integer(element, field=GF)
            if is_irreducible(poly):
                break

    return poly


@set_module("galois")
def irreducible_polys(characteristic, degree):
    """
    Returns all degree-:math:`m` irreducible polynomials :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired polynomial that produces the field extension :math:`\\mathrm{GF}(p^m)`
        of :math:`\\mathrm{GF}(p)`.

    Returns
    -------
    list
        All degree-:math:`m` irreducible polynomials over :math:`\\mathrm{GF}(p)`.

    Examples
    --------
    .. ipython:: python

        galois.irreducible_polys(2, 5)
    """
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
def primitive_poly(characteristic, degree, method="random"):
    """
    Returns a degree-:math:`m` primitive polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired polynomial that produces the field extension :math:`\\mathrm{GF}(p^m)`
        of :math:`\\mathrm{GF}(p)`.
    method : str, optional
        The search method for finding the primitive polynomial, either `"random"` (default), `"smallest"`, or `"largest"`. The random
        search method will randomly generate degree-:math:`m` polynomials and test for primitivity. The smallest/largest search
        method will produce polynomials in increasing/decreasing lexicographical order and test for primitivity.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` primitive polynomial over :math:`\\mathrm{GF}(p)`.

    Examples
    --------
    For the extension field :math:`\\mathrm{GF}(2^8)`, notice the lexicographically-smallest irreducible polynomial
    is not primitive. The Conway polynomial :math:`C_{2,8}` is the lexicographically-smallest primitive *and primitive*
    polynomial.

    .. ipython:: python

        p = galois.irreducible_poly(2, 8, method="smallest"); p
        galois.is_irreducible(p), galois.is_primitive(p)

        # This is the same as the Conway polynomial C_2,8
        p = galois.primitive_poly(2, 8, method="smallest"); p
        galois.is_irreducible(p), galois.is_primitive(p)

        p = galois.conway_poly(2, 8); p
        galois.is_irreducible(p), galois.is_primitive(p)
    """
    if not method in ["random", "smallest", "largest"]:
        raise ValueError(f"Argument `method` must be in ['random', 'smallest', 'largest'], not {method}.")
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
        if method == "smallest":
            elements = range(min_, max_)
        else:
            elements = range(max_ - 1, min_ - 1, -1)

        for element in elements:
            poly = Poly.Integer(element, field=GF)
            if is_primitive(poly):
                break

    return poly


@set_module("galois")
def primitive_polys(characteristic, degree):
    """
    Returns all degree-:math:`m` primitive polynomials :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired polynomial that produces the field extension :math:`\\mathrm{GF}(p^m)`
        of :math:`\\mathrm{GF}(p)`.

    Returns
    -------
    list
        All degree-:math:`m` primitive polynomials over :math:`\\mathrm{GF}(p)`.

    Examples
    --------
    .. ipython:: python

        galois.primitive_polys(2, 5)
    """
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
def minimal_poly(element):
    """
    Computes the minimal polynomial :math:`m_e(x) \\in \\mathrm{GF}(p)[x]` of a Galois field
    element :math:`e \\in \\mathrm{GF}(p^m)`.

    The *minimal polynomial* of a Galois field element :math:`e \\in \\mathrm{GF}(p^m)` is the polynomial of
    minimal degree over :math:`\\mathrm{GF}(p)` for which :math:`e` is a root when evaluated in :math:`\\mathrm{GF}(p^m)`.
    Namely, :math:`m_e(x) \\in \\mathrm{GF}(p)[x] \\in \\mathrm{GF}(p^m)[x]` and :math:`m_e(e) = 0` over :math:`\\mathrm{GF}(p^m)`.

    Parameters
    ----------
    element : galois.FieldArray
        Any element :math:`e` of the Galois field :math:`\\mathrm{GF}(p^m)`. This must be a 0-dim array.

    Returns
    -------
    galois.Poly
        The minimal polynomial :math:`m_e(x)` over :math:`\\mathrm{GF}(p)` of the element :math:`e`.

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

    Not all elements of :math:`\\mathrm{GF}(2^4)` have minimal polynomials with degree-:math:`4`.

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
    if not is_field(element):
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
    """
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
    """
    Checks whether the polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)` is irreducible.

    A polynomial :math:`f(x) \\in \\mathrm{GF}(p)[x]` is *reducible* over :math:`\\mathrm{GF}(p)` if it can
    be represented as :math:`f(x) = g(x) h(x)` for some :math:`g(x), h(x) \\in \\mathrm{GF}(p)[x]` of strictly
    lower degree. If :math:`f(x)` is not reducible, it is said to be *irreducible*. Since Galois fields are not algebraically
    closed, such irreducible polynomials exist.

    This function implements Rabin's irreducibility test. It says a degree-:math:`n` polynomial :math:`f(x)`
    over :math:`\\mathrm{GF}(p)` for prime :math:`p` is irreducible if and only if :math:`f(x)\\ |\\ (x^{p^n} - x)`
    and :math:`\\textrm{gcd}(f(x),\\ x^{p^{m_i}} - x) = 1` for :math:`1 \\le i \\le k`, where :math:`m_i = n/p_i` for
    the :math:`k` prime divisors :math:`p_i` of :math:`n`.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    Returns
    -------
    bool
        `True` if the polynomial is irreducible.

    References
    ----------
    * M. O. Rabin. Probabilistic algorithms in finite fields. SIAM Journal on Computing (1980), 273–280. https://apps.dtic.mil/sti/pdfs/ADA078416.pdf
    * S. Gao and D. Panarino. Tests and constructions of irreducible polynomials over finite fields. https://www.math.clemson.edu/~sgao/papers/GP97a.pdf
    * https://en.wikipedia.org/wiki/Factorization_of_polynomials_over_finite_fields

    Examples
    --------
    .. ipython:: python

        # Conway polynomials are always irreducible (and primitive)
        f = galois.conway_poly(2, 5); f

        # f(x) has no roots in GF(2), a requirement of being irreducible
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
        raise TypeError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")
    if not poly.field.is_prime_field:
        raise ValueError(f"We can only check irreducibility of polynomials over prime fields GF(p), not {poly.field.name}.")

    if poly.coeffs[-1] == 0:
        # We can factor out (x), therefore it is not irreducible.
        return False

    if poly.degree == 1:
        # x + a in any Galois field is irreducible
        return True

    field = poly.field
    p = field.order
    n = poly.degree
    zero = Poly.Zero(field)
    one = Poly.One(field)
    x = Poly.Identity(field)

    primes, _ = prime_factors(n)
    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([n // pi for pi in primes]):
        # The GCD of f(x) and (x^(p^(n/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of n
        hi = poly_pow(h0, p**(ni - n0), poly)
        g = poly_gcd(poly, hi - x)[0]
        if g != one:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(p^n) - x) to be irreducible
    h = poly_pow(h0, p**(n - n0), poly)
    g = (h - x) % poly
    if g != zero:
        return False

    return True


@set_module("galois")
def is_primitive(poly):
    """
    Checks whether the polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)` is primitive.

    A degree-:math:`n` polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)` is *primitive* if it is irreducible and
    :math:`f(x)\\ |\\ (x^k - 1)` for :math:`k = p^n - 1` and no :math:`k` less than :math:`p^n - 1`.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

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

    The irreducible polynomial of :math:`\\mathrm{GF}(2^8)` for AES is not primitive.

    .. ipython:: python

        f = galois.Poly.Degrees([8,4,3,1,0]); f
        galois.is_primitive(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise TypeError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")
    if not poly.field.is_prime_field:
        raise ValueError(f"We can only check irreducibility of polynomials over prime fields GF(p), not {poly.field.name}.")

    if not is_irreducible(poly):
        # A polynomial must be irreducible to be primitive
        return False

    field = poly.field
    p = field.order
    n = poly.degree
    zero = Poly.Zero(field)
    one = Poly.One(field)

    primes, _ = prime_factors(p**n - 1)
    h0 = Poly.Identity(field)
    k0 = 1
    for ki in sorted([(p**n - 1) // pi for pi in primes]):
        # Multiply h0(x) by x^(ki - k0) and reduce by p(x) such that hi(x) = x^ki mod p(x)
        hi = (h0 * Poly.Degrees([ki - k0], field=field)) % poly
        g = hi - one  # Equivalent to g(x) = (x^ki - 1) mod p(x)

        # f(x) must not divide (x^((p^n - 1)/pi) - 1) for f(x) to be primitive, where pi are the prime factors of p**n - 1
        if ki < p**n - 1 and not g != zero:
            return False

        # f(x) must divide (x^(p^n - 1) - 1) for f(x) to be primitive
        if ki == p**n - 1 and not g == zero:
            return False

        h0, k0 = hi, ki

    return True


@set_module("galois")
def is_primitive_element(element, irreducible_poly):  # pylint: disable=redefined-outer-name
    """
    Determines if :math:`g(x)` is a primitive element of the Galois field :math:`\\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    The number of primitive elements of :math:`\\mathrm{GF}(p^m)` is :math:`\\phi(p^m - 1)`, where
    :math:`\\phi(n)` is the Euler totient function, see :obj:`galois.euler_totient`.

    Parameters
    ----------
    element : galois.Poly
        An element :math:`g(x)` of :math:`\\mathrm{GF}(p^m)` as a polynomial over :math:`\\mathrm{GF}(p)` with degree
        less than :math:`m`.
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)` that defines the extension field :math:`\\mathrm{GF}(p^m)`.

    Returns
    -------
    bool
        `True` if :math:`g(x)` is a primitive element of :math:`\\mathrm{GF}(p^m)` with irreducible polynomial
        :math:`f(x)`.

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
    assert isinstance(element, Poly) and isinstance(irreducible_poly, Poly)
    assert element.field is irreducible_poly.field

    field = irreducible_poly.field
    p = field.order
    m = irreducible_poly.degree
    one = Poly.One(field)

    order = p**m - 1  # Multiplicative order of GF(p^n)
    primes, _ = prime_factors(order)

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
    """
    Finds the smallest primitive element :math:`g(x)` of the Galois field :math:`\\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    Parameters
    ----------
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)` that defines the extension field :math:`\\mathrm{GF}(p^m)`.
    start : int, optional
        Starting value (inclusive, integer representation of the polynomial) in the search for a primitive element :math:`g(x)` of :math:`\\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p`, which corresponds to :math:`g(x) = x` over :math:`\\mathrm{GF}(p)`.
    stop : int, optional
        Stopping value (exclusive, integer representation of the polynomial) in the search for a primitive element :math:`g(x)` of :math:`\\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p^m`, which corresponds to :math:`g(x) = x^m` over :math:`\\mathrm{GF}(p)`.
    reverse : bool, optional
        Search for a primitive element in reverse order, i.e. find the largest primitive element first. Default is `False`.

    Returns
    -------
    galois.Poly
        A primitive element of :math:`\\mathrm{GF}(p^m)` with irreducible polynomial :math:`f(x)`. The primitive element :math:`g(x)` is
        a polynomial over :math:`\\mathrm{GF}(p)` with degree less than :math:`m`.

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
    """
    Finds all primitive elements :math:`g(x)` of the Galois field :math:`\\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)`.

    The number of primitive elements of :math:`\\mathrm{GF}(p^m)` is :math:`\\phi(p^m - 1)`, where
    :math:`\\phi(n)` is the Euler totient function. See :obj:galois.euler_totient`.

    Parameters
    ----------
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p)` that defines the extension field :math:`\\mathrm{GF}(p^m)`.
    start : int, optional
        Starting value (inclusive, integer representation of the polynomial) in the search for primitive elements :math:`g(x)` of :math:`\\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p`, which corresponds to :math:`g(x) = x` over :math:`\\mathrm{GF}(p)`.
    stop : int, optional
        Stopping value (exclusive, integer representation of the polynomial) in the search for primitive elements :math:`g(x)` of :math:`\\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p^m`, which corresponds to :math:`g(x) = x^m` over :math:`\\mathrm{GF}(p)`.
    reverse : bool, optional
        Search for primitive elements in reverse order, i.e. largest to smallest. Default is `False`.

    Returns
    -------
    list
        List of all primitive elements of :math:`\\mathrm{GF}(p^m)` with irreducible polynomial :math:`f(x)`. Each primitive element :math:`g(x)` is
        a polynomial over :math:`\\mathrm{GF}(p)` with degree less than :math:`m`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        g = galois.primitive_elements(f); g
        len(g) == galois.euler_totient(3**2 - 1)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        g = galois.primitive_elements(f); g
        len(g) == galois.euler_totient(3**2 - 1)
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
