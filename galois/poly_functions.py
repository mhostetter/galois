import numpy as np

from .gf_prime import GF_prime
from .modular import totatives
from .poly import Poly
from .prime import prime_factors


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
        Polynomial :math:`x(x)`, such that :math:`a x + b y = gcd(a, b)`.
    galois.Poly
        Polynomial :math:`y(x)`, such that :math:`a x + b y = gcd(a, b)`.

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

    if a == zero:
        return b, 0, 1
    if b == zero:
        return a, 1, 0

    r2, r1 = a, b
    s2, s1 = one, zero
    t2, t1 = zero, one

    while True:
        qi = r2 // r1
        ri = r2 % r1
        r2, r1 = r1, ri
        s2, s1 = s1, s2 - qi*s1
        t2, t1 = t1, t2 - qi*t1
        if ri == zero:
            break

    # Non-zero scalar is considered a unit in a fintie field
    if r2.degree == 0 and r2.coeffs[0] > 0:
        r2 /= r2
        s2 /= r2
        t2 /= r2

    return r2, s2, t2


def poly_exp_mod(poly, power, modulus):
    """
    Efficiently exponentiates a polynomial :math:`f(x)` to the power :math:`k` reducing by modulo :math:`g(x)`,
    :math:`f^k\\ \\textrm{mod}\\ g`.

    The algorithm is more efficient than exponentiating first and then reducing modulo :math:`g(x)`. Instead,
    this algorithm repeatedly squares :math:`f`, reducing modulo :math:`g` at each step.

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

        # %timeit galois.poly_exp_mod(f, 200, g)
        # 41.7 ms ± 468 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        galois.poly_exp_mod(f, 200, g)
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
    if not poly.field.degree == 1:
        raise ValueError(f"We can only check irreducibility of polynomials over prime fields GF(p), not {poly.field.name}.")

    field = poly.field
    p = field.order
    n = poly.degree
    primes, _ = prime_factors(n)
    zero = Poly.Zero(field)
    one = Poly.One(field)
    x = Poly.Identity(field)

    if poly.coeffs[-1] == 0:
        # We can factor out (x), therefore it is not irreducible.
        return False

    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([n // pi for pi in primes]):
        # The GCD of f(x) and (x^(p^(n/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of n
        hi = poly_exp_mod(h0, p**(ni - n0), poly)
        g = poly_gcd(poly, hi - x)[0]
        if g != one:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(p^n) - x) to be irreducible
    h = poly_exp_mod(h0, p**(n - n0), poly)
    g = (h - x) % poly
    if g != zero:
        return False

    return True


def is_primitive(poly):
    assert isinstance(poly, Poly)
    assert poly.degree > 1
    assert poly.field.degree == 1

    field = poly.field
    p = field.order
    n = poly.degree
    zero = Poly.Zero(field)
    one = Poly.One(field)
    x = Poly.Identity(field)

    # f(x) must divide (x^(p^n) - x) to be irreducible
    h = Poly.One(field)
    for k in range(1, p**n):
        h = h * x
        g = (h - one) % poly
        if g == zero and k < p**n - 1:
            return False
        if g != zero and k == p**n - 1:
            return False

    return True


def is_primitive_element(element, irreducible_poly):
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
        g = poly_exp_mod(element, k, irreducible_poly)
        if g == one:
            return False

    g = poly_exp_mod(element, order, irreducible_poly)
    if g != one:
        return False

    return True


def primitive_element(irreducible_poly, start=None, stop=None, reverse=False):
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


def primitive_elements(irreducible_poly, start=None, stop=None, reverse=False):
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

    elements = []
    for totative in totatives(p**m - 1):
        h = poly_exp_mod(element, totative, irreducible_poly)
        elements.append(h)

    elements = [e for e in elements if start <= e.integer < stop]  # Only return elements in the search range
    elements = sorted(elements, key=lambda e: e.integer, reverse=reverse)  # Sort element lexicographically

    return elements


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
