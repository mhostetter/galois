"""
A module with various tests of polynomial properties.
"""
from .._factor import factors
from .._overrides import set_module

from ._functions import poly_gcd, poly_pow
from ._poly import Poly

__all__ = ["is_monic", "is_irreducible", "is_primitive", "is_primitive_element"]


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
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)` is irreducible.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    bool
        `True` if the polynomial is irreducible.

    Notes
    -----
    A polynomial :math:`f(x) \in \mathrm{GF}(p^m)[x]` is *reducible* over :math:`\mathrm{GF}(p^m)` if it can
    be represented as :math:`f(x) = g(x) h(x)` for some :math:`g(x), h(x) \in \mathrm{GF}(p^m)[x]` of strictly
    lower degree. If :math:`f(x)` is not reducible, it is said to be *irreducible*. Since Galois fields are not algebraically
    closed, such irreducible polynomials exist.

    This function implements Rabin's irreducibility test. It says a degree-:math:`m` polynomial :math:`f(x)`
    over :math:`\mathrm{GF}(q)` for prime power :math:`q` is irreducible if and only if :math:`f(x)\ |\ (x^{q^m} - x)`
    and :math:`\textrm{gcd}(f(x),\ x^{q^{m_i}} - x) = 1` for :math:`1 \le i \le k`, where :math:`m_i = m/p_i` for
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

        g = galois.irreducible_poly(2**4, 2, method="random"); g
        h = galois.irreducible_poly(2**4, 3, method="random"); h
        f = g * h; f

        galois.is_irreducible(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")

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
    q = field.order
    m = poly.degree
    zero = Poly.Zero(field)
    one = Poly.One(field)
    x = Poly.Identity(field)

    primes, _ = factors(m)
    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([m // pi for pi in primes]):
        # The GCD of f(x) and (x^(q^(m/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of m
        hi = poly_pow(h0, q**(ni - n0), poly)
        g = poly_gcd(poly, hi - x)
        if g != one:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(q^m) - x) to be irreducible
    h = poly_pow(h0, q**(m - n0), poly)
    g = (h - x) % poly
    if g != zero:
        return False

    return True


@set_module("galois")
def is_primitive(poly):
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` is primitive.

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
