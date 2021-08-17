"""
A module that contains polymorphic math functions that work on integers and polynomials.
"""
import builtins

import numpy as np

from . import _factor as integer_factor
from . import _math as integer_math
from ._overrides import set_module
from ._polys import Poly
from ._polys import _factor as poly_factor
from ._polys import _math as poly_math

__all__ = [
    "gcd", "egcd", "lcm", "prod", "are_coprime",
    "pow", "crt",
    "factors", "is_square_free",
]


###############################################################################
# Divisibility
###############################################################################

@set_module("galois")
def gcd(a, b):
    r"""
    Finds the greatest common divisor of :math:`a` and :math:`b`.

    Parameters
    ----------
    a : int, galois.Poly
        The first integer or polynomial argument.
    b : int, galois.Poly
        The second integer or polynomial argument.

    Returns
    -------
    int, galois.Poly
        Greatest common divisor of :math:`a` and :math:`b`.

    Notes
    -----
    This function implements the Euclidean Algorithm.

    References
    ----------
    * Algorithm 2.104 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    * Algorithm 2.218 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    Compute the GCD of two integers.

    .. ipython:: python

        galois.gcd(12, 16)

    Compute the GCD of two polynomials.

    .. ipython:: python

        GF = galois.GF(7)
        p1 = galois.irreducible_poly(7, 1); p1
        p2 = galois.irreducible_poly(7, 2); p2
        p3 = galois.irreducible_poly(7, 3); p3

        a = p1**2 * p2; a
        b = p1 * p3; b
        gcd = galois.gcd(a, b); gcd
    """
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return integer_math.gcd(a, b)
    elif isinstance(a, Poly) and isinstance(b, Poly):
        return poly_math.gcd(a, b)
    else:
        raise TypeError(f"Arguments `a` and `b` must both be either int or galois.Poly, not {type(a)} and {type(b)}.")


@set_module("galois")
def egcd(a, b):
    r"""
    Finds the multiplicands of :math:`a` and :math:`b` such that :math:`a s + b t = \mathrm{gcd}(a, b)`.

    Parameters
    ----------
    a : int, galois.Poly
        The first integer or polynomial argument.
    b : int, galois.Poly
        The second integer or polynomial argument.

    Returns
    -------
    int, galois.Poly
        Greatest common divisor of :math:`a` and :math:`b`.
    int, galois.Poly
        The multiplicand :math:`s` of :math:`a`, such that :math:`a s + b t = \mathrm{gcd}(a, b)`.
    int, galois.Poly
        The multiplicand :math:`t` of :math:`b`, such that :math:`a s + b t = \mathrm{gcd}(a, b)`.

    Notes
    -----
    This function implements the Extended Euclidean Algorithm.

    References
    ----------
    * Algorithm 2.107 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    * Algorithm 2.221 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    * T. Moon, "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181

    Examples
    --------
    Compute the extended GCD of two integers.

    .. ipython:: python

        a, b = 12, 16
        gcd, s, t = galois.egcd(a, b)
        gcd, s, t
        a*s + b*t == gcd

    Compute the extended GCD of two polynomials.

    .. ipython:: python

        GF = galois.GF(7)
        p1 = galois.irreducible_poly(7, 1); p1
        p2 = galois.irreducible_poly(7, 2); p2
        p3 = galois.irreducible_poly(7, 3); p3

        a = p1**2 * p2; a
        b = p1 * p3; b
        gcd, s, t = galois.egcd(a, b)
        gcd, s, t
        a*s + b*t == gcd
    """
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return integer_math.egcd(a, b)
    elif isinstance(a, Poly) and isinstance(b, Poly):
        return poly_math.egcd(a, b)
    else:
        raise TypeError(f"Arguments `a` and `b` must both be either int or galois.Poly, not {type(a)} and {type(b)}.")


@set_module("galois")
def lcm(*values):
    r"""
    Computes the least common multiple of the arguments.

    Parameters
    ----------
    *values : int, galois.Poly
        Each argument must be an integer or polynomial.

    Returns
    -------
    int, galois.Poly
        The least common multiple of the arguments.

    Examples
    --------
    Compute the LCM of three integers.

    .. ipython:: python

        galois.lcm(2, 4, 14)

    Compute the LCM of three polynomials.

    .. ipython:: python

        GF = galois.GF(7)
        p1 = galois.irreducible_poly(7, 1); p1
        p2 = galois.irreducible_poly(7, 2); p2
        p3 = galois.irreducible_poly(7, 3); p3

        a = p1**2 * p2; a
        b = p1 * p3; b
        c = p2 * p3; c
        galois.lcm(a, b, c)
        p1**2 * p2 * p3
    """
    if not len(values) > 0:
        raise ValueError("At least one argument must be provided.")

    if all(isinstance(value, (int, np.integer)) for value in values):
        return integer_math.lcm(*values)
    elif all(isinstance(value, Poly) for value in values):
        return poly_math.lcm(*values)
    else:
        raise TypeError(f"All arguments must be either int or galois.Poly, not {[type(value) for value in values]}.")


@set_module("galois")
def prod(*values):
    r"""
    Computes the product of the arguments.

    Parameters
    ----------
    *values : int, galois.Poly
        Each argument must be an integer or polynomial.

    Returns
    -------
    int, galois.Poly
        The product of the arguments.

    Examples
    --------
    Compute the product of three integers.

    .. ipython:: python

        galois.prod(2, 4, 14)

    Compute the product of three polynomials.

    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Random(2, field=GF)
        b = galois.Poly.Random(3, field=GF)
        c = galois.Poly.Random(4, field=GF)
        galois.prod(a, b, c)
        a * b * c
    """
    if not len(values) > 0:
        raise ValueError("At least one argument must be provided.")

    if all(isinstance(value, (int, np.integer)) for value in values):
        return integer_math.prod(*values)
    elif all(isinstance(value, Poly) for value in values):
        return poly_math.prod(*values)
    else:
        raise TypeError(f"All arguments must be either int or galois.Poly, not {[type(value) for value in values]}.")


@set_module("galois")
def are_coprime(*values):
    r"""
    Determines if the arguments are pairwise coprime.

    Parameters
    ----------
    *values : int, galois.Poly
        Each argument must be an integer or polynomial.

    Returns
    -------
    bool
        `True` if the arguments are pairwise coprime.

    Notes
    -----
    A set of integers or polynomials are pairwise coprime if their LCM is equal to their product.

    Examples
    --------
    Determine if a set of integers are pairwise coprime.

    .. ipython:: python

        galois.are_coprime(3, 4, 5)
        galois.are_coprime(3, 7, 9, 11)

    Determine if a set of polynomials are pairwise coprime.

    .. ipython:: python

        GF = galois.GF(7)
        p1 = galois.irreducible_poly(7, 1); p1
        p2 = galois.irreducible_poly(7, 2); p2
        p3 = galois.irreducible_poly(7, 3); p3

        galois.are_coprime(p1, p2, p3)
        galois.are_coprime(p1*p2, p2, p3)
    """
    if not (all(isinstance(value, (int, np.integer)) for value in values) or all(isinstance(value, Poly) for value in values)):
        raise TypeError(f"All arguments must be either int or galois.Poly, not {[type(value) for value in values]}.")
    if not len(values) > 0:
        raise ValueError("At least one argument must be provided.")

    return lcm(*values) == prod(*values)


###############################################################################
# Congruences
###############################################################################

@set_module("galois")
def pow(base, exponent, modulus):  # pylint: disable=redefined-builtin
    r"""
    Efficiently performs modular exponentiation.

    Parameters
    ----------
    base : int, galois.Poly
        The integer or polynomial base :math:`a`.
    exponent : int
        The non-negative integer exponent :math:`k`.
    modulus : int, galois.Poly
        The integer or polynomial modulus :math:`m`.

    Returns
    -------
    int, galois.Poly
        The modular exponentiation :math:`a^k\ \textrm{mod}\ m`.

    Notes
    -----
    This function implements the Square-and-Multiply Algorithm. The algorithm is more efficient than exponentiating
    first and then reducing modulo :math:`m`, especially for very large exponents. Instead, this algorithm repeatedly squares :math:`a`,
    reducing modulo :math:`m` at each step.

    References
    ----------
    * Algorithm 2.143 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    * Algorithm 2.227 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    Compute the modular exponentiation of an integer.

    .. ipython:: python

        galois.pow(3, 100, 7)
        3**100 % 7

    Compute the modular exponentiation of a polynomial.

    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Random(3, field=GF)
        m = galois.Poly.Random(10, field=GF)
        galois.pow(a, 100, m)
        a**100 % m
    """
    if not isinstance(exponent, (int, np.integer)):
        raise TypeError(f"Argument `exponent` must be an integer, not {exponent}.")
    if not exponent >= 0:
        raise ValueError(f"Argument `exponent` must be non-negative, not {exponent}.")

    if isinstance(base, (int, np.integer)) and isinstance(modulus, (int, np.integer)):
        return builtins.pow(base, exponent, modulus)
    elif isinstance(base, Poly) and isinstance(modulus, Poly):
        return poly_math.pow(base, exponent, modulus)
    else:
        raise TypeError(f"Arguments `base` and `modulus` must both be either int or galois.Poly, not {type(base)} and {type(modulus)}.")


@set_module("galois")
def crt(remainders, moduli):
    r"""
    Solves the simultaneous system of congruences for :math:`x`.

    Parameters
    ----------
    remainders : tuple, list
        The integer or polynomial remainders :math:`a_i`.
    moduli : tuple, list
        The integer or polynomial moduli :math:`m_i`.

    Returns
    -------
    int
        The simultaneous solution :math:`x` to the system of congruences.

    Notes
    -----
    This function implements the Chinese Remainder Theorem.

    .. math::
        x &\equiv a_1\ (\textrm{mod}\ m_1)

        x &\equiv a_2\ (\textrm{mod}\ m_2)

        x &\equiv \ldots

        x &\equiv a_n\ (\textrm{mod}\ m_n)

    References
    ----------
    * Section 14.5 from https://cacr.uwaterloo.ca/hac/about/chap14.pdf

    Examples
    --------
    Solve a system of integer congruences.

    .. ipython:: python

        a = [0, 3, 4]
        m = [3, 4, 5]
        x = galois.crt(a, m); x

        for i in range(len(a)):
            ai = x % m[i]
            print(f"{x} = {ai} (mod {m[i]}), Valid congruence: {ai == a[i]}")

    Solve a system of polynomial congruences.

    .. ipython:: python

        GF = galois.GF(7)
        x_truth = galois.Poly.Random(10, field=GF); x_truth
        m = [galois.irreducible_poly(7, 2), galois.irreducible_poly(7, 3), galois.irreducible_poly(7, 4)]; m
        a = [x_truth % mi for mi in m]; a
        x = galois.crt(a, m); x

        for i in range(len(a)):
            ai = x % m[i]
            print(f"{x} = {ai} (mod {m[i]}), Valid congruence: {ai == a[i]}")
    """
    if not (isinstance(remainders, (tuple, list)) and (all(isinstance(x, (int, np.integer)) for x in remainders) or all(isinstance(x, Poly) for x in remainders))):
        raise TypeError(f"Argument `remainders` must be a tuple or list of int or galois.Poly, not {remainders}.")
    if not (isinstance(moduli, (tuple, list)) and (all(isinstance(x, (int, np.integer)) for x in moduli) or all(isinstance(x, Poly) for x in moduli))):
        raise TypeError(f"Argument `moduli` must be a tuple or list of int or galois.Poly, not {moduli}.")
    if not len(remainders) == len(moduli) >= 2:
        raise ValueError(f"Arguments `remainders` and `moduli` must be the same length of at least 2, not {len(remainders)} and {len(moduli)}.")
    if not are_coprime(*moduli):
        raise ValueError(f"Elements of argument `moduli` must be pairwise coprime, {moduli} are not.")

    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1, m1 = remainders[0], moduli[0]
    for a2, m2 in zip(remainders[1:], moduli[1:]):
        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = 1,
        # where 1 is the GCD(m1, m2) because m1 and m2 are pairwise relatively coprime
        _, b1, b2 = egcd(m1, m2)

        # Compute x through explicit construction
        x = a1*b2*m2 + a2*b1*m1

        m1 = m1 * m2  # The new modulus
        a1 = x % m1  # The new equivalent remainder

    # Align x to be within [0, prod(m))
    x = x % prod(*moduli)

    return x


###############################################################################
# Factorization
###############################################################################

@set_module("galois")
def factors(value):
    r"""
    Computes the prime factors of a positive integer or the irreducible factors of a non-constant, monic polynomial.

    Parameters
    ----------
    value : int, galois.Poly
        A positive integer :math:`n` or a non-constant, monic polynomial :math:`f(x)`.

    Returns
    -------
    list
        Sorted list of prime factors :math:`\{p_1, p_2, \dots, p_k\}` of :math:`n` with :math:`p_1 < p_2 < \dots < p_k` or
        irreducible factors :math:`\{g_1(x), g_2(x), \dots, g_k(x)\}` of :math:`f(x)` sorted in lexicographically-increasing order.
    list
        List of corresponding multiplicities :math:`\{e_1, e_2, \dots, e_k\}`.

    Notes
    -----
    **Integer Factorization**

    This function factors a positive integer :math:`n` into its :math:`k` prime factors such that :math:`n = p_1^{e_1} p_2^{e_2} \dots p_k^{e_k}`.

    Steps:

    1. Test if :math:`n` is prime. If so, return `[n], [1]`.
    2. Test if :math:`n` is a perfect power, such that :math:`n = x^k`. If so, prime factor :math:`x` and multiply the exponents by :math:`k`.
    3. Use trial division with a list of primes up to :math:`10^6`. If no residual factors, return the discovered prime factors.
    4. Use Pollard's Rho algorithm to find a non-trivial factor of the residual. Continue until all are found.

    **Polynomial Factorization**

    This function factors a monic polynomial :math:`f(x)` into its :math:`k` irreducible factors such that :math:`f(x) = g_1(x)^{e_1} g_2(x)^{e_2} \dots g_k(x)^{e_k}`.

    Steps:

    1. Apply the Square-Free Factorization algorithm to factor the monic polynomial into square-free polynomials.
    2. Apply the Distinct-Degree Factorization algorithm to factor each square-free polynomial into a product of factors with the same degree.
    3. Apply the Equal-Degree Factorization algorithm to factor the product of factors of equal degree into their irreducible factors.

    References
    ----------
    * D. Hachenberger, D. Jungnickel. Topics in Galois Fields. Algorithm 6.1.7.
    * Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples
    --------
    Factor a positive integer.

    .. ipython:: python

        galois.factors(120)

    Factor a polynomial over :math:`\mathrm{GF}(3)`.

    .. ipython:: python

        GF = galois.GF(3)
        g1, g2, g3 = galois.irreducible_poly(3, 3), galois.irreducible_poly(3, 4), galois.irreducible_poly(3, 5)
        g1, g2, g3
        e1, e2, e3 = 5, 4, 3
        # Construct the composite polynomial
        f = g1**e1 * g2**e2 * g3**e3
        galois.factors(f)
    """
    if isinstance(value, (int, np.integer)):
        return integer_factor.factors(value)
    elif isinstance(value, Poly):
        return poly_factor.factors(value)
    else:
        raise TypeError(f"Argument `value` must be either int or galois.Poly, not {type(value)}.")


@set_module("galois")
def is_square_free(value):
    r"""
    Determines if the positive integer or the non-constant, monic polynomial is square-free.

    Parameters
    ----------
    value : int, galois.Poly
        A positive integer :math:`n` or a non-constant, monic polynomial :math:`f(x)`.

    Returns
    -------
    bool:
        `True` if the integer or polynomial is square-free.

    Notes
    -----
    A square-free integer :math:`n` is divisible by no perfect squares. As a consequence, the prime factorization
    of a square-free integer :math:`n` is

    .. math:: n = \prod_{i=1}^{k} p_i^{e_i} = \prod_{i=1}^{k} p_i .

    Similarly, a square-free polynomial :math:`f(x)` has no irreducible factors with multiplicity greater than one. Therefore,
    its canonical factorization is

    .. math:: f(x) = \prod_{i=1}^{k} g_i(x)^{e_i} = \prod_{i=1}^{k} g_i(x) .

    Examples
    --------
    Determine if an integer is square-free.

    .. ipython:: python

        galois.is_square_free(10)
        galois.is_square_free(16)

    Determine if a polynomial is square-free over :math:`\mathrm{GF}(3)`.

    .. ipython:: python

        GF = galois.GF(3)
        g3 = galois.irreducible_poly(3, 3); g3
        g4 = galois.irreducible_poly(3, 4); g4
        galois.is_square_free(g3 * g4)
        galois.is_square_free(g3**2 * g4)
    """
    if isinstance(value, (int, np.integer)):
        return integer_factor.is_square_free(value)
    elif isinstance(value, Poly):
        return poly_factor.is_square_free(value)
    else:
        raise TypeError(f"Argument `value` must be either int or galois.Poly, not {type(value)}.")
