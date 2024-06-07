"""
A module that contains polymorphic math functions that work on integers and polynomials.
"""

from __future__ import annotations

from typing import Sequence, overload

import numpy as np

from ._helper import export
from ._math import egcd as int_egcd
from ._math import gcd as int_gcd
from ._math import lcm as int_lcm
from ._math import prod as int_prod
from ._polys import Poly
from ._polys._functions import egcd as poly_egcd
from ._polys._functions import gcd as poly_gcd
from ._polys._functions import lcm as poly_lcm
from ._polys._functions import prod as poly_prod
from ._prime import factors as int_factors
from ._prime import is_square_free as int_is_square_free

###############################################################################
# Divisibility
###############################################################################


@overload
def gcd(a: int, b: int) -> int: ...


@overload
def gcd(a: Poly, b: Poly) -> Poly: ...


@export
def gcd(a, b):
    r"""
    Finds the greatest common divisor of $a$ and $b$.

    Arguments:
        a: The first integer or polynomial argument.
        b: The second integer or polynomial argument.

    Returns:
        Greatest common divisor of $a$ and $b$.

    See Also:
        egcd, lcm, prod

    Notes:
        This function implements the Euclidean Algorithm.

    References:
        - Algorithm 2.104 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - Algorithm 2.218 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Compute the GCD of two integers.

                .. ipython:: python

                    galois.gcd(12, 16)

            .. md-tab-item:: Polynomials

                Generate irreducible polynomials over $\mathrm{GF}(7)$.

                .. ipython:: python

                    GF = galois.GF(7)
                    f1 = galois.irreducible_poly(7, 1); f1
                    f2 = galois.irreducible_poly(7, 2); f2
                    f3 = galois.irreducible_poly(7, 3); f3

                Compute the GCD of $f_1(x)^2 f_2(x)$ and $f_1(x) f_3(x)$, which is $f_1(x)$.

                .. ipython:: python

                    galois.gcd(f1**2 * f2, f1 * f3)

    Group:
        number-theory-divisibility
    """
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int_gcd(a, b)
    if isinstance(a, Poly) and isinstance(b, Poly):
        return poly_gcd(a, b)
    raise TypeError(f"Arguments `a` and `b` must both be either int or galois.Poly, not {type(a)} and {type(b)}.")


@overload
def egcd(a: int, b: int) -> tuple[int, int, int]: ...


@overload
def egcd(a: Poly, b: Poly) -> tuple[Poly, Poly, Poly]: ...


@export
def egcd(a, b):
    r"""
    Finds the multiplicands of $a$ and $b$ such that $a s + b t = \mathrm{gcd}(a, b)$.

    Arguments:
        a: The first integer or polynomial argument.
        b: The second integer or polynomial argument.

    Returns:
        - Greatest common divisor of $a$ and $b$.
        - The multiplicand $s$ of $a$.
        - The multiplicand $t$ of $b$.

    See Also:
        gcd, lcm, prod

    Notes:
        This function implements the Extended Euclidean Algorithm.

    References:
        - Algorithm 2.107 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - Algorithm 2.221 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - Moon, T. "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Compute the extended GCD of two integers.

                .. ipython:: python

                    a, b = 12, 16
                    gcd, s, t = galois.egcd(a, b)
                    gcd, s, t
                    a*s + b*t == gcd

            .. md-tab-item:: Polynomials

                Generate irreducible polynomials over $\mathrm{GF}(7)$.

                .. ipython:: python

                    GF = galois.GF(7)
                    f1 = galois.irreducible_poly(7, 1); f1
                    f2 = galois.irreducible_poly(7, 2); f2
                    f3 = galois.irreducible_poly(7, 3); f3

                Compute the extended GCD of $f_1(x)^2 f_2(x)$ and $f_1(x) f_3(x)$.

                .. ipython:: python

                    a = f1**2 * f2
                    b = f1 * f3
                    gcd, s, t = galois.egcd(a, b)
                    gcd, s, t
                    a*s + b*t == gcd

    Group:
        number-theory-divisibility
    """
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int_egcd(a, b)
    if isinstance(a, Poly) and isinstance(b, Poly):
        return poly_egcd(a, b)
    raise TypeError(f"Arguments `a` and `b` must both be either int or galois.Poly, not {type(a)} and {type(b)}.")


@overload
def lcm(*values: int) -> int: ...


@overload
def lcm(*values: Poly) -> Poly: ...


@export
def lcm(*values):
    r"""
    Computes the least common multiple of the arguments.

    Arguments:
        values: Each argument must be an integer or polynomial.

    Returns:
        The least common multiple of the arguments.

    See Also:
        gcd, egcd, prod

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Compute the LCM of three integers.

                .. ipython:: python

                    galois.lcm(2, 4, 14)

            .. md-tab-item:: Polynomials

                Generate irreducible polynomials over $\mathrm{GF}(7)$.

                .. ipython:: python

                    GF = galois.GF(7)
                    f1 = galois.irreducible_poly(7, 1); f1
                    f2 = galois.irreducible_poly(7, 2); f2
                    f3 = galois.irreducible_poly(7, 3); f3

                Compute the LCM of three polynomials $f_1(x)^2 f_2(x)$, $f_1(x) f_3(x)$, and
                $f_2(x) f_3(x)$, which is $f_1(x)^2 f_2(x) f_3(x)$.

                .. ipython:: python

                    galois.lcm(f1**2 * f2, f1 * f3, f2 * f3)
                    f1**2 * f2 * f3

    Group:
        number-theory-divisibility
    """
    if not len(values) > 0:
        raise ValueError("At least one argument must be provided.")

    if all(isinstance(value, (int, np.integer)) for value in values):
        return int_lcm(*values)
    if all(isinstance(value, Poly) for value in values):
        return poly_lcm(*values)
    raise TypeError(f"All arguments must be either int or galois.Poly, not {[type(value) for value in values]}.")


@overload
def prod(*values: int) -> int: ...


@overload
def prod(*values: Poly) -> Poly: ...


@export
def prod(*values):
    r"""
    Computes the product of the arguments.

    Arguments:
        values: Each argument must be an integer or polynomial.

    Returns:
        The product of the arguments.

    See Also:
        gcd, egcd, lcm

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Compute the product of three integers.

                .. ipython:: python

                    galois.prod(2, 4, 14)

            .. md-tab-item:: Polynomials

                Generate random polynomials over $\mathrm{GF}(7)$.

                .. ipython:: python

                    GF = galois.GF(7)
                    f1 = galois.Poly.Random(2, field=GF); f1
                    f2 = galois.Poly.Random(3, field=GF); f2
                    f3 = galois.Poly.Random(4, field=GF); f3

                Compute the product of three polynomials.

                .. ipython:: python

                    galois.prod(f1, f2, f3)
                    f1 * f2 * f3

    Group:
        number-theory-divisibility
    """
    if not len(values) > 0:
        raise ValueError("At least one argument must be provided.")

    if all(isinstance(value, (int, np.integer)) for value in values):
        return int_prod(*values)
    if all(isinstance(value, Poly) for value in values):
        return poly_prod(*values)
    raise TypeError(f"All arguments must be either int or galois.Poly, not {[type(value) for value in values]}.")


@overload
def are_coprime(*values: int) -> bool: ...


@overload
def are_coprime(*values: Poly) -> bool: ...


@export
def are_coprime(*values):
    r"""
    Determines if the arguments are pairwise coprime.

    Arguments:
        values: Each argument must be an integer or polynomial.

    Returns:
        `True` if the arguments are pairwise coprime.

    See Also:
        lcm, prod

    Notes:
        A set of integers or polynomials are pairwise coprime if their LCM is equal to their product.

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Determine if a set of integers are pairwise coprime.

                .. ipython:: python

                    galois.are_coprime(3, 4, 5)
                    galois.are_coprime(3, 7, 9, 11)

            .. md-tab-item:: Polynomials

                Generate irreducible polynomials over $\mathrm{GF}(7)$.

                .. ipython:: python

                    GF = galois.GF(7)
                    f1 = galois.irreducible_poly(7, 1); f1
                    f2 = galois.irreducible_poly(7, 2); f2
                    f3 = galois.irreducible_poly(7, 3); f3

                Determine if combinations of the irreducible polynomials are pairwise coprime.

                .. ipython:: python

                    galois.are_coprime(f1, f2, f3)
                    galois.are_coprime(f1 * f2, f2, f3)

    Group:
        number-theory-divisibility
    """
    if not (
        all(isinstance(value, (int, np.integer)) for value in values)
        or all(isinstance(value, Poly) for value in values)
    ):
        raise TypeError(f"All arguments must be either int or galois.Poly, not {[type(value) for value in values]}.")
    if not len(values) > 0:
        raise ValueError("At least one argument must be provided.")

    return lcm(*values) == prod(*values)


###############################################################################
# Congruences
###############################################################################


@overload
def crt(remainders: Sequence[int], moduli: Sequence[int]) -> int: ...


@overload
def crt(remainders: Sequence[Poly], moduli: Sequence[Poly]) -> Poly: ...


@export
def crt(remainders, moduli):
    r"""
    Solves the simultaneous system of congruences for $x$.

    Arguments:
        remainders: The integer or polynomial remainders $a_i$.
        moduli: The integer or polynomial moduli $m_i$.

    Returns:
        The simultaneous solution $x$ to the system of congruences.

    Notes:
        This function implements the Chinese Remainder Theorem.

        $$
        x &\equiv a_1\ (\textrm{mod}\ m_1) \\
        x &\equiv a_2\ (\textrm{mod}\ m_2) \\
        x &\equiv \ldots \\
        x &\equiv a_n\ (\textrm{mod}\ m_n)
        $$

    References:
        - Section 14.5 from https://cacr.uwaterloo.ca/hac/about/chap14.pdf

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Define a system of integer congruences.

                .. ipython:: python

                    a = [0, 3, 4]
                    m = [3, 4, 5]

                Solve the system of congruences.

                .. ipython:: python

                    x = galois.crt(a, m); x

                Show that the solution satisfies each congruence.

                .. ipython:: python

                    for i in range(len(a)):
                        ai = x % m[i]
                        print(ai, ai == a[i])

            .. md-tab-item:: Polynomials

                Define a system of polynomial congruences over $\mathrm{GF}(7)$.

                .. ipython:: python

                    GF = galois.GF(7)
                    x_truth = galois.Poly.Random(6, field=GF); x_truth
                    m3 = galois.Poly.Random(3, field=GF)
                    m4 = galois.Poly.Random(4, field=GF)
                    m5 = galois.Poly.Random(5, field=GF)
                    m = [m3, m4, m5]; m
                    a = [x_truth % m3, x_truth % m4, x_truth % m5]; a

                Solve the system of congruences.

                .. ipython:: python

                    x = galois.crt(a, m); x

                Show that the solution satisfies each congruence.

                .. ipython:: python

                    for i in range(len(a)):
                        ai = x % m[i]
                        print(ai, ai == a[i])

    Group:
        number-theory-congruences
    """
    if not (
        isinstance(remainders, (tuple, list))
        and (all(isinstance(x, (int, np.integer)) for x in remainders) or all(isinstance(x, Poly) for x in remainders))
    ):
        raise TypeError(f"Argument 'remainders' must be a tuple or list of int or Poly, not {remainders}.")
    if not (
        isinstance(moduli, (tuple, list))
        and (all(isinstance(x, (int, np.integer)) for x in moduli) or all(isinstance(x, Poly) for x in moduli))
    ):
        raise TypeError(f"Argument 'moduli' must be a tuple or list of int or Poly, not {moduli}.")
    if not len(remainders) == len(moduli) >= 2:
        raise ValueError(
            f"Arguments 'remainders' and 'moduli' must be the same length of at least 2, "
            f"not {len(remainders)} and {len(moduli)}."
        )

    # Ensure polynomial arguments have each remainder have degree less than its modulus
    if isinstance(remainders[0], Poly):
        for remainder, modulus in zip(remainders, moduli):
            if not (remainder == 0 or remainder.degree < modulus.degree):
                raise ValueError(
                    f"Each remainder must have degree strictly less than its modulus. "
                    f"Remainder {remainder} with modulus {modulus} does not satisfy that condition."
                )

    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1, m1 = remainders[0], moduli[0]
    for a2, m2 in zip(remainders[1:], moduli[1:]):
        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = gcd(m1, m2).
        d, b1, b2 = egcd(m1, m2)

        if d == 1:
            # The moduli (m1, m2) are coprime
            x = (a1 * b2 * m2) + (a2 * b1 * m1)  # Compute x through explicit construction
            m1 = m1 * m2  # The new modulus
        else:
            # The moduli (m1, m2) are not coprime, however if a1 == b2 (mod d)
            # then a unique solution still exists.
            if not (a1 % d) == (a2 % d):
                raise ValueError(
                    f"Moduli {[m1, m2]} are not coprime "
                    f"and their residuals {[a1, a2]} are not equal modulo their GCD {d}, "
                    "therefore a unique solution does not exist."
                )
            x = ((a1 * b2 * m2) + (a2 * b1 * m1)) // d  # Compute x through explicit construction
            m1 = (m1 * m2) // d  # The new modulus

        a1 = x % m1  # The new equivalent remainder

    # At the end of the process x == a1 (mod m1) where a1 and m1 are the new/modified residual
    # and remainder.

    return a1


###############################################################################
# Factorization
###############################################################################


@overload
def factors(value: int) -> tuple[list[int], list[int]]: ...


@overload
def factors(value: Poly) -> tuple[list[Poly], list[int]]: ...


@export
def factors(value):
    r"""
    Computes the prime factors of a positive integer or the irreducible factors of a non-constant, monic polynomial.

    Arguments:
        value: A positive integer $n$ or a non-constant, monic polynomial $f(x)$.

    Returns:
        - Sorted list of prime factors $\{p_1, p_2, \dots, p_k\}$ of $n$ with
          $p_1 < p_2 < \dots < p_k$ or irreducible factors $\{g_1(x), g_2(x), \dots, g_k(x)\}$ of
          $f(x)$ sorted in lexicographical order.
        - List of corresponding multiplicities $\{e_1, e_2, \dots, e_k\}$.

    Notes:
        .. md-tab-set::

            .. md-tab-item:: Integers

                This function factors a positive integer $n$ into its $k$ prime factors such that
                $n = p_1^{e_1} p_2^{e_2} \dots p_k^{e_k}$.

                Steps:

                0. Test if $n$ is in the `Cunningham Book's
                   <https://homes.cerias.purdue.edu/~ssw/cun/third/index.html>`_ database of $n = p^m \pm 1$
                   factorizations. If so, return the prime factorization.
                1. Test if $n$ is prime. If so, return `[n], [1]`. See :func:`~galois.is_prime`.
                2. Test if $n$ is a perfect power, such that $n = x^k$. If so, prime factor $x$ and
                   multiply the exponents by $k$. See :func:`~galois.perfect_power`.
                3. Use trial division with a list of primes up to $10^6$. If no residual factors, return the
                   discovered prime factors. See :func:`~galois.trial_division`.
                4. Use Pollard's Rho algorithm to find a non-trivial factor of the residual. Continue until all are
                   found. See :func:`~galois.pollard_rho`.

            .. md-tab-item:: Polynomials

                This function factors a monic polynomial $f(x)$ into its $k$ irreducible factors such that
                $f(x) = g_1(x)^{e_1} g_2(x)^{e_2} \dots g_k(x)^{e_k}$.

                Steps:

                1. Apply the Square-Free Factorization algorithm to factor the monic polynomial into square-free
                   polynomials. See :func:`Poly.square_free_factors`.
                2. Apply the Distinct-Degree Factorization algorithm to factor each square-free polynomial into a
                   product of factors with the same degree. See :func:`Poly.distinct_degree_factors`.
                3. Apply the Equal-Degree Factorization algorithm to factor the product of factors of equal degree into
                   their irreducible factors. See :func:`Poly.equal_degree_factors`.

                This factorization is also available in :func:`Poly.factors`.

    References:
        - Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.1.7.
        - Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Construct a composite integer from prime factors.

                .. ipython:: python

                    n = 2**3 * 3 * 5; n

                Factor the integer into its prime factors.

                .. ipython:: python

                    galois.factors(n)

            .. md-tab-item:: Polynomials

                Generate irreducible polynomials over $\mathrm{GF}(3)$.

                .. ipython:: python

                    GF = galois.GF(3)
                    g1 = galois.irreducible_poly(3, 3); g1
                    g2 = galois.irreducible_poly(3, 4); g2
                    g3 = galois.irreducible_poly(3, 5); g3

                Construct a composite polynomial.

                .. ipython:: python

                    e1, e2, e3 = 5, 4, 3
                    f = g1**e1 * g2**e2 * g3**e3; f

                Factor the polynomial into its irreducible factors over $\mathrm{GF}(3)$.

                .. ipython:: python

                    galois.factors(f)

    Group:
        factorization-prime
    """
    if isinstance(value, (int, np.integer)):
        return int_factors(value)
    if isinstance(value, Poly):
        return value.factors()
    raise TypeError(f"Argument 'value' must be either int or Poly, not {type(value)}.")


@overload
def is_square_free(value: int) -> bool: ...


@overload
def is_square_free(value: Poly) -> bool: ...


@export
def is_square_free(value):
    r"""
    Determines if an integer or polynomial is square-free.

    Arguments:
        value: An integer $n$ or polynomial $f(x)$.

    Returns:
        `True` if the integer or polynomial is square-free.

    See Also:
        is_prime_power, is_perfect_power

    Notes:
        .. md-tab-set::

            .. md-tab-item:: Integers

                A square-free integer $n$ is divisible by no perfect squares. As a consequence, the prime
                factorization of a square-free integer $n$ is

                $$n = \prod_{i=1}^{k} p_i^{e_i} = \prod_{i=1}^{k} p_i .$$

            .. md-tab-item:: Polynomials

                A square-free polynomial $f(x)$ has no irreducible factors with multiplicity greater than one.
                Therefore, its canonical factorization is

                $$f(x) = \prod_{i=1}^{k} g_i(x)^{e_i} = \prod_{i=1}^{k} g_i(x) .$$

                This test is also available in :func:`Poly.is_square_free`.

    Examples:
        .. md-tab-set::

            .. md-tab-item:: Integers

                Determine if integers are square-free.

                .. ipython:: python

                    galois.is_square_free(10)
                    galois.is_square_free(18)

            .. md-tab-item:: Polynomials

                Generate irreducible polynomials over $\mathrm{GF}(3)$.

                .. ipython:: python

                    GF = galois.GF(3)
                    f1 = galois.irreducible_poly(3, 3); f1
                    f2 = galois.irreducible_poly(3, 4); f2

                Determine if composite polynomials are square-free over $\mathrm{GF}(3)$.

                .. ipython:: python

                    galois.is_square_free(f1 * f2)
                    galois.is_square_free(f1**2 * f2)

    Group:
        primes-tests
    """
    if isinstance(value, (int, np.integer)):
        return int_is_square_free(value)
    if isinstance(value, Poly):
        return value.is_square_free()
    raise TypeError(f"Argument 'value' must be either int or Poly, not {type(value)}.")
