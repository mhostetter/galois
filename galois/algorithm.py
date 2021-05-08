import math
from itertools import combinations

import numpy as np

from .overrides import set_module
from .math_ import prod

__all__ = ["gcd", "crt"]


@set_module("galois")
def gcd(a, b):
    """
    Finds the integer multiplicands of :math:`a` and :math:`b` such that :math:`a x + b y = \\mathrm{gcd}(a, b)`.

    This function implements the Extended Euclidean Algorithm.

    Parameters
    ----------
    a : int
        Any integer.
    b : int
        Any integer.

    Returns
    -------
    int
        Greatest common divisor of :math:`a` and :math:`b`.
    int
        Integer :math:`x`, such that :math:`a x + b y = \\mathrm{gcd}(a, b)`.
    int
        Integer :math:`y`, such that :math:`a x + b y = \\mathrm{gcd}(a, b)`.

    References
    ----------
    * T. Moon, "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181
    * https://en.wikipedia.org/wiki/Euclidean_algorithm#Extended_Euclidean_algorithm

    Examples
    --------
    .. ipython:: python

        a = 2
        b = 13
        gcd, x, y = galois.gcd(a, b)
        gcd, x, y
        a*x + b*y == gcd
    """
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(b, (int, np.integer)):
        raise TypeError(f"Argument `b` must be an integer, not {type(b)}.")

    r2, r1 = a, b
    s2, s1 = 1, 0
    t2, t1 = 0, 1

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q*r1
        s2, s1 = s1, s2 - q*s1
        t2, t1 = t1, t2 - q*t1

    return r2, s2, t2


@set_module("galois")
def crt(a, m):
    """
    Solves the simultaneous system of congruences for :math:`x`.

    This function implements the Chinese Remainder Theorem.

    .. math::
        x &\\equiv a_1\\ (\\textrm{mod}\\ m_1)

        x &\\equiv a_2\\ (\\textrm{mod}\\ m_2)

        x &\\equiv \\ldots

        x &\\equiv a_n\\ (\\textrm{mod}\\ m_n)

    Parameters
    ----------
    a : array_like
        The integer remainders :math:`a_i`.
    m : array_like
        The integer modulii :math:`m_i`.

    Returns
    -------
    int
        The simultaneous solution :math:`x` to the system of congruences.

    Examples
    --------
    .. ipython:: python

        a = [0, 3, 4]
        m = [3, 4, 5]
        x = galois.crt(a, m); x

        for i in range(len(a)):
            ai = x % m[i]
            print(f"{x} = {ai} (mod {m[i]}), Valid congruence: {ai == a[i]}")
    """
    if not len(a) == len(m):
        raise ValueError(f"Arguments `a` and `m` are not the same length, {len(a)} != {len(m)}.")
    for pair in combinations(m, 2):
        if not math.gcd(pair[0], pair[1]) == 1:
            raise ValueError(f"Elements of argument `m` must be pairwise coprime, {pair} are not.")

    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1, m1 = a[0], m[0]
    for a2, m2 in zip(a[1:], m[1:]):
        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = 1,
        # where 1 is the GCD(m1, m2) because m1 and m2 are pairwise relatively coprime
        b1, b2 = gcd(m1, m2)[1:]

        # Compute x through explicit construction
        x = a1*b2*m2 + a2*b1*m1

        m1 = m1 * m2  # The new modulus
        a1 = x % m1  # The new equivalent remainder

    # Align x to be within [0, prod(m))
    x = x % prod(m)

    return x
