import math
from itertools import combinations

import numba
import numpy as np


def gcd(a, b):
    """
    Finds the integer multiplicands of :math:`a` and :math:`b` such that :math:`a x + b y = \\mathrm{gcd}(a, b)`.

    This implementation uses the Extended Euclidean Algorithm.

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

    r = [a, b]
    s = [1, 0]
    t = [0, 1]

    while True:
        qi = r[-2] // r[-1]
        ri = r[-2] % r[-1]
        r.append(ri)
        s.append(s[-2] - qi*s[-1])
        t.append(t[-2] - qi*t[-1])
        if ri == 0:
            break

    return r[-2], s[-2], t[-2]


@numba.jit("int64[:](int64, int64)", nopython=True)
def gcd_jit(a, b):  # pragma: no cover
    r = [a, b]
    s = [1, 0]
    t = [0, 1]

    while True:
        qi = r[-2] // r[-1]
        ri = r[-2] % r[-1]
        r.append(ri)
        s.append(s[-2] - qi*s[-1])
        t.append(t[-2] - qi*t[-1])
        if ri == 0:
            break

    return np.array([r[-2], s[-2], t[-2]], dtype=np.int64)


def chinese_remainder_theorem(a, m):
    """
    Solves the simultaneous system of congruences for :math:`x`.

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
        x = galois.chinese_remainder_theorem(a, m); x

        for i in range(len(a)):
            ai = x % m[i]
            print(f"{x} = {ai} (mod {m[i]}), Valid congruence: {ai == a[i]}")
    """
    a = np.array(a)
    m = np.array(m)
    if not m.size == a.size:
        raise ValueError(f"Arguments `a` and `m` are not the same size, {a.size} != {m.size}.")
    for pair in combinations(m, 2):
        if not math.gcd(pair[0], pair[1]) == 1:
            raise ValueError(f"Elements of argument `m` must be pairwise coprime, {pair} are not.")

    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1 = a[0]
    m1 = m[0]
    for i in range(1, m.size):
        a2 = a[i]
        m2 = m[i]

        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = 1,
        # where 1 is the GCD(m1, m2) because m1 and m2 are pairwise relatively coprime
        b1, b2 = gcd(m1, m2)[1:3]

        # Compute x through explicit construction
        x = a1*b2*m2 + a2*b1*m1

        m1 = m1 * m2  # The new modulus
        a1 = x % m1  # The new equivalent remainder

    # Align x to be within [0, prod(m))
    x = x % np.prod(m)

    return x
