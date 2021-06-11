"""
A module that contains some future features of the math stdlib for earlier Python versions.
"""
import math
import sys

import numpy as np

from .overrides import set_module

__all__ = ["pow", "isqrt", "iroot", "ilog", "lcm", "prod"]


@set_module("galois")
def pow(base, exp, mod):  # pylint: disable=redefined-builtin
    """
    Efficiently exponentiates an integer :math:`a^k (\\textrm{mod}\\ m)`.

    The algorithm is more efficient than exponentiating first and then reducing modulo :math:`m`. This
    is the integer equivalent of :func:`galois.poly_pow`.

    Note
    ----
    This function is an alias of :func:`pow` in the standard library.

    Parameters
    ----------
    base : int
        The integer base :math:`a`.
    exp : int
        The integer exponent :math:`k`.
    mod : int
        The integer modulus :math:`m`.

    Returns
    -------
    int
        The modular exponentiation :math:`a^k (\\textrm{mod}\\ m)`.

    Examples
    --------
    .. ipython:: python

        galois.pow(3, 5, 7)
        (3**5) % 7
    """
    import builtins  # pylint: disable=import-outside-toplevel
    return builtins.pow(base, exp, mod)


@set_module("galois")
def isqrt(n):
    """
    Computes the integer square root of :math:`n` such that :math:`\\textrm{isqrt}(n)^2 \\le n`.

    Note
    ----
    This function is included for Python versions before 3.8. For Python 3.8 and later, this function
    calls :func:`math.isqrt` from the standard library.

    Parameters
    ----------
    n : int
        A non-negative integer.

    Returns
    -------
    int
        The integer square root of :math:`n` such that :math:`\\textrm{isqrt}(n)^2 \\le n`.

    Examples
    --------
    .. ipython:: python

        galois.isqrt(27**2 - 1)
        galois.isqrt(27**2)
        galois.isqrt(27**2 + 1)
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        return math.isqrt(n)  # pylint: disable=no-member
    else:
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
        if not n >= 0:
            raise ValueError(f"Argument `n` must be non-negative, not {n}.")

        n = int(n)
        if n < 2:
            return n

        small_candidate = isqrt(n >> 2) << 1
        large_candidate = small_candidate + 1
        if large_candidate * large_candidate > n:
            return small_candidate
        else:
            return large_candidate


@set_module("galois")
def iroot(n, k):
    """
    Finds the integer :math:`k`-th root :math:`x` of :math:`n`, such that :math:`x^k \\le n`.

    Parameters
    ----------
    n : int
        A positive integer.
    k : int
        The root :math:`k`, must be at least 2.

    Returns
    -------
    int
        The integer :math:`k`-th root :math:`x` of :math:`n`, such that :math:`x^k \\le n`

    Examples
    --------
    .. ipython :: python

        galois.iroot(27**5 - 1, 5)
        galois.iroot(27**5, 5)
        galois.iroot(27**5 + 1, 5)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be positive, not {n}.")
    if not k >= 2:
        raise ValueError(f"Argument `k` must be at least 2, not {k}.")

    # https://stackoverflow.com/a/39191163/11694321
    u = n
    x = n + 1
    k1 = k - 1

    while u < x:
        x = u
        u = (k1*u + n // u**k1) // k

    return x


@set_module("galois")
def ilog(n, b):
    """
    Finds the integer :math:`\\textrm{log}_b(n) = k`, such that :math:`b^k \\le n`.

    Parameters
    ----------
    n : int
        A positive integer.
    b : int
        The logarithm base :math:`b`.

    Returns
    -------
    int
        The integer :math:`\\textrm{log}_b(n) = k`, such that :math:`b^k \\le n`.

    Examples
    --------
    .. ipython :: python

        galois.ilog(27**5 - 1, 27)
        galois.ilog(27**5, 27)
        galois.ilog(27**5 + 1, 27)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(b, (int, np.integer)):
        raise TypeError(f"Argument `b` must be an integer, not {type(b)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be positive, not {n}.")
    if not b >= 2:
        raise ValueError(f"Argument `b` must be at least 2, not {b}.")

    # https://stackoverflow.com/a/39191163/11694321
    low, b_low, high, b_high = 0, 1, 1, b

    while b_high < n:
        low, b_low, high, b_high = high, b_high, high*2, b_high**2

    while high - low > 1:
        mid = (low + high) // 2
        b_mid = b_low * b**(mid - low)
        if n < b_mid:
            high, b_high = mid, b_mid
        elif b_mid < n:
            low, b_low = mid, b_mid
        else:
            return mid

    if b_high == n:
        return high

    return low


@set_module("galois")
def lcm(*integers):
    """
    Computes the least common multiple of the integer arguments.

    Note
    ----
    This function is included for Python versions before 3.9. For Python 3.9 and later, this function
    calls :func:`math.lcm` from the standard library.

    Returns
    -------
    int
        The least common multiple of the integer arguments. If any argument is 0, the LCM is 0. If no
        arguments are provided, 1 is returned.

    Examples
    --------
    .. ipython:: python

        galois.lcm()
        galois.lcm(2, 4, 14)
        galois.lcm(3, 0, 9)

    This function also works on arbitrarily-large integers.

    .. ipython:: python

        prime1, prime2 = galois.mersenne_primes(100)[-2:]
        prime1, prime2
        lcm = galois.lcm(prime1, prime2); lcm
        lcm == prime1 * prime2
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 9:
        return math.lcm(*integers)  # pylint: disable=no-member
    else:
        _lcm  = 1
        for integer in integers:
            _lcm = _lcm * integer // math.gcd(_lcm, integer)
        return _lcm


@set_module("galois")
def prod(iterable, start=1):
    """
    Computes the product of the integer arguments.

    Note
    ----
    This function is included for Python versions before 3.8. For Python 3.8 and later, this function
    calls :func:`math.prod` from the standard library.

    Returns
    -------
    int
        The product of the integer arguments.

    Examples
    --------
    .. ipython:: python

        galois.prod([2, 4, 14])
        galois.prod([2, 4, 14], start=2)
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        return math.prod(iterable, start=start)  # pylint: disable=no-member
    else:
        result = start
        for integer in iterable:
            result *= integer
        return result
