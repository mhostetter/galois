"""
A module that contains integer arithmetic routines.
"""
import math
import sys

import numpy as np

from ._overrides import set_module

__all__ = ["isqrt", "iroot", "ilog"]


@set_module("galois")
def isqrt(n):
    r"""
    Computes :math:`x = \lfloor\sqrt{n}\rfloor` such that :math:`x^2 \le n < (x + 1)^2`.

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
        The integer square root of :math:`n`.

    Examples
    --------
    .. ipython:: python

        n = 1000
        x = galois.isqrt(n); x
        print(f"{x**2} <= {n} < {(x + 1)**2}")
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

        # Recursively compute the integer square root
        x = isqrt(n >> 2) << 1

        if (x + 1)**2 > n:
            return x
        else:
            return x + 1


@set_module("galois")
def iroot(n, k):
    r"""
    Computes :math:`x = \lfloor n^{\frac{1}{k}} \rfloor` such that :math:`x^k \le n < (x + 1)^k`.

    Parameters
    ----------
    n : int
        A non-negative integer.
    k : int
        The root :math:`k`, must be at least 2.

    Returns
    -------
    int
        The integer :math:`k`-th root of :math:`n`.

    Examples
    --------
    .. ipython :: python

        n = 1000
        x = galois.iroot(n, 5); x
        print(f"{x**5} <= {n} < {(x + 1)**5}")
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
    if not n >= 0:
        raise ValueError(f"Argument `n` must be non-negative, not {n}.")
    if not k >= 2:
        raise ValueError(f"Argument `k` must be at least 2, not {k}.")
    n, k = int(n), int(k)

    if n == 0:
        return 0

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
    r"""
    Computes :math:`x = \lfloor\textrm{log}_b(n)\rfloor` such that :math:`b^x \le n < b^{x + 1}`.

    Parameters
    ----------
    n : int
        A positive integer.
    b : int
        The logarithm base :math:`b`.

    Returns
    -------
    int
        The integer logarithm base :math:`b` of :math:`n`.

    Examples
    --------
    .. ipython :: python

        n = 1000
        x = galois.ilog(n, 5); x
        print(f"{5**x} <= {n} < {5**(x + 1)}")
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(b, (int, np.integer)):
        raise TypeError(f"Argument `b` must be an integer, not {type(b)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be positive, not {n}.")
    if not b >= 2:
        raise ValueError(f"Argument `b` must be at least 2, not {b}.")
    n, b = int(n), int(b)

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
