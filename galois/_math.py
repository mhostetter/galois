"""
A module that contains some future features of the math stdlib for earlier Python versions.
"""
import numpy as np


def isqrt(n):
    """
    Computes the integer square root of :math:`n` such that :math:`\\textrm{isqrt}(n)^2 \\le n`.

    Note
    ----
        This function is included for Python versions before 3.8. For Python 3.8 and later, use
        `math.isqrt` from the standard library.

    Parameters
    ----------
    n : int
        A non-negative integer

    Returns
    -------
    int
        The integer square root of :math:`n` such that :math:`\\textrm{isqrt}(n)^2 \\le n`.

    Examples
    --------
    .. ipython:: python

        # Use a large Mersenne prime
        p = galois.mersenne_primes(2000)[-1]; p
        sqrt_p = galois.isqrt(p); sqrt_p
        sqrt_p**2 <= p
        (sqrt_p + 1)**2 <= p
    """
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
