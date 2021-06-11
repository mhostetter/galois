"""
A module containing routines for integer factorization.
"""
import bisect
import functools
import math
import random

import numpy as np

from .math_ import isqrt, iroot, ilog
from .overrides import set_module
from .prime import PRIMES, is_prime

__all__ = ["prime_factors", "is_smooth"]


def perfect_power(n):
    max_prime_idx = bisect.bisect_right(PRIMES, ilog(n, 2))

    for prime in PRIMES[0:max_prime_idx]:
        x = iroot(n, prime)
        if x**prime == n:
            return x, prime

    return None


def trial_division_factor(n):
    max_factor = isqrt(n)
    max_prime_idx = bisect.bisect_right(PRIMES, max_factor)

    p, e = [], []
    for prime in PRIMES[0:max_prime_idx]:
        degree = 0
        while n % prime == 0:
            degree += 1
            n //= prime
        if degree > 0:
            p.append(prime)
            e.append(degree)
            if n == 1:
                break

    return p, e, n


@functools.lru_cache(maxsize=1024)
def pollard_rho_factor(n, c=1):
    """
    References
    ----------
    * Section 3.2.2 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf
    """
    f = lambda x: (x**2 + c) % n

    a, b, d = 2, 2, 1
    while True:
        a = f(a)
        b = f(f(b))
        b = f(f(b))
        d = math.gcd(a - b, n)

        if 1 < d < n:
            return d
        if d == n:
            return None  # Failure


# def fermat_factors(n):
#     a = isqrt(n) + 1
#     b2 = a**2 - n
#     while isqrt(b2)**2 != b2:
#         a += 1
#         b2 = a**2 - n
#     b = isqrt(b2)
#     return a - b, a + b


@set_module("galois")
def prime_factors(n):
    """
    Computes the prime factors of the positive integer :math:`n`.

    The integer :math:`n` can be factored into :math:`n = p_1^{e_1} p_2^{e_2} \\dots p_{k-1}^{e_{k-1}}`.

    **Steps**:

    1. Test if :math:`n` is prime. If so, return `[n], [1]`.
    2. Test if :math:`n` is a perfect power, such that :math:`n = x^k`. If so, prime factor :math:`x` and multiply its exponents by :math:`k`.
    3. Use trial division with a list of primes up to :math:`10^6`. If no residual factors, return the discovered prime factors.
    4. Use Pollard's Rho algorithm to find a non-trivial factor of the residual. Continue until all are found.

    Parameters
    ----------
    n : int
        The positive integer to be factored.

    Returns
    -------
    list
        Sorted list of :math:`k` prime factors :math:`p = [p_1, p_2, \\dots, p_{k-1}]` with :math:`p_1 < p_2 < \\dots < p_{k-1}`.
    list
        List of corresponding prime powers :math:`e = [e_1, e_2, \\dots, e_{k-1}]`.

    Examples
    --------
    .. ipython:: python

        p, e = galois.prime_factors(120)
        p, e

        # The product of the prime powers is the factored integer
        np.multiply.reduce(np.array(p) ** np.array(e))

    Prime factorization of 1 less than a large prime.

    .. ipython:: python

        prime =1000000000000000035000061
        galois.is_prime(prime)
        p, e = galois.prime_factors(prime - 1)
        p, e
        np.multiply.reduce(np.array(p) ** np.array(e))
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 1:
        raise ValueError(f"Argument `n` must be greater than 1, not {n}.")
    n = int(n)

    # Step 1
    if is_prime(n):
        return [n], [1]

    # Step 2
    result = perfect_power(n)
    if result is not None:
        base, exponent = result
        p, e = prime_factors(base)
        e = [ei * exponent for ei in e]
        return p, e

    # Step 3
    p, e, n = trial_division_factor(n)

    # Step 4
    while n > 1 and not is_prime(n):
        f = pollard_rho_factor(n)  # A non-trivial factor
        while f is None:
            # Try again with a different random function f(x)
            f = pollard_rho_factor(n, c=random.randint(2, n // 2))
        if is_prime(f):
            degree = 0
            while n % f == 0:
                degree += 1
                n //= f
            p.append(f)
            e.append(degree)
        else:
            raise RuntimeError(f"Encountered a very large composite {f}. Please report this as a GitHub issue at https://github.com/mhostetter/galois/issues.")

    if n > 1:
        p.append(n)
        e.append(1)

    return p, e


@set_module("galois")
def is_smooth(n, B):
    """
    Determines if the positive integer :math:`n` is :math:`B`-smooth, i.e. all its prime factors satisfy :math:`p \\le B`.

    The :math:`2`-smooth numbers are the powers of :math:`2`. The :math:`5`-smooth numbers are known
    as *regular numbers*. The :math:`7`-smooth numbers are known as *humble numbers* or *highly composite numbers*.

    Parameters
    ----------
    n : int
        A positive integer.
    B : int
        The smoothness bound.

    Returns
    -------
    bool
        `True` if :math:`n` is :math:`B`-smooth.

    Examples
    --------
    .. ipython:: python

        galois.is_smooth(2**10, 2)
        galois.is_smooth(10, 5)
        galois.is_smooth(12, 5)
        galois.is_smooth(60**2, 5)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(B, (int, np.integer)):
        raise TypeError(f"Argument `B` must be an integer, not {type(B)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be non-negative, not {n}.")
    if not B >= 2:
        raise ValueError(f"Argument `B` must be at least 2, not {B}.")

    if n == 1:
        return True
    else:
        p, _ = prime_factors(n)
        return p[-1] <= B
