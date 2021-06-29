"""
A module containing routines for integer factorization.
"""
import bisect
import functools
import itertools
import math
import random

import numpy as np

from .integer import isqrt, iroot, ilog
from .math_ import prod
from .overrides import set_module
from .prime import PRIMES, MAX_PRIME, is_prime

__all__ = [
    "legendre_symbol", "jacobi_symbol", "kronecker_symbol",
    "factors", "perfect_power",
    "divisors", "divisor_sigma",
    "is_prime_power", "is_perfect_power", "is_square_free", "is_smooth", "is_powersmooth"
]


###############################################################################
# Legendre, Jacobi, and Kronecker symbols
###############################################################################

@set_module("galois")
def legendre_symbol(a, p):
    r"""
    Computes the Legendre symbol :math:`(\frac{a}{p})`.

    The Legendre symbol is useful for determining if :math:`a` is a quadratic residue modulo :math:`p`, namely
    :math:`a \in Q_p`. A quadratic residue :math:`a` modulo :math:`p` satisfies :math:`x^2 \equiv a\ (\textrm{mod}\ p)`
    for some :math:`x`.

    .. math::

        \bigg(\frac{a}{p}\bigg) =
            \begin{cases}
                0, & p\ |\ a

                1, & a \in Q_p

                -1, & a \in \overline{Q}_p
            \end{cases}

    Parameters
    ----------
    a : int
        An integer.
    p : int
        An odd prime :math:`p \ge 3`.

    Returns
    -------
    int
        The Legendre symbol :math:`(\frac{a}{p})` with value in :math:`\{0, 1, -1\}`.

    References
    ----------
    * Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    The quadratic residues modulo :math:`7` are :math:`Q_7 = \{1, 2, 4\}`. The quadratic non-residues
    modulo :math:`7` are :math:`\overline{Q}_7 = \{3, 5, 6\}`.

    .. ipython:: python

        [pow(x, 2, 7) for x in range(7)]
        for a in range(7):
            print(f"({a} / 7) = {galois.legendre_symbol(a, 7)}")
    """
    if not is_prime(p):
        raise ValueError(f"Argument `p` must be an odd prime greater than 2, not {p}.")

    return jacobi_symbol(a, p)


@set_module("galois")
def jacobi_symbol(a, n):
    r"""
    Computes the Jacobi symbol :math:`(\frac{a}{n})`.

    The Jacobi symbol extends the Legendre symbol for odd :math:`n \ge 3`. Unlike the Legendre symbol, :math:`(\frac{a}{n}) = 1`
    does not imply :math:`a` is a quadratic residue modulo :math:`n`. However, all :math:`a \in Q_n` have :math:`(\frac{a}{n}) = 1`.

    Parameters
    ----------
    a : int
        An integer.
    n : int
        An odd integer :math:`n \ge 3`.

    Returns
    -------
    int
        The Jacobi symbol :math:`(\frac{a}{n})` with value in :math:`\{0, 1, -1\}`.

    References
    ----------
    * Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    The quadratic residues modulo :math:`9` are :math:`Q_9 = \{1, 4, 7\}` and these all satisfy :math:`(\frac{a}{9}) = 1`.
    The quadratic non-residues modulo :math:`9` are :math:`\overline{Q}_9 = \{2, 3, 5, 6, 8\}`, but notice :math:`\{2, 5, 8\}`
    also satisfy :math:`(\frac{a}{9}) = 1`. The set of integers :math:`\{3, 6\}` not coprime to :math:`n` satisfies
    :math:`(\frac{a}{9}) = 0`.

    .. ipython:: python

        [pow(x, 2, 9) for x in range(9)]
        for a in range(9):
            print(f"({a} / 9) = {galois.jacobi_symbol(a, 9)}")
    """
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not (n > 2 and n % 2 == 1):
        raise ValueError(f"Argument `n` must be an odd integer greater than 2, not {n}.")

    a = a % n
    if a == 0:
        return 0
    if a == 1:
        return 1

    # Write a = 2^e * a1
    a1, e = a, 0
    while a1 % 2 == 0:
        a1, e = a1 // 2, e + 1
    assert 2**e * a1 == a

    if e % 2 == 0:
        s = 1
    else:
        if n % 8 in [1, 7]:
            s = 1
        else:
            s = -1

    if n % 4 == 3 and a1 % 4 == 3:
        s = -s

    n1 = n % a1
    if a1 == 1:
        return s
    else:
        return s * jacobi_symbol(n1, a1)


@set_module("galois")
def kronecker_symbol(a, n):
    r"""
    Computes the Kronecker symbol :math:`(\frac{a}{n})`.

    The Kronecker symbol extends the Jacobi symbol for all :math:`n`.

    Parameters
    ----------
    a : int
        An integer.
    n : int
        An integer.

    Returns
    -------
    int
        The Kronecker symbol :math:`(\frac{a}{n})` with value in :math:`\{0, -1, 1\}`.

    References
    ----------
    * Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    if n == 0:
        return 1 if a in [1, -1] else 0
    if n == 1:
        return 1
    if n == -1:
        return -1 if a < 0 else 1
    if n == 2:
        if a % 2 == 0:
            return 0
        elif a % 8 in [1, 7]:
            return 1
        else:
            return -1

    # Factor out the unit +/- 1
    u = -1 if n < 0 else 1
    n //= u

    # Factor out the powers of 2 so the resulting n is odd
    e = 0
    while n % 2 == 0:
        n, e = n // 2, e + 1

    if n >=3 :
        # Handle the remaining odd n using the Jacobi symbol
        return kronecker_symbol(a, u) * kronecker_symbol(a, 2)**e * jacobi_symbol(a, n)
    else:
        return kronecker_symbol(a, u) * kronecker_symbol(a, 2)**e


###############################################################################
# Prime factorization
###############################################################################

@set_module("galois")
@functools.lru_cache(maxsize=2048)
def factors(n):
    r"""
    Computes the prime factors of the positive integer :math:`n`.

    The integer :math:`n` can be factored into :math:`n = p_1^{e_1} p_2^{e_2} \dots p_{k-1}^{e_{k-1}}`.

    **Steps**:

    1. Test if :math:`n` is prime. If so, return `[n], [1]`.
    2. Test if :math:`n` is a perfect power, such that :math:`n = x^k`. If so, prime factor :math:`x` and multiply its exponents by :math:`k`.
    3. Use trial division with a list of primes up to :math:`10^6`. If no residual factors, return the discovered prime factors.
    4. Use Pollard's Rho algorithm to find a non-trivial factor of the residual. Continue until all are found.

    Parameters
    ----------
    n : int
        Any positive integer.

    Returns
    -------
    list
        Sorted list of :math:`k` prime factors :math:`p = [p_1, p_2, \dots, p_{k-1}]` with :math:`p_1 < p_2 < \dots < p_{k-1}`.
    list
        List of corresponding prime powers :math:`e = [e_1, e_2, \dots, e_{k-1}]`.

    Examples
    --------
    .. ipython:: python

        p, e = galois.factors(120)
        p, e

        # The product of the prime powers is the factored integer
        np.multiply.reduce(np.array(p) ** np.array(e))

    Prime factorization of 1 less than a large prime.

    .. ipython:: python

        prime =1000000000000000035000061
        galois.is_prime(prime)
        p, e = galois.factors(prime - 1)
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
        p, e = factors(base)
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
@functools.lru_cache(maxsize=512)
def perfect_power(n):
    r"""
    Returns the integer base :math:`m > 1` and exponent :math:`e > 1` of :math:`n = m^e` if :math:`n` is a perfect power.

    Parameters
    ----------
    n : int
        A positive integer :math:`n > 1`.

    Returns
    -------
    None, tuple
        `None` is :math:`n` is not a perfect power. Otherwise, :math:`(c, e)` such that :math:`n = c^e`. :math:`c`
        may be composite.

    Examples
    --------
    .. ipython:: python

        # Primes are not perfect powers
        galois.perfect_power(5)
        # Products of primes are not perfect powers
        galois.perfect_power(6)
        # Products of prime powers were the GCD of the exponents is 1 are not perfect powers
        galois.perfect_power(36*125)
        galois.perfect_power(36)
        galois.perfect_power(125)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 1:
        raise TypeError(f"Argument `n` must be greater than 1, not {n}.")
    n = abs(int(n))
    max_prime_idx = bisect.bisect_right(PRIMES, ilog(n, 2))

    for k in PRIMES[0:max_prime_idx]:
        x = iroot(n, k)
        if x**k == n:
            # Recursively determine if x is a perfect power
            ret = perfect_power(x)
            if ret is None:
                return x, k
            else:
                x, kk = ret
                return x, k * kk

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


###############################################################################
# Compsite factorization
###############################################################################

@set_module("galois")
def divisors(n):
    r"""
    Computes all positive integer divisors :math:`d` of the integer :math:`n` such that :math:`d\ |\ n`.

    Parameters
    ----------
    n : int
        Any integer.

    Returns
    -------
    list
        Sorted list of integer divisors :math:`d`.

    Examples
    --------
    .. ipython:: python

        galois.divisors(0)
        galois.divisors(1)
        galois.divisors(24)
        galois.divisors(-24)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    n = abs(int(n))

    if n == 0:
        return []
    if n == 1:
        return [1]

    # Factor n into its unique k prime factors and their exponents
    p, e = factors(n)
    k = len(p)

    # Enumerate all the prime powers, i.e. [p1, p1^2, p1^3, p2, p2^2, ...]
    prime_powers = []
    for pi, ei in zip(p, e):
        prime_powers += [pi**j for j in range(1, ei + 1)]

    d = [1, n]
    for ki in range(1, k + 1):
        # For all prime powers choose ki for ki in [1, 2, ..., k]
        for pair in itertools.combinations(prime_powers, ki):
            d1 = prod(pair)  # One possible divisor
            if n % d1 == 0:
                d2 = n // d1  # The other divisor
                d += [d1, d2]

    # Reduce the list to unique divisors and sort ascending
    d = sorted(list(set(d)))

    return d


@set_module("galois")
def divisor_sigma(n, k=1):
    r"""
    Returns the sum of :math:`k`-th powers of the positive divisors of :math:`n`.

    This function implements the :math:`\sigma_k(n)` function. It is defined as:

    .. math:: \sigma_k(n) = \sum_{d\ |\ n} d^k

    Parameters
    ----------
    n : int
        Any integer.
    k : int, optional
        The degree of the positive divisors. The default is 1 which corresponds to :math:`\sigma_1(n)` which is the
        sum of positive divisors.

    Returns
    -------
    int
        The sum of divisors function :math:`\sigma_k(n)`.

    Examples
    --------
    .. ipython:: python

        galois.divisors(9)
        galois.divisor_sigma(9, k=0)
        galois.divisor_sigma(9, k=1)
        galois.divisor_sigma(9, k=2)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    d = divisors(n)

    if n == 0:
        return len(d)
    else:
        return sum([di**k for di in d])


###############################################################################
# Primality tests
###############################################################################

@set_module("galois")
def is_prime_power(n):
    r"""
    Determines if :math:`n` is a prime power :math:`n = p^k` for prime :math:`p` and :math:`k \ge 1`.

    There is some controversy over whether :math:`1` is a prime power :math:`p^0`. Since :math:`1` is the :math:`0`-th power
    of all primes, it is often regarded not as a prime power. This function returns `False` for :math:`1`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is a prime power.

    Examples
    --------
    .. ipython:: python

        galois.is_prime_power(8)
        galois.is_prime_power(6)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return False

    if is_prime(n):
        return True

    # Determine is n is a perfect power and then check is the base is prime or composite
    ret = perfect_power(n)
    if ret is not None and is_prime(ret[0]):
        return True

    return False


@set_module("galois")
def is_perfect_power(n):
    r"""
    Determines if :math:`n` is a perfect power :math:`n = x^k` for :math:`x > 0` and :math:`k \ge 2`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is a perfect power.

    Examples
    --------
    .. ipython:: python

        galois.is_perfect_power(8)
        galois.is_perfect_power(16)
        galois.is_perfect_power(20)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return True

    if perfect_power(n) is not None:
        return True

    return False


@set_module("galois")
def is_square_free(n):
    r"""
    Determines if :math:`n` is square-free, such that :math:`n = p_1 p_2 \dots p_k`.

    A square-free integer :math:`n` is divisible by no perfect squares. As a consequence, the prime factorization
    of a square-free integer :math:`n` is

    .. math:: n = \prod_{i=1}^{k} p_i^{e_i} = \prod_{i=1}^{k} p_i .

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is square-free.

    Examples
    --------
    .. ipython:: python

        galois.is_square_free(10)
        galois.is_square_free(16)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return True

    _, e = factors(n)

    return e == [1,]*len(e)


@set_module("galois")
def is_smooth(n, B):
    r"""
    Determines if the positive integer :math:`n` is :math:`B`-smooth.

    An integer :math:`n` with prime factorization :math:`n = p_1^{e_1} \dots p_k^{e_k}` is :math:`B`-smooth
    if :math:`p_k \le B`. The :math:`2`-smooth numbers are the powers of :math:`2`. The :math:`5`-smooth numbers
    are known as *regular numbers*. The :math:`7`-smooth numbers are known as *humble numbers* or *highly composite numbers*.

    Parameters
    ----------
    n : int
        A positive integer.
    B : int
        The smoothness bound :math:`B \ge 2`.

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

    # https://math.stackexchange.com/a/3150231
    assert B <= MAX_PRIME
    factor_base = PRIMES[0:bisect.bisect_right(PRIMES, B)]
    k = prod(factor_base)

    while True:
        d = math.gcd(n, k)
        if d > 1:
            while n % d == 0:
                n //= d
            if n == 1:
                return True
        else:
            break

    return False


@set_module("galois")
def is_powersmooth(n, B):
    r"""
    Determines if the positive integer :math:`n` is :math:`B`-powersmooth.

    An integer :math:`n` with prime factorization :math:`n = p_1^{e_1} \dots p_k^{e_k}` is :math:`B`-powersmooth
    if :math:`p_i^{e_i} \le B` for :math:`1 \le i \le k`.

    Parameters
    ----------
    n : int
        A positive integer.
    B : int
        The smoothness bound :math:`B \ge 2`.

    Returns
    -------
    bool
        `True` if :math:`n` is :math:`B`-powersmooth.

    Examples
    --------
    Comparison of :math:`B`-smooth and :math:`B`-powersmooth. Necessarily, any :math:`n` that is
    :math:`B`-powersmooth must be :math:`B`-smooth.

    .. ipython:: python

        galois.is_smooth(2**4 * 3**2 * 5, 5)
        galois.is_powersmooth(2**4 * 3**2 * 5, 5)
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

    assert B <= MAX_PRIME
    D = 1  # The product of all GCDs with the prime powers
    for p in PRIMES[0:bisect.bisect_right(PRIMES, B)]:
        e = ilog(B, p) + 1  # Find the exponent e of p such that p^e > B
        d = math.gcd(p**e, n)
        D *= d

        # If the GCD is p^e, then p^e > B divides n and therefor n cannot be B-powersmooth
        if d == p**e:
            return False

    # If the product of GCDs of n with each prime power is less than n, then n has a prime factor greater than B.
    # Therefore, n cannot be B-powersmooth.
    if D < n:
        return False

    return True
