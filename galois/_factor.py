"""
A module containing routines for integer factorization.
"""
import bisect
import functools
import itertools
import math
import random

import numpy as np

from ._math import prod, isqrt, iroot, ilog
from ._overrides import set_module
from ._prime import primes, is_prime

__all__ = [
    "legendre_symbol", "jacobi_symbol", "kronecker_symbol",
    "perfect_power", "trial_division", "pollard_p1", "pollard_rho",
    "divisors", "divisor_sigma",
    "is_prime_power", "is_perfect_power", "is_smooth", "is_powersmooth"
]


###############################################################################
# Legendre, Jacobi, and Kronecker symbols
###############################################################################

@set_module("galois")
def legendre_symbol(a, p):
    r"""
    Computes the Legendre symbol :math:`(\frac{a}{p})`.

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

    Notes
    -----
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
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(p, (int, np.integer)):
        raise TypeError(f"Argument `p` must be an integer, not {type(p)}.")
    if not (is_prime(p) and p > 2):
        raise ValueError(f"Argument `p` must be an odd prime greater than 2, not {p}.")

    return jacobi_symbol(a, p)


@set_module("galois")
def jacobi_symbol(a, n):
    r"""
    Computes the Jacobi symbol :math:`(\frac{a}{n})`.

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

    Notes
    -----
    The Jacobi symbol extends the Legendre symbol for odd :math:`n \ge 3`. Unlike the Legendre symbol, :math:`(\frac{a}{n}) = 1`
    does not imply :math:`a` is a quadratic residue modulo :math:`n`. However, all :math:`a \in Q_n` have :math:`(\frac{a}{n}) = 1`.

    References
    ----------
    * Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples
    --------
    The quadratic residues modulo :math:`9` are :math:`Q_9 = \{1, 4, 7\}` and these all satisfy :math:`(\frac{a}{9}) = 1`.
    The quadratic non-residues modulo :math:`9` are :math:`\overline{Q}_9 = \{2, 3, 5, 6, 8\}`, but notice :math:`\{2, 5, 8\}`
    also satisfy :math:`(\frac{a}{9}) = 1`. The set of integers :math:`\{3, 6\}` not coprime to :math:`9` satisfies
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

@functools.lru_cache(maxsize=2048)
def factors(n):
    """
    This function is wrapped and documented in `_polymorphic.factors()`.
    """
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
    p, e, n = trial_division(n, 10_000_000)

    # Step 4
    while n > 1 and not is_prime(n):
        f = pollard_rho(n)  # A non-trivial factor
        while f is None:
            # Try again with a different random function f(x)
            f = pollard_rho(n, c=random.randint(2, n // 2))
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
    Returns the integer base :math:`c > 1` and exponent :math:`e > 1` of :math:`n = c^e` if :math:`n` is a perfect power.

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
        galois.perfect_power(2*3)

        # Products of prime powers were the GCD of the exponents is 1 are not perfect powers
        galois.perfect_power(2 * 3 * 5**3)

        # Products of prime powers were the GCD of the exponents is > 1 are perfect powers
        galois.perfect_power(2**2 * 3**2 * 5**4)
        galois.perfect_power(36)
        galois.perfect_power(125)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 1:
        raise ValueError(f"Argument `n` must be greater than 1, not {n}.")
    n = int(n)

    for k in primes(ilog(n, 2)):
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


@set_module("galois")
def trial_division(n, B=None):
    r"""
    Finds all the prime factors :math:`p_i^{e_i}` of :math:`n` for :math:`p_i \le B`.

    The trial division factorization will find all prime factors :math:`p_i \le B` such that :math:`n` factors
    as :math:`n = p_1^{e_1} \dots p_k^{e_k} n_r` where :math:`n_r` is a residual factor (which may be composite).

    Parameters
    ----------
    n : int
        A positive integer.
    B : int, optional
        The max divisor in the trial division. The default is `None` which corresponds to :math:`B = \sqrt{n}`.
        If :math:`B > \sqrt{n}`, the algorithm will only search up to :math:`\sqrt{n}`, since a prime factor of :math:`n`
        cannot be larger than :math:`\sqrt{n}`.

    Returns
    -------
    list
        The discovered prime factors :math:`\{p_1, \dots, p_k\}`.
    list
        The corresponding prime exponents :math:`\{e_1, \dots, e_k\}`.
    int
        The residual factor :math:`n_r`.

    Examples
    --------
    .. ipython:: python

        n = 2**4 * 17**3 * 113 * 15013
        galois.trial_division(n)
        galois.trial_division(n, B=500)
        galois.trial_division(n, B=100)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(B, (type(None), int, np.integer)):
        raise TypeError(f"Argument `B` must be an integer, not {type(B)}.")
    B = isqrt(n) if B is None else B
    if not n > 1:
        raise ValueError(f"Argument `n` must be greater than 1, not {n}.")
    if not B > 2:
        raise ValueError(f"Argument `B` must be greater than 2, not {B}.")
    n = int(n)
    B = min(isqrt(n), B)  # There cannot be a prime factor greater than sqrt(n)

    p, e = [], []
    for prime in primes(B):
        degree = 0
        while n % prime == 0:
            degree += 1
            n //= prime
        if degree > 0:
            p.append(prime)
            e.append(degree)

            # Check if we've fully factored n and if so break early
            if n == 1:
                break

    return p, e, n


@set_module("galois")
def pollard_p1(n, B, B2=None):
    r"""
    Attempts to find a non-trivial factor of :math:`n` if it has a prime factor :math:`p` such that
    :math:`p-1` is :math:`B`-smooth.

    For a given odd composite :math:`n` with a prime factor :math:`p`, Pollard's :math:`p-1` algorithm can discover a non-trivial factor
    of :math:`n` if :math:`p-1` is :math:`B`-smooth. Specifically, the prime factorization must satisfy :math:`p-1 = p_1^{e_1} \dots p_k^{e_k}`
    with each :math:`p_i \le B`.

    A extension of Pollard's :math:`p-1` algorithm allows a prime factor :math:`p` to be :math:`B`-smooth with the exception of one
    prime factor :math:`B < p_{k+1} \le B_2`. In this case, the prime factorization is :math:`p-1 = p_1^{e_1} \dots p_k^{e_k} p_{k+1}`.
    Often :math:`B_2` is chosen such that :math:`B_2 \gg B`.

    Parameters
    ----------
    n : int
        An odd composite integer :math:`n > 2` that is not a prime power.
    B : int
        The smoothness bound :math:`B > 2`.
    B2 : int, optional
        The smoothness bound :math:`B_2` for the optional second step of the algorithm. The default is `None`, which
        will not perform the second step.

    Returns
    -------
    None, int
        A non-trivial factor of :math:`n`, if found. `None` if not found.

    References
    ----------
    * Section 3.2.3 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples
    --------
    Here, :math:`n = pq` where :math:`p-1` is :math:`1039`-smooth and :math:`q-1` is :math:`17`-smooth.

    .. ipython:: python

        p, q = 1458757, 1326001
        galois.factors(p - 1)
        galois.factors(q - 1)

    Searching with :math:`B=15` will not recover a prime factor.

    .. ipython:: python

        galois.pollard_p1(p*q, 15)

    Searching with :math:`B=17` will recover the prime factor :math:`q`.

    .. ipython:: python

        galois.pollard_p1(p*q, 17)

    Searching :math:`B=15` will not recover a prime factor in the first step, but will find :math:`q` in the second
    step because :math:`p_{k+1} = 17` satisfies :math:`15 < 17 \le 100`.

    .. ipython:: python

        galois.pollard_p1(p*q, 15, B2=100)

    Pollard's :math:`p-1` algorithm may return a composite factor.

    .. ipython:: python

        n = 2133861346249
        galois.factors(n)
        galois.pollard_p1(n, 10)
        37*41
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(B, (int, np.integer)):
        raise TypeError(f"Argument `B` must be an integer, not {type(B)}.")
    if not isinstance(B2, (type(None), int, np.integer)):
        raise TypeError(f"Argument `B2` must be an integer, not {type(B2)}.")
    if not (n % 2 == 1 and n > 2):
        raise ValueError(f"Argument `n` must be odd and greater than 2, not {n}.")
    if not B > 2:
        raise ValueError(f"Argument `B` must be greater than 2, not {B}.")
    n = int(n)

    a = 2  # A value that is coprime to n (since n is odd)
    check_stride = 10

    for i, p in enumerate(primes(B)):
        e = ilog(n, p)
        a = pow(a, p**e, n)

        # Check the GCD periodically to return early without checking all primes less than the
        # smoothness bound
        if i % check_stride == 0 and math.gcd(a - 1, n) not in [1, n]:
            return math.gcd(a - 1, n)

    d = math.gcd(a - 1, n)

    if d not in [1, n]:
        return d
    if d == n:
        return None

    # Try to find p such that p - 1 has a single prime factor larger than B
    if B2 is not None:
        P = primes(B2)
        P = P[bisect.bisect_right(P, B):bisect.bisect_right(P, B2)]  # Only select primes between B < prime <= B2
        for i, p in enumerate(P):
            a = pow(a, p, n)

            # Check the GCD periodically to return early without checking all primes less than the
            # smoothness bound
            if i % check_stride == 0 and math.gcd(a - 1, n) not in [1, n]:
                return math.gcd(a - 1, n)

        d = math.gcd(a - 1, n)
        if d not in [1, n]:
            return d

    return None


# @functools.lru_cache(maxsize=1024)
def pollard_rho(n, c=1):
    r"""
    Attempts to find a non-trivial factor of :math:`n` using cycle detection.

    Pollard's :math:`\rho` algorithm seeks to find a non-trivial factor of :math:`n` by finding a cycle in a sequence
    of integers :math:`x_0, x_1, \dots` defined by :math:`x_i = f(x_{i-1}) = x_{i-1}^2 + 1\ \textrm{mod}\ p` where :math:`p`
    is an unknown small prime factor of :math:`n`. This happens when :math:`x_{m} \equiv x_{2m}\ (\textrm{mod}\ p)`.
    Because :math:`p` is unknown, this is accomplished by computing the sequence modulo :math:`n` and looking for
    :math:`\textrm{gcd}(x_m - x_{2m}, n) > 1`.

    Parameters
    ----------
    n : int
        An odd composite integer :math:`n > 2` that is not a prime power.
    c : int, optional
        The constant offset in the function :math:`f(x) = x^2 + c\ \textrm{mod}\ n`. The default is 1. A requirement
        of the algorithm is that :math:`c \not\in \{0, -2\}`.

    Returns
    -------
    None, int
        A non-trivial factor :math:`m` of :math:`n`, if found. `None` if not found.

    References
    ----------
    * Section 3.2.2 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples
    --------
    Pollard's :math:`\rho` is especially good at finding small factors.

    .. ipython:: python

        n = 503**7 * 10007 * 1000003
        galois.pollard_rho(n)

    It is also efficient for finding relatively small factors.

    .. ipython:: python

        n = 1182640843 * 1716279751
        galois.pollard_rho(n)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(c, (type(None), int, np.integer)):
        raise TypeError(f"Argument `c` must be an integer, not {type(c)}.")
    if not n > 1:
        raise ValueError(f"Argument `n` must be greater than 1, not {n}.")
    if not c not in [0, -2]:
        raise ValueError("Argument `c` cannot be -2 or 0.")
    n = abs(int(n))

    f = lambda x: (x**2 + c) % n

    a, b, d = 2, 2, 1
    while d == 1:
        a = f(a)
        b = f(f(b))
        d = math.gcd(a - b, n)

    if d == n:
        return None

    return d


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
        Sorted list of positive integer divisors :math:`d`.

    Notes
    -----
    :func:`galois.divisors` find *all* positive integer divisors or factors of :math:`n`, where :func:`galois.factors` only finds the prime
    factors of :math:`n`.

    Examples
    --------
    .. ipython:: python

        galois.divisors(0)
        galois.divisors(1)
        galois.divisors(24)
        galois.divisors(-24)
        galois.factors(24)
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
        for items in itertools.combinations(prime_powers, ki):
            d1 = prod(*items)  # One possible divisor
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

    Notes
    -----
    This function implements the :math:`\sigma_k(n)` function. It is defined as:

    .. math:: \sigma_k(n) = \sum_{d\ |\ n} d^k

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

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is a prime power.

    Notes
    -----
    There is some controversy over whether :math:`1` is a prime power :math:`p^0`. Since :math:`1` is the :math:`0`-th power
    of all primes, it is often regarded not as a prime power. This function returns `False` for :math:`1`.

    Examples
    --------
    .. ipython:: python

        galois.is_prime_power(8)
        galois.is_prime_power(6)
        galois.is_prime_power(1)
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


def is_square_free(n):
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

    Notes
    -----
    An integer :math:`n` with prime factorization :math:`n = p_1^{e_1} \dots p_k^{e_k}` is :math:`B`-smooth
    if :math:`p_k \le B`. The :math:`2`-smooth numbers are the powers of :math:`2`. The :math:`5`-smooth numbers
    are known as *regular numbers*. The :math:`7`-smooth numbers are known as *humble numbers* or *highly composite numbers*.

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

    for p in primes(B):
        e = ilog(n, p)
        d = math.gcd(p**e, n)
        if d > 1:
            n //= d
        if n == 1:
            # n can be fully factored by the primes already test, therefore it is B-smooth
            return True

    # n has residual prime factors larger than B and therefore is not B-smooth
    return False


@set_module("galois")
def is_powersmooth(n, B):
    r"""
    Determines if the positive integer :math:`n` is :math:`B`-powersmooth.

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

    Notes
    -----
    An integer :math:`n` with prime factorization :math:`n = p_1^{e_1} \dots p_k^{e_k}` is :math:`B`-powersmooth
    if :math:`p_i^{e_i} \le B` for :math:`1 \le i \le k`.

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

    D = 1  # The product of all GCDs with the prime powers
    for p in primes(B):
        e = ilog(B, p) + 1  # Find the exponent e of p such that p^e > B
        d = math.gcd(p**e, n)
        D *= d

        # If the GCD is p^e, then p^e > B divides n and therefore n cannot be B-powersmooth
        if d == p**e:
            return False

    # If the product of GCDs of n with each prime power is less than n, then n has a prime factor greater than B.
    # Therefore, n cannot be B-powersmooth.
    if D < n:
        return False

    return True
