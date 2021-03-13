from itertools import combinations, islice
from bisect import bisect_left, bisect_right
from math import gcd, sqrt
from typing import List, Tuple
import numba  # TODO force it to work with globals
import numpy as np

from .poly import Poly


def primes(limit: int) -> List[int]:
    """ TODO docstring  """
    chanks = (limit + 29) // 30
    lim, sieve = chanks * 15 - 1, bytearray(b'\0\0\1\0\1\1\0\1\1\0\1\0\0\1\1') * chanks
    for b, p in zip(sieve, range(3, int(sqrt(chanks * 30 + 1)) + 1, 2)):
        if b:
            pp = (p * p - 3) // 2
            sieve[pp::p] = b'\0' * ((lim - pp) // p + 1)
    return [2, 3, 5, *(p for b, p in zip(sieve, range(3, limit, 2)) if b)]


PRIMES = primes(10_000_000)  # TODO kick out magic number
TOP_PRIME = PRIMES[-1]
TOP_PRIME_SQUARE = TOP_PRIME ** 2


def prev_prime(n: int) -> int:
    """
    Returns the nearest prime `p <= n`.
    """
    assert 2 < n <= TOP_PRIME
    i = bisect_left(PRIMES, n)
    return PRIMES[i - (n < PRIMES[i])]


def next_prime(n: int) -> int:
    """
    Returns the nearest prime `p > n`.
    """
    assert 1 < n < TOP_PRIME
    return PRIMES[bisect_right(PRIMES, n)]


def trace(n: int) -> List[int]:  # TODO numba acceleration?
    """
    sample:
    trace(360) -> [2, 3, 5] #  all prime divisors of 360
    """
    res, lo, step = [], 0, 100  # TODO kick out magic number
    for hi in range(step, len(PRIMES), step):
        for p in islice(PRIMES, lo, hi):
            if not n % p:
                res.append(p)
                while not n % p:
                    n //= p
        if n < p * p:
            if n != 1:
                res.append(n)
            return res
        lo = hi
    assert n < TOP_PRIME_SQUARE  # TODO more information when out of range


def factors(n: int) -> List[int]:
    """
    Computes the positive factors of the integer `n`.

    Parameters
    ----------
    n : int
        An integer to be factored.

    Returns
    -------
    list res:
        Sorted list of factors of `n`.
    """
    res = [1]
    for p in trace(n):
        shift = -len(res)
        while not n % p:
            n //= p
            for _ in range(shift, 0):
                res.append(res[shift] * p)
    return sorted(res)


def prime_factors(n: int) -> List[Tuple[int, int]]:
    """
    Computes the prime factors of the positive integer `n`.

    The integer :math:`n` can be factored into :math:`x = p_1^{k_1} p_2^{k_2} ... p_{n-1}^{k_{n-1}}`.

    Parameters
    ----------
    n : int
        The positive integer to be factored (`n > 1`).

    Returns
    -------
    list of pairs (prime: int, power: int)
    """
    assert isinstance(n, int)
    res = []
    for prime in trace(n):
        power = 0
        while not n % prime:
            n //= prime
            power += 1
        res.append((prime, power))
    return res


def is_prime(n: int) -> bool:
    """
    Determines if `n` is prime.

    Parameters
    ----------
    n : int
        A positive integer (`n > 1`).

    Returns
    -------
    bool:
        `True` if the integer `n` is prime. `False` if it isn't.
    """
    assert isinstance(n, int) and 1 < n <= TOP_PRIME_SQUARE
    if n <= TOP_PRIME:
        return PRIMES[bisect_left(PRIMES, n)] == n
    return trace(n)[-1] == n


def euclidean_algorithm(a, b):
    """
    Implements the Euclidean Algorithm to find the greatest common divisor of two integers.

    Parameters
    ----------
    a : int
        Any integer.
    b : int
        Any integer.

    Returns
    -------
    int
        Greatest common divisor of `a` and `b`, i.e. `gcd(a,b)`.

    References
    ----------
    * T. Moon, "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181
    * https://en.wikipedia.org/wiki/Euclidean_algorithm
    """
    assert isinstance(a, (int, np.integer))
    assert isinstance(b, (int, np.integer))
    r = [a, b]

    while True:
        ri = r[-2] % r[-1]
        r.append(ri)
        if ri == 0:
            break

    return r[-2]


@numba.jit("int64[:](int64, int64)", nopython=True)
def extended_euclidean_algorithm(a, b):
    """
    Implements the Extended Euclidean Algorithm to find the integer multiplicands of `a` and `b`,
    such that `a*x + b*y = gcd(a,b)`.

    Parameters
    ----------
    a : int
        Any integer.
    b : int
        Any integer.

    Returns
    -------
    int
        Integer `x`, such that `a*x + b*y = gcd(a, b)`.
    int
        Integer `y`, such that `a*x + b*y = gcd(a, b)`.
    int
        Greatest common divisor of `a` and `b`.

    References
    ----------
    * T. Moon, "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181
    * https://en.wikipedia.org/wiki/Euclidean_algorithm#Extended_Euclidean_algorithm
    """
    r = [a, b]
    s = [1, 0]
    t = [0, 1]

    while True:
        qi = r[-2] // r[-1]
        ri = r[-2] % r[-1]
        r.append(ri)
        s.append(s[-2] - qi * s[-1])
        t.append(t[-2] - qi * t[-1])
        if ri == 0:
            break

    return np.array([s[-2], t[-2], r[-2]])


def chinese_remainder_theorem(a, m):
    """
    Implements the Chinese Remainder Theorem (CRT). The CRT is a method for finding the simultaneous
    solution to a system of congruences.

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
    """
    a = np.array(a, dtype=int)
    m = np.array(m, dtype=int)
    assert m.size == a.size

    # Check that modulii are pairwise relatively coprime
    for pair in combinations(m, 2):
        assert gcd(pair[0], pair[1]) == 1, "{} and {} are not pairwise relatively coprime".format(pair[0], pair[1])

    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1 = a[0]
    m1 = m[0]
    for i in range(1, m.size):
        a2 = a[i]
        m2 = m[i]

        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = 1,
        # where 1 is the GCD(m1, m2) because m1 and m2 are pairwise relatively coprime
        b1, b2 = extended_euclidean_algorithm(m1, m2)[0:2]

        # Compute x through explicit construction
        x = a1 * b2 * m2 + a2 * b1 * m1

        m1 = m1 * m2  # The new modulus
        a1 = x % m1  # The new equivalent remainder

    # Align x to be within [0, prod(m))
    x = x % np.prod(m)

    return x


def euler_totient(n: int) -> int:
    """
    Implements the Euler Totient function to count the positive integers (totatives) in `1 <= k < n` that
    are relatively prime to `n`, i.e. `gcd(n, k) = 1`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The number of totatives that are relatively prime to `n`.
    """
    assert n > 0
    for p in trace(n):
        n = n // p * (p - 1)
    return n


def carmichael(n: int) -> int:
    """
    Implements the Carmichael function to find the smallest positive integer `m` such that `a^m = 1 (mod n)`
    for `1 <= a < n`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The smallest positive integer `m` such that `a^m = 1 (mod n)` for `1 <= a < n`.
    """
    assert n > 0
    res = 1
    for prime, power in prime_factors(n if n & 7 else n // 2):
        res //= gcd(res, prime - 1)
        res *= prime ** (power - 1) * (prime - 1)
    return res


@np.vectorize
def modular_exp(base, exponent, modulus):
    """
    Compute the modular exponentiation `base**exponent % modulus`.

    Parameters
    ----------
    base : array_like
        The base of exponential, an int or an array (follows numpy broadcasting rules).
    exponent : array_like
        The exponent, an int or an array (follows numpy broadcasting rules).
    modulus : array_like
        The modulus of the computation, an int or an array (follows numpy broadcasting rules).

    Returns
    -------
    array_like
        The results of `base**exponent % modulus`.
    """
    if modulus == 1:
        return 0
    result = 1
    # base = base % modulus
    # while exponent > 0:
    #     if exponent % 2 == 0:
    #         result = (result * base) % modulus
    #     exponent //= 2
    #     base = (base * base) % modulus
    for _ in range(0, exponent):
        result = (result * base) % modulus
    return result


def primitive_root(n):
    """
    Finds the first, smallest primitive n-th root of unity that satisfy `x^k = 1 (mod n)`.

    Parameters
    ----------
    n : int
        A positive integer `n > 1`.

    Returns
    -------
    int
        The first, smallest primitive root of unity modulo `n`.
    """
    assert n > 0
    if n == 1:
        return [0]
    if n == 2:
        return [1]

    phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    # lambda_ = carmichael(n)  # The smallest m such that a^m = 1 (mod n) for all a coprime to n
    elements = np.arange(1, n)  # Non-zero integers less than n

    # According to Euler's theorem, a**phi(n) = 1 (mod n) for every a coprime to n
    congruenes = elements[np.where(modular_exp(elements, phi, n) == 1)]
    assert len(congruenes) == phi, "The number of congruences ({} found) is phi(n) = {}".format(len(congruenes), phi)

    root = None
    degrees = np.arange(1, phi + 1)
    for m in congruenes:
        y = modular_exp(m, degrees, n)
        if set(y) == set(congruenes):
            root = m
            break

    return root


def primitive_roots(n):
    """
    Finds all primitive n-th roots of unity that satisfy `x^k = 1 (mod n)`.

    Parameters
    ----------
    n : int
        A positive integer `n > 1`.

    Returns
    -------
    np.ndarray
        An array of integer roots of unity modulo `n`.
    """
    assert n > 0
    if n == 1:
        return [0]
    if n == 2:
        return [1]
    # phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    # p, k = prime_factors(phi)  # Prime factorization of phi(n)
    # print("prime_factors", p, k)

    # roots = []
    # for m in range(1, n):
    #     y = np.array([modular_exp(m, phi // pi, n) for pi in p])
    #     print(m, y)
    #     if np.all(y != 1):
    #         roots.append(m)

    # print(roots)

    # if len(roots) > 0:
    #     N_roots = euler_totient(phi)
    #     assert len(roots) == N_roots, "The number of primitive roots ({} found), if there are any, is phi(phi(n)) = {}".format(len(roots), N_roots)

    # return roots

    phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    # lambda_ = carmichael(n)  # The smallest m such that a^m = 1 (mod n) for all a coprime to n

    elements = np.arange(1, n)  # Non-zero integers less than n

    # According to Euler's theorem, a**phi(n) = 1 (mod n) for every a coprime to n
    congruenes = elements[np.where(modular_exp(elements, phi, n) == 1)]
    assert len(congruenes) == phi, "The number of congruences ({} found) is phi(n) = {}".format(len(congruenes), phi)

    roots = []
    degrees = np.arange(1, phi + 1)
    for m in congruenes:
        y = modular_exp(m, degrees, n)
        if set(y) == set(congruenes):
            roots.append(m)

    if len(roots) > 0:
        assert len(roots) == euler_totient(phi), "The number of primitive roots, if there are any, is phi(phi(n))"

    return roots


def min_poly(a, field, n):
    """
    Finds the minimal polynomial of `a` over the specified Galois field.

    Parameters
    ----------
    a : int
        Field element in the extension field GF(q^n).
    field : galois.gf.GFBase
        The base field GF(q).
    n : int
        The degree of the extension field.

    Returns
    -------
    galois.Poly
        The minimal polynomial of n-th degree with coefficients in `field`, i.e. GF(q), for which
        `x` in GF(q^n) is a root. p(x) over GF(q) and p(a) = 0 in GF(q^n).
    """
    if n == 1:
        poly = Poly([1, -a], field=field)
    else:
        poly = None
        # Loop over all degree-n polynomials
        for poly_dec in range(field.order ** 1, field.order ** (n + 1)):
            # Polynomial over GF(2^m) with coefficients in GF2
            poly = Poly.Decimal(poly_dec, field=field)
            if poly(a) == 0:
                break
    return poly
