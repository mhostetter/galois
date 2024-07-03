"""
A module containing routines for prime number generation, prime factorization, and primality testing.
"""

from __future__ import annotations

import bisect
import functools
import itertools
import math
import random

import numpy as np

from ._databases import PrimeFactorsDatabase
from ._helper import export, verify_isinstance
from ._math import ilog, iroot, isqrt, prod

# Global variables to store the prime lookup table (will generate a larger list after defining the `primes()` function)
PRIMES = [2, 3, 5, 7]
MAX_K = len(PRIMES)  # The max prime index (1-indexed)
MAX_N = 10  # The max value for which all primes <= N are contained in the lookup table


###############################################################################
# Prime generation
###############################################################################


@export
def primes(n: int) -> list[int]:
    r"""
    Returns all primes $p$ for $p \le n$.

    Arguments:
        n: An integer.

    Returns:
        All primes up to and including $n$. If $n < 2$, the function returns an empty list.

    See Also:
        kth_prime, prev_prime, next_prime

    Notes:
        This function implements the Sieve of Eratosthenes to efficiently find the primes.

    References:
        - https://oeis.org/A000040

    Examples:
        .. ipython:: python

            galois.primes(19)
            galois.primes(20)

    Group:
        primes-generation
    """
    global PRIMES, MAX_K, MAX_N
    verify_isinstance(n, int)

    if n <= MAX_N:
        # Return a subset of the pre-computed global primes list
        return PRIMES[0 : bisect.bisect_right(PRIMES, n)]

    N_odd = int(math.ceil(n / 2)) - 1  # Number of odd integers (including n) to check starting at 3, i.e. skip 1
    composite = np.zeros(N_odd, dtype=bool)  # Indices correspond to integers 3,5,7,9,...

    # We only need to test integers for compositeness up to sqrt(n) because at that point the
    # integers above sqrt(n) have already been marked by the multiples of earlier primes
    max_composite = isqrt(n)  # Python 3.8 has math.isqrt(). Use this until the supported Python versions are bumped.
    max_composite_idx = (max_composite - 3) // 2

    for i in range(0, max_composite_idx + 1):
        if not composite[i]:
            prime = i * 2 + 3  # Convert index back to integer value

            # We want to mark `2*prime, 3*prime, 4*prime, ...` as composite. We don't need to mark the
            # even multiples because they're not in the composite array (which only has odds). So we'll
            # mark `3*prime, 5*prime, ...`

            delta = prime  # A delta of `2*prime` converted to an index of the odd composite array, i.e. `2*prime//2`
            first_multiple = i + delta  # First odd multiple of the prime, i.e. `3*prime`

            # Mark multiples of the prime that are odd (and in the composite array) as composite
            composite[first_multiple::delta] = True

    prime_idxs = np.where(composite == False)[0]  # noqa: E712
    p = (prime_idxs * 2 + 3).tolist()  # Convert indices back to odd integers
    p.insert(0, 2)  # Add the only even prime, 2

    # Replace the global primes lookup table with the newly created, larger list
    PRIMES = p
    MAX_K = len(PRIMES)
    MAX_N = n

    return p


# TODO: Don't build a large lookup table at import time. Instead, use the progressively growing nature of PRIMES.
# Generate a prime lookup table for efficient lookup in other algorithms
PRIMES = primes(10_000_000)
MAX_K = len(PRIMES)
MAX_N = 10_000_000


@export
def kth_prime(k: int) -> int:
    r"""
    Returns the $k$-th prime, where $k = \{1,2,3,4,\dots\}$ for primes $p = \{2,3,5,7,\dots\}$.

    Arguments:
        k: The prime index (1-indexed).

    Returns:
        The $k$-th prime.

    See Also:
        primes, prev_prime, next_prime

    Examples:
        .. ipython:: python

            galois.kth_prime(1)
            galois.kth_prime(2)
            galois.kth_prime(3)
            galois.kth_prime(1000)

    Group:
        primes-generation
    """
    verify_isinstance(k, int)
    if not 1 <= k <= MAX_K:
        raise ValueError(
            f"Argument 'k' is out of range of the prime lookup table. "
            f"The lookup table only contains the first {MAX_K} primes (up to {MAX_N})."
        )

    return PRIMES[k - 1]


@export
def prev_prime(n: int) -> int:
    r"""
    Returns the nearest prime $p$, such that $p \le n$.

    Arguments:
        n: An integer $n \ge 2$.

    Returns:
        The nearest prime $p \le n$.

    See Also:
        primes, kth_prime, next_prime

    Examples:
        .. ipython:: python

            galois.prev_prime(13)
            galois.prev_prime(15)
            galois.prev_prime(6298891201241929548477199440981228280038)

    Group:
        primes-generation
    """
    verify_isinstance(n, int)
    if not n >= 2:
        raise ValueError("There are no primes less than 2.")

    # Directly use lookup table
    if n <= MAX_N:
        return PRIMES[bisect.bisect_right(PRIMES, n) - 1]

    shifts = [29, 23, 19, 17, 13, 11, 7, 1]  # Factorization wheel for basis {2, 3, 5}
    base = n // 30 * 30  # Wheel factorization starting point

    while True:
        for shift in shifts:
            i = base + shift  # May be bigger than n
            if i > n:
                continue
            if is_prime(i):
                return i
        base -= 30


@export
def next_prime(n: int) -> int:
    r"""
    Returns the nearest prime $p$, such that $p > n$.

    Arguments:
        n: An integer.

    Returns:
        The nearest prime $p > n$.

    See Also:
        primes, kth_prime, prev_prime

    Examples:
        .. ipython:: python

            galois.next_prime(13)
            galois.next_prime(15)
            galois.next_prime(6852976918500265458318414454675831645298)

    Group:
        primes-generation
    """
    verify_isinstance(n, int)

    # Directly use lookup table
    if n < PRIMES[-1]:
        return PRIMES[bisect.bisect_right(PRIMES, n)]

    shifts = [1, 7, 11, 13, 17, 19, 23, 29]  # Factorization wheel for basis {2, 3, 5}
    base = n // 30 * 30  # Wheel factorization starting point. May be less than n.

    while True:
        for shift in shifts:
            i = base + shift
            if i <= n:
                continue
            if is_prime(i):
                return i
        base += 30


@export
def random_prime(bits: int, seed: int | None = None) -> int:
    r"""
    Returns a random prime $p$ with $b$ bits, such that $2^b \le p < 2^{b+1}$.

    Arguments:
        bits: The number of bits in the prime $p$.
        seed: Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed.

    Returns:
        A random prime in $2^b \le p < 2^{b+1}$.

    See Also:
        prev_prime, next_prime

    Notes:
        This function randomly generates integers with $b$ bits and uses the primality tests in
        :func:`~galois.is_prime` to determine if $p$ is prime.

    References:
        - https://en.wikipedia.org/wiki/Prime_number_theorem

    Examples:
        Generate a random 1024-bit prime.

        .. ipython:: python

            p = galois.random_prime(1024, seed=1); p
            galois.is_prime(p)

        Verify that $p$ is prime using the OpenSSL library.

        .. code-block:: console

            $ openssl prime 327845897586213436751081882871255331286648902836386839087617368608439574698192016043769533823474001379935585889197488144338014865193967937011638431094821943416361149113909692569658970713864593781874423564706915495970135894084612689487074397782022398597547611189482697523681694691585678818112329605903872356773
            1D2DE38DE88C67E1EAFDEEAE77C40B8709ED9C275522C6D5578976B1ABCBE7E0F8C6DE1271EEC6EB3827649164189788F9F3A622AEA5F4039761EC708B5841DE88566D9B5BAF49BA92DCE5A300297A9E0E890E4103ED2AD4B5E0553CE56E8C34758CD45900125DBA1553AE73AA0CBD6018A2A8713D46E475BF058D1AAA52EF1A5 (327845897586213436751081882871255331286648902836386839087617368608439574698192016043769533823474001379935585889197488144338014865193967937011638431094821943416361149113909692569658970713864593781874423564706915495970135894084612689487074397782022398597547611189482697523681694691585678818112329605903872356773) is prime

    Group:
        primes-generation
    """
    verify_isinstance(bits, int)
    verify_isinstance(seed, int, optional=True)
    if not bits > 0:
        raise ValueError(f"Argument 'bits' must be positive, not {bits}.")

    random.seed(seed)
    while True:
        p = random.randint(2**bits, 2 ** (bits + 1) - 1)
        if is_prime(p):
            break

    return p


# https://www.mersenne.org/primes/
MERSENNE_EXPONENTS = [
    2,
    3,
    5,
    7,
    13,
    17,
    19,
    31,
    61,
    89,
    107,
    127,
    521,
    607,
    1279,
    2203,
    2281,
    3217,
    4253,
    4423,
    9689,
    9941,
    11213,
    19937,
    21701,
    23209,
    44497,
    86243,
    110503,
    132049,
    216091,
    756839,
    859433,
    1257787,
    1398269,
    2976221,
    3021377,
    6972593,
    13466917,
    20996011,
    24036583,
    25964951,
    30402457,
    32582657,
    37156667,
    42643801,
    43112609,
    57885161,  # Found Jan 25, 2013 by Curtis Cooper
    74207281,  # Found Jan 07, 2016 by Curtis Cooper
    77232917,  # Found Dec 26, 2017 by Jon Pace
    82589933,  # Found Dec 07, 2018 by Patrick Laroche
]


@export
def mersenne_exponents(n: int | None = None) -> list[int]:
    r"""
    Returns all known Mersenne exponents $e$ for $e \le n$.

    Arguments:
        n: The max exponent of 2. The default is `None` which returns all known Mersenne exponents.

    Returns:
        The list of Mersenne exponents $e$ for $e \le n$.

    See Also:
        mersenne_primes

    Notes:
        A Mersenne exponent $e$ is an exponent of 2 such that $2^e - 1$ is prime.

    References:
        - https://oeis.org/A000043

    Examples:
        .. ipython:: python

            # List all Mersenne exponents for Mersenne primes up to 2000 bits
            e = galois.mersenne_exponents(2000); e

            # Select one Merseene exponent and compute its Mersenne prime
            p = 2**e[-1] - 1; p
            galois.is_prime(p)

    Group:
        primes-generation
    """
    if n is None:
        return MERSENNE_EXPONENTS

    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be positive, not {n}.")

    return MERSENNE_EXPONENTS[0 : bisect.bisect_right(MERSENNE_EXPONENTS, n)]


@export
def mersenne_primes(n: int | None = None) -> list[int]:
    r"""
    Returns all known Mersenne primes $p$ for $p \le 2^n - 1$.

    Arguments:
        n: The max power of 2. The default is `None` which returns all known Mersenne exponents.

    Returns:
        The list of known Mersenne primes $p$ for $p \le 2^n - 1$.

    See Also:
        mersenne_exponents

    Notes:
        Mersenne primes are primes that are one less than a power of 2.

    References:
        - https://oeis.org/A000668

    Examples:
        .. ipython:: python

            # List all Mersenne primes up to 2000 bits
            p = galois.mersenne_primes(2000); p
            galois.is_prime(p[-1])

    Group:
        primes-generation
    """
    return [2**e - 1 for e in mersenne_exponents(n)]


###############################################################################
# Primality tests
###############################################################################


@export
def fermat_primality_test(n: int, a: int | None = None, rounds: int = 1) -> bool:
    r"""
    Determines if $n$ is composite using Fermat's primality test.

    Arguments:
        n: An odd integer $n \ge 3$.
        a: An integer in $2 \le a \le n - 2$. The default is `None` which selects a random $a$.
        rounds: The number of iterations attempting to detect $n$ as composite. Additional rounds will choose
            a new $a$. The default is 1.

    Returns:
        `False` if $n$ is shown to be composite. `True` if $n$ is a probable prime.

    See Also:
        is_prime, miller_rabin_primality_test

    Notes:
        Fermat's theorem says that for prime $p$ and $1 \le a \le p-1$, the congruence
        $a^{p-1} \equiv 1\ (\textrm{mod}\ p)$ holds. Fermat's primality test of $n$ computes
        $a^{n-1}\ \textrm{mod}\ n$ for some $1 \le a \le n-1$. If $a$ is such that
        $a^{p-1} \not\equiv 1\ (\textrm{mod}\ p)$, then $a$ is said to be a *Fermat witness* to the
        compositeness of $n$. If $n$ is composite and $a^{p-1} \equiv 1\ (\textrm{mod}\ p)$, then
        $a$ is said to be a *Fermat liar* to the primality of $n$.

        Since $a = \{1, n-1\}$ are Fermat liars for all composite $n$, it is common to reduce the range of
        possible $a$ to $2 \le a \le n - 2$.

    References:
        - Section 4.2.1 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf

    Examples:
        Fermat's primality test will never mark a true prime as composite.

        .. ipython:: python

            primes = [257, 24841, 65497]
            [galois.is_prime(p) for p in primes]
            [galois.fermat_primality_test(p) for p in primes]

        However, Fermat's primality test may mark a composite as probable prime. Here are pseudoprimes base 2 from
        `A001567 <https://oeis.org/A001567>`_.

        .. ipython:: python

            # List of some Fermat pseudoprimes to base 2
            pseudoprimes = [2047, 29341, 65281]
            [galois.is_prime(p) for p in pseudoprimes]

            # The pseudoprimes base 2 satisfy 2^(p-1) = 1 (mod p)
            [galois.fermat_primality_test(p, a=2) for p in pseudoprimes]

            # But they may not satisfy a^(p-1) = 1 (mod p) for other a
            [galois.fermat_primality_test(p) for p in pseudoprimes]

        And the pseudoprimes base 3 from `A005935 <https://oeis.org/A005935>`_.

        .. ipython:: python

            # List of some Fermat pseudoprimes to base 3
            pseudoprimes = [2465, 7381, 16531]
            [galois.is_prime(p) for p in pseudoprimes]

            # The pseudoprimes base 3 satisfy 3^(p-1) = 1 (mod p)
            [galois.fermat_primality_test(p, a=3) for p in pseudoprimes]

            # But they may not satisfy a^(p-1) = 1 (mod p) for other a
            [galois.fermat_primality_test(p) for p in pseudoprimes]

    Group:
        primes-specific-tests
    """
    verify_isinstance(n, int)
    verify_isinstance(a, int, optional=True)
    verify_isinstance(rounds, int)

    a = random.randint(2, n - 2) if a is None else a
    if not (n > 2 and n % 2 == 1):
        raise ValueError(f"Argument 'n' must be odd and greater than 2, not {n}.")
    if not 2 <= a <= n - 2:
        raise ValueError(f"Argument 'a' must satisfy 2 <= a <= {n - 2}, not {a}.")
    if not rounds >= 1:
        raise ValueError(f"Argument 'rounds' must be at least 1, not {rounds}.")

    for _ in range(rounds):
        if pow(a, n - 1, n) != 1:
            return False  # n is definitely composite
        a = random.randint(2, n - 2)

    return True  # n is a probable prime


@export
def miller_rabin_primality_test(n: int, a: int = 2, rounds: int = 1) -> bool:
    r"""
    Determines if $n$ is composite using the Miller-Rabin primality test.

    Arguments:
        n: An odd integer $n \ge 3$.
        a: An integer in $2 \le a \le n - 2$. The default is 2.
        rounds: The number of iterations attempting to detect $n$ as composite. Additional rounds will choose
            consecutive primes for $a$. The default is 1.

    Returns:
        `False` if $n$ is shown to be composite. `True` if $n$ is probable prime.

    See Also:
        is_prime, fermat_primality_test

    Notes:
        The Miller-Rabin primality test is based on the fact that for odd $n$ with factorization
        $n = 2^s r$ for odd $r$ and integer $a$ such that $\textrm{gcd}(a, n) = 1$, then either
        $a^r \equiv 1\ (\textrm{mod}\ n)$ or $a^{2^j r} \equiv -1\ (\textrm{mod}\ n)$ for some $j$ in
        $0 \le j \le s - 1$.

        In the Miller-Rabin primality test, if $a^r \not\equiv 1\ (\textrm{mod}\ n)$ and
        $a^{2^j r} \not\equiv -1\ (\textrm{mod}\ n)$ for all $j$ in $0 \le j \le s - 1$, then
        $a$ is called a *strong witness* to the compositeness of $n$. If not, namely
        $a^r \equiv 1\ (\textrm{mod}\ n)$ or $a^{2^j r} \equiv -1\ (\textrm{mod}\ n)$ for any $j$
        in $0 \le j \le s - 1$, then $a$ is called a *strong liar* to the primality of $n$ and
        $n$ is called a *strong pseudoprime to the base a*.

        Since $a = \{1, n-1\}$ are strong liars for all composite $n$, it is common to reduce the range
        of possible $a$ to $2 \le a \le n - 2$.

        For composite odd $n$, the probability that the Miller-Rabin test declares it a probable prime is less
        than $(\frac{1}{4})^t$, where $t$ is the number of rounds, and is often much lower.

    References:
        - Section 4.2.3 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf
        - https://math.dartmouth.edu/~carlp/PDF/paper25.pdf

    Examples:
        The Miller-Rabin primality test will never mark a true prime as composite.

        .. ipython:: python

            primes = [257, 24841, 65497]
            [galois.is_prime(p) for p in primes]
            [galois.miller_rabin_primality_test(p) for p in primes]

        However, a composite $n$ may have strong liars. 91 has
        $\{9,10,12,16,17,22,29,38,53,62,69,74,75,79,81,82\}$ as strong liars.

        .. ipython:: python

            strong_liars = [9,10,12,16,17,22,29,38,53,62,69,74,75,79,81,82]
            witnesses = [a for a in range(2, 90) if a not in strong_liars]

            # All strong liars falsely assert that 91 is prime
            [galois.miller_rabin_primality_test(91, a=a) for a in strong_liars] == [True,]*len(strong_liars)

            # All other a are witnesses to the compositeness of 91
            [galois.miller_rabin_primality_test(91, a=a) for a in witnesses] == [False,]*len(witnesses)

    Group:
        primes-specific-tests
    """
    verify_isinstance(n, int)
    verify_isinstance(a, int)
    verify_isinstance(rounds, int)

    # To avoid this test `2 <= a <= n - 2` which doesn't apply for n=3
    if n == 3:
        return True

    if not (n > 2 and n % 2 == 1):
        raise ValueError(f"Argument 'n' must be odd and greater than 2, not {n}.")
    if not 2 <= a <= n - 2:
        raise ValueError(f"Argument 'a' must satisfy 2 <= a <= {n - 2}, not {a}.")
    if not rounds >= 1:
        raise ValueError(f"Argument 'rounds' must be at least 1, not {rounds}.")

    # Write (n - 1) = 2^s * r, for odd r
    r, s = n - 1, 0
    while r % 2 == 0:
        r, s = r // 2, s + 1
    assert 2**s * r == n - 1

    for t in range(rounds):
        y = pow(a, r, n)
        if y not in [1, n - 1]:
            j = 0
            while j < s - 1 and y != n - 1:
                y = pow(y, 2, n)
                if y == 1:
                    return False  # n is definitely composite
                j += 1

            if y != n - 1:
                return False  # a is a strong witness to the compositness of n

        a = PRIMES[t]

    return True  # n is a probable prime


###############################################################################
# Legendre, Jacobi, and Kronecker symbols
###############################################################################


@export
def legendre_symbol(a: int, p: int) -> int:
    r"""
    Computes the Legendre symbol $(\frac{a}{p})$.

    Arguments:
        a: An integer.
        p: An odd prime $p \ge 3$.

    Returns:
        The Legendre symbol $(\frac{a}{p})$ with value in $\{0, 1, -1\}$.

    See Also:
        jacobi_symbol, kronecker_symbol

    Notes:
        The Legendre symbol is useful for determining if $a$ is a quadratic residue modulo $p$, namely
        $a \in Q_p$. A quadratic residue $a$ modulo $p$ satisfies
        $x^2 \equiv a\ (\textrm{mod}\ p)$ for some $x$.

        .. math::
            \bigg(\frac{a}{p}\bigg) =
                \begin{cases}
                    0, & p \mid a

                    1, & a \in Q_p

                    -1, & a \in \overline{Q}_p
                \end{cases}

    References:
        - Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples:
        The quadratic residues modulo 7 are $Q_7 = \{1, 2, 4\}$. The quadratic non-residues
        modulo 7 are $\overline{Q}_7 = \{3, 5, 6\}$.

        .. ipython:: python

            [pow(x, 2, 7) for x in range(7)]
            for a in range(7):
                print(f"({a} / 7) = {galois.legendre_symbol(a, 7)}")

    Group:
        number-theory-congruences
    """
    verify_isinstance(a, int)
    verify_isinstance(p, int)
    if not (is_prime(p) and p > 2):
        raise ValueError(f"Argument 'p' must be an odd prime greater than 2, not {p}.")

    return jacobi_symbol(a, p)


@export
def jacobi_symbol(a: int, n: int) -> int:
    r"""
    Computes the Jacobi symbol $(\frac{a}{n})$.

    Arguments:
        a: An integer.
        n: An odd integer $n \ge 3$.

    Returns:
        The Jacobi symbol $(\frac{a}{n})$ with value in $\{0, 1, -1\}$.

    See Also:
        legendre_symbol, kronecker_symbol

    Notes:
        The Jacobi symbol extends the Legendre symbol for odd $n \ge 3$. Unlike the Legendre symbol,
        $(\frac{a}{n}) = 1$ does not imply $a$ is a quadratic residue modulo $n$. However, all
        $a \in Q_n$ have $(\frac{a}{n}) = 1$.

    References:
        - Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Examples:
        The quadratic residues modulo 9 are $Q_9 = \{1, 4, 7\}$ and these all satisfy $(\frac{a}{9}) = 1$.
        The quadratic non-residues modulo 9 are $\overline{Q}_9 = \{2, 3, 5, 6, 8\}$, but notice
        $\{2, 5, 8\}$ also satisfy $(\frac{a}{9}) = 1$. The set of integers $\{3, 6\}$ not coprime
        to 9 satisfies $(\frac{a}{9}) = 0$.

        .. ipython:: python

            [pow(x, 2, 9) for x in range(9)]
            for a in range(9):
                print(f"({a} / 9) = {galois.jacobi_symbol(a, 9)}")

    Group:
        number-theory-congruences
    """
    verify_isinstance(a, int)
    verify_isinstance(n, int)
    if not (n > 2 and n % 2 == 1):
        raise ValueError(f"Argument 'n' must be an odd integer greater than 2, not {n}.")

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
    if a1 != 1:
        s *= jacobi_symbol(n1, a1)

    return s


@export
def kronecker_symbol(a: int, n: int) -> int:
    r"""
    Computes the Kronecker symbol $(\frac{a}{n})$. The Kronecker symbol extends the Jacobi symbol for
    all $n$.

    Arguments:
        a: An integer.
        n: An integer.

    Returns:
        The Kronecker symbol $(\frac{a}{n})$ with value in $\{0, -1, 1\}$.

    See Also:
        legendre_symbol, jacobi_symbol

    References:
        - Algorithm 2.149 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf

    Group:
        number-theory-congruences
    """
    verify_isinstance(a, int)
    verify_isinstance(n, int)

    if n == 0:
        return 1 if a in [1, -1] else 0
    if n == 1:
        return 1
    if n == -1:
        return -1 if a < 0 else 1
    if n == 2:
        if a % 2 == 0:
            return 0
        if a % 8 in [1, 7]:
            return 1
        return -1

    # Factor out the unit +/- 1
    u = -1 if n < 0 else 1
    n //= u

    # Factor out the powers of 2 so the resulting n is odd
    e = 0
    while n % 2 == 0:
        n, e = n // 2, e + 1

    s = kronecker_symbol(a, u) * kronecker_symbol(a, 2) ** e
    if n >= 3:
        # Handle the remaining odd n using the Jacobi symbol
        s *= jacobi_symbol(a, n)

    return s


###############################################################################
# Prime factorization
###############################################################################


@functools.lru_cache(maxsize=2048)
def factors(n: int) -> tuple[list[int], list[int]]:
    """
    This function is wrapped and documented in `_polymorphic.factors()`.
    """
    verify_isinstance(n, int)
    if not n > 1:
        raise ValueError(f"Argument 'n' must be greater than 1, not {n}.")

    # Step 0: Check if n is in the prime factors database.
    try:
        p, e, n = PrimeFactorsDatabase().fetch(n)
        if n == 1:
            return p, e
        # Else, there still may be a residual composite
        # Although, we're probably not powerful enough to factor it...
    except LookupError:
        p, e = [], []

    # Step 1: Test is n is prime.
    if is_prime(n):
        p.append(n)
        e.append(1)
        return p, e

    # Step 2: Test if n is a perfect power. The base may be composite.
    base, exponent = perfect_power(n)
    if base != n:
        pp, ee = factors(base)
        ee = [eei * exponent for eei in ee]
        p.extend(pp)
        e.extend(ee)
        return p, e

    # Step 3: Perform trial division up to medium-sized primes.
    pp, ee, n = trial_division(n, 10_000_000)
    p.extend(pp)
    e.extend(ee)

    # Step 4: Use Pollard's rho algorithm to find a non-trivial factor.
    while n > 1 and not is_prime(n):
        c = 1
        while True:
            try:
                f = pollard_rho(n, c=c)  # A non-trivial factor
                break  # Found a factor
            except RuntimeError:
                # Could not find one -- keep searching
                c = random.randint(2, n // 2)

        if is_prime(f):
            degree = 0
            while n % f == 0:
                degree += 1
                n //= f
            p.append(f)
            e.append(degree)
        else:
            raise RuntimeError(
                f"Encountered a very large composite {f}. "
                f"Please report this in a GitHub issue at https://github.com/mhostetter/galois/issues."
            )

    if n > 1:
        p.append(n)
        e.append(1)

    return p, e


@export
def perfect_power(n: int) -> tuple[int, int]:
    r"""
    Returns the integer base $c$ and exponent $e$ of $n = c^e$. If $n$ is *not* a
    perfect power, then $c = n$ and $e = 1$.

    Arguments:
        n: An integer.

    Returns:
        - The *potentially* composite base $c$.
        - The exponent $e$.

    See Also:
        factors, is_perfect_power, is_prime_power

    Examples:
        Primes are not perfect powers because their exponent is 1.

        .. ipython:: python

            n = 13
            galois.perfect_power(n)
            galois.is_perfect_power(n)

        Products of primes are not perfect powers.

        .. ipython:: python

            n = 5 * 7
            galois.perfect_power(n)
            galois.is_perfect_power(n)

        Products of prime powers where the GCD of the exponents is 1 are not perfect powers.

        .. ipython:: python

            n = 2 * 3 * 5**3
            galois.perfect_power(n)
            galois.is_perfect_power(n)

        Products of prime powers where the GCD of the exponents is greater than 1 are perfect powers.

        .. ipython:: python

            n = 2**2 * 3**2 * 5**4
            galois.perfect_power(n)
            galois.is_perfect_power(n)

        Negative integers can be perfect powers if they can be factored with an odd exponent.

        .. ipython:: python

            n = -64
            galois.perfect_power(n)
            galois.is_perfect_power(n)

        Negative integers that are only factored with an even exponent are not perfect powers.

        .. ipython:: python

            n = -100
            galois.perfect_power(n)
            galois.is_perfect_power(n)

    Group:
        factorization-specific
    """
    return _perfect_power(n)


@functools.lru_cache(maxsize=512)
def _perfect_power(n: int) -> tuple[int, int]:
    verify_isinstance(n, int)

    if n == 0:
        return 0, 1

    abs_n = abs(n)

    # Test if n is a prime power of small primes. This saves a very costly calculation below for n like 2^571.
    # See https://github.com/mhostetter/galois/issues/443. There may be a bug remaining in perfect_power() still
    # left to resolve.
    for prime in primes(1_000):
        exponent = ilog(abs_n, prime)
        if prime**exponent == abs_n:
            return _adjust_base_and_exponent(n, prime, exponent)

    for k in primes(ilog(abs_n, 2)):
        x = iroot(abs_n, k)
        if x**k == abs_n:
            # Recursively determine if x is a perfect power
            c, e = perfect_power(x)
            e *= k  # Multiply the exponent of c by the exponent of x
            return _adjust_base_and_exponent(n, c, e)

    # n is composite and cannot be factored into a perfect power
    return n, 1


def _adjust_base_and_exponent(n: int, base: int, exponent: int) -> tuple[int, int]:
    """
    Adjusts the base and exponent of a perfect power to account for negative integers.
    """
    if n < 0:
        # Try to convert the even exponent of a factored negative number into the next largest odd power
        while exponent > 2:
            if exponent % 2 == 0:
                base *= 2
                exponent //= 2
            else:
                return -base, exponent

        # Failed to convert the even exponent to and odd one, therefore there is no real factorization of
        # this negative integer
        return n, 1

    return base, exponent


@export
def trial_division(n: int, B: int | None = None) -> tuple[list[int], list[int], int]:
    r"""
    Finds all the prime factors $p_i^{e_i}$ of $n$ for $p_i \le B$.

    The trial division factorization will find all prime factors $p_i \le B$ such that $n$ factors
    as $n = p_1^{e_1} \dots p_k^{e_k} n_r$ where $n_r$ is a residual factor (which may be composite).

    Arguments:
        n: A positive integer.
        B: The max divisor in the trial division. The default is `None` which corresponds to $B = \sqrt{n}$.
            If $B > \sqrt{n}$, the algorithm will only search up to $\sqrt{n}$, since a prime factor of
            $n$ cannot be larger than $\sqrt{n}$.

    Returns:
        - The discovered prime factors $\{p_1, \dots, p_k\}$.
        - The corresponding prime exponents $\{e_1, \dots, e_k\}$.
        - The residual factor $n_r$.

    See Also:
        factors

    Examples:
        .. ipython:: python

            n = 2**4 * 17**3 * 113 * 15013
            galois.trial_division(n)
            galois.trial_division(n, B=500)
            galois.trial_division(n, B=100)

    Group:
        factorization-specific
    """
    verify_isinstance(n, int)
    verify_isinstance(B, int, optional=True)

    B = isqrt(n) if B is None else B
    if not n > 1:
        raise ValueError(f"Argument 'n' must be greater than 1, not {n}.")
    if not B > 2:
        raise ValueError(f"Argument 'B' must be greater than 2, not {B}.")
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


@export
def pollard_p1(n: int, B: int, B2: int | None = None) -> int:
    r"""
    Attempts to find a non-trivial factor of $n$ if it has a prime factor $p$ such that
    $p-1$ is $B$-smooth.

    Arguments:
        n: An odd composite integer $n > 2$ that is not a prime power.
        B: The smoothness bound $B > 2$.
        B2: The smoothness bound $B_2$ for the optional second step of the algorithm. The default is `None` which
            will not perform the second step.

    Returns:
        A non-trivial factor of $n$.

    Raises:
        RuntimeError: If a non-trivial factor cannot be found.

    See Also:
        factors, pollard_rho

    Notes:
        For a given odd composite $n$ with a prime factor $p$, Pollard's $p-1$ algorithm can discover
        a non-trivial factor of $n$ if $p-1$ is $B$-smooth. Specifically, the prime factorization
        must satisfy $p-1 = p_1^{e_1} \dots p_k^{e_k}$ with each $p_i \le B$.

        A extension of Pollard's $p-1$ algorithm allows a prime factor $p$ to be $B$-smooth with the
        exception of one prime factor $B < p_{k+1} \le B_2$. In this case, the prime factorization is
        $p-1 = p_1^{e_1} \dots p_k^{e_k} p_{k+1}$. Often $B_2$ is chosen such that $B_2 \gg B$.

    References:
        - Section 3.2.3 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples:
        Here, $n = pq$ where $p-1$ is 1039-smooth and $q-1$ is 17-smooth.

        .. ipython:: python

            p, q = 1458757, 1326001
            galois.factors(p - 1)
            galois.factors(q - 1)

        Searching with $B=15$ will not recover a prime factor.

        .. ipython:: python
            :okexcept:

            galois.pollard_p1(p*q, 15)

        Searching with $B=17$ will recover the prime factor $q$.

        .. ipython:: python

            galois.pollard_p1(p*q, 17)

        Searching $B=15$ will not recover a prime factor in the first step, but will find $q$ in the second
        step because $p_{k+1} = 17$ satisfies $15 < 17 \le 100$.

        .. ipython:: python

            galois.pollard_p1(p*q, 15, B2=100)

        Pollard's $p-1$ algorithm may return a composite factor.

        .. ipython:: python

            n = 2133861346249
            galois.factors(n)
            galois.pollard_p1(n, 10)
            37*41

    Group:
        factorization-specific
    """
    verify_isinstance(n, int)
    verify_isinstance(B, int)
    verify_isinstance(B2, int, optional=True)

    if not (n % 2 == 1 and n > 2):
        raise ValueError(f"Argument 'n' must be odd and greater than 2, not {n}.")
    if not B > 2:
        raise ValueError(f"Argument 'B' must be greater than 2, not {B}.")

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
        raise RuntimeError(
            f"A non-trivial factor of {n} could not be found using the Pollard p-1 algorithm "
            f"with smoothness bound {B} and secondary bound {B2}."
        )

    # Try to find p such that p - 1 has a single prime factor larger than B
    if B2 is not None:
        P = primes(B2)
        P = P[bisect.bisect_right(P, B) : bisect.bisect_right(P, B2)]  # Only select primes between B < prime <= B2
        for i, p in enumerate(P):
            a = pow(a, p, n)

            # Check the GCD periodically to return early without checking all primes less than the
            # smoothness bound
            if i % check_stride == 0 and math.gcd(a - 1, n) not in [1, n]:
                return math.gcd(a - 1, n)

        d = math.gcd(a - 1, n)
        if d not in [1, n]:
            return d

    raise RuntimeError(
        f"A non-trivial factor of {n} could not be found using the Pollard p-1 algorithm "
        f"with smoothness bound {B} and secondary bound {B2}."
    )


@export
# @functools.lru_cache(maxsize=1024)
def pollard_rho(n: int, c: int = 1) -> int:
    r"""
    Attempts to find a non-trivial factor of $n$ using cycle detection.

    Arguments:
        n: An odd composite integer $n > 2$ that is not a prime power.
        c: The constant offset in the function $f(x) = x^2 + c\ \textrm{mod}\ n$. The default is 1. A requirement
            of the algorithm is that $c \not\in \{0, -2\}$.

    Returns:
        A non-trivial factor $m$ of $n$.

    Raises:
        RuntimeError: If a non-trivial factor cannot be found.

    See Also:
        factors, pollard_p1

    Notes:
        Pollard's $\rho$ algorithm seeks to find a non-trivial factor of $n$ by finding a cycle in a
        sequence of integers $x_0, x_1, \dots$ defined by
        $x_i = f(x_{i-1}) = x_{i-1}^2 + 1\ \textrm{mod}\ p$ where $p$ is an unknown small prime factor
        of $n$. This happens when $x_{m} \equiv x_{2m}\ (\textrm{mod}\ p)$. Because $p$ is unknown,
        this is accomplished by computing the sequence modulo $n$ and looking for
        $\textrm{gcd}(x_m - x_{2m}, n) > 1$.

    References:
        - Section 3.2.2 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples:
        Pollard's $\rho$ is especially good at finding small factors.

        .. ipython:: python

            n = 503**7 * 10007 * 1000003
            galois.pollard_rho(n)

        It is also efficient for finding relatively small factors.

        .. ipython:: python

            n = 1182640843 * 1716279751
            galois.pollard_rho(n)

    Group:
        factorization-specific
    """
    verify_isinstance(n, int)
    verify_isinstance(c, int, optional=True)

    if not n > 1:
        raise ValueError(f"Argument 'n' must be greater than 1, not {n}.")
    if not c not in [0, -2]:
        raise ValueError("Argument 'c' cannot be -2 or 0.")
    n = abs(n)

    def f(x):
        return (x**2 + c) % n

    a, b, d = 2, 2, 1
    while d == 1:
        a = f(a)
        b = f(f(b))
        d = math.gcd(a - b, n)

    if d == n:
        raise RuntimeError(
            f"A non-trivial factor of {n} could not be found using the Pollard Rho algorithm with f(x) = x^2 + {c}."
        )

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
# Composite factorization
###############################################################################


@export
def divisors(n: int) -> list[int]:
    r"""
    Computes all positive integer divisors $d$ of the integer $n$ such that $d \mid n$.

    Arguments:
        n: An integer.

    Returns:
        Sorted list of positive integer divisors $d$ of $n$.

    See Also:
        factors, divisor_sigma

    Notes:
        The :func:`~galois.divisors` function finds *all* positive integer divisors or factors of $n$, where the
        :func:`~galois.factors` function only finds the prime factors of $n$.

    Examples:
        .. ipython:: python

            galois.divisors(0)
            galois.divisors(1)
            galois.divisors(24)
            galois.divisors(-24)
            galois.factors(24)

    Group:
        factorization-composite
    """
    verify_isinstance(n, int)
    n = abs(n)

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


@export
def divisor_sigma(n: int, k: int = 1) -> int:
    r"""
    Returns the sum of $k$-th powers of the positive divisors of $n$.

    Arguments:
        n: An integer.
        k: The degree of the positive divisors. The default is 1 which corresponds to $\sigma_1(n)$ which is the
            sum of positive divisors.

    Returns:
        The sum of divisors function $\sigma_k(n)$.

    See Also:
        factors, divisors

    Notes:
        This function implements the $\sigma_k(n)$ function. It is defined as:

        $$\sigma_k(n) = \sum_{d\ \mid\ n} d^k$$

    Examples:
        .. ipython:: python

            galois.divisors(9)
            galois.divisor_sigma(9, k=0)
            galois.divisor_sigma(9, k=1)
            galois.divisor_sigma(9, k=2)

    Group:
        factorization-composite
    """
    verify_isinstance(n, int)

    d = divisors(n)

    if n == 0:
        return len(d)

    return sum(di**k for di in d)


###############################################################################
# Integer tests
###############################################################################


@export
def is_prime(n: int) -> bool:
    r"""
    Determines if $n$ is prime.

    Arguments:
        n: An integer.

    Returns:
        `True` if the integer $n$ is prime.

    See Also:
        is_composite, is_prime_power, is_perfect_power

    Notes:
        This algorithm will first run Fermat's primality test to check $n$ for compositeness, see
        :func:`~galois.fermat_primality_test`. If it determines $n$ is composite, the function will quickly
        return.

        If Fermat's primality test returns `True`, then $n$ could be prime or pseudoprime. If so, then the
        algorithm will run 10 rounds of Miller-Rabin's primality test, see :func:`~galois.miller_rabin_primality_test`.
        With this many rounds, a result of `True` should have high probability of $n$ being a true prime,
        not a pseudoprime.

    Examples:
        .. ipython:: python

            galois.is_prime(13)
            galois.is_prime(15)

        The algorithm is also efficient on very large $n$.

        .. ipython:: python

            galois.is_prime(1000000000000000035000061)

    Group:
        primes-tests
    """
    verify_isinstance(n, int)

    if n < 2:
        return False

    # Test n against the first few primes. If n is a multiple of them, it cannot be prime. This is very fast
    # and can quickly rule out many composites.
    for p in PRIMES[0:250]:
        if n == p:
            return True
        if n % p == 0:
            return False

    if not fermat_primality_test(n):
        return False  # n is definitely composite

    if not miller_rabin_primality_test(n, rounds=10):
        return False  # n is definitely composite

    return True  # n is a probable prime with high confidence


@export
def is_composite(n: int) -> bool:
    r"""
    Determines if $n$ is composite.

    Arguments:
        n: An integer.

    Returns:
        `True` if the integer $n$ is composite.

    See Also:
        is_prime, is_square_free, is_perfect_power

    Examples:
        .. ipython:: python

            galois.is_composite(13)
            galois.is_composite(15)

    Group:
        primes-tests
    """
    verify_isinstance(n, int)

    if n < 2:
        return False

    return not is_prime(n)


@export
def is_prime_power(n: int) -> bool:
    r"""
    Determines if $n$ is a prime power $n = p^k$ for prime $p$ and $k \ge 1$.

    Arguments:
        n: An integer.

    Returns:
        `True` if the integer $n$ is a prime power.

    See Also:
        is_perfect_power, is_prime

    Notes:
        There is some controversy over whether 1 is a prime power $p^0$. Since 1 is the 0-th power
        of all primes, it is often regarded not as a prime power. This function returns `False` for 1.

    Examples:
        .. ipython:: python

            galois.is_prime_power(8)
            galois.is_prime_power(6)
            galois.is_prime_power(1)

    Group:
        primes-tests
    """
    verify_isinstance(n, int)

    if n < 2:
        return False

    if is_prime(n):
        return True

    # Determine is n is a perfect power and then check is the base is prime or composite
    c, _ = perfect_power(n)

    return is_prime(c)


@export
def is_perfect_power(n: int) -> bool:
    r"""
    Determines if $n$ is a perfect power $n = c^e$ with $e > 1$.

    Arguments:
        n: An integer.

    Returns:
        `True` if the integer $n$ is a perfect power.

    See Also:
        is_prime_power, is_square_free

    Examples:
        Primes are not perfect powers because their exponent is 1.

        .. ipython:: python

            galois.perfect_power(13)
            galois.is_perfect_power(13)

        Products of primes are not perfect powers.

        .. ipython:: python

            galois.perfect_power(5*7)
            galois.is_perfect_power(5*7)

        Products of prime powers where the GCD of the exponents is 1 are not perfect powers.

        .. ipython:: python

            galois.perfect_power(2 * 3 * 5**3)
            galois.is_perfect_power(2 * 3 * 5**3)

        Products of prime powers where the GCD of the exponents is greater than 1 are perfect powers.

        .. ipython:: python

            galois.perfect_power(2**2 * 3**2 * 5**4)
            galois.is_perfect_power(2**2 * 3**2 * 5**4)

        Negative integers can be perfect powers if they can be factored with an odd exponent.

        .. ipython:: python

            galois.perfect_power(-64)
            galois.is_perfect_power(-64)

        Negative integers that are only factored with an even exponent are not perfect powers.

        .. ipython:: python

            galois.perfect_power(-100)
            galois.is_perfect_power(-100)

    Group:
        primes-tests
    """
    verify_isinstance(n, int)

    # Special cases: -1 = -1^3, 0 = 0^2, 1 = 1^3
    if n in [-1, 0, 1]:
        return True

    c, _ = perfect_power(n)

    return n != c


def is_square_free(n: int) -> bool:
    """
    This function is wrapped and documented in `_polymorphic.is_square_free()`.
    """
    # Since -1 can factored out of the prime factorization is_square_free(-n) == is_square_free(n)
    n = abs(n)

    if n == 0:
        return False
    if n == 1:
        return True

    # For n to be square-free it must have no prime factors with exponent greater than 1
    _, e = factors(n)

    return e == [1] * len(e)


@export
def is_smooth(n: int, B: int) -> bool:
    r"""
    Determines if the integer $n$ is $B$-smooth.

    Arguments:
        n: An integer.
        B: The smoothness bound $B \ge 2$.

    Returns:
        `True` if $n$ is $B$-smooth.

    See Also:
        factors, is_powersmooth

    Notes:
        An integer $n$ with prime factorization $n = p_1^{e_1} \dots p_k^{e_k}$ is $B$-smooth
        if $p_k \le B$. The 2-smooth numbers are the powers of 2. The 5-smooth numbers
        are known as *regular numbers*. The 7-smooth numbers are known as *humble numbers* or *highly composite
        numbers*.

    Examples:
        .. ipython:: python

            galois.is_smooth(2**10, 2)
            galois.is_smooth(10, 5)
            galois.is_smooth(12, 5)
            galois.is_smooth(60**2, 5)

    Group:
        primes-tests
    """
    verify_isinstance(n, int)
    verify_isinstance(B, int)
    if not B >= 2:
        raise ValueError(f"Argument 'B' must be at least 2, not {B}.")

    # Since -1 can factored out of the prime factorization is_square_free(-n) == is_square_free(n)
    n = abs(n)

    if n == 0:
        return False
    if n == 1:
        return True

    for p in primes(B):
        e = ilog(n, p)
        d = math.gcd(p**e, n)
        if d > 1:
            n //= d
        if n == 1:
            # n can be fully factored by the primes already tested, therefore it is B-smooth
            return True

    # n has residual prime factors larger than B, therefore it is not B-smooth
    return False


@export
def is_powersmooth(n: int, B: int) -> bool:
    r"""
    Determines if the integer $n$ is $B$-powersmooth.

    Arguments:
        n: An integer.
        B: The smoothness bound $B \ge 2$.

    Returns:
        `True` if $n$ is $B$-powersmooth.

    See Also:
        factors, is_smooth

    Notes:
        An integer $n$ with prime factorization $n = p_1^{e_1} \dots p_k^{e_k}$ is $B$-powersmooth
        if $p_i^{e_i} \le B$ for $1 \le i \le k$.

    Examples:
        Comparison of $B$-smooth and $B$-powersmooth. Necessarily, any $n$ that is
        $B$-powersmooth must be $B$-smooth.

        .. ipython:: python

            galois.is_smooth(2**4 * 3**2 * 5, 5)
            galois.is_powersmooth(2**4 * 3**2 * 5, 5)

    Group:
        primes-tests
    """
    verify_isinstance(n, int)
    verify_isinstance(B, int)
    if not B >= 2:
        raise ValueError(f"Argument 'B' must be at least 2, not {B}.")

    # Since -1 can factored out of the prime factorization is_square_free(-n) == is_square_free(n)
    n = abs(n)

    if n == 0:
        return False
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
