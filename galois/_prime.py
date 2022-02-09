"""
A module containing routines for prime number generation, prime factorization, and primality testing.
"""
import bisect
import functools
import itertools
import math
import random
from typing import Tuple, List, Optional

import numpy as np

from ._math import prod, isqrt, iroot, ilog
from ._overrides import set_module

__all__ = [
    "primes", "kth_prime", "prev_prime", "next_prime", "random_prime", "mersenne_exponents", "mersenne_primes",
    "fermat_primality_test", "miller_rabin_primality_test",
    "legendre_symbol", "jacobi_symbol", "kronecker_symbol",
    "perfect_power", "trial_division", "pollard_p1", "pollard_rho",
    "divisors", "divisor_sigma",
    "is_prime", "is_composite", "is_prime_power", "is_perfect_power", "is_smooth", "is_powersmooth",
]

# Global variables to store the prime lookup table (will generate a larger list after defining the `primes()` function)
PRIMES = [2, 3, 5, 7]
MAX_K = len(PRIMES)  # The max prime index (1-indexed)
MAX_N = 10  # The max value for which all primes <= N are contained in the lookup table


###############################################################################
# Prime generation
###############################################################################

@set_module("galois")
def primes(n: int) -> List[int]:
    r"""
    Returns all primes :math:`p` for :math:`p \le n`.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    list
        All primes up to and including :math:`n`. If :math:`n < 2`, the function returns an empty list.

    Notes
    -----
    This function implements the Sieve of Eratosthenes to efficiently find the primes.

    References
    ----------
    * https://oeis.org/A000040

    Examples
    --------
    .. ipython:: python

        galois.primes(19)
        galois.primes(20)
    """
    global PRIMES, MAX_K, MAX_N
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    if n <= MAX_N:
        # Return a subset of the pre-computed global primes list
        return PRIMES[0:bisect.bisect_right(PRIMES, n)]

    N_odd = int(math.ceil(n/2)) - 1  # Number of odd integers (including n) to check starting at 3, i.e. skip 1
    composite = np.zeros(N_odd, dtype=bool)  # Indices correspond to integers 3,5,7,9,...

    # We only need to test integers for compositeness up to sqrt(n) because at that point the
    # integers above sqrt(n) have already been marked by the multiples of earlier primes
    max_composite = isqrt(n)  # Python 3.8 has math.isqrt(). Use this until the supported python versions are bumped.
    max_composite_idx = (max_composite - 3)//2

    for i in range(0, max_composite_idx + 1):
        if not composite[i]:
            prime = i*2 + 3  # Convert index back to integer value

            # We want to mark `2*prime, 3*prime, 4*prime, ...` as composite. We don't need to mark the
            # even multiples because they're not in the composite array (which only has odds). So we'll
            # mark `3*prime, 5*prime, ...`

            delta = prime  # A delta of `2*prime` converted to an index of the odd composite array, i.e. `2*prime//2`
            first_multiple = i + delta  # First odd multiple of the prime, i.e. `3*prime`

            # Mark multiples of the prime that are odd (and in the composite array) as composite
            composite[first_multiple::delta] = True

    prime_idxs = np.where(composite == False)[0]  # pylint: disable=singleton-comparison
    p = (prime_idxs*2 + 3).tolist()  # Convert indices back to odd integers
    p.insert(0, 2)  # Add the only even prime, 2

    # Replace the global primes lookup table with the newly-created, larger list
    PRIMES = p
    MAX_K = len(PRIMES)
    MAX_N = n

    return p


# TODO: Don't build a large lookup table at import time. Instead, use the progressively-growing nature of PRIMES.
# Generate a prime lookup table for efficient lookup in other algorithms
PRIMES = primes(10_000_000)
MAX_K = len(PRIMES)
MAX_N = 10_000_000


@set_module("galois")
def kth_prime(k: int) -> int:
    r"""
    Returns the :math:`k`-th prime.

    Parameters
    ----------
    k : int
        The prime index (1-indexed), where :math:`k = \{1,2,3,4,\dots\}` for primes :math:`p = \{2,3,5,7,\dots\}`.

    Returns
    -------
    int
        The :math:`k`-th prime.

    Examples
    --------
    .. ipython:: python

        galois.kth_prime(1)
        galois.kth_prime(3)
        galois.kth_prime(1000)
    """
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
    if not 1 <= k <= MAX_K:
        raise ValueError(f"Argument `k` is out of range of the prime lookup table. The lookup table only contains the first {MAX_K} primes (up to {MAX_N}).")

    return PRIMES[k - 1]


@set_module("galois")
def prev_prime(n: int) -> int:
    r"""
    Returns the nearest prime :math:`p`, such that :math:`p \le n`.

    Parameters
    ----------
    n : int
        An integer :math:`n \ge 2`.

    Returns
    -------
    int
        The nearest prime :math:`p \le n`.

    Examples
    --------
    .. ipython:: python

        galois.prev_prime(13)
        galois.prev_prime(15)
        galois.prev_prime(6298891201241929548477199440981228280038)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n >= 2:
        raise ValueError("There are no primes less than 2.")

    # Directly use lookup table
    if n <= MAX_N:
        return PRIMES[bisect.bisect_right(PRIMES, n) - 1]

    # TODO: Make this faster using wheel factorization
    n = n - 1 if n % 2 == 0 else n  # The next possible prime (which is odd)
    while True:
        n -= 2  # Only check odds
        if is_prime(n):
            break

    return n


@set_module("galois")
def next_prime(n: int) -> int:
    r"""
    Returns the nearest prime :math:`p`, such that :math:`p > n`.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    int
        The nearest prime :math:`p > n`.

    Examples
    --------
    .. ipython:: python

        galois.next_prime(13)
        galois.next_prime(15)
        galois.next_prime(6852976918500265458318414454675831645298)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    # Directly use lookup table
    if n < PRIMES[-1]:
        return PRIMES[bisect.bisect_right(PRIMES, n)]

    # TODO: Make this faster using wheel factorization
    n = n + 1 if n % 2 == 0 else n + 2  # The next possible prime (which is odd)
    while True:
        n += 2  # Only check odds
        if is_prime(n):
            break

    return n


@set_module("galois")
def random_prime(bits: int) -> int:
    r"""
    Returns a random prime :math:`p` with :math:`b` bits, such that :math:`2^b \le p < 2^{b+1}`.

    This function randomly generates integers with :math:`b` bits and uses the primality tests in
    :func:`galois.is_prime` to determine if :math:`p` is prime.

    Parameters
    ----------
    bits : int
        The number of bits in the prime :math:`p`.

    Returns
    -------
    int
        A random prime in :math:`2^b \le p < 2^{b+1}`.

    References
    ----------
    * https://en.wikipedia.org/wiki/Prime_number_theorem

    Examples
    --------
    Generate a random 1024-bit prime.

    .. ipython::

        In [2]: p = galois.random_prime(1024); p
        Out[2]: 236861787926957382206996886087214592029752524078026392358936844479667423570833116126506927878773110287700754280996224768092589904231910149528080012692722763539766058401127758399272786475279348968866620857161889678512852050561604969208679095086283103827661300743342847921567132587459205365243815835763830067933

        In [3]: galois.is_prime(p)
        Out[3]: True

    .. code-block::

        $ openssl prime 236861787926957382206996886087214592029752524078026392358936844479667423570833116126506927878773110287700754280996224768092589904231910149528080012692722763539766058401127758399272786475279348968866620857161889678512852050561604969208679095086283103827661300743342847921567132587459205365243815835763830067933
        1514D68EDB7C650F1FF713531A1A43255A4BE6D66EE1FDBD96F4EB32757C1B1BAF16A5933E24D45FAD6C6A814F3C8C14F3CB98F24FEA74C43C349D6FA3AB76EB0156811A1FBAA64EB4AC525CCEF9278AF78886DC6DBF46C4463A34C0E53B0FA2F784BB2DC5FDF076BB6E145AA15AA6D616ACC1D5F95B8BE757670B9AAF53292DD (236861787926957382206996886087214592029752524078026392358936844479667423570833116126506927878773110287700754280996224768092589904231910149528080012692722763539766058401127758399272786475279348968866620857161889678512852050561604969208679095086283103827661300743342847921567132587459205365243815835763830067933) is prime
    """
    if not isinstance(bits, (int, np.integer)):
        raise TypeError(f"Argument `bits` must be an integer, not {type(bits)}.")
    if not bits > 0:
        raise ValueError(f"Argument `bits` must be positive, not {bits}.")

    while True:
        p = random.randint(2**bits, 2**(bits + 1) - 1)
        if is_prime(p):
            break

    return p


MERSENNE_EXPONENTS = [2,3,5,7,13,17,19,31,61,89,107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213,19937,21701,23209,44497,86243,110503,132049,216091,756839,859433,1257787,1398269,2976221,3021377,6972593,13466917,20996011,24036583,25964951,30402457,32582657,37156667,42643801,43112609]


@set_module("galois")
def mersenne_exponents(n: Optional[int] = None) -> List[int]:
    r"""
    Returns all known Mersenne exponents :math:`e` for :math:`e \le n`.

    A Mersenne exponent :math:`e` is an exponent of :math:`2` such that :math:`2^e - 1` is prime.

    Parameters
    ----------
    n : int, optional
        The max exponent of 2. The default is `None` which returns all known Mersenne exponents.

    Returns
    -------
    list
        The list of Mersenne exponents :math:`e` for :math:`e \le n`.

    References
    ----------
    * https://oeis.org/A000043

    Examples
    --------
    .. ipython:: python

        # List all Mersenne exponents for Mersenne primes up to 2000 bits
        e = galois.mersenne_exponents(2000); e

        # Select one Merseene exponent and compute its Mersenne prime
        p = 2**e[-1] - 1; p
        galois.is_prime(p)
    """
    if n is None:
        return MERSENNE_EXPONENTS
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be positive, not {n}.")

    return MERSENNE_EXPONENTS[0:bisect.bisect_right(MERSENNE_EXPONENTS, n)]


@set_module("galois")
def mersenne_primes(n: Optional[int] = None) -> List[int]:
    r"""
    Returns all known Mersenne primes :math:`p` for :math:`p \le 2^n - 1`.

    Mersenne primes are primes that are one less than a power of 2.

    Parameters
    ----------
    n : int, optional
        The max power of 2. The default is `None` which returns all known Mersenne exponents.

    Returns
    -------
    list
        The list of known Mersenne primes :math:`p` for :math:`p \le 2^n - 1`.

    References
    ----------
    * https://oeis.org/A000668

    Examples
    --------
    .. ipython:: python

        # List all Mersenne primes up to 2000 bits
        p = galois.mersenne_primes(2000); p
        galois.is_prime(p[-1])
    """
    return [2**e - 1 for e in mersenne_exponents(n)]


###############################################################################
# Primality tests
###############################################################################

@set_module("galois")
def fermat_primality_test(n: int, a: Optional[int] = None, rounds: int = 1) -> bool:
    r"""
    Determines if :math:`n` is composite using Fermat's primality test.

    Parameters
    ----------
    n : int
        An odd integer :math:`n \ge 3`.
    a : int, optional
        An integer in :math:`2 \le a \le n - 2`. The default is `None` which selects a random :math:`a`.
    rounds : int, optional
        The number of iterations attempting to detect :math:`n` as composite. Additional rounds will choose
        a new :math:`a`. The default is 1.

    Returns
    -------
    bool
        `False` if :math:`n` is shown to be composite. `True` if :math:`n` is probable prime.

    Notes
    -----
    Fermat's theorem says that for prime :math:`p` and :math:`1 \le a \le p-1`, the congruence :math:`a^{p-1} \equiv 1\ (\textrm{mod}\ p)`
    holds. Fermat's primality test of :math:`n` computes :math:`a^{n-1}\ \textrm{mod}\ n` for some :math:`1 \le a \le n-1`.
    If :math:`a` is such that :math:`a^{p-1} \not\equiv 1\ (\textrm{mod}\ p)`, then :math:`a` is said to be a *Fermat witness* to the
    compositeness of :math:`n`. If :math:`n` is composite and :math:`a^{p-1} \equiv 1\ (\textrm{mod}\ p)`, then :math:`a` is said to be
    a *Fermat liar* to the primality of :math:`n`.

    Since :math:`a = \{1, n-1\}` are Fermat liars for all composite :math:`n`, it is common to reduce the range of possible :math:`a`
    to :math:`2 \le a \le n - 2`.

    References
    ----------
    * Section 4.2.1 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf

    Examples
    --------
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
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(a, (type(None), int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(rounds, (int, np.integer)):
        raise TypeError(f"Argument `rounds` must be an integer, not {type(rounds)}.")

    n = int(n)
    a = random.randint(2, n - 2) if a is None else a
    if not (n > 2 and n % 2 == 1):
        raise ValueError(f"Argument `n` must be odd and greater than 2, not {n}.")
    if not 2 <= a <= n - 2:
        raise ValueError(f"Argument `a` must satisfy 2 <= a <= {n - 2}, not {a}.")
    if not rounds >= 1:
        raise ValueError(f"Argument `rounds` must be at least 1, not {rounds}.")

    for _ in range(rounds):
        if pow(a, n - 1, n) != 1:
            return False  # n is definitely composite
        a = random.randint(2, n - 2)

    return True  # n is a probable prime


@set_module("galois")
def miller_rabin_primality_test(n: int, a: int = 2, rounds: int = 1) -> bool:
    r"""
    Determines if :math:`n` is composite using the Miller-Rabin primality test.

    Parameters
    ----------
    n : int
        An odd integer :math:`n \ge 3`.
    a : int, optional
        An integer in :math:`2 \le a \le n - 2`. The default is 2.
    rounds : int, optional
        The number of iterations attempting to detect :math:`n` as composite. Additional rounds will choose
        consecutive primes for :math:`a`.

    Returns
    -------
    bool
        `False` if :math:`n` is shown to be composite. `True` if :math:`n` is probable prime.

    Notes
    -----
    The Miller-Rabin primality test is based on the fact that for odd :math:`n` with factorization :math:`n = 2^s r` for odd :math:`r`
    and integer :math:`a` such that :math:`\textrm{gcd}(a, n) = 1`, then either :math:`a^r \equiv 1\ (\textrm{mod}\ n)`
    or :math:`a^{2^j r} \equiv -1\ (\textrm{mod}\ n)` for some :math:`j` in :math:`0 \le j \le s - 1`.

    In the Miller-Rabin primality test, if :math:`a^r \not\equiv 1\ (\textrm{mod}\ n)` and :math:`a^{2^j r} \not\equiv -1\ (\textrm{mod}\ n)`
    for all :math:`j` in :math:`0 \le j \le s - 1`, then :math:`a` is called a *strong witness* to the compositeness of :math:`n`. If not, namely
    :math:`a^r \equiv 1\ (\textrm{mod}\ n)` or :math:`a^{2^j r} \equiv -1\ (\textrm{mod}\ n)` for any :math:`j` in :math:`0 \le j \le s - 1`,
    then :math:`a` is called a *strong liar* to the primality of :math:`n` and :math:`n` is called a *strong pseudoprime to the base a*.

    Since :math:`a = \{1, n-1\}` are strong liars for all composite :math:`n`, it is common to reduce the range of possible :math:`a`
    to :math:`2 \le a \le n - 2`.

    For composite odd :math:`n`, the probability that the Miller-Rabin test declares it a probable prime is less than :math:`(\frac{1}{4})^t`,
    where :math:`t` is the number of rounds, and is often much lower.

    References
    ----------
    * Section 4.2.3 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf
    * https://math.dartmouth.edu/~carlp/PDF/paper25.pdf

    Examples
    --------
    The Miller-Rabin primality test will never mark a true prime as composite.

    .. ipython:: python

        primes = [257, 24841, 65497]
        [galois.is_prime(p) for p in primes]
        [galois.miller_rabin_primality_test(p) for p in primes]

    However, a composite :math:`n` may have strong liars. :math:`91` has :math:`\{9,10,12,16,17,22,29,38,53,62,69,74,75,79,81,82\}`
    as strong liars.

    .. ipython:: python

        strong_liars = [9,10,12,16,17,22,29,38,53,62,69,74,75,79,81,82]
        witnesses = [a for a in range(2, 90) if a not in strong_liars]

        # All strong liars falsely assert that 91 is prime
        [galois.miller_rabin_primality_test(91, a=a) for a in strong_liars] == [True,]*len(strong_liars)

        # All other a are witnesses to the compositeness of 91
        [galois.miller_rabin_primality_test(91, a=a) for a in witnesses] == [False,]*len(witnesses)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(rounds, (int, np.integer)):
        raise TypeError(f"Argument `rounds` must be an integer, not {type(rounds)}.")

    # To avoid this test `2 <= a <= n - 2` which doesn't apply for n=3
    if n == 3:
        return True

    n = int(n)
    if not (n > 2 and n % 2 == 1):
        raise ValueError(f"Argument `n` must be odd and greater than 2, not {n}.")
    if not 2 <= a <= n - 2:
        raise ValueError(f"Argument `a` must satisfy 2 <= a <= {n - 2}, not {a}.")
    if not rounds >= 1:
        raise ValueError(f"Argument `rounds` must be at least 1, not {rounds}.")

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

@set_module("galois")
def legendre_symbol(a: int, p: int) -> int:
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
def jacobi_symbol(a: int, n: int) -> int:
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
def kronecker_symbol(a: int, n: int) -> int:
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
def factors(n: int) -> Tuple[List[int], List[int]]:
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
    base, exponent = perfect_power(n)
    if base != n:
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
def perfect_power(n: int) -> Tuple[int, int]:
    r"""
    Returns the integer base :math:`c` and exponent :math:`e` of :math:`n = c^e`.

    If :math:`n` is a *not* perfect power, then :math:`c = n` and :math:`e = 1`.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    int
        The *potentially* composite base :math:`c`.
    int
        The exponent :math:`e`.

    Examples
    --------
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
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    if n == 0:
        return 0, 1

    n = int(n)
    abs_n = abs(n)

    for k in primes(ilog(abs_n, 2)):
        x = iroot(abs_n, k)
        if x**k == abs_n:
            # Recursively determine if x is a perfect power
            c, e = perfect_power(x)
            e *= k  # Multiply the exponent of c by the exponent of x

            if n < 0:
                # Try to convert the even exponent of a factored negative number into the next largest odd power
                while e > 2:
                    if e % 2 == 0:
                        c *= 2
                        e //= 2
                    else:
                        return -c, e

                # Failed to convert the even exponent to and odd one, therefore there is no real factorization of this negative integer
                return n, 1

            return c, e

    # n is composite and cannot be factored into a perfect power
    return n, 1


@set_module("galois")
def trial_division(n: int, B: Optional[int] = None) -> Tuple[List[int], List[int], int]:
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
def pollard_p1(n: int, B: int, B2: Optional[int] = None) -> Optional[int]:
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
def pollard_rho(n: int, c: int = 1) -> Optional[int]:
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
def divisors(n: int) -> List[int]:
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
def divisor_sigma(n: int, k: int = 1) -> int:
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
# Integer tests
###############################################################################

@set_module("galois")
def is_prime(n: int) -> bool:
    r"""
    Determines if :math:`n` is prime.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is prime.

    Notes
    -----
    This algorithm will first run Fermat's primality test to check :math:`n` for compositeness, see
    :func:`galois.fermat_primality_test`. If it determines :math:`n` is composite, the function will quickly return.
    If Fermat's primality test returns `True`, then :math:`n` could be prime or pseudoprime. If so, then the algorithm
    will run 10 rounds of Miller-Rabin's primality test, see :func:`galois.miller_rabin_primality_test`. With this many rounds,
    a result of `True` should have high probability of :math:`n` being a true prime, not a pseudoprime.

    Examples
    --------
    .. ipython:: python

        galois.is_prime(13)
        galois.is_prime(15)

    The algorithm is also efficient on very large :math:`n`.

    .. ipython:: python

        galois.is_prime(1000000000000000035000061)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    if n < 2:
        return False

    # Test n against the first few primes. If n is a multiple of them, it cannot be prime. This is very fast
    # and can quickly rule out many composites.
    for p in PRIMES[0:250]:
        if n == p:
            return True
        elif n % p == 0:
            return False

    if not fermat_primality_test(n):
        return False  # n is definitely composite

    if not miller_rabin_primality_test(n, rounds=10):
        return False  # n is definitely composite

    return True  # n is a probable prime with high confidence


@set_module("galois")
def is_composite(n: int) -> bool:
    r"""
    Determines if :math:`n` is composite.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is composite.

    Examples
    --------
    .. ipython:: python

        galois.is_composite(13)
        galois.is_composite(15)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

    if n < 2:
        return False

    return not is_prime(n)


@set_module("galois")
def is_prime_power(n: int) -> bool:
    r"""
    Determines if :math:`n` is a prime power :math:`n = p^k` for prime :math:`p` and :math:`k \ge 1`.

    Parameters
    ----------
    n : int
        An integer.

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

    if n < 2:
        return False

    if is_prime(n):
        return True

    # Determine is n is a perfect power and then check is the base is prime or composite
    c, _ = perfect_power(n)

    return is_prime(c)


@set_module("galois")
def is_perfect_power(n: int) -> bool:
    r"""
    Determines if :math:`n` is a perfect power :math:`n = c^e` with :math:`e > 1`.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    bool
        `True` if the integer :math:`n` is a perfect power.

    Examples
    --------
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
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")

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

    return e == [1,]*len(e)


@set_module("galois")
def is_smooth(n: int, B: int) -> bool:
    r"""
    Determines if the integer :math:`n` is :math:`B`-smooth.

    Parameters
    ----------
    n : int
        An integer.
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
    if not B >= 2:
        raise ValueError(f"Argument `B` must be at least 2, not {B}.")

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


@set_module("galois")
def is_powersmooth(n: int, B: int) -> bool:
    r"""
    Determines if the integer :math:`n` is :math:`B`-powersmooth.

    Parameters
    ----------
    n : int
        An integer.
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
    if not B >= 2:
        raise ValueError(f"Argument `B` must be at least 2, not {B}.")

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
