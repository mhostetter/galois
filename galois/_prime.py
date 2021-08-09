import bisect
import math
import random

import numpy as np

from ._integer import isqrt
from ._overrides import set_module

__all__ = [
    "primes", "kth_prime", "prev_prime", "next_prime", "random_prime", "mersenne_exponents", "mersenne_primes",
    "is_prime", "is_composite", "fermat_primality_test", "miller_rabin_primality_test"
]

# Global variables to store the prime lookup table (will generate a larger list after defining the `primes()` function)
PRIMES = [2, 3, 5, 7]
MAX_K = len(PRIMES)  # The max prime index (1-indexed)
MAX_N = 10  # The max value for which all primes <= N are contained in the lookup table


###############################################################################
# Prime generation
###############################################################################

@set_module("galois")
def primes(n):
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


# Generate a prime lookup table for efficient lookup in other algorithms
PRIMES = primes(10_000_000)
MAX_K = len(PRIMES)
MAX_N = 10_000_000


@set_module("galois")
def kth_prime(k):
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
def prev_prime(n):
    r"""
    Returns the nearest prime :math:`p`, such that :math:`p \le n`.

    Parameters
    ----------
    n : int
        An integer.

    Returns
    -------
    int
        The nearest prime :math:`p \le n`. If :math:`n < 2`, the function returns `None`.

    Examples
    --------
    .. ipython:: python

        galois.prev_prime(13)
        galois.prev_prime(15)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n <= MAX_N:
        raise ValueError(f"Argument `n` is out of range of the prime lookup table. The lookup table only stores primes <= {MAX_N}.")

    return PRIMES[bisect.bisect_right(PRIMES, n) - 1] if n >= 2 else None


@set_module("galois")
def next_prime(n):
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
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n < PRIMES[-1]:
        raise ValueError(f"Argument `n` is out of range of the prime lookup table. The lookup table only stores primes <= {MAX_N}.")

    return PRIMES[bisect.bisect_right(PRIMES, n)]


@set_module("galois")
def random_prime(bits):
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
def mersenne_exponents(n=None):
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
def mersenne_primes(n=None):
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
def is_prime(n):
    r"""
    Determines if :math:`n` is prime.

    Parameters
    ----------
    n : int
        A positive integer.

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
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
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
def is_composite(n):
    r"""
    Determines if :math:`n` is composite.

    Parameters
    ----------
    n : int
        A positive integer.

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
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return False

    return not is_prime(n)


@set_module("galois")
def fermat_primality_test(n, a=None, rounds=1):
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
def miller_rabin_primality_test(n, a=2, rounds=1):
    r"""
    Determines if :math:`n` is composite using the Miller-Rabin primality test.

    Parameters
    ----------
    n : int
        An odd integer :math:`n \ge 3`.
    a : int, optional
        An integer in :math:`2 \le a \le n - 2`. The default is `2`.
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
    if not isinstance(a, (type(None), int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(rounds, (int, np.integer)):
        raise TypeError(f"Argument `rounds` must be an integer, not {type(rounds)}.")

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
