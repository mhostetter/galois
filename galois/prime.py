import bisect
import math
import random

import numpy as np

from .math_ import isqrt
from .overrides import set_module

__all__ = [
    "primes", "kth_prime", "prev_prime", "next_prime", "random_prime", "mersenne_exponents", "mersenne_primes",
    "is_prime", "is_prime_fermat", "is_prime_miller_rabin"
]


@set_module("galois")
def primes(n):
    """
    Returns all primes :math:`p` for :math:`p \\le n`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    list
        The primes up to and including :math:`n`.

    References
    ----------
    * https://oeis.org/A000040

    Examples
    --------
    .. ipython:: python

        galois.primes(19)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n >= 2:
        raise ValueError(f"Argument `n` must be at least 2, not {n}.")

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

    return p


# Generate a prime lookup table for efficient lookup in other algorithms
PRIMES = primes(10_000_000)
MAX_K = len(PRIMES)
MAX_PRIME = PRIMES[-1]


@set_module("galois")
def kth_prime(k):
    """
    Returns the :math:`k`-th prime.

    Parameters
    ----------
    k : int
        The prime index, where :math:`k = \\{1,2,3,4,\\dots\\}` for primes :math:`p = \\{2,3,5,7,\\dots\\}`.

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
        raise ValueError(f"Argument `k` is out of range of the prime lookup table. The lookup table only stores the first {MAX_K} primes.")
    return PRIMES[k - 1]


@set_module("galois")
def prev_prime(n):
    """
    Returns the nearest prime :math:`p`, such that :math:`p \\le n`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The nearest prime :math:`p \\le n`.

    Examples
    --------
    .. ipython:: python

        galois.prev_prime(13)
        galois.prev_prime(15)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not 2 <= n <= MAX_PRIME:
        raise ValueError(f"Argument `n` is out of range of the prime lookup table. The lookup table only stores primes <= {MAX_PRIME}.")
    return PRIMES[bisect.bisect_right(PRIMES, n) - 1]


@set_module("galois")
def next_prime(n):
    """
    Returns the nearest prime :math:`p`, such that :math:`p > n`.

    Parameters
    ----------
    n : int
        A positive integer.

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
    if not n < MAX_PRIME:
        raise ValueError(f"Argument `n` is out of range of the prime lookup table. The lookup table only stores primes <= {MAX_PRIME}.")
    return PRIMES[bisect.bisect_right(PRIMES, n)]


@set_module("galois")
def random_prime(bits):
    """
    Returns a random prime :math:`p` with :math:`b` bits, such that :math:`2^b \\le p < 2^{b+1}`.

    This function randomly generates integers with :math:`b` bits and uses the primality tests in
    :func:`galois.is_prime` to determine if :math:`p` is prime.

    Parameters
    ----------
    bits : int
        The number of bits in the prime :math:`p`.

    Returns
    -------
    int
        A random prime in :math:`2^b \\le p < 2^{b+1}`.

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
    while True:
        p = random.randint(2**bits, 2**(bits + 1) - 1)
        if is_prime(p):
            return p


MERSENNE_EXPONENTS = [2,3,5,7,13,17,19,31,61,89,107,127,521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213,19937,21701,23209,44497,86243,110503,132049,216091,756839,859433,1257787,1398269,2976221,3021377,6972593,13466917,20996011,24036583,25964951,30402457,32582657,37156667,42643801,43112609]


@set_module("galois")
def mersenne_exponents(n=None):
    """
    Returns all known Mersenne exponents :math:`e` for :math:`e \\le n`.

    A Mersenne exponent :math:`e` is an exponent of :math:`2` such that :math:`2^e - 1` is prime.

    Parameters
    ----------
    n : int, optional
        The max exponent of 2. The default is `None` which returns all known Mersenne exponents.

    Returns
    -------
    list
        The list of Mersenne exponents :math:`e` for :math:`e \\le n`.

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
    else:
        return MERSENNE_EXPONENTS[0:bisect.bisect_right(MERSENNE_EXPONENTS, n)]


@set_module("galois")
def mersenne_primes(n=None):
    """
    Returns all known Mersenne primes :math:`p` for :math:`p \\le 2^n - 1`.

    Mersenne primes are primes that are one less than a power of 2.

    Parameters
    ----------
    n : int, optional
        The max power of 2. The default is `None` which returns all known Mersenne exponents.

    Returns
    -------
    list
        The list of known Mersenne primes :math:`p` for :math:`p \\le 2^n - 1`.

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


@set_module("galois")
def is_prime(n):
    """
    Determines if :math:`n` is prime.

    This algorithm will first run Fermat's primality test to check :math:`n` for compositeness, see
    :obj:`galois.is_prime_fermat`. If it determines :math:`n` is composite, the function will quickly return.
    If Fermat's primality test returns `True`, then :math:`n` could be prime or pseudoprime. If so, then the algorithm
    will run seven rounds of Miller-Rabin's primality test, see :obj:`galois.is_prime_miller_rabin`. With this many rounds,
    a result of `True` should have high probability of :math:`n` being a true prime, not a pseudoprime.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool:
        `True` if the integer :math:`n` is prime.

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
    if n == 2:
        return True

    if not is_prime_fermat(n):
        # If the test returns False, then n is definitely composite
        return False

    if not is_prime_miller_rabin(n, rounds=7):
        # If the test returns False, then n is definitely composite
        return False

    return True


@set_module("galois")
def is_prime_fermat(n):
    """
    Determines if :math:`n` is composite.

    This function implements Fermat's primality test. The test says that for an integer :math:`n`, select an integer
    :math:`a` coprime with :math:`n`. If :math:`a^{n-1} \\equiv 1 (\\textrm{mod}\\ n)`, then :math:`n` is prime or pseudoprime.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool
        `False` if :math:`n` is known to be composite. `True` if :math:`n` is prime or pseudoprime.

    References
    ----------
    * https://oeis.org/A001262
    * https://oeis.org/A001567

    Examples
    --------
    .. ipython:: python

        # List of some primes
        primes = [257, 24841, 65497]

        for prime in primes:
            is_prime = galois.is_prime_fermat(prime)
            p, k = galois.prime_factors(prime)
            print("Prime = {:5d}, Fermat's Prime Test = {}, Prime factors = {}".format(prime, is_prime, list(p)))

        # List of some strong pseudoprimes with base 2
        pseudoprimes = [2047, 29341, 65281]

        for pseudoprime in pseudoprimes:
            is_prime = galois.is_prime_fermat(pseudoprime)
            p, k = galois.prime_factors(pseudoprime)
            print("Pseudoprime = {:5d}, Fermat's Prime Test = {}, Prime factors = {}".format(pseudoprime, is_prime, list(p)))
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 2:
        raise ValueError(f"Argument `n` must be greater than 2, not {n}.")

    a = 2  # A value coprime with n

    if pow(a, n - 1, n) != 1:
        # n is definitely composite
        return False

    # n is a pseudoprime, but still may be composite
    return True


@set_module("galois")
def is_prime_miller_rabin(n, a=None, rounds=1):
    """
    Determines if :math:`n` is composite.

    This function implements the Miller-Rabin primality test. The test says that for an integer :math:`n`, select an integer
    :math:`a` such that :math:`a < n`. Factor :math:`n - 1` such that :math:`2^s d = n - 1`. Then, :math:`n` is composite,
    if :math:`a^d \\not\\equiv 1 (\\textrm{mod}\\ n)` and :math:`a^{2^r d} \\not\\equiv n - 1 (\\textrm{mod}\\ n)` for
    :math:`1 \\le r < s`.

    Parameters
    ----------
    n : int
        A positive integer.
    a : int, optional
        Initial composite witness value, :math:`1 \\le a < n`. On subsequent rounds, :math:`a` will
        be a different value. The default is a random value.
    rounds : int, optional
        The number of iterations attempting to detect :math:`n` as composite. Additional rounds will choose
        new :math:`a`. Sufficient rounds have arbitrarily-high probability of detecting a composite.

    Returns
    -------
    bool
        `False` if :math:`n` is known to be composite. `True` if :math:`n` is prime or pseudoprime.

    References
    ----------
    * https://math.dartmouth.edu/~carlp/PDF/paper25.pdf
    * https://oeis.org/A001262

    Examples
    --------
    .. ipython:: python

        # List of some primes
        primes = [257, 24841, 65497]

        for prime in primes:
            is_prime = galois.is_prime_miller_rabin(prime)
            p, k = galois.prime_factors(prime)
            print("Prime = {:5d}, Miller-Rabin Prime Test = {}, Prime factors = {}".format(prime, is_prime, list(p)))

        # List of some strong pseudoprimes with base 2
        pseudoprimes = [2047, 29341, 65281]

        # Single round of Miller-Rabin, sometimes fooled by pseudoprimes
        for pseudoprime in pseudoprimes:
            is_prime = galois.is_prime_miller_rabin(pseudoprime)
            p, k = galois.prime_factors(pseudoprime)
            print("Pseudoprime = {:5d}, Miller-Rabin Prime Test = {}, Prime factors = {}".format(pseudoprime, is_prime, list(p)))

        # 7 rounds of Miller-Rabin, never fooled by pseudoprimes
        for pseudoprime in pseudoprimes:
            is_prime = galois.is_prime_miller_rabin(pseudoprime, rounds=7)
            p, k = galois.prime_factors(pseudoprime)
            print("Pseudoprime = {:5d}, Miller-Rabin Prime Test = {}, Prime factors = {}".format(pseudoprime, is_prime, list(p)))
    """
    a = random.randint(1, n - 1) if a is None else a
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 1:
        raise ValueError(f"Argument `n` must be greater than 1, not {n}.")
    if not 1 <= a < n:
        raise ValueError(f"Arguments must satisfy `1 <= a < n`, not `1 <= {a} < {n}`.")

    # Factor (n - 1) by 2
    x = n -1
    s = 0
    while x % 2 == 0:
        s += 1
        x //= 2

    d = (n - 1) // 2**s
    assert d % 2 != 0

    # Write (n - 1) = 2^s * d
    assert 2**s * d == n - 1

    composite_tests = []
    composite_tests.append(pow(a, d, n) != 1)
    for r in range(0, s):
        composite_tests.append(pow(a, 2**r * d, n) != n - 1)
    composite_test = all(composite_tests)

    if composite_test:
        # n is definitely composite
        return False
    else:
        # Optionally iterate tests
        rounds -= 1
        if rounds > 0:
            # On subsequent rounds, don't specify a -- we want new, different a's
            return is_prime_miller_rabin(n, rounds=rounds)
        else:
            # n might be a prime
            return True
