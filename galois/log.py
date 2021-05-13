"""
A module containing algorithms for computing discrete logarithms.
"""
import math

from .algorithm import gcd, crt
from .factor import prime_factors
from .math_ import isqrt
from .modular import euler_totient, is_cyclic, is_primitive_root, order
from .overrides import set_module

__all__ = ["log_naive", "log_baby_giant_step", "log_pollard_rho", "log_pohlig_hellman"]


@set_module("galois")
def log_naive(beta, alpha, modulus):
    """
    Computes the discrete logarithm :math:`x = \\textrm{log}_{\\alpha}(\\beta)\\ (\\textrm{mod}\\ m)`.

    This function implements the naive algorithm. It is included for testing and reference.
    This method take :math:`O(n)` multiplications, where :math:`n` is the multiplicative order of :math:`\\alpha`.

    Arguments
    ---------
    beta : int
        The integer :math:`\\beta` to compute the logarithm of. :math:`\\beta` must be coprime to :math:`m`.
    alpha : int
        The base :math:`\\alpha`. :math:`\\alpha` must be coprime to :math:`m`.
    modulus : int
        The modulus :math:`m`.

    References
    ----------
    * Chapter 3.6.1 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples
    --------
    .. ipython:: python

        N = 17
        galois.totatives(N)
        galois.primitive_roots(N)
        x = galois.log_naive(3, 7, N); x
        7**x % N

    .. ipython:: python

        N = 18

        # alpha and beta must be totatives of the modulus, i.e. coprime to N
        galois.totatives(N)

        # The discrete log is defined for all totatives base a primitive root
        galois.primitive_roots(N)

        x = galois.log_naive(11, 5, N); x
        pow(5, x, N)
    """
    if not math.gcd(beta, modulus) == 1:
        raise ValueError(f"Argument `beta` must be coprime with `modulus`, {beta} is not coprime with {modulus}.")
    if not math.gcd(alpha, modulus) == 1:
        raise ValueError(f"Argument `alpha` must be coprime with `modulus`, {alpha} is not coprime with {modulus}.")

    order_ = order(alpha, modulus)
    for k in range(order_):
        if pow(alpha, k, modulus) == beta:
            return k

    raise ValueError(f"The discrete logarithm of {beta} base {alpha} mod {modulus} does not exist.")


@set_module("galois")
def log_baby_giant_step(beta, alpha, modulus, cache=True):
    """
    Computes the discrete logarithm :math:`x = \\textrm{log}_{\\alpha}(\\beta)\\ (\\textrm{mod}\\ m)`.

    This function implements the Baby-Step Giant-Step algorithm. It requires :math:`O(\\sqrt{n})` storage, where
    :math:`n` is the multiplicative order of the group. The running time of the algorithm is :math:`O(\\sqrt{n})`.

    Arguments
    ---------
    beta : int
        The integer :math:`\\beta` to compute the logarithm of. :math:`\\beta` must be coprime to :math:`m`.
    alpha : int
        The base :math:`\\alpha`. :math:`\\alpha` must be a primitive root of :math:`m`.
    modulus : int
        The modulus :math:`m`.
    cache : bool, optional
        Optionally save the generated lookup tables for :math:`(\\alpha, m)`. The default is `True`.

    References
    ----------
    * Chapter 3.6.2 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples
    --------
    .. ipython:: python

        prime = galois.random_prime(20); prime
        alpha = galois.primitive_root(prime); alpha
        x = galois.log_baby_giant_step(123456, alpha, prime); x
        pow(alpha, x, prime)
    """
    # pylint: disable=protected-access
    if not is_cyclic(modulus):
        raise ValueError(f"Argument `modulus` must produce a multiplicative cyclic group, (ℤ/{modulus}ℤ)* is not cyclic.")
    if not math.gcd(beta, modulus) == 1:
        raise ValueError(f"Argument `beta` must be coprime with `modulus`, {beta} is not coprime with {modulus}.")
    # if not math.gcd(alpha, modulus) == 1:
    #     raise ValueError(f"Argument `alpha` must be coprime with `modulus`, {alpha} is not coprime with {modulus}.")
    if not is_primitive_root(alpha, modulus):
        raise ValueError(f"Argument `alpha` must be a primitive root modulo `modulus`, {alpha} is not a primitive root modulo {modulus}.")

    if beta == alpha:
        return 1

    n = euler_totient(modulus)  # The multiplicative order of the group
    # n = order(alpha, modulus)
    m = isqrt(n)
    if m**2 < n:
        m += 1

    # Recall the LUT from cache or generate it
    key = (alpha, modulus)
    if cache and key in log_baby_giant_step._LUTs:
        LUT = log_baby_giant_step._LUTs[key]
    else:
        js = list(range(0, m))
        alpha_js = [pow(alpha, j, modulus) for j in js]
        LUT = dict(zip(alpha_js, js))
        if cache:
            log_baby_giant_step._LUTs[key] = LUT

    alpha_inv = gcd(alpha, modulus)[1]
    alpha_inv_m = pow(alpha_inv, m, modulus)
    gamma = beta

    for i in range(0, m):
        if gamma in LUT.keys():
            j = LUT[gamma]
            return i*m + j
        else:
            gamma = (gamma * alpha_inv_m) % modulus

    raise ValueError(f"The discrete logarithm of {beta} base {alpha} mod {modulus} does not exist.")

log_baby_giant_step._LUTs = {}  # pylint: disable=protected-access


@set_module("galois")
def log_pollard_rho(beta, alpha, modulus):
    """
    Computes the discrete logarithm :math:`x = \\textrm{log}_{\\alpha}(\\beta)\\ (\\textrm{mod}\\ m)`.

    This function implements the Pollard's rho algorithm. The running time of the algorithm is :math:`O(\\sqrt{n})`.

    Arguments
    ---------
    beta : int
        The integer :math:`\\beta` to compute the logarithm of. :math:`\\beta` must be coprime to :math:`m`.
    alpha : int
        The base :math:`\\alpha`. :math:`\\alpha` must be coprime to :math:`m`.
    modulus : int
        The modulus :math:`m`.

    References
    ----------
    * Chapter 3.6.3 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples
    --------
    .. ipython:: python

        N = 383
        galois.is_cyclic(N)
        galois.primitive_roots(N, stop=20)
        alpha = 2
        galois.order(alpha, N)
        x = galois.log_pollard_rho(228, alpha, N); x
        pow(alpha, x, N)
    """
    # pylint: disable=protected-access
    if not is_cyclic(modulus):
        raise ValueError(f"Argument `modulus` must produce a multiplicative cyclic group, (ℤ/{modulus}ℤ)* is not cyclic.")
    if not math.gcd(beta, modulus) == 1:
        raise ValueError(f"Argument `beta` must be coprime with `modulus`, {beta} is not coprime with {modulus}.")
    if not math.gcd(alpha, modulus) == 1:
        raise ValueError(f"Argument `alpha` must be coprime with `modulus`, {alpha} is not coprime with {modulus}.")

    if beta == alpha:
        return 1

    n = order(alpha, modulus)  # The multiplicative order of alpha

    xi, ai, bi = 1, 0, 0
    x2i, a2i, b2i = xi, 0, 0

    def update(x, a, b):
        if x % 3 == 1:  # Set 1
            x = (beta * x) % modulus
            b = (b + 1) % n
        elif x % 3 == 0:  # Set 2
            x = (x * x) % modulus
            a = (2*a) % n
            b = (2*b) % n
        else:
            x = (alpha * x) % modulus
            a = (a + 1) % n
        return x, a, b

    while True:
        xi, ai, bi = update(xi, ai, bi)
        x2i, a2i, b2i = update(*update(x2i, a2i, b2i))  # Runs twice as fast
        if xi == x2i:  # Cycle found
            break

    r = (bi - b2i) % n
    if r == 0:
        return RuntimeError("Failed")
    r_inv = gcd(r, n)[1] % n
    x = (r_inv * (a2i - ai)) % n

    return x


@set_module("galois")
def log_pohlig_hellman(beta, alpha, modulus):
    """
    Computes the discrete logarithm :math:`x = \\textrm{log}_{\\alpha}(\\beta)\\ (\\textrm{mod}\\ m)`.

    This function implements the Pohlig-Hellman algorithm.

    Arguments
    ---------
    beta : int
        The integer :math:`\\beta` to compute the logarithm of. :math:`\\beta` must be coprime to :math:`m`.
    alpha : int
        The base :math:`\\alpha`. :math:`\\alpha` must be coprime to :math:`m`.
    modulus : int
        The modulus :math:`m`.

    References
    ----------
    * Chapter 3.6.4 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf

    Examples
    --------
    .. ipython:: python

        prime = 22708823198678103974314518195029102158525052496759285596453269189798311427475159776411276642277139650833937
        galois.is_prime(prime)
        alpha = 3
        galois.is_primitive_root(alpha, prime)

        # Because the prime-1 is smooth, Pohlig-Hellman is efficient
        galois.prime_factors(prime - 1)

        x = galois.log_pohlig_hellman(123456789, alpha, prime)
        pow(alpha, x, prime)
    """
    # pylint: disable=protected-access
    if not is_cyclic(modulus):
        raise ValueError(f"Argument `modulus` must produce a multiplicative cyclic group, (ℤ/{modulus}ℤ)* is not cyclic.")
    if not math.gcd(beta, modulus) == 1:
        raise ValueError(f"Argument `beta` must be coprime with `modulus`, {beta} is not coprime with {modulus}.")
    if not math.gcd(alpha, modulus) == 1:
        raise ValueError(f"Argument `alpha` must be coprime with `modulus`, {alpha} is not coprime with {modulus}.")

    n = euler_totient(modulus)  # The multiplicative order of the group
    primes, exponents = prime_factors(n)

    x = [0,]*len(primes)
    for i in range(len(primes)):
        q, e = primes[i], exponents[i]
        l = [0,]*(e + 1)  # One longer than needed
        gamma = 1
        l[-1] = 0
        alpha_bar = pow(alpha, n // q, modulus)
        for j in range(e):
            gamma = gamma * pow(alpha, int(l[j - 1] * q**(j - 1)), modulus) % modulus
            gamma_inv = gcd(gamma, modulus)[1]
            beta_bar = pow(beta * gamma_inv, n // q**(j + 1), modulus)
            # l[j] = log_baby_giant_step(beta_bar, alpha_bar, modulus)
            # l[j] = log_pollard_rho(beta_bar, alpha_bar, modulus)
            l[j] = log_naive(beta_bar, alpha_bar, modulus)
            x[i] += l[j]*q**(j)

    m = [pi**ei for pi, ei in zip(primes, exponents)]

    return crt(x, m)
