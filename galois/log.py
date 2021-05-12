"""
A module containing algorithms for computing discrete logarithms.
"""
import math

from .algorithm import gcd
from .math_ import isqrt
from .modular import euler_totient, is_cyclic, is_primitive_root, order
from .overrides import set_module

__all__ = ["log_naive", "log_baby_giant_step"]


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
    if not is_primitive_root(alpha, modulus):
        raise ValueError(f"Argument `alpha` must be a primitive root modulo `modulus`, {alpha} is not a primitive root modulo {modulus}.")

    n = euler_totient(modulus)  # The multiplicative order of the group
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
