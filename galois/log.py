"""
A module containing algorithms for computing discrete logarithms.
"""
import math

from .modular import order
from .overrides import set_module

__all__ = ["log_naive"]


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
