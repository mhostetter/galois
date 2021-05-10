"""
A module containing algorithms for computing discrete logarithms.
"""
import math

from .modular import euler_totient, is_cyclic, is_primitive_root
from .overrides import set_module

__all__ = ["log_naive"]


@set_module("galois")
def log_naive(beta, alpha, modulus):
    """
    Computes the discrete logarithm :math:`x = \\textrm{log}_{\\alpha}(\\beta)\\ (\\textrm{mod}\\ m)`.

    This function implements the naive algorithm. It is included for testing and reference.

    Arguments
    ---------
    beta : int
        The integer :math:`\\beta` to compute the logarithm of.
    alpha : int
        The base :math:`\\alpha`.
    modulus : int
        The modulus :math:`m`.

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
        galois.totatives(N)
        galois.primitive_roots(N)
        x = galois.log_naive(11, 5, N); x
        5**x % N
    """
    if not is_cyclic(modulus):
        raise ValueError(f"Argument `modulus` must produce a multiplicative group that is cyclic, Z({modulus})* is not cyclic.")
    if not math.gcd(beta, modulus) == 1:
        raise ValueError(f"Argument `beta` must be coprime with `modulus`, {beta} is not coprime with {modulus}.")
    if not is_primitive_root(alpha, modulus):
        raise ValueError(f"Argument `alpha` must be a primitive root of `modulus`, {alpha} is not a primitive root of {modulus}.")

    order = euler_totient(modulus)
    for k in range(order):
        if pow(alpha, k, modulus) == beta:
            return k

    raise RuntimeError(f"Discrete logarithm of {beta} base {alpha} mod {modulus} was not found.")
