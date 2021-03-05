"""
A Python 3 numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import prev_prime, next_prime, factors, prime_factors, is_prime, euclidean_algorithm, extended_euclidean_algorithm, \
                       chinese_remainder_theorem, euler_totient, carmichael, modular_exp, primitive_roots, \
                       min_poly
from .gf2 import GF2
from .gfp import GFp, GFp_factory
from .poly import Poly

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2.prim_poly = min_poly(GF2.alpha, GF2, 1)


def GF_factory(p, m, prim_poly=None, rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type GF(p^m).

    If `p = 2` and `m = 1`, this function will return `galois.GF2`. If `p = 2` and `m > 1`, this function will
    invoke `galois.GF2m_factory()`. If `p is prime` and `m = 1`, this function will invoke `galois.GFp_factory()`.
    If `p is prime` and `m > 1`, this function will invoke `galois.GFpm_factory()`.

    Parameters
    ----------
    p : int
        The prime characteristic of the field GF(p^m).
    m : int
        The degree of the prime of the field GF(p^m).
    prim_poly : galois.Poly, optional
        The primitive polynomial of the field. Default is `None` which will auto-determine the primitive polynomial.
    rebuild : bool, optional
        A flag to force a rebuild of the class and its lookup tables. Default is `False` which will return the cached,
        previously-built class if it exists.

    Returns
    -------
    galois.GF2 or galois.GFp
        A new Galois field class that is a sublcass of `galois._GF`.
    """
    assert m >= 1

    if p == 2 and m == 1:
        assert prim_poly is None or prim_poly == GF2.prim_poly
        cls = GF2
    # elif p == 2:
    #     cls = GF2m_factory(m, rebuild=rebuild)
    else:
        cls = GFp_factory(p, rebuild=rebuild)

    return cls
