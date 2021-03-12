"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import prev_prime, next_prime, factors, prime_factors, is_prime, euclidean_algorithm, extended_euclidean_algorithm, \
                       chinese_remainder_theorem, euler_totient, carmichael, modular_exp, primitive_roots, primitive_root, \
                       min_poly
from .conway import conway_polynomial
from .gf2 import GF2
from .gf2m import GF2m, GF2m_factory
from .gfp import GFp, GFp_factory
from .poly import Poly

# Create the default GF2 class and target the numba ufuncs for "cpu" (lowest overhead)
GF2.target("cpu")

GF2.alpha = GF2(1)

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2.prim_poly = min_poly(GF2.alpha, GF2, 1)


def GF_factory(p, m, prim_poly=None, target="cpu", rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(p^m)`.

    If :math:`p = 2` and :math:`m = 1`, this function will return `galois.GF2`. If :math:`p = 2` and :math:`m > 1`, this function will
    invoke `galois.GF2m_factory()`. If :math:`p` is prime and :math:`m = 1`, this function will invoke `galois.GFp_factory()`.
    If :math:`p` is prime and :math:`m > 1`, this function will invoke `galois.GFpm_factory()`.

    Parameters
    ----------
    p : int
        The prime characteristic of the field :math:`\\mathrm{GF}(p^m)`.
    m : int
        The degree of the prime characteristic of the field :math:`\\mathrm{GF}(p^m)`.
    prim_poly : Poly, optional
        The primitive polynomial of the field. Default is `None` which will auto-determine the primitive polynomial.
    target : str, optional
        The `target` from `numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. See: https://numba.readthedocs.io/en/stable/user/vectorize.html.
    rebuild : bool, optional
        Indicates whether to force a rebuild of the lookup tables. The default is `False`.

    Returns
    -------
    GF2, GF2m, GFp, GFpm
        A new Galois field class that is a sublcass of `galois.GFBase`.
    """
    if not isinstance(p, int):
        raise TypeError(f"Galois field prime characteristic `p` must be an integer, not {type(p)}")
    if not isinstance(m, int):
        raise TypeError(f"Galois field characteristic degree `m` must be an integer, not {type(m)}")
    if not (prim_poly is None or isinstance(prim_poly, Poly)):
        raise TypeError(f"Primitive polynomial `prim_poly` must be either None or galois.Poly, not {type(prim_poly)}")
    if not isinstance(rebuild, bool):
        raise TypeError(f"Rebuild Galois field class flag `rebuild` must be a bool, not {type(rebuild)}")
    if not is_prime(p):
        raise ValueError(f"Galois field prime characteristic `p` must be prime, not {p}")
    if not m >= 1:
        raise ValueError(f"Galois field characteristic degree `m` must be >= 1, not {m}")

    if p == 2 and m == 1:
        if not (prim_poly is None or prim_poly is GF2.prim_poly):
            raise ValueError(f"In GF(2), the primitive polynomial `prim_poly` must be either None or {GF2.prim_poly}, not {prim_poly}")
        GF2.target(target)
        cls = GF2
    elif p == 2:
        cls = GF2m_factory(m, target=target, rebuild=rebuild)
    else:
        cls = GFp_factory(p, target=target, rebuild=rebuild)

    return cls
