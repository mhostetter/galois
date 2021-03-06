"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import prev_prime, next_prime, factors, prime_factors, is_prime, euclidean_algorithm, extended_euclidean_algorithm, \
                       chinese_remainder_theorem, euler_totient, carmichael, modular_exp, primitive_roots, primitive_root, \
                       min_poly
from .factory import GF_factory, GFp_factory
from .gf import GFBase
from .gf2 import GF2
from .gfp import GFpBase
from .poly import Poly

# Create the default GF2 class and target the numba ufuncs for "cpu" (lowest overhead)
GF2.target("cpu")

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2.prim_poly = min_poly(GF2.alpha, GF2, 1)
