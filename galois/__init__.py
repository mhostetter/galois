"""
A Python 3 package for Galois field arithmetic.
"""
from .version import __version__

from .algorithm import prev_prime, next_prime, factors, prime_factors, is_prime, euclidean_algorithm, extended_euclidean_algorithm, \
                       chinese_remainder_theorem, euler_totient, carmichael, modular_exp, primitive_roots, \
                       min_poly
from .gf2 import GF2
from .poly import Poly

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency
GF2.prim_poly = Poly(GF2([1,1]))
