"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import gcd, chinese_remainder_theorem
from .array import GFArray
from .conway import conway_poly
from .gf import GF
from .gf2 import GF2
from .math_ import isqrt, lcm
from .meta_gf import GFMeta
from .modular import totatives, euler_totient, carmichael, is_cyclic, is_primitive_root, primitive_root, primitive_roots
from .poly import Poly
from .poly_functions import poly_gcd, poly_exp_mod, is_irreducible, is_primitive, is_primitive_element, primitive_element, primitive_elements, is_monic
from .prime import primes, kth_prime, prev_prime, next_prime, mersenne_exponents, mersenne_primes, prime_factors, is_prime, fermat_primality_test, miller_rabin_primality_test

# Define the GF2 primitive polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2._irreducible_poly = Poly([1, 1])  # pylint: disable=protected-access
