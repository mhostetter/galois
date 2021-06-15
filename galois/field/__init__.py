"""
A subpackage containing arrays and polynomials over Galois fields.
"""
from .array import *
from .conway import *
from .factory import *
from .gf2 import *
from .meta_class import *
from .oakley import *
from .poly import *
from .poly_functions import *

# Define the GF2 primitive polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2._irreducible_poly = Poly([1, 1])  # pylint: disable=protected-access
