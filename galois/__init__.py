"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import *
from .array import *
from .conway import *
from .factor import *
from .gf import *
from .gf2 import *
from .math_ import *
from .meta_gf import *
from .modular import *
from .poly import *
from .poly_functions import *
from .prime import *

# Define the GF2 primitive polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2._irreducible_poly = Poly([1, 1])  # pylint: disable=protected-access
