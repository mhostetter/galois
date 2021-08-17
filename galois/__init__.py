"""
A performant NumPy extension for Galois fields and their applications.
"""
# pylint: disable=redefined-builtin

from ._version import __version__

# Subpackages
from ._codes import *
from ._fields import *
from ._polys import *

# Modules
from ._factor import *
from ._factory import *
from ._lfsr import *
from ._math import *
from ._modular import *
from ._polymorphic import *
from ._prime import *

# Define the GF(2) primitive polynomial here, not in _fields/_gf2.py, to avoid a circular dependency with `Poly`.
# The primitive polynomial is p(x) = x - alpha, where alpha = 1. Over GF(2), this is equivalent
# to p(x) = x + 1.
GF2._irreducible_poly = Poly([1, 1])  # pylint: disable=protected-access
