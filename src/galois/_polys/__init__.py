"""
A subpackage containing arrays over Galois fields.
"""
from . import _constructors
from ._conway import *
from ._factor import *
from ._functions import *
from ._irreducible import *
from ._poly import *
from ._primitive import *

_constructors.POLY = Poly
_constructors.POLY_DEGREES = Poly.Degrees
_constructors.POLY_INT = Poly.Int
_constructors.POLY_RANDOM = Poly.Random
