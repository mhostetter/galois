"""
A subpackage containing arrays over Galois fields.
"""
from ._functions import *
from ._irreducible import *
from ._poly import *
from ._primitive import *

# pylint: disable=undefined-variable
_poly.GCD = _functions.gcd
