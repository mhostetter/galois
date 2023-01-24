"""
A subpackage containing arrays over Galois fields.
"""
from ._conway import *
from ._factor import *
from ._functions import *
from ._irreducible import *
from ._poly import *
from ._primitive import *

# pylint: disable=undefined-variable
Poly.square_free_factors = _factor.square_free_factors
Poly.distinct_degree_factors = _factor.distinct_degree_factors
Poly.equal_degree_factors = _factor.equal_degree_factors
Poly.factors = _factor.factors
Poly.is_irreducible = _irreducible.is_irreducible
Poly.is_primitive = _primitive.is_primitive
