from ._conway import CONWAY_POLYS
from .gf2 import GF2
from .gfp import GFp_factory
from .poly import Poly


def conway_polynomial(p, n):
    if (p,n) not in CONWAY_POLYS.keys():
        raise ValueError(f"Frank Luebek's Conway polynomial lookup table does not contain an entry for {(p,n)}\n\nSee: http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html")
    field = GF2 if p == 2 else GFp_factory(p)
    poly = Poly(CONWAY_POLYS[(p,n)][::-1], field=field)
    return poly
