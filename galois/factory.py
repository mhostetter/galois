import numpy as np

from . import gfp
from .algorithm import is_prime, primitive_root, min_poly
from .gf import DTYPES
from .gf2 import GF2
from .poly import Poly


def GF_factory(p, m, prim_poly=None, rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type GF(p^m).

    If `p = 2` and `m = 1`, this function will return `galois.GF2`. If `p = 2` and `m > 1`, this function will
    invoke `galois.GF2m_factory()`. If `p is prime` and `m = 1`, this function will invoke `galois.GFp_factory()`.
    If `p is prime` and `m > 1`, this function will invoke `galois.GFpm_factory()`.

    Parameters
    ----------
    p : int
        The prime characteristic of the field GF(p^m).
    m : int
        The degree of the prime characteristic of the field GF(p^m).
    prim_poly : galois.Poly, optional
        The primitive polynomial of the field. Default is `None` which will auto-determine the primitive polynomial.
    rebuild : bool, optional
        A flag to force a rebuild of the class and its lookup tables. Default is `False` which will return the cached,
        previously-built class if it exists.

    Returns
    -------
    galois.GF2Base or galois.GFpBase
        A new Galois field class that is a sublcass of `galois.GFBase`.
    """
    if not isinstance(p, int):
        raise TypeError(f"Galois field prime characteristic `p` must be an integer, not {type(p)}")
    if not isinstance(m, int):
        raise TypeError(f"Galois field characteristic degree `m` must be an integer, not {type(m)}")
    if not (prim_poly is None or isinstance(prim_poly, Poly)):
        raise TypeError(f"Primitive polynomial `prim_poly` must be either None or galois.Poly, not {type(prim_poly)}")
    if not isinstance(rebuild, bool):
        raise TypeError(f"Rebuild Galois field class flag `rebuild` must be a bool, not {type(rebuild)}")
    if not is_prime(p):
        raise ValueError(f"Galois field prime characteristic `p` must be prime, not {p}")
    if not m >= 1:
        raise ValueError(f"Galois field characteristic degree `m` must be >= 1, not {m}")

    if p == 2 and m == 1:
        if not (prim_poly is None or prim_poly is GF2.prim_poly):
            raise ValueError(f"In GF(2), the primitive polynomial `prim_poly` must be either None or {GF2.prim_poly}, not {prim_poly}")
        cls = GF2
    # elif p == 2:
    #     cls = GF2m_factory(m, rebuild=rebuild)
    else:
        cls = GFp_factory(p, rebuild=rebuild)

    return cls


def GFp_factory(p, rebuild=False):
    """
    Factory function to construct Galois field array classes of type GF(p).

    Parameters
    ----------
    p : int
        The prime characteristic of the field GF(p).
    rebuild : bool, optional
        A flag to force a rebuild of the class and its lookup tables. Default is `False` which will return the cached,
        previously-built class if it exists.

    Returns
    -------
    galois.GFpBase
        A new Galois field class that is a sublcass of `galois.GFpBase`.
    """
    # If the requested field has already been constructed, return it instead of rebuilding
    key = (p,)
    if not rebuild and key in GFp_factory.classes:
        return GFp_factory.classes[key]

    if not is_prime(p):
        return ValueError(f"GF(p) fields must have a prime characteristic `p`, not {p}")
    if not 2 <= p <= 2**16:
        return ValueError(f"GF(p) classes are only supported for 2 <= p <= 2**16, not {p}")

    order = p
    name = "GF{}".format(order)
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = primitive_root(p)

    # Create new class type
    cls = type(name, (gfp.GFpBase,), {
        "characteristic": p,
        "power": 1,
        "order": p,
        "alpha": alpha,
        "dtypes": dtypes
    })

    # Construct the field-specific  lookup tables
    cls._build_luts()  # pylint: disable=protected-access

    # JIT compile the numba ufuncs
    cls.target("cpu")

    # Assign the primitive polynomial with coefficients in the field
    cls.prim_poly = min_poly(cls.alpha, cls, 1)

    # Add class to dictionary of flyweights
    GFp_factory.classes[p] = cls

    return cls

GFp_factory.classes = {}
