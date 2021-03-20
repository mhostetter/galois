"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import prev_prime, next_prime, factors, prime_factors, is_prime, euclidean_algorithm, extended_euclidean_algorithm, \
                       chinese_remainder_theorem, euler_totient, carmichael, modular_exp, primitive_roots, primitive_root, \
                       min_poly
from .gf2 import GF2
from .gf2m import GF2m
from .gfp import GFp
from .poly import Poly

# Create the default GF2 class and target the numba ufuncs for "cpu" (lowest overhead)
GF2.target("cpu")

GF2.alpha = GF2(1)

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2.prim_poly = min_poly(GF2.alpha, GF2, 1)


def GF_factory(characteristic, degree, prim_poly=None, target="cpu", mode="auto", rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(p^m)`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p^m)`.
    degree : int
        The degree :math:`m` of the prime characteristic of the field :math:`\\mathrm{GF}(p^m)`.
    prim_poly : Poly, optional
        The primitive polynomial of the field. Default is `None` which will use the Conway polynomial `galois.conway_polynomial(p, m)`.
    target : str, optional
        The `target` from `numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. See: https://numba.readthedocs.io/en/stable/user/vectorize.html.
    mode : str, optional
        The type of field computation, either `"auto"`, `"lookup"`, or `"calculate"`. The default is `"auto"`. GF(2) fields
        only support `"auto"` and `"calculate"` modes. This is because it's more efficient to just compute the arithmetic than store
        lookup tables. The "lookup" mode will use Zech log, log, and anti-log lookup table for speed. The `"calculate"` mode will
        not store any lookup tables, but calculate the field arithmetic on the fly. The `"calculate"` mode is slower
        than `"lookup"` but uses less RAM. The "auto" mode will determine whether to use `"lookup"` or `"calculate"` based
        on field order.
    rebuild : bool, optional
        Indicates whether to force a rebuild of the lookup tables. The default is `False`.

    Returns
    -------
    GF2, GF2m, GFp, GFpm
        A new Galois field class that is a sublcass of `galois.GFBase`.
    """
    if not isinstance(characteristic, int):
        raise TypeError(f"Galois field GF(p^m) prime characteristic `p` must be an integer, not {type(characteristic)}")
    if not isinstance(degree, int):
        raise TypeError(f"Galois field GF(p^m) characteristic degree `m` must be an integer, not {type(degree)}")
    if not (prim_poly is None or isinstance(prim_poly, Poly)):
        raise TypeError(f"Primitive polynomial `prim_poly` must be either None or galois.Poly, not {type(prim_poly)}")
    if not isinstance(rebuild, bool):
        raise TypeError(f"Rebuild Galois field GF(p^m) class flag `rebuild` must be a bool, not {type(rebuild)}")
    if not is_prime(characteristic):
        raise ValueError(f"Galois field GF(p^m) prime characteristic `p` must be prime, not {characteristic}")
    if not degree >= 1:
        raise ValueError(f"Galois field GF(p^m) characteristic degree `m` must be >= 1, not {degree}")

    # If the requested field has already been constructed, return it instead of rebuilding
    key = (characteristic, degree, mode)
    if not rebuild and key in GF_factory.classes:
        return GF_factory.classes[key]

    if characteristic == 2 and degree == 1:
        if not (prim_poly is None or prim_poly is GF2.prim_poly):
            raise ValueError(f"In GF(2), the primitive polynomial `prim_poly` must be either None or {GF2.prim_poly}, not {prim_poly}")
        GF2.target(target)
        cls = GF2
    elif characteristic == 2:
        cls = _GF2m_factory(degree, prim_poly=prim_poly, target=target, mode=mode)
    else:
        cls = _GFp_factory(characteristic, prim_poly=prim_poly, target=target, mode=mode)

    # Add class to dictionary of flyweights
    GF_factory.classes[key] = cls

    return cls

GF_factory.classes = {}


def _GF2m_factory(m, prim_poly=None, target="cpu", mode="auto"):
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(2^m)`.

    Parameters
    ----------
    m : int
        The prime characteristic degree :math:`m` of the field :math:`\\mathrm{GF}(2^m)`.
    prim_poly : Poly, optional
        The primitive polynomial of the field. Default is `None` which will use the Conway polynomial `galois.conway_polynomial(2, m)`.
    target : str, optional
        The `target` from `numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. See: https://numba.readthedocs.io/en/stable/user/vectorize.html.
    mode : str, optional
        The type of field computation, either `"auto"`, `"lookup"`, or `"calculate"`. The default is `"auto"`.
        The "lookup" mode will use Zech log, log, and anti-log lookup table for speed. The `"calculate"` mode will
        not store any lookup tables, but calculate the field arithmetic on the fly. The `"calculate"` mode is slower
        than `"lookup"` but uses less RAM. The "auto" mode will determine whether to use `"lookup"` or `"calculate"` based
        on field order.

    Returns
    -------
    GF2m
        A new Galois field class that is a sublcass of `galois.GF2m`.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from .gf import DTYPES

    if not isinstance(m, (int, np.integer)):
        raise TypeError(f"GF(2^m) characteristic degree `m` must be an integer, not {type(m)}")
    if not 1 <= m <= 32:
        return ValueError(f"GF(2^m) classes are only supported for 2 <= m <= 2**32, not {m}")
    if not (prim_poly is None or isinstance(prim_poly, Poly)):
        raise TypeError(f"Primitive polynomial `prim_poly` must be either None or galois.Poly, not {type(prim_poly)}")

    characteristic = 2
    degree = m
    order = characteristic**degree
    # name = "GF{}".format(order)
    name = f"GF{characteristic}^{degree}"
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order - 1]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = 2
    if prim_poly is None:
        prim_poly = conway_polynomial(characteristic, degree)

    # Create new class type
    cls = type(name, (GF2m,), {
        "characteristic": characteristic,
        "degree": degree,
        "order": order,
        "prim_poly": prim_poly,
        "dtypes": dtypes
    })

    # Define the primitive element as a 0-dim array in the newly created Galois field array class
    cls.alpha = cls(alpha)

    # JIT compile the numba ufuncs
    cls.target(target, mode=mode)

    return cls


def _GFp_factory(p, prim_poly=None, target="cpu", mode="auto"):  # pylint: disable=unused-argument
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(p)`.

    Parameters
    ----------
    p : int
        The prime characteristic of the field :math:`\\mathrm{GF}(p)`.
    target : str, optional
        The `target` from `numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. See: https://numba.readthedocs.io/en/stable/user/vectorize.html.
    mode : str, optional
        The type of field computation, either `"auto"`, `"lookup"`, or `"calculate"`. The default is `"auto"`.
        The "lookup" mode will use Zech log, log, and anti-log lookup table for speed. The `"calculate"` mode will
        not store any lookup tables, but calculate the field arithmetic on the fly. The `"calculate"` mode is slower
        than `"lookup"` but uses less RAM. The "auto" mode will determine whether to use `"lookup"` or `"calculate"` based
        on field order.

    Returns
    -------
    GFp
        A new Galois field class that is a sublcass of `galois.GFp`.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from .gf import DTYPES

    if not isinstance(p, (int, np.integer)):
        raise TypeError(f"GF(p) prime characteristic `p` must be an integer, not {type(p)}")
    if not is_prime(p):
        raise ValueError(f"GF(p) fields must have a prime characteristic `p`, not {p}")
    # if not 2 <= p <= 2**16:
    #     raise ValueError(f"GF(p) classes are only supported for 2 <= p <= 2**16, not {p}")

    characteristic = p
    degree = 1
    order = characteristic**degree

    if mode == "auto":
        mode = "lookup" if order <= 2**16 else "calculate"

    name = "GF{}".format(order)
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order - 1]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = primitive_root(p)

    # Create new class type
    cls = type(name, (GFp,), {
        "characteristic": characteristic,
        "degree": degree,
        "order": order,
        "dtypes": dtypes
    })

    # Define the primitive element as a 0-dim array in the newly created Galois field array class
    cls.alpha = cls(alpha)

    # JIT compile the numba ufuncs
    cls.target(target, mode)

    cls.prim_poly = Poly([1, -alpha], field=cls)  # pylint: disable=invalid-unary-operand-type

    return cls


def conway_polynomial(p, n):
    # pylint: disable=import-outside-toplevel
    from ._conway import CONWAY_POLYS

    if (p,n) not in CONWAY_POLYS.keys():
        raise ValueError(f"Frank Luebek's Conway polynomial lookup table does not contain an entry for {(p,n)}\n\nSee: http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html")

    field = GF2 if p == 2 else _GFp_factory(p)
    poly = Poly(CONWAY_POLYS[(p,n)][::-1], field=field)

    return poly
