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


def GF_factory(p, m, prim_poly=None, target="cpu", rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(p^m)`.

    If :math:`p = 2` and :math:`m = 1`, this function will return `galois.GF2`. If :math:`p = 2` and :math:`m > 1`, this function will
    invoke `galois.GF2m_factory()`. If :math:`p` is prime and :math:`m = 1`, this function will invoke `galois.GFp_factory()`.
    If :math:`p` is prime and :math:`m > 1`, this function will invoke `galois.GFpm_factory()`.

    Parameters
    ----------
    p : int
        The prime characteristic of the field :math:`\\mathrm{GF}(p^m)`.
    m : int
        The degree of the prime characteristic of the field :math:`\\mathrm{GF}(p^m)`.
    prim_poly : Poly, optional
        The primitive polynomial of the field. Default is `None` which will use the Conway polynomial `galois.conway_polynomial(p, m)`.
    target : str, optional
        The `target` from `numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. See: https://numba.readthedocs.io/en/stable/user/vectorize.html.
    rebuild : bool, optional
        Indicates whether to force a rebuild of the lookup tables. The default is `False`.

    Returns
    -------
    GF2, GF2m, GFp, GFpm
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
        GF2.target(target)
        cls = GF2
    elif p == 2:
        cls = GF2m_factory(m, prim_poly=prim_poly, target=target, rebuild=rebuild)
    else:
        cls = GFp_factory(p, prim_poly=prim_poly, target=target, rebuild=rebuild)

    return cls


def GF2m_factory(m, prim_poly=None, target="cpu", mode="auto", rebuild=False):
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
    rebuild : bool, optional
        Indicates whether to force a rebuild of the lookup tables. The default is `False`.

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

    # If the requested field has already been constructed, return it instead of rebuilding
    key = (m,)
    if not rebuild and key in GF2m_factory.classes:
        return GF2m_factory.classes[key]

    characteristic = 2
    power = m
    order = characteristic**power
    name = "GF{}".format(order)
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = 2
    if prim_poly is None:
        prim_poly = conway_polynomial(characteristic, power)

    # Create new class type
    cls = type(name, (GF2m,), {
        "characteristic": characteristic,
        "power": power,
        "order": order,
        "prim_poly": prim_poly,
        "dtypes": dtypes
    })

    # Define the primitive element as a 0-dim array in the newly created Galois field array class
    cls.alpha = cls(alpha)

    # JIT compile the numba ufuncs
    cls.target(target, mode=mode, rebuild=rebuild)

    # Add class to dictionary of flyweights
    GF2m_factory.classes[key] = cls

    return cls

GF2m_factory.classes = {}


def GFp_factory(p, prim_poly=None, target="cpu", mode="auto", rebuild=False):  # pylint: disable=unused-argument
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
    rebuild : bool, optional
        Indicates whether to force a rebuild of the lookup tables. The default is `False`.

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
        return ValueError(f"GF(p) fields must have a prime characteristic `p`, not {p}")
    if not 2 <= p <= 2**16:
        return ValueError(f"GF(p) classes are only supported for 2 <= p <= 2**16, not {p}")

    # If the requested field has already been constructed, return it instead of rebuilding
    key = (p,)
    if not rebuild and key in GFp_factory.classes:
        return GFp_factory.classes[key]

    characteristic = p
    power = 1
    order = characteristic**power
    name = "GF{}".format(order)
    dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = primitive_root(p)

    # Create new class type
    cls = type(name, (GFp,), {
        "characteristic": characteristic,
        "power": power,
        "order": order,
        "dtypes": dtypes
    })

    # Define the primitive element as a 0-dim array in the newly created Galois field array class
    cls.alpha = cls(alpha)

    # JIT compile the numba ufuncs
    cls.target(target, mode=mode, rebuild=rebuild)

    cls.prim_poly = Poly([1, -alpha], field=cls)  # pylint: disable=invalid-unary-operand-type

    # Add class to dictionary of flyweights
    GFp_factory.classes[key] = cls

    return cls

GFp_factory.classes = {}


def conway_polynomial(p, n):
    # pylint: disable=import-outside-toplevel
    from ._conway import CONWAY_POLYS

    if (p,n) not in CONWAY_POLYS.keys():
        raise ValueError(f"Frank Luebek's Conway polynomial lookup table does not contain an entry for {(p,n)}\n\nSee: http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html")

    field = GF2 if p == 2 else GFp_factory(p)
    poly = Poly(CONWAY_POLYS[(p,n)][::-1], field=field)

    return poly
