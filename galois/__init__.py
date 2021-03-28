"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import factors, euclidean_algorithm, extended_euclidean_algorithm, chinese_remainder_theorem, euler_totient, carmichael, primitive_root
from .gf import GF
from .gf2 import GF2
from .modular import modular_exp
from .poly import Poly
from .prime import prev_prime, next_prime, prime_factors, is_prime, fermat_primality_test, miller_rabin_primality_test


def GF_factory(characteristic, degree, prim_poly=None, target="cpu", mode="auto", rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(p^m)`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p^m)`.
    degree : int
        The prime characteristic's degree :math:`m` of the field :math:`\\mathrm{GF}(p^m)`.
    prim_poly : galois.Poly, optional
        The primitive polynomial of the field. Default is `None` which will use the Conway polynomial
        obtained from :obj:`galois.conway_poly`.
    target : str
        The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`.
    mode : str, optional
        The type of field computation, either `"auto"`, `"lookup"`, or `"calculate"`. The default is `"auto"`.
        The "lookup" mode will use Zech log, log, and anti-log lookup tables for speed. The "calculate" mode will
        not store any lookup tables, but perform field arithmetic on the fly. The "calculate" mode is designed
        for large fields that cannot store lookup tables in RAM. Generally, "calculate" will be slower than "lookup".
        The "auto" mode will determine whether to use "lookup" or "calculate" based on the field size. For "auto",
        field's with `order <= 2**16` will use the "lookup" mode.
    rebuild : bool, optional
        Indicates whether to force a rebuild of the lookup tables. The default is `False`.

    Returns
    -------
    galois.GF
        A new Galois field array class that is a sublcass of :obj:`galois.GF`.

    Examples
    --------

    Construct various kinds of Galois fields.

    .. ipython:: python

        # Construct a GF(2^m) class
        GF256 = galois.GF_factory(2, 8); print(GF256)

        # Construct a GF(p) class
        GF571 = galois.GF_factory(571, 1); print(GF571)

        # Construct a very large GF(2^m) class
        GF2m = galois.GF_factory(2, 100); print(GF2m)

        # Construct a very large GF(p) class
        GFp = galois.GF_factory(36893488147419103183, 1); print(GFp)

    See :obj:`galois.GF` for more examples of what a Galois field array can do.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np

    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Galois field GF(p^m) prime characteristic `p` must be an int, not {type(characteristic)}")
    if not isinstance(degree, int):
        raise TypeError(f"Galois field GF(p^m) characteristic degree `m` must be an int, not {type(degree)}")
    if not (prim_poly is None or isinstance(prim_poly, Poly)):
        raise TypeError(f"Primitive polynomial `prim_poly` must be either None or galois.Poly, not {type(prim_poly)}")
    if not isinstance(rebuild, bool):
        raise TypeError(f"The 'rebuild' argument must be a bool, not {type(rebuild)}")

    if not is_prime(characteristic):
        raise ValueError(f"Galois field GF(p^m) characteristic `p` must be prime, not {characteristic}")
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
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from .gf import DTYPES
    from .gf2m import GF2m

    characteristic = 2
    degree = m
    order = characteristic**degree
    # name = "GF{}".format(order)
    name = f"GF{characteristic}^{degree}"

    if order - 1 > 2**62:
        # TODO: Double check these conditions
        dtypes = [np.object_]
    else:
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= order - 1]

    # Use the smallest primitive root as the multiplicative generator for the field
    alpha = 2
    if prim_poly is None:
        prim_poly = conway_poly(characteristic, degree)

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
    if mode == "auto":
        mode = "lookup" if cls.order <= 2**16 else "calculate"
    cls.target(target, mode)

    # Add helper variables for python ufuncs. This prevents the ufuncs from having to repeatedly calculate them.
    cls._alpha_dec = int(cls.alpha)  # pylint: disable=protected-access
    cls._prim_poly_dec = cls.prim_poly.integer  # pylint: disable=protected-access

    return cls


def _GFp_factory(p, prim_poly=None, target="cpu", mode="auto"):  # pylint: disable=unused-argument
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from .gf import DTYPES
    from .gfp import GFp

    characteristic = p
    degree = 1
    order = characteristic**degree

    name = "GF{}".format(order)

    if order - 1 > 2**31:
        # Orders of 2^31 or less can be stored as uint32 or int64. This is because we need to sometimes calculate
        # `(a * b) % prime` in a field and we need (2^31) * (2^31) = 2^62 < 2^63 to fit in an int64. Values larger
        # than 2^31 need to be stored as dtype=np.object_.
        # TODO: Double check these conditions
        dtypes = [np.object_]
    else:
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
    if mode == "auto":
        mode = "lookup" if cls.order <= 2**16 else "calculate"
    cls.target(target, mode)

    cls.prim_poly = Poly([1, -alpha], field=cls)  # pylint: disable=invalid-unary-operand-type

    # Add helper variables for python ufuncs. This prevents the ufuncs from having to repeatedly calculate them.
    cls._alpha_dec = int(cls.alpha)  # pylint: disable=protected-access
    cls._prim_poly_dec = cls.prim_poly.integer  # pylint: disable=protected-access

    return cls


def conway_poly(characteristic, degree):
    """
    Returns the Conway polynomial for :math:`\\mathrm{GF}(p^m)`.

    This function uses Frank Luebeck's Conway polynomial database. See: http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\\mathrm{GF}(p^m)`.
    degree : int
        The prime characteristic's degree :math:`m` of the field :math:`\\mathrm{GF}(p^m)`.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` polynomial in :math:`\\mathrm{GF}(p)[x]`.

    Note
    ----
        If the :math:`\\mathrm{GF}(p)` field hasn't already been created, it will be created in this function
        since it's needed in the return polynomial.

    Examples
    --------
    .. ipython:: python

        galois.conway_poly(2, 100)
        galois.conway_poly(7, 13)
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np
    from .conway import ConwayDatabase

    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"GF(p^m) prime characteristic `p` must be an integer, not {type(characteristic)}")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"GF(p^m) characteristic degree `m` must be an integer, not {type(degree)}")

    db = ConwayDatabase()
    coeffs = db.fetch(characteristic, degree)
    if coeffs is None:
        raise ValueError(f"Frank Luebeck's database of Conway polynomials doesn't contain an entry for GF({characteristic}^{degree})")

    coeffs = list(map(int, coeffs[1:-1].split(",")))  # List of degree-ascending coefficients
    field = GF2 if characteristic == 2 else _GFp_factory(characteristic)

    return Poly(coeffs[::-1], field=field)


# Create the default GF2 class and target the numba ufuncs for "cpu" (lowest overhead)
GF2.target("cpu")

GF2.alpha = GF2(1)

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2.prim_poly = conway_poly(2, 1)
