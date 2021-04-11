"""
A performant numpy extension for Galois fields.
"""
from .version import __version__

from .algorithm import factors, gcd, chinese_remainder_theorem
from .gf import GFArray
from .gf2 import GF2
from .modular import totatives, euler_totient, carmichael, is_cyclic, primitive_root, primitive_roots, is_primitive_root
from .poly import Poly, poly_gcd, poly_exp_mod, is_irreducible
from .prime import isqrt, primes, kth_prime, prev_prime, next_prime, mersenne_exponents, mersenne_primes, prime_factors, is_prime, fermat_primality_test, miller_rabin_primality_test


def GF(order, prim_poly=None, target="cpu", mode="auto", rebuild=False):  # pylint: disable=redefined-outer-name
    """
    Factory function to construct Galois field array classes of type :math:`\\mathrm{GF}(p^m)`.

    Parameters
    ----------
    order : int
        The order :math:`p^m` of the field :math:`\\mathrm{GF}(p^m)`. Order must be a prime power.
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
    type
        A new Galois field array class that is a subclass of :obj:`galois.GFArray`.

    Examples
    --------
    Construct various Galois field array classes.

    .. ipython:: python

        # Construct a GF(2^m) class
        GF256 = galois.GF(2**8); print(GF256)

        # Construct a GF(p) class
        GF571 = galois.GF(571); print(GF571)

        # Construct a very large GF(2^m) class
        GF2m = galois.GF(2**100); print(GF2m)

        # Construct a very large GF(p) class
        GFp = galois.GF(36893488147419103183); print(GFp)

    See :obj:`galois.GFArray` for more examples of what Galois field arrays can do.
    """
    # pylint: disable=import-outside-toplevel
    import numpy as np

    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not (prim_poly is None or isinstance(prim_poly, Poly)):
        raise TypeError(f"Argument `prim_poly` must be either None or galois.Poly, not {type(prim_poly)}.")
    if not isinstance(rebuild, bool):
        raise TypeError(f"Argument `rebuild` must be a bool, not {type(rebuild)}.")

    p, k = prime_factors(order)
    if not len(p) == 1:
        s = " + ".join([f"{pp}**{kk}" for pp, kk in zip(p, k)])
        raise ValueError(f"Argument `order` must be a prime power, not {order} = {s}.")
    characteristic = p[0]
    degree = k[0]

    # If the requested field has already been constructed, return it instead of rebuilding
    key = (characteristic, degree, mode)
    if not rebuild and key in GF.classes:
        return GF.classes[key]

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
    GF.classes[key] = cls

    return cls

GF.classes = {}


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


def conway_poly(p, n):
    """
    Returns the degree-:math:`n` Conway polynomial :math:`C_{p,n}` over :math:`\\mathrm{GF}(p)`.

    A Conway polynomial is a an irreducible polynomial over :math:`\\mathrm{GF}(p)` that provides a standard
    representation of :math:`\\mathrm{GF}(p^n)` as a splitting field of :math:`C_{p,n}`. Conway polynomials
    provide compatability between fields and their subfields, and hence are the common way to represent extension
    fields.

    The Conway polynomial :math:`C_{p,n}` is defined as the lexicographically-minimal monic irreducible polynomial
    of degree :math:`n` over :math:`\\mathrm{GF}(p)` that is compatible with all :math:`C_{p,m}`, for :math:`m` dividing
    :math:`n`.

    This function uses Frank Luebeck's Conway polynomial database for fast lookup, not construction.

    Parameters
    ----------
    p : int
        The prime characteristic of the field :math:`\\mathrm{GF}(p)`.
    n : int
        The degree :math:`n` of the Conway polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`n` Conway polynomial :math:`C_{p,n}` over :math:`\\mathrm{GF}(p)`.

    Raises
    ------
    LookupError
        If the Conway polynomial :math:`C_{p,n}` is not found in Frank Luebeck's database.

    Warning
    -------
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

    if not isinstance(p, (int, np.integer)):
        raise TypeError(f"Argument `p` must be an integer, not {type(p)}")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}")
    if not is_prime(p):
        raise ValueError(f"Argument `p` must be prime, not {p}")
    if not n >= 1:
        raise ValueError(f"Argument `n` must be at least 1, not {n}")

    coeffs = ConwayDatabase().fetch(p, n)
    if p == 2:
        field = GF2
    else:
        field = GF(p)
    poly = Poly(coeffs, field=field)

    return poly


# Create the default GF2 class and target the numba ufuncs for "cpu" (lowest overhead)
GF2.target("cpu")

GF2.alpha = GF2(1)

# Define the GF2 primitve polynomial here, not in gf2.py, to avoid a circular dependency.
# The primitive polynomial is p(x) = x - alpha, where alpha=1. Over GF2, this is equivalent
# to p(x) = x + 1
GF2.prim_poly = conway_poly(2, 1)
