from .._factor import factors
from .._overrides import set_module

from ._factory_extension import GF_extension
from ._factory_prime import GF_prime

__all__ = ["GF", "Field"]


@set_module("galois")
def GF(order, irreducible_poly=None, primitive_element=None, verify=True, mode="auto"):
    r"""
    Factory function to construct a Galois field array class for :math:`\mathrm{GF}(p^m)`.

    The created class will be a subclass of :obj:`galois.FieldArray` and instance of :obj:`galois.FieldClass`.
    The :obj:`galois.FieldArray` inheritance provides the :obj:`numpy.ndarray` functionality. The :obj:`galois.FieldClass` metaclass
    provides a variety of class attributes and methods relating to the finite field.

    Parameters
    ----------
    order : int
        The order :math:`p^m` of the field :math:`\mathrm{GF}(p^m)`. The order must be a prime power.
    irreducible_poly : int, tuple, list, numpy.ndarray, galois.Poly, optional
        Optionally specify an irreducible polynomial of degree :math:`m` over :math:`\mathrm{GF}(p)` that will
        define the Galois field arithmetic. An integer may be provided, which is the integer representation of the
        irreducible polynomial. A tuple, list, or ndarray may be provided, which represents the polynomial coefficients in
        degree-descending order. The default is `None` which uses the Conway polynomial :math:`C_{p,m}` obtained from :func:`galois.conway_poly`.
    primitive_element : int, tuple, list, numpy.ndarray, galois.Poly, optional
        Optionally specify a primitive element of the field :math:`\mathrm{GF}(p^m)`. A primitive element is a generator of
        the multiplicative group of the field. For prime fields :math:`\mathrm{GF}(p)`, the primitive element must be an integer
        and is a primitive root modulo :math:`p`. For extension fields :math:`\mathrm{GF}(p^m)`, the primitive element is a polynomial
        of degree less than :math:`m` over :math:`\mathrm{GF}(p)`. An integer may be provided, which is the integer representation of the polynomial.
        A tuple, list, or ndarray may be provided, which represents the polynomial coefficients in degree-descending order. The default is `None`
        which uses :obj:`galois.primitive_root(p)` for prime fields and :obj:`galois.primitive_element(irreducible_poly)` for extension fields.
    verify : bool, optional
        Indicates whether to verify that the specified irreducible polynomial is in fact irreducible and that the specified primitive element
        is in fact a generator of the multiplicative group. The default is `True`. For large fields and irreducible polynomials that are already
        known to be irreducible (and may take a long time to verify), this argument can be set to `False`. If the default irreducible polynomial
        and primitive element are used, no verification is performed because the defaults are already guaranteed to be irreducible and a multiplicative
        generator, respectively.
    mode : str, optional
        The type of field computation, either `"auto"`, `"jit-lookup"`, or `"jit-calculate"`. The default is `"auto"`.
        The "jit-lookup" mode will use Zech log, log, and anti-log lookup tables for efficient calculation. The "jit-calculate"
        mode will not store any lookup tables, but instead perform field arithmetic on the fly. The "jit-calculate" mode is
        designed for large fields that cannot or should not store lookup tables in RAM. Generally, "jit-calculate" mode will
        be slower than "jit-lookup". The "auto" mode will determine whether to use "jit-lookup" or "jit-calculate" based on the field's
        size. In "auto" mode, field's with `order <= 2**20` will use the "jit-lookup" mode.

    Returns
    -------
    galois.FieldClass
        A new Galois field array class that is a subclass of :obj:`galois.FieldArray` and instance of :obj:`galois.FieldClass`.

    Examples
    --------
    Construct a Galois field array class with default irreducible polynomial and primitive element.

    .. ipython:: python

        # Construct a GF(2^m) class
        GF256 = galois.GF(2**8)

        # Notice the irreducible polynomial is primitive
        print(GF256.properties)

        poly = GF256.irreducible_poly

    Construct a Galois field specifying a specific irreducible polynomial.

    .. ipython:: python

        # Field used in AES
        GF256_AES = galois.GF(2**8, irreducible_poly=galois.Poly.Degrees([8,4,3,1,0]))
        print(GF256_AES.properties)

        # Construct a GF(p) class
        GF571 = galois.GF(571); print(GF571.properties)

        # Construct a very large GF(2^m) class
        GF2m = galois.GF(2**100); print(GF2m.properties)

        # Construct a very large GF(p) class
        GFp = galois.GF(36893488147419103183); print(GFp.properties)

    See :obj:`galois.FieldArray` for more examples of what Galois field arrays can do.
    """
    if not isinstance(order, int):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    p, e = factors(order)
    if not len(p) == len(e) == 1:
        s = " + ".join([f"{pi}**{ei}" for pi, ei in zip(p, e)])
        raise ValueError(f"Argument `order` must be a prime power, not {order} = {s}.")
    p, m = p[0], e[0]

    if m == 1:
        if not irreducible_poly is None:
            raise ValueError(f"Argument `irreducible_poly` can only be specified for extension fields, not the prime field GF({p}).")
        return GF_prime(p, primitive_element=primitive_element, verify=verify, mode=mode)
    else:
        return GF_extension(p, m, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, mode=mode)


@set_module("galois")
def Field(order, irreducible_poly=None, primitive_element=None, verify=True, mode="auto"):
    """
    Alias of :func:`galois.GF`.
    """
    return GF(order, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, mode=mode)
