from .._factor import factors
from .._overrides import set_module

from ._factory_extension import GF_extension
from ._factory_prime import GF_prime

__all__ = ["GF", "Field"]

# pylint: disable=redefined-builtin


@set_module("galois")
def GF(order, irreducible_poly=None, primitive_element=None, verify=True, compile=None, display=None):
    r"""
    Factory function to construct a Galois field array class for :math:`\mathrm{GF}(p^m)`.

    Parameters
    ----------
    order : int
        The order :math:`p^m` of the field :math:`\mathrm{GF}(p^m)`. The order must be a prime power.

    irreducible_poly : int, str, tuple, list, numpy.ndarray, galois.Poly, optional
        Optionally specify an irreducible polynomial of degree :math:`m` over :math:`\mathrm{GF}(p)` that will
        define the Galois field arithmetic.

        * :obj:`None` (default): Uses the Conway polynomial :math:`C_{p,m}`, see :func:`galois.conway_poly`.
        * :obj:`int`: The integer representation of the irreducible polynomial.
        * :obj:`str`: The irreducible polynomial expressed as a string, e.g. `"x^2 + 1"`.
        * :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`: The irreducible polynomial coefficients in degree-descending order.
        * :obj:`galois.Poly`: The irreducible polynomial as a polynomial object.

    primitive_element : int, str, tuple, list, numpy.ndarray, galois.Poly, optional
        Optionally specify a primitive element of the field :math:`\mathrm{GF}(p^m)`. This value is used when building the log/anti-log
        lookup tables and when computing :func:`np.log`. A primitive element is a generator of the multiplicative group of the field.
        For prime fields :math:`\mathrm{GF}(p)`, the primitive element must be an integer and is a primitive root modulo :math:`p`. For extension
        fields :math:`\mathrm{GF}(p^m)`, the primitive element is a polynomial of degree less than :math:`m` over :math:`\mathrm{GF}(p)`.

        **For prime fields:**

        * :obj:`None` (default): Uses the minimal primitive root modulo :math:`p`, see :func:`galois.primitive_root`.
        * :obj:`int`: A primitive root modulo :math:`p`.

        **For extension fields:**

        * :obj:`None` (default): Uses the lexicographically-minimal primitive element, see :func:`galois.primitive_element`.
        * :obj:`int`: The integer representation of the primitive element.
        * :obj:`str`: The primitive element expressed as a string, e.g. `"x + 1"`.
        * :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`: The primitive element's polynomial coefficients in degree-descending order.
        * :obj:`galois.Poly`: The primitive element as a polynomial object.

    verify : bool, optional
        Indicates whether to verify that the specified irreducible polynomial is in fact irreducible and whether the specified primitive element
        is in fact a generator of the multiplicative group. The default is `True`. For large fields and irreducible polynomials that are already
        known to be irreducible (which may take a long time to verify), this argument can be set to `False`. If the default irreducible polynomial
        and primitive element are used, no verification is performed because the defaults are guaranteed to be irreducible and a multiplicative
        generator, respectively.

    compile : str, optional
        The ufunc calculation mode. This can be modified after class construction with the :func:`galois.FieldClass.compile` method.

        * `None` (default): For newly-created classes, `None` corresponds to `"auto"`. For Galois field array classes of this type that were
          previously created, `None` does not modify the current ufunc compilation mode.
        * `"auto"`: Selects "jit-lookup" for fields with order less than :math:`2^{20}`, "jit-calculate" for larger fields, and "python-calculate"
          for fields whose elements cannot be represented with :obj:`numpy.int64`.
        * `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
          In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
        * `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The "jit-calculate" mode is designed for large
          fields that cannot or should not store lookup tables in RAM. Generally, the "jit-calculate" mode is slower than "jit-lookup".
        * `"python-calculate"`: Uses pure-python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
          represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_` with python :obj:`int` (which has arbitrary precision).

    display : str, optional
        The field element display representation. This can be modified after class consstruction with the :func:`galois.FieldClass.display` method.

        * `None` (default): For newly-created classes, `None` corresponds to the integer representation (`"int"`). For Galois field array classes
          of this type that were previously created, `None` does not modify the current display mode.
        * `"int"`: The element displayed as the integer representation of the polynomial. For example, :math:`2x^2 + x + 2` is an element of
          :math:`\mathrm{GF}(3^3)` and is equivalent to the integer :math:`23 = 2 \cdot 3^2 + 3 + 2`.
        * `"poly"`: The element as a polynomial over :math:`\mathrm{GF}(p)` of degree less than :math:`m`. For example, :math:`2x^2 + x + 2` is an element
          of :math:`\mathrm{GF}(3^3)`.
        * `"power"`: The element as a power of the primitive element, see :obj:`galois.FieldClass.primitive_element`. For example, :math:`2x^2 + x + 2 = \alpha^5`
          in :math:`\mathrm{GF}(3^3)` with irreducible polynomial :math:`x^3 + 2x + 1` and primitive element :math:`\alpha = x`.

    Returns
    -------
    galois.FieldClass
        A Galois field array class for :math:`\mathrm{GF}(p^m)`. If this class has already been created, a reference to that class is returned.

    Notes
    -----
    The created class is a subclass of :obj:`galois.FieldArray` and an instance of :obj:`galois.FieldClass`.
    The :obj:`galois.FieldArray` inheritance provides the :obj:`numpy.ndarray` functionality and some additional methods on
    Galois field arrays, such as :func:`galois.FieldArray.row_reduce`. The :obj:`galois.FieldClass` metaclass provides a variety
    of class attributes and methods relating to the finite field, such as the :func:`galois.FieldClass.display` method to
    change the field element display representation.

    Galois field array classes of the same type (order, irreducible polynomial, and primitive element) are singletons. So, calling this
    class factory with arguments that correspond to the same class will return the same class object.

    Examples
    --------
    Construct various Galois field array class for :math:`\mathrm{GF}(2)`, :math:`\mathrm{GF}(2^m)`, :math:`\mathrm{GF}(p)`, and :math:`\mathrm{GF}(p^m)`
    with the default irreducible polynomials and primitive elements. For the extension fields, notice the irreducible polynomials are primitive and
    :math:`x` is a primitive element.

    .. ipython:: python

        # Construct a GF(2) class
        GF2 = galois.GF(2); print(GF2.properties)

        # Construct a GF(2^m) class
        GF256 = galois.GF(2**8); print(GF256.properties)

        # Construct a GF(p) class
        GF3 = galois.GF(3); print(GF3.properties)

        # Construct a GF(p^m) class
        GF243 = galois.GF(3**5); print(GF243.properties)

    Or construct a Galois field array class and specify the irreducible polynomial. Here is an example using the :math:`\mathrm{GF}(2^8)`
    field from AES. Notice the irreducible polynomial is not primitive and :math:`x` is not a primitive element.

    .. ipython:: python

        poly = galois.Poly.Degrees([8,4,3,1,0]); poly
        GF256_AES = galois.GF(2**8, irreducible_poly=poly)
        print(GF256_AES.properties)

    Very large fields are also supported but they use :obj:`numpy.object_` dtypes with python :obj:`int` and, therefore, do not have compiled ufuncs.

    .. ipython:: python

        # Construct a very large GF(2^m) class
        GF2m = galois.GF(2**100); print(GF2m.properties)
        GF2m.dtypes, GF2m.ufunc_mode

        # Construct a very large GF(p) class
        GFp = galois.GF(36893488147419103183); print(GFp.properties)
        GFp.dtypes, GFp.ufunc_mode

    The default display mode for field elements is the integer representation. This can be modified by using the `display` keyword argument. It
    can also be changed after class construction by calling the :func:`galois.FieldClass.display` method.

    .. ipython:: python

        GF = galois.GF(2**8)
        GF.Random()
        GF = galois.GF(2**8, display="poly")
        GF.Random()
        @suppress
        GF.display();

    Galois field array classes of the same type (order, irreducible polynomial, and primitive element) are singletons. So, calling this
    class factory with arguments that correspond to the same class will return the same field class object.

    .. ipython:: python

        poly1 = galois.Poly([1, 0, 0, 0, 1, 1, 0, 1, 1])
        poly2 = poly1.integer
        galois.GF(2**8, irreducible_poly=poly1) is galois.GF(2**8, irreducible_poly=poly2)

    See :obj:`galois.FieldArray` and :obj:`galois.FieldClass` for more examples of what Galois field arrays can do.
    """
    if not isinstance(order, int):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(verify, bool):
        raise TypeError(f"Argument `verify` must be a bool, not {type(verify)}.")
    if not isinstance(compile, (type(None), str)):
        raise TypeError(f"Argument `compile` must be a string, not {type(compile)}.")
    if not isinstance(display, (type(None), str)):
        raise TypeError(f"Argument `display` must be a string, not {type(display)}.")

    p, e = factors(order)
    if not len(p) == len(e) == 1:
        s = " + ".join([f"{pi}**{ei}" for pi, ei in zip(p, e)])
        raise ValueError(f"Argument `order` must be a prime power, not {order} = {s}.")
    if not compile in [None, "auto", "jit-lookup", "jit-calculate", "python-calculate"]:
        raise ValueError(f"Argument `compile` must be in ['auto', 'jit-lookup', 'jit-calculate', 'python-calculate'], not {compile!r}.")
    if not display in [None, "int", "poly", "power"]:
        raise ValueError(f"Argument `display` must be in ['int', 'poly', 'power'], not {display!r}.")

    p, m = p[0], e[0]

    if m == 1:
        if not irreducible_poly is None:
            raise ValueError(f"Argument `irreducible_poly` can only be specified for extension fields, not the prime field GF({p}).")
        return GF_prime(p, primitive_element=primitive_element, verify=verify, compile=compile, display=display)
    else:
        return GF_extension(p, m, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, compile=compile, display=display)


@set_module("galois")
def Field(order, irreducible_poly=None, primitive_element=None, verify=True, compile=None, display=None):
    """
    Alias of :func:`galois.GF`.
    """
    return GF(order, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, compile=compile, display=display)
