"""
A module to implement the Galois field class factory `GF()`.
"""
import types
from typing import Sequence, Optional, Union, Type
from typing_extensions import Literal

import numpy as np

from .._modular import primitive_root, is_primitive_root
from .._overrides import set_module
from .._polys import Poly, conway_poly, primitive_element, is_irreducible, is_primitive_element
from .._prime import factors

from ._array import FieldArray
from ._gf2 import GF2
from ._gfp import GFpMeta
from ._gf2m import GF2mMeta
from ._gfpm import GFpmMeta

__all__ = ["GF", "Field"]

PolyLike = Union[int, str, Sequence[int], np.ndarray, FieldArray, Poly]


@set_module("galois")
def GF(
    order: int,
    irreducible_poly: Optional[PolyLike] = None,
    primitive_element: Optional[PolyLike] = None,
    verify: bool = True,
    compile: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    r"""
    Creates a :ref:`Galois field array class` for :math:`\mathrm{GF}(p^m)`.

    See :ref:`Galois Field Classes` for a detailed discussion of the relationship between :obj:`galois.FieldArrayClass` and
    :obj:`galois.FieldArray`.

    Parameters
    ----------
    order
        The order :math:`p^m` of the field :math:`\mathrm{GF}(p^m)`. The order must be a prime power.

    irreducible_poly
        Optionally specify an irreducible polynomial of degree :math:`m` over :math:`\mathrm{GF}(p)` that will
        define the Galois field arithmetic.

        * :obj:`None` (default): Uses the Conway polynomial :math:`C_{p,m}`. See :func:`galois.conway_poly`.
        * :obj:`int`: The integer representation of the irreducible polynomial.
        * :obj:`str`: The irreducible polynomial expressed as a string, e.g. `"x^2 + 1"`.
        * :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`: The irreducible polynomial coefficients in degree-descending order.
        * :obj:`galois.Poly`: The irreducible polynomial as a polynomial object.

    primitive_element
        Optionally specify a primitive element of the field. This value is used when building the exponential and logarithm
        lookup tables and when computing :obj:`numpy.log`. A primitive element is a generator of the multiplicative group of the
        field.

        .. tab-set::

            .. tab-item:: Prime fields

                For prime fields :math:`\mathrm{GF}(p)`, the primitive element must be an integer and is a primitive root modulo :math:`p`.

                * :obj:`None` (default): Uses the minimal primitive root modulo :math:`p`. See :func:`galois.primitive_root`.
                * :obj:`int`: A primitive root modulo :math:`p`.

            .. tab-item:: Extension fields

                For extension fields :math:`\mathrm{GF}(p^m)`, the primitive element is a polynomial of degree less than :math:`m` over
                :math:`\mathrm{GF}(p)`.

                * :obj:`None` (default): Uses the lexicographically-minimal primitive element. See :func:`galois.primitive_element`.
                * :obj:`int`: The integer representation of the primitive element.
                * :obj:`str`: The primitive element expressed as a string, e.g. `"x + 1"`.
                * :obj:`tuple`, :obj:`list`, :obj:`numpy.ndarray`: The primitive element's polynomial coefficients in degree-descending order.
                * :obj:`galois.Poly`: The primitive element as a polynomial object.

    verify
        Indicates whether to verify that the user-specified irreducible polynomial is in fact irreducible and that the user-specified
        primitive element is in fact a generator of the multiplicative group. The default is `True`.

        For large fields and irreducible polynomials that are already known to be irreducible (which may take a while to verify),
        this argument may be set to `False`.

        The default irreducible polynomial and primitive element are never verified because they are known to be irreducible and
        a multiplicative generator.

    compile
        The ufunc calculation mode. This can be modified after class construction with the :func:`galois.FieldArrayClass.compile` method.
        See :ref:`Compilation Modes` for a further discussion.

        * `None` (default): For newly-created classes, `None` corresponds to `"auto"`. For *Galois field array classes* of this type that were
          previously created, `None` does not modify the current ufunc compilation mode.
        * `"auto"`: Selects `"jit-lookup"` for fields with order less than :math:`2^{20}`, `"jit-calculate"` for larger fields, and `"python-calculate"`
          for fields whose elements cannot be represented with :obj:`numpy.int64`.
        * `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
          In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
        * `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"` mode is designed for large
          fields that cannot or should not store lookup tables in RAM. Generally, the `"jit-calculate"` mode is slower than `"jit-lookup"`.
        * `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
          represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_` with Python :obj:`int` (which has arbitrary precision).

    display
        The field element display representation. This can be modified after class construction with the :func:`galois.FieldArrayClass.display` method.
        See :ref:`Field Element Representation` for a further discussion.

        * `None` (default): For newly-created classes, `None` corresponds to `"int"`. For *Galois field array classes*
          of this type that were previously created, `None` does not modify the current display mode.
        * `"int"`: Sets the display mode to the :ref:`integer representation <Integer representation>`.
        * `"poly"`: Sets the display mode to the :ref:`polynomial representation <Polynomial representation>`.
        * `"power"`: Sets the display mode to the :ref:`power representation <Power representation>`.

    Returns
    -------
    :
        A *Galois field array class* for :math:`\mathrm{GF}(p^m)`. If this class has already been created, a reference to that class is returned.

    Notes
    -----
    The created *Galois field array class* is a subclass of :obj:`galois.FieldArray` and an instance of :obj:`galois.FieldArrayClass`.
    The :obj:`galois.FieldArray` inheritance provides the :obj:`numpy.ndarray` functionality and some additional methods on
    *Galois field arrays*. The :obj:`galois.FieldArrayClass` metaclass provides a variety of class attributes and methods relating to the
    finite field.

    *Galois field array classes* of the same type (order, irreducible polynomial, and primitive element) are singletons. So, calling this
    class factory with arguments that correspond to the same class will return the same class object.

    Examples
    --------
    Create a *Galois field array class* for each type of finite field.

    .. tab-set::

        .. tab-item:: GF(2)

            Construct the binary field.

            .. ipython:: python

                GF = galois.GF(2)
                print(GF)

        .. tab-item:: GF(p)

            Construct a prime field.

            .. ipython:: python

                GF = galois.GF(31)
                print(GF)

        .. tab-item:: GF(2^m)

            Construct a binary extension field. Notice the default irreducible polynomial is primitive and :math:`x`
            is a primitive element.

            .. ipython:: python

                GF = galois.GF(2**8)
                print(GF)

        .. tab-item:: GF(p^m)

            Construct a prime extension field. Notice the default irreducible polynomial is primitive and :math:`x`
            is a primitive element.

            .. ipython:: python

                GF = galois.GF(3**5)
                print(GF)

    Create a *Galois field array class* for extension fields and specify their irreducible polynomial.

    .. tab-set::

        .. tab-item:: GF(2^m)

            Construct the :math:`\mathrm{GF}(2^8)` field that is used in AES. Notice the irreducible polynomial is not primitive and
            :math:`x` is not a primitive element.

            .. ipython:: python

                GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
                print(GF)

        .. tab-item:: GF(p^m)

            Construct :math:`\mathrm{GF}(3^5)` with an irreducible, but not primitive, polynomial. Notice that :math:`x` is not a
            primitive element.

            .. ipython:: python

                GF = galois.GF(3**5, irreducible_poly="x^5 + 2x + 2")
                print(GF)

    Arbitrarily-large finite fields are also supported.

    .. tab-set::

        .. tab-item:: GF(p)

            Construct an arbitrarily-large prime field.

            .. ipython:: python

                GF = galois.GF(36893488147419103183)
                print(GF)

        .. tab-item:: GF(2^m)

            Construct an arbitrarily-large binary extension field.

            .. ipython:: python

                GF = galois.GF(2**100)
                print(GF)

        .. tab-item:: GF(p^m)

            Construct an arbitrarily-large prime extension field.

            .. ipython:: python

                GF = galois.GF(109987**4)
                print(GF)
    """
    # pylint: disable=redefined-outer-name,redefined-builtin
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
        return _GF_prime(p, primitive_element_=primitive_element, verify=verify, compile_=compile, display=display)
    else:
        return _GF_extension(p, m, irreducible_poly_=irreducible_poly, primitive_element_=primitive_element, verify=verify, compile_=compile, display=display)


@set_module("galois")
def Field(
    order: int,
    irreducible_poly: Optional[PolyLike] = None,
    primitive_element: Optional[PolyLike] = None,
    verify: bool = True,
    compile: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    """
    Alias of :func:`galois.GF`.
    """
    # pylint: disable=redefined-outer-name,redefined-builtin
    return GF(order, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, compile=compile, display=display)


def _GF_prime(
    characteristic: int,
    primitive_element_: Optional[PolyLike] = None,
    verify: bool = True,
    compile_: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    """
    Class factory for prime fields GF(p).
    """
    degree = 1
    order = characteristic**degree
    name = f"GF{characteristic}"

    # Get default primitive element
    if primitive_element_ is None:
        primitive_element_ = primitive_root(characteristic)

    # Check primitive element range
    if not 0 < primitive_element_ < order:
        raise ValueError(f"Argument `primitive_element` must be non-zero in the field 0 < x < {order}, not {primitive_element_}.")

    # If the requested field has already been constructed, return it
    key = (order, primitive_element_)
    if key in _GF_prime._classes:
        cls = _GF_prime._classes[key]
        if compile_ is not None:
            cls.compile(compile_)
        if display is not None:
            cls.display(display)
        return cls

    # Since this is a new class, set `compile` and `display` to their default values
    if compile_ is None:
        compile_ = "auto"
    if display is None:
        display = "int"

    if verify and not is_primitive_root(primitive_element_, characteristic):
        raise ValueError(f"Argument `primitive_element` must be a primitive root modulo {characteristic}, {primitive_element_} is not.")

    if characteristic == 2:
        cls = GF2
        cls.compile(compile_)
    else:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GFpMeta,
            "characteristic": characteristic,
            "degree": degree,
            "order": order,
            "is_primitive_poly": True,
            "primitive_element": primitive_element_,
            "compile": compile_
        })

    cls.__module__ = "galois"
    cls._irreducible_poly = Poly([1, -int(primitive_element_)], field=cls)
    cls.display(display)

    # Add class to dictionary of flyweights
    _GF_prime._classes[key] = cls

    return cls

_GF_prime._classes = {}


def _GF_extension(
    characteristic: int,
    degree: int,
    irreducible_poly_: Optional[PolyLike] = None,
    primitive_element_: Optional[PolyLike] = None,
    verify: bool = True,
    compile_: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    """
    Class factory for extension fields GF(p^m).
    """
    # pylint: disable=too-many-statements
    order = characteristic**degree
    name = f"GF{characteristic}_{degree}"
    prime_subfield = _GF_prime(characteristic)
    is_primitive_poly = None
    verify_poly = verify
    verify_element = verify

    # Get default irreducible polynomial
    if irreducible_poly_ is None:
        irreducible_poly_ = conway_poly(characteristic, degree)
        is_primitive_poly = True
        verify_poly = False  # We don't need to verify Conway polynomials are irreducible
        if primitive_element_ is None:
            primitive_element_ = Poly.Identity(prime_subfield)
            verify_element = False  # We know `g(x) = x` is a primitive element of the Conway polynomial because Conway polynomials are primitive polynomials
    else:
        irreducible_poly_ = Poly._PolyLike(irreducible_poly_, field=prime_subfield)

    # Get default primitive element
    if primitive_element_ is None:
        primitive_element_ = primitive_element(irreducible_poly_)
        verify_element = False
    else:
        primitive_element_ = Poly._PolyLike(primitive_element_, field=prime_subfield)

    # Check polynomial fields and degrees
    if not irreducible_poly_.field.order == characteristic:
        raise ValueError(f"Argument `irreducible_poly` must be over {prime_subfield.name}, not {irreducible_poly_.field.name}.")
    if not irreducible_poly_.degree == degree:
        raise ValueError(f"Argument `irreducible_poly` must have degree equal to {degree}, not {irreducible_poly_.degree}.")
    if not primitive_element_.field.order == characteristic:
        raise ValueError(f"Argument `primitive_element` must be a polynomial over {prime_subfield.name}, not {primitive_element_.field.name}.")
    if not primitive_element_.degree < degree:
        raise ValueError(f"Argument `primitive_element` must have degree strictly less than {degree}, not {primitive_element_.degree}.")

    # If the requested field has already been constructed, return it
    key = (order, int(primitive_element_), int(irreducible_poly_))
    if key in _GF_extension._classes:
        cls = _GF_extension._classes[key]
        if compile_ is not None:
            cls.compile(compile_)
        if display is not None:
            cls.display(display)
        return cls

    # Since this is a new class, set `compile` and `display` to their default values
    if compile_ is None:
        compile_ = "auto"
    if display is None:
        display = "int"

    if verify_poly and not is_irreducible(irreducible_poly_):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly_} is not.")
    if verify_element and not is_primitive_element(primitive_element_, irreducible_poly_):
        raise ValueError(f"Argument `primitive_element` must be a multiplicative generator of GF({characteristic}^{degree}), {primitive_element_} is not.")

    if characteristic == 2:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GF2mMeta,
            "characteristic": characteristic,
            "degree": degree,
            "order": order,
            "irreducible_poly": irreducible_poly_,
            "is_primitive_poly": is_primitive_poly,
            "primitive_element": int(primitive_element_),
            "prime_subfield": prime_subfield,
            "compile": compile_
        })
    else:
        cls = types.new_class(name, bases=(FieldArray,), kwds={
            "metaclass": GFpmMeta,
            "characteristic": characteristic,
            "degree": degree,
            "order": order,
            "irreducible_poly": irreducible_poly_,
            "is_primitive_poly": is_primitive_poly,
            "primitive_element": int(primitive_element_),
            "prime_subfield": prime_subfield,
            "compile": compile_
        })

    cls.__module__ = "galois"
    cls.display(display)

    # Add class to dictionary of flyweights
    _GF_extension._classes[key] = cls

    return cls

_GF_extension._classes = {}
