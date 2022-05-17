"""
A module to implement the Galois field class factory `GF()`.
"""
from __future__ import annotations

import types
from typing import Optional, Type
from typing_extensions import Literal

from .._modular import primitive_root, is_primitive_root
from .._overrides import set_module
from .._polys import Poly, conway_poly, primitive_element, is_primitive_element
from .._prime import factors
from ..typing import PolyLike

from ._array import FieldArray
from ._gf2 import GF2
from ._ufunc import UFuncMixin_2_m, UFuncMixin_p_1, UFuncMixin_p_m

__all__ = ["GF", "Field"]


@set_module("galois")
def GF(
    order: int,
    irreducible_poly: Optional[PolyLike] = None,
    primitive_element: Optional[PolyLike] = None,  # pylint: disable=redefined-outer-name
    verify: bool = True,
    compile: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,  # pylint: disable=redefined-builtin
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    r"""
    Creates a :obj:`~galois.FieldArray` subclass for :math:`\mathrm{GF}(p^m)`.

    Parameters
    ----------
    order
        The order :math:`p^m` of the field :math:`\mathrm{GF}(p^m)`. The order must be a prime power.
    irreducible_poly
        Optionally specify an irreducible polynomial of degree :math:`m` over :math:`\mathrm{GF}(p)` that
        defines the finite field arithmetic. The default is `None` which uses the Conway polynomial :math:`C_{p,m}`,
        see :func:`~galois.conway_poly`.
    primitive_element
        Optionally specify a primitive element of the field. This value is used when building the exponential and logarithm
        lookup tables and as the base of :obj:`numpy.log`. A primitive element is a generator of the multiplicative group of the
        field.

        For prime fields :math:`\mathrm{GF}(p)`, the primitive element must be an integer and is a primitive root modulo :math:`p`.
        The default is `None` which uses :func:`~galois.primitive_root`.

        For extension fields :math:`\mathrm{GF}(p^m)`, the primitive element is a polynomial of degree less than :math:`m` over
        :math:`\mathrm{GF}(p)`. The default is `None` which uses :func:`~galois.primitive_element`.
    verify
        Indicates whether to verify that the user-provided irreducible polynomial is in fact irreducible and that the user-provided
        primitive element is in fact a generator of the multiplicative group. The default is `True`.

        For large fields and irreducible polynomials that are already known to be irreducible (which may take a while to verify),
        this argument may be set to `False`.

        The default irreducible polynomial and primitive element are never verified because they are already known to be irreducible
        and a multiplicative generator, respectively.
    compile
        The ufunc calculation mode. This can be modified after class construction with the :func:`~galois.FieldArray.compile` method.
        See :doc:`/basic-usage/compilation-modes` for a further discussion.

        - `None` (default): For a newly-created :obj:`~galois.FieldArray` subclass, `None` corresponds to `"auto"`. If the
          :obj:`~galois.FieldArray` subclass already exists, `None` does not modify its current compilation mode.
        - `"auto"`: Selects `"jit-lookup"` for fields with order less than :math:`2^{20}`, `"jit-calculate"` for larger fields, and `"python-calculate"`
          for fields whose elements cannot be represented with :obj:`numpy.int64`.
        - `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
          In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
        - `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"` mode is designed for large
          fields that cannot or should not store lookup tables in RAM. Generally, the `"jit-calculate"` mode is slower than `"jit-lookup"`.
        - `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
          represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_` with Python :obj:`int` (which has arbitrary precision).

    display
        The field element display representation. This can be modified after class construction with the :func:`~galois.FieldArray.display` method.
        See :doc:`/basic-usage/element-representation` for a further discussion.

        - `None` (default): For a newly-created :obj:`~galois.FieldArray` subclass, `None` corresponds to `"int"`. If the
          :obj:`~galois.FieldArray` subclass already exists, `None` does not modify its current display mode.
        - `"int"`: Sets the display mode to the :ref:`integer representation <int-repr>`.
        - `"poly"`: Sets the display mode to the :ref:`polynomial representation <poly-repr>`.
        - `"power"`: Sets the display mode to the :ref:`power representation <power-repr>`.

    Returns
    -------
    :
        A :obj:`~galois.FieldArray` subclass for :math:`\mathrm{GF}(p^m)`.

    Notes
    -----
    :obj:`~galois.FieldArray` subclasses of the same type (order, irreducible polynomial, and primitive element) are singletons. So,
    calling this class factory with arguments that correspond to the same subclass will return the same class object.

    Examples
    --------
    Create a :obj:`~galois.FieldArray` subclass for each type of finite field.

    .. tab-set::

        .. tab-item:: GF(2)

            Construct the binary field.

            .. ipython:: python

                GF = galois.GF(2)
                print(GF.properties)

        .. tab-item:: GF(p)

            Construct a prime field.

            .. ipython:: python

                GF = galois.GF(31)
                print(GF.properties)

        .. tab-item:: GF(2^m)

            Construct a binary extension field. Notice the default irreducible polynomial is primitive and :math:`x`
            is a primitive element.

            .. ipython:: python

                GF = galois.GF(2**8)
                print(GF.properties)

        .. tab-item:: GF(p^m)

            Construct a prime extension field. Notice the default irreducible polynomial is primitive and :math:`x`
            is a primitive element.

            .. ipython:: python

                GF = galois.GF(3**5)
                print(GF.properties)

    Create a :obj:`~galois.FieldArray` subclass for extension fields and specify their irreducible polynomials.

    .. tab-set::

        .. tab-item:: GF(2^m)

            Construct the :math:`\mathrm{GF}(2^8)` field that is used in AES. Notice the irreducible polynomial is not primitive and
            :math:`x` is not a primitive element.

            .. ipython:: python

                GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
                print(GF.properties)

        .. tab-item:: GF(p^m)

            Construct :math:`\mathrm{GF}(3^5)` with an irreducible, but not primitive, polynomial. Notice that :math:`x` is not a
            primitive element.

            .. ipython:: python

                GF = galois.GF(3**5, irreducible_poly="x^5 + 2x + 2")
                print(GF.properties)

    Finite fields with arbitrarily-large orders are supported.

    .. tab-set::

        .. tab-item:: GF(p)

            Construct a large prime field.

            .. ipython:: python

                GF = galois.GF(36893488147419103183)
                print(GF.properties)

        .. tab-item:: GF(2^m)

            Construct a large binary extension field.

            .. ipython:: python

                GF = galois.GF(2**100)
                print(GF.properties)

        .. tab-item:: GF(p^m)

            Construct a large prime extension field.

            .. ipython:: python

                GF = galois.GF(109987**4)
                print(GF.properties)
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
        return _GF_prime(p, alpha=primitive_element, verify=verify, compile=compile, display=display)
    else:
        return _GF_extension(p, m, irreducible_poly_=irreducible_poly, alpha=primitive_element, verify=verify, compile=compile, display=display)


@set_module("galois")
def Field(
    order: int,
    irreducible_poly: Optional[PolyLike] = None,
    primitive_element: Optional[PolyLike] = None,  # pylint: disable=redefined-outer-name
    verify: bool = True,
    compile: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,  # pylint: disable=redefined-builtin
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    """
    Alias of :func:`~galois.GF`.
    """
    return GF(order, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, compile=compile, display=display)


def _GF_prime(
    p: int,
    alpha: Optional[int] = None,
    verify: bool = True,
    compile: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,  # pylint: disable=redefined-builtin
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    """
    Class factory for prime fields GF(p).
    """
    name = f"GF({p})"

    # Get default primitive element
    if alpha is None:
        alpha = primitive_root(p)

    # Check primitive element range
    if not 0 < alpha < p:
        raise ValueError(f"Argument `primitive_element` must be non-zero in the field 0 < x < {p}, not {alpha}.")

    # If the requested field has already been constructed, return it
    key = (p, alpha)
    if key in _GF_prime._classes:
        field = _GF_prime._classes[key]
        if compile is not None:
            field.compile(compile)
        if display is not None:
            field.display(display)
        return field

    if verify and not is_primitive_root(alpha, p):
        raise ValueError(f"Argument `primitive_element` must be a primitive root modulo {p}, {alpha} is not.")

    if p == 2:
        field = GF2
    else:
        field = types.new_class(name, bases=(FieldArray, UFuncMixin_p_1), kwds={
            "p": p,
            "m": 1,
            "characteristic": p,
            "degree": 1,
            "order": p,
            "irreducible_poly_int": 2*p - alpha,  # f(x) = x - e
            "primitive_element": alpha,
        })

    # Add the class to the "galois" namespace
    field.__module__ = "galois"

    # Since this is a new class, compile the ufuncs and set the display mode
    field.compile("auto" if compile is None else compile)
    field.display("int" if display is None else display)

    field._is_primitive_poly = field._irreducible_poly(field._primitive_element, field=field) == 0

    # Add class to dictionary of flyweights
    _GF_prime._classes[key] = field

    return field

_GF_prime._classes = {}


def _GF_extension(
    p: int,
    m: int,
    irreducible_poly_: Optional[PolyLike] = None,
    alpha: Optional[PolyLike] = None,
    verify: bool = True,
    compile: Optional[Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]] = None,  # pylint: disable=redefined-builtin
    display: Optional[Literal["int", "poly", "power"]] = None
) -> Type[FieldArray]:
    """
    Class factory for extension fields GF(p^m).
    """
    # pylint: disable=too-many-statements
    name = f"GF({p}^{m})"
    prime_subfield = _GF_prime(p)
    is_primitive_poly = None
    verify_poly = verify
    verify_element = verify

    # Get default irreducible polynomial
    if irreducible_poly_ is None:
        irreducible_poly_ = conway_poly(p, m)
        is_primitive_poly = True
        verify_poly = False  # We don't need to verify Conway polynomials are irreducible
        if alpha is None:
            alpha = Poly.Identity(prime_subfield)
            verify_element = False  # We know `g(x) = x` is a primitive element of the Conway polynomial because Conway polynomials are primitive polynomials
    else:
        irreducible_poly_ = Poly._PolyLike(irreducible_poly_, field=prime_subfield)

    # Get default primitive element
    if alpha is None:
        alpha = primitive_element(irreducible_poly_)
        verify_element = False
    else:
        alpha = Poly._PolyLike(alpha, field=prime_subfield)

    # Check polynomial fields and degrees
    if not irreducible_poly_.field.order == p:
        raise ValueError(f"Argument `irreducible_poly` must be over {prime_subfield.name}, not {irreducible_poly_.field.name}.")
    if not irreducible_poly_.degree == m:
        raise ValueError(f"Argument `irreducible_poly` must have degree equal to {m}, not {irreducible_poly_.degree}.")
    if not alpha.field.order == p:
        raise ValueError(f"Argument `primitive_element` must be a polynomial over {prime_subfield.name}, not {alpha.field.name}.")
    if not alpha.degree < m:
        raise ValueError(f"Argument `primitive_element` must have degree strictly less than {m}, not {alpha.degree}.")

    # If the requested field has already been constructed, return it
    key = (p, m, int(alpha), int(irreducible_poly_))
    if key in _GF_extension._classes:
        field = _GF_extension._classes[key]
        if compile is not None:
            field.compile(compile)
        if display is not None:
            field.display(display)
        return field

    if verify_poly and not irreducible_poly_.is_irreducible():
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly_} is not.")
    if verify_element and not is_primitive_element(alpha, irreducible_poly_):
        raise ValueError(f"Argument `primitive_element` must be a multiplicative generator of {name}, {alpha} is not.")

    ufunc_mixin = UFuncMixin_2_m if p == 2 else UFuncMixin_p_m

    field = types.new_class(name, bases=(FieldArray, ufunc_mixin), kwds={
        "p": p,
        "m": m,
        "characteristic": p,
        "degree": m,
        "order": p**m,
        "irreducible_poly_int": int(irreducible_poly_),
        "primitive_element": int(alpha),
        "prime_subfield": prime_subfield,
    })

    # Add the class to the "galois" namespace
    field.__module__ = "galois"

    # Since this is a new class, compile the ufuncs and set the display mode
    field.compile("auto" if compile is None else compile)
    field.display("int" if display is None else display)

    if is_primitive_poly is not None:
        field._is_primitive_poly = is_primitive_poly
    else:
        field._is_primitive_poly = field._irreducible_poly(field._primitive_element, field=field) == 0

    # Add class to dictionary of flyweights
    _GF_extension._classes[key] = field

    return field

_GF_extension._classes = {}
