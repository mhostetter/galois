"""
A module to implement the Galois field class factory `GF()`. This module also includes functions to generate
irreducible, primitive, and Conway polynomials. They are included here due to a circular dependence with the
Galois field class factory.
"""
import functools
import random
import types
from typing import Sequence, List, Optional, Union, Iterator, Type
from typing_extensions import Literal

import numpy as np

from .._databases import ConwayPolyDatabase
from .._modular import totatives, primitive_root, is_primitive_root
from .._overrides import set_module
from .._prime import factors, is_prime, is_prime_power

from . import _poly_functions as poly_functions
from ._gfp import GFpMeta
from ._gf2m import GF2mMeta
from ._gfpm import GFpmMeta
from ._main import FieldArray, GF2, Poly

__all__ = [
    "GF", "Field",
    "irreducible_poly", "irreducible_polys", "is_irreducible",
    "primitive_poly", "primitive_polys", "is_primitive",
    "matlab_primitive_poly", "conway_poly",
    "primitive_element", "primitive_elements", "is_primitive_element",
]

PolyLike = Union[int, str, Sequence[int], np.ndarray, FieldArray, Poly]

###############################################################################
# Construct Galois field array classes
###############################################################################

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

    See :ref:`Galois Field Classes` for a detailed discussion of the relationship between :obj:`galois.FieldClass` and
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
        The ufunc calculation mode. This can be modified after class construction with the :func:`galois.FieldClass.compile` method.
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
        The field element display representation. This can be modified after class construction with the :func:`galois.FieldClass.display` method.
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
    The created *Galois field array class* is a subclass of :obj:`galois.FieldArray` and an instance of :obj:`galois.FieldClass`.
    The :obj:`galois.FieldArray` inheritance provides the :obj:`numpy.ndarray` functionality and some additional methods on
    *Galois field arrays*. The :obj:`galois.FieldClass` metaclass provides a variety of class attributes and methods relating to the
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
        return GF_prime(p, primitive_element_=primitive_element, verify=verify, compile_=compile, display=display)
    else:
        return GF_extension(p, m, irreducible_poly_=irreducible_poly, primitive_element_=primitive_element, verify=verify, compile_=compile, display=display)


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


def GF_prime(
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
    if key in GF_prime._classes:
        cls = GF_prime._classes[key]
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
    GF_prime._classes[key] = cls

    return cls

GF_prime._classes = {}


def GF_extension(
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
    prime_subfield = GF_prime(characteristic)
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
    if key in GF_extension._classes:
        cls = GF_extension._classes[key]
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
    GF_extension._classes[key] = cls

    return cls

GF_extension._classes = {}


###############################################################################
# Irreducible polynomials
###############################################################################

@set_module("galois")
def irreducible_poly(order: int, degree: int, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired irreducible polynomial.
    method
        The search method for finding the irreducible polynomial.

        * `"min"` (default): Returns the lexicographically-minimal monic irreducible polynomial.
        * `"max"`: Returns the lexicographically-maximal monic irreducible polynomial.
        * `"random"`: Returns a randomly generated degree-:math:`m` monic irreducible polynomial.

    Returns
    -------
    :
        The degree-:math:`m` monic irreducible polynomial over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    is_irreducible, primitive_poly, conway_poly

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Search methods

            Find the lexicographically-minimal monic irreducible polynomial.

            .. ipython:: python

                galois.irreducible_poly(7, 3)

            Find the lexicographically-maximal monic irreducible polynomial.

            .. ipython:: python

                galois.irreducible_poly(7, 3, method="max")

            Find a random monic irreducible polynomial.

            .. ipython:: python

                galois.irreducible_poly(7, 3, method="random")

        .. tab-item:: Properties

            Find a random monic irreducible polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

            .. ipython:: python

                f = galois.irreducible_poly(7, 5, method="random"); f
                galois.is_irreducible(f)

            Monic irreducible polynomials scaled by non-zero field elements (now non-monic) are also irreducible.

            .. ipython:: python

                GF = galois.GF(7)
                g = f * GF(3); g
                galois.is_irreducible(g)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no irreducible polynomials with degree 0.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    if method == "min":
        return next(irreducible_polys(order, degree))
    elif method == "max":
        return next(irreducible_polys(order, degree, reverse=True))
    else:
        return _irreducible_poly_random_search(order, degree)


@set_module("galois")
def irreducible_polys(order: int, degree: int, reverse: bool = False) -> Iterator[Poly]:
    r"""
    Iterates through all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired irreducible polynomial.
    reverse
        Indicates to return the irreducible polynomials from lexicographically maximal to minimal. The default is `False`.

    Returns
    -------
    :
        An iterator over all degree-:math:`m` monic irreducible polynomials over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    is_irreducible, primitive_polys

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Return full list

            All monic irreducible polynomials over :math:`\mathrm{GF}(3)` with degree :math:`4`. You may also use :func:`tuple` on
            the returned generator.

            .. ipython:: python

                list(galois.irreducible_polys(3, 4))

        .. tab-item:: For loop

            Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the polynomials that would
            have been found after the `break` condition is never incurred.

            .. ipython:: python

                for poly in galois.irreducible_polys(3, 4, reverse=True):
                    if poly.coeffs[1] < 2:  # Early exit condition
                        break
                    print(poly)

        .. tab-item:: Manual iteration

            Or, manually iterate over the generator.

            .. ipython:: python

                generator = galois.irreducible_polys(3, 4, reverse=True); generator
                next(generator)
                next(generator)
                next(generator)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 0:
        raise ValueError(f"Argument `degree` must be at least 0, not {degree}.")

    field = GF(order)

    # Only search monic polynomials of degree m over GF(q)
    start = order**degree
    stop = 2*order**degree
    step = 1

    if reverse:
        start, stop, step = stop - 1, start - 1, -1

    while True:
        poly = _irreducible_poly_deterministic_search(field, start, stop, step)
        if poly is not None:
            start = int(poly) + step
            yield poly
        else:
            break


@functools.lru_cache(maxsize=4096)
def _irreducible_poly_deterministic_search(field, start, stop, step) -> Optional[Poly]:
    """
    Searches for an irreducible polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if is_irreducible(poly):
            return poly

    return None


def _irreducible_poly_random_search(order, degree) -> Poly:
    """
    Searches for a random irreducible polynomial.
    """
    field = GF(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2*order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if is_irreducible(poly):
            return poly


@set_module("galois")
def is_irreducible(poly: Poly) -> bool:
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)` is irreducible.

    Parameters
    ----------
    poly
        A polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    :
        `True` if the polynomial is irreducible.

    See Also
    --------
    is_primitive, irreducible_poly, irreducible_polys

    Notes
    -----
    A polynomial :math:`f(x) \in \mathrm{GF}(p^m)[x]` is *reducible* over :math:`\mathrm{GF}(p^m)` if it can
    be represented as :math:`f(x) = g(x) h(x)` for some :math:`g(x), h(x) \in \mathrm{GF}(p^m)[x]` of strictly
    lower degree. If :math:`f(x)` is not reducible, it is said to be *irreducible*. Since Galois fields are not algebraically
    closed, such irreducible polynomials exist.

    This function implements Rabin's irreducibility test. It says a degree-:math:`m` polynomial :math:`f(x)`
    over :math:`\mathrm{GF}(q)` for prime power :math:`q` is irreducible if and only if :math:`f(x)\ |\ (x^{q^m} - x)`
    and :math:`\textrm{gcd}(f(x),\ x^{q^{m_i}} - x) = 1` for :math:`1 \le i \le k`, where :math:`m_i = m/p_i` for
    the :math:`k` prime divisors :math:`p_i` of :math:`m`.

    References
    ----------
    * M. O. Rabin. Probabilistic algorithms in finite fields. SIAM Journal on Computing (1980), 273–280. https://apps.dtic.mil/sti/pdfs/ADA078416.pdf
    * S. Gao and D. Panarino. Tests and constructions of irreducible polynomials over finite fields. https://www.math.clemson.edu/~sgao/papers/GP97a.pdf
    * Section 4.5.1 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf
    * https://en.wikipedia.org/wiki/Factorization_of_polynomials_over_finite_fields

    Examples
    --------
    .. ipython:: python

        # Conway polynomials are always irreducible (and primitive)
        f = galois.conway_poly(2, 5); f

        # f(x) has no roots in GF(2), a necessary but not sufficient condition of being irreducible
        f.roots()

        galois.is_irreducible(f)

    .. ipython:: python

        g = galois.irreducible_poly(2**4, 2, method="random"); g
        h = galois.irreducible_poly(2**4, 3, method="random"); h
        f = g * h; f

        galois.is_irreducible(f)
    """
    # pylint: disable=too-many-return-statements
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")

    if poly.degree == 0:
        # Over fields, f(x) = 0 is the zero element of GF(p^m)[x] and f(x) = c are the units of GF(p^m)[x]. Both the
        # zero element and the units are not irreducible over the polynomial ring GF(p^m)[x].
        return False

    if poly.degree == 1:
        # f(x) = x + a (even a = 0) in any Galois field is irreducible
        return True

    if poly.coeffs[-1] == 0:
        # g(x) = x can be factored, therefore it is not irreducible
        return False

    if poly.field.order == 2 and poly.nonzero_coeffs.size % 2 == 0:
        # Polynomials over GF(2) with degree at least 2 and an even number of terms satisfy f(1) = 0, hence
        # g(x) = x + 1 can be factored. Section 4.5.2 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf.
        return False

    field = poly.field
    q = field.order
    m = poly.degree
    x = Poly.Identity(field)

    primes, _ = factors(m)
    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([m // pi for pi in primes]):
        # The GCD of f(x) and (x^(q^(m/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of m
        hi = pow(h0, q**(ni - n0), poly)
        g = poly_functions.gcd(poly, hi - x)
        if g != 1:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(q^m) - x) to be irreducible
    h = pow(h0, q**(m - n0), poly)
    g = (h - x) % poly
    if g != 0:
        return False

    return True


###############################################################################
# Primitive polynomials
###############################################################################

@set_module("galois")
def primitive_poly(order: int, degree: int, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Returns a monic primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired primitive polynomial.
    method
        The search method for finding the primitive polynomial.

        * `"min"` (default): Returns the lexicographically-minimal monic primitive polynomial.
        * `"max"`: Returns the lexicographically-maximal monic primitive polynomial.
        * `"random"`: Returns a randomly generated degree-:math:`m` monic primitive polynomial.

    Returns
    -------
    :
        The degree-:math:`m` monic primitive polynomial over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    is_primitive, matlab_primitive_poly, conway_poly

    Notes
    -----
    If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also primitive.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Search methods

            Find the lexicographically-minimal monic primitive polynomial.

            .. ipython:: python

                galois.primitive_poly(7, 3)

            Find the lexicographically-maximal monic primitive polynomial.

            .. ipython:: python

                galois.primitive_poly(7, 3, method="max")

            Find a random monic primitive polynomial.

            .. ipython:: python

                galois.primitive_poly(7, 3, method="random")

        .. tab-item:: Primitive vs. Conway

            Notice :func:`galois.primitive_poly` returns the lexicographically-minimal primitive polynomial but
            :func:`galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
            with smaller Conway polynomials.

            This is sometimes the same polynomial.

            .. ipython:: python

                galois.primitive_poly(2, 4)
                galois.conway_poly(2, 4)

            However, it is not always.

            .. ipython:: python

                galois.primitive_poly(7, 10)
                galois.conway_poly(7, 10)

        .. tab-item:: Properties

            Find a random monic primitive polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

            .. ipython:: python

                f = galois.primitive_poly(7, 5, method="random"); f
                galois.is_primitive(f)

            Monic primitive polynomials scaled by non-zero field elements (now non-monic) are also primitive.

            .. ipython:: python

                GF = galois.GF(7)
                g = f * GF(3); g
                galois.is_primitive(g)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    if method == "min":
        return next(primitive_polys(order, degree))
    elif method == "max":
        return next(primitive_polys(order, degree, reverse=True))
    else:
        return _primitive_poly_random_search(order, degree)


@set_module("galois")
def primitive_polys(order: int, degree: int, reverse: bool = False) -> Iterator[Poly]:
    r"""
    Iterates through all monic primitive polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired primitive polynomial.
    reverse
        Indicates to return the primitive polynomials from lexicographically maximal to minimal. The default is `False`.

    Returns
    -------
    :
        An iterator over all degree-:math:`m` monic primitive polynomials over :math:`\mathrm{GF}(q)`.

    See Also
    --------
    is_primitive, irreducible_polys

    Notes
    -----
    If :math:`f(x)` is a primitive polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also primitive.

    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    .. tab-set::

        .. tab-item:: Return full list

            All monic primitive polynomials over :math:`\mathrm{GF}(3)` with degree :math:`4`. You may also use :func:`tuple` on
            the returned generator.

            .. ipython:: python

                list(galois.primitive_polys(3, 4))

        .. tab-item:: For loop

            Loop over all the polynomials in reversed order, only finding them as needed. The search cost for the polynomials that would
            have been found after the `break` condition is never incurred.

            .. ipython:: python

                for poly in galois.primitive_polys(3, 4, reverse=True):
                    if poly.coeffs[1] < 2:  # Early exit condition
                        break
                    print(poly)

        .. tab-item:: Manual iteration

            Or, manually iterate over the generator.

            .. ipython:: python

                generator = galois.primitive_polys(3, 4, reverse=True); generator
                next(generator)
                next(generator)
                next(generator)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 0:
        raise ValueError(f"Argument `degree` must be at least 0, not {degree}.")

    field = GF(order)

    # Only search monic polynomials of degree m over GF(q)
    start = order**degree
    stop = 2*order**degree
    step = 1

    if reverse:
        start, stop, step = stop - 1, start - 1, -1

    while True:
        poly = _primitive_poly_deterministic_search(field, start, stop, step)
        if poly is not None:
            start = int(poly) + step
            yield poly
        else:
            break


@functools.lru_cache(maxsize=4096)
def _primitive_poly_deterministic_search(field, start, stop, step) -> Optional[Poly]:
    """
    Searches for an primitive polynomial in the range using the specified deterministic method.
    """
    for element in range(start, stop, step):
        poly = Poly.Int(element, field=field)
        if is_primitive(poly):
            return poly

    return None


def _primitive_poly_random_search(order, degree) -> Poly:
    """
    Searches for a random primitive polynomial.
    """
    field = GF(order)

    # Only search monic polynomials of degree m over GF(p)
    start = order**degree
    stop = 2*order**degree

    while True:
        integer = random.randint(start, stop - 1)
        poly = Poly.Int(integer, field=field)
        if is_primitive(poly):
            return poly


@set_module("galois")
def is_primitive(poly: Poly) -> bool:
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is primitive.

    Parameters
    ----------
    poly
        A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Returns
    -------
    :
        `True` if the polynomial is primitive.

    See Also
    --------
    is_irreducible, primitive_poly, matlab_primitive_poly, primitive_polys

    Notes
    -----
    A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is *primitive* if it is irreducible and
    :math:`f(x)\ |\ (x^k - 1)` for :math:`k = q^m - 1` and no :math:`k` less than :math:`q^m - 1`.

    References
    ----------
    * Algorithm 4.77 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf

    Examples
    --------
    All Conway polynomials are primitive.

    .. ipython:: python

        f = galois.conway_poly(2, 8); f
        galois.is_primitive(f)

        f = galois.conway_poly(3, 5); f
        galois.is_primitive(f)

    The irreducible polynomial of :math:`\mathrm{GF}(2^8)` for AES is not primitive.

    .. ipython:: python

        f = galois.Poly.Degrees([8, 4, 3, 1, 0]); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")

    if poly.degree == 0:
        # Over fields, f(x) = 0 is the zero element of GF(p^m)[x] and f(x) = c are the units of GF(p^m)[x]. Both the
        # zero element and the units are not irreducible over the polynomial ring GF(p^m)[x], and therefore cannot
        # be primitive.
        return False

    if poly.field.order == 2 and poly.degree == 1:
        # There is only one primitive polynomial in GF(2)
        return poly == Poly([1, 1])

    if poly.coeffs[-1] == 0:
        # A primitive polynomial cannot have zero constant term
        # TODO: Why isn't f(x) = x primitive? It's irreducible and passes the primitivity tests.
        return False

    if not is_irreducible(poly):
        # A polynomial must be irreducible to be primitive
        return False

    field = poly.field
    q = field.order
    m = poly.degree
    one = Poly.One(field)

    primes, _ = factors(q**m - 1)
    x = Poly.Identity(field)
    for ki in sorted([(q**m - 1) // pi for pi in primes]):
        # f(x) must not divide (x^((q^m - 1)/pi) - 1) for f(x) to be primitive, where pi are the prime factors of q**m - 1
        h = pow(x, ki, poly)
        g = (h - one) % poly
        if g == 0:
            return False

    return True


###############################################################################
# Special primitive polynomials
###############################################################################

@set_module("galois")
def conway_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree
        The degree :math:`m` of the Conway polynomial.

    Returns
    -------
    :
        The degree-:math:`m` Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)`.

    See Also
    --------
    is_primitive, primitive_poly, matlab_primitive_poly

    Raises
    ------
    LookupError
        If the Conway polynomial :math:`C_{p,m}(x)` is not found in Frank Luebeck's database.

    Notes
    -----
    A Conway polynomial is an irreducible and primitive polynomial over :math:`\mathrm{GF}(p)` that provides a standard
    representation of :math:`\mathrm{GF}(p^m)` as a splitting field of :math:`C_{p,m}(x)`. Conway polynomials
    provide compatability between fields and their subfields and, hence, are the common way to represent extension
    fields.

    The Conway polynomial :math:`C_{p,m}(x)` is defined as the lexicographically-minimal monic primitive polynomial
    of degree :math:`m` over :math:`\mathrm{GF}(p)` that is compatible with all :math:`C_{p,n}(x)` for :math:`n` dividing
    :math:`m`.

    This function uses `Frank Luebeck's Conway polynomial database <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_
    for fast lookup, not construction.

    Examples
    --------
    Notice :func:`galois.primitive_poly` returns the lexicographically-minimal primitive polynomial but
    :func:`galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials.

    This is sometimes the same polynomial.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

    However, it is not always.

    .. ipython:: python

        galois.primitive_poly(7, 10)
        galois.conway_poly(7, 10)
    """
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")

    coeffs = ConwayPolyDatabase().fetch(characteristic, degree)
    field = GF_prime(characteristic)
    poly = Poly(coeffs, field=field)

    return poly


@set_module("galois")
def matlab_primitive_poly(characteristic: int, degree: int) -> Poly:
    r"""
    Returns Matlab's default primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree
        The degree :math:`m` of the desired primitive polynomial.

    Returns
    -------
    :
        Matlab's default degree-:math:`m` primitive polynomial over :math:`\mathrm{GF}(p)`.

    See Also
    --------
    is_primitive, primitive_poly, conway_poly

    Notes
    -----
    This function returns the same result as Matlab's `gfprimdf(m, p)`. Matlab uses the primitive polynomial with minimum terms
    (equivalent to `galois.primitive_poly(p, m, method="min-terms")`) as the default... *mostly*. There are three
    notable exceptions:

    1. :math:`\mathrm{GF}(2^7)` uses :math:`x^7 + x^3 + 1`, not :math:`x^7 + x + 1`.
    2. :math:`\mathrm{GF}(2^{14})` uses :math:`x^{14} + x^{10} + x^6 + x + 1`, not :math:`x^{14} + x^5 + x^3 + x + 1`.
    3. :math:`\mathrm{GF}(2^{16})` uses :math:`x^{16} + x^{12} + x^3 + x + 1`, not :math:`x^{16} + x^5 + x^3 + x^2 + 1`.

    Warning
    -------
    This has been tested for all the :math:`\mathrm{GF}(2^m)` fields for :math:`2 \le m \le 16` (Matlab doesn't support
    larger than 16). And it has been spot-checked for :math:`\mathrm{GF}(p^m)`. There may exist other exceptions. Please
    submit a GitHub issue if you discover one.

    References
    ----------
    * S. Lin and D. Costello. Error Control Coding. Table 2.7.

    Examples
    --------
    .. ipython:: python

        galois.primitive_poly(2, 6)
        galois.matlab_primitive_poly(2, 6)

    .. ipython:: python

        galois.primitive_poly(2, 7)
        galois.matlab_primitive_poly(2, 7)
    """
    if not isinstance(characteristic, (int, np.integer)):
        raise TypeError(f"Argument `characteristic` must be an integer, not {type(characteristic)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime(characteristic):
        raise ValueError(f"Argument `characteristic` must be prime, not {characteristic}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}. There are no primitive polynomials with degree 0.")

    # Textbooks and Matlab use the lexicographically-minimal primitive polynomial for the default. But for some
    # reason, there are three exceptions. I can't determine why.
    if characteristic == 2 and degree == 7:
        # Not the lexicographically-minimal of `x^7 + x + 1`
        return Poly.Degrees([7, 3, 0])
    elif characteristic == 2 and degree == 14:
        # Not the lexicographically-minimal of `x^14 + x^5 + x^3 + x + 1`
        return Poly.Degrees([14, 10, 6, 1, 0])
    elif characteristic == 2 and degree == 16:
        # Not the lexicographically-minimal of `x^16 + x^5 + x^3 + x^2 + 1`
        return Poly.Degrees([16, 12, 3, 1, 0])
    else:
        return primitive_poly(characteristic, degree)


###############################################################################
# Generate and test primitive elements
###############################################################################

@set_module("galois")
def primitive_element(
    irreducible_poly: Poly,  # pylint: disable=redefined-outer-name
    method: Literal["min", "max", "random"] = "min"
) -> Poly:
    r"""
    Finds a primitive element :math:`g` of the Galois field :math:`\mathrm{GF}(q^m)` with degree-:math:`m` irreducible polynomial
    :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Parameters
    ----------
    irreducible_poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` that defines the extension field :math:`\mathrm{GF}(q^m)`.
    method
        The search method for finding the primitive element.

    Returns
    -------
    :
        A primitive element :math:`g` of :math:`\mathrm{GF}(q^m)` with irreducible polynomial :math:`f(x)`. The primitive element :math:`g` is
        a polynomial over :math:`\mathrm{GF}(q)` with degree less than :math:`m`.

    See Also
    --------
    is_primitive_element, FieldClass.primitive_element

    Examples
    --------
    .. tab-set::

        .. tab-item:: Min

            Find the smallest primitive element for the degree :math:`5` extension of :math:`\mathrm{GF}(7)`.

            .. ipython:: python

                f = galois.conway_poly(7, 5); f
                g = galois.primitive_element(f); g

            Construct the extension field :math:`\mathrm{GF}(7^5)`. Note, by default, :func:`galois.GF` uses a Conway polynomial
            as its irreducible polynomial.

            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF)
                int(g) == GF.primitive_element

        .. tab-item:: Max

            Find the largest primitive element for the degree :math:`5` extension of :math:`\mathrm{GF}(7)`.

            .. ipython:: python

                f = galois.conway_poly(7, 5); f
                g = galois.primitive_element(f, method="max"); g

            Construct the extension field :math:`\mathrm{GF}(7^5)`. Note, by default, :func:`galois.GF` uses a Conway polynomial
            as its irreducible polynomial.

            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF)
                int(g) in GF.primitive_elements

        .. tab-item:: Random

            Find a random primitive element for the degree :math:`5` extension of :math:`\mathrm{GF}(7)`.

            .. ipython:: python

                f = galois.conway_poly(7, 5); f
                g = galois.primitive_element(f, method="random"); g

            Construct the extension field :math:`\mathrm{GF}(7^5)`. Note, by default, :func:`galois.GF` uses a Conway polynomial
            as its irreducible polynomial.

            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF)
                int(g) in GF.primitive_elements
    """
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument `irreducible_poly` must have degree greater than 1, not {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    field = irreducible_poly.field
    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    start = q
    stop = q**m

    if method == "min":
        for integer in range(start, stop):
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                break
    elif method == "max":
        for integer in range(stop - 1, start - 1, -1):
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                break
    else:
        while True:
            integer = random.randint(start, stop - 1)
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                break

    return element


@set_module("galois")
def primitive_elements(irreducible_poly: Poly) -> List[Poly]:  # pylint: disable=redefined-outer-name
    r"""
    Finds all primitive elements :math:`g` of the Galois field :math:`\mathrm{GF}(q^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Parameters
    ----------
    irreducible_poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` that defines the extension
        field :math:`\mathrm{GF}(q^m)`.

    Returns
    -------
    :
        List of all primitive elements of :math:`\mathrm{GF}(q^m)` with irreducible polynomial :math:`f(x)`. Each primitive
        element :math:`g` is a polynomial over :math:`\mathrm{GF}(q)` with degree less than :math:`m`.

    See Also
    --------
    is_primitive_element, FieldClass.primitive_elements

    Notes
    -----
    The number of primitive elements of :math:`\mathrm{GF}(q^m)` is :math:`\phi(q^m - 1)`, where
    :math:`\phi(n)` is the Euler totient function. See :obj:`galois.euler_phi`.

    Examples
    --------
    Find all primitive elements for the degree :math:`4` extension of :math:`\mathrm{GF}(3)`.

    .. ipython:: python

        f = galois.conway_poly(3, 4); f
        g = galois.primitive_elements(f); g

    Construct the extension field :math:`\mathrm{GF}(3^4)`. Note, by default, :func:`galois.GF` uses a Conway polynomial
    as its irreducible polynomial.

    .. ipython:: python

        GF = galois.GF(3**4)
        print(GF)
        np.array_equal([int(gi) for gi in g], GF.primitive_elements)

    The number of primitive elements is given by :math:`\phi(q^m - 1)`.

    .. ipython:: python

        phi = galois.euler_phi(3**4 - 1); phi
        len(g) == phi
    """
    # Find one primitive element first
    element = primitive_element(irreducible_poly)

    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    elements = []
    for totative in totatives(q**m - 1):
        h = pow(element, totative, irreducible_poly)
        elements.append(h)

    elements = sorted(elements, key=int)  # Sort element lexicographically

    return elements


@set_module("galois")
def is_primitive_element(
    element: Union[int, str, Sequence[int], np.ndarray, FieldArray, "Poly"],
    irreducible_poly: Poly  # pylint: disable=redefined-outer-name
) -> bool:
    r"""
    Determines if :math:`g` is a primitive element of the Galois field :math:`\mathrm{GF}(q^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Parameters
    ----------
    element
        An element :math:`g` of :math:`\mathrm{GF}(q^m)` is a polynomial over :math:`\mathrm{GF}(q)` with degree
        less than :math:`m`.

        * :obj:`int`: The integer representation of the primitive element.
        * :obj:`str`: The primitive element expressed as a string, e.g. `"x + 1"`.
        * :obj:`galois.Poly`: The primitive element as a polynomial object.

    irreducible_poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` that defines the extension
        field :math:`\mathrm{GF}(q^m)`.

    Returns
    -------
    :
        `True` if :math:`g` is a primitive element of :math:`\mathrm{GF}(q^m)`.

    See Also
    --------
    primitive_element, FieldClass.primitive_element

    Examples
    --------
    Find all primitive elements for the degree :math:`4` extension of :math:`\mathrm{GF}(3)`.

    .. ipython:: python

        f = galois.conway_poly(3, 4); f
        g = galois.primitive_elements(f); g

    Note from the list above that :math:`x + 2` is a primitive element, but :math:`x + 1` is not.

    .. ipython:: python

        galois.is_primitive_element("x + 2", f)
        # x + 1 over GF(3) has integer equivalent of 4
        galois.is_primitive_element(4, f)
    """
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    field = irreducible_poly.field

    # Convert element into a Poly object
    element = Poly._PolyLike(element, field=field)

    if not element.field == irreducible_poly.field:
        raise ValueError(f"Arguments `element` and `irreducible_poly` must be over the same field, not {element.field.name} and {irreducible_poly.field.name}.")
    if not element.degree < irreducible_poly.degree:
        raise ValueError(f"Argument `element` must have degree less than `irreducible_poly`, not {element.degree} and {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")

    return _is_primitive_element(element, irreducible_poly)


def _is_primitive_element(element: Poly, irreducible_poly: Poly) -> bool:  # pylint: disable=redefined-outer-name
    """
    A private version of `is_primitive_element()` without type checking/conversion for internal use.
    """
    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    order = q**m - 1  # Multiplicative order of GF(q^m)
    primes, _ = factors(order)

    for k in sorted([order // pi for pi in primes]):
        g = pow(element, k, irreducible_poly)
        if g == 1:
            return False

    g = pow(element, order, irreducible_poly)
    if g != 1:
        return False

    return True
