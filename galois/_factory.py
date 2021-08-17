"""
A module to implement the Galois field class factory `GF()`. This module also includes functions to generate irreducible, primitive,
and Conway polynomials. They are included here due to a circular dependence with the Galois field class factory.
"""
import random
import types

import numpy as np

from ._databases import ConwayPolyDatabase
from ._factor import factors, is_prime_power
from ._fields import FieldArray, GF2
from ._fields._gfp import GFpMeta
from ._fields._gf2m import GF2mMeta
from ._fields._gfpm import GFpmMeta
from ._modular import totatives, primitive_root, is_primitive_root
from ._overrides import set_module
from ._polys import Poly
from ._polys import _math as poly_math
from ._prime import is_prime

__all__ = [
    "GF", "Field",
    "irreducible_poly", "irreducible_polys", "is_irreducible",
    "primitive_poly", "primitive_polys", "matlab_primitive_poly", "conway_poly", "is_primitive",
    "primitive_element", "primitive_elements", "is_primitive_element",
]


###############################################################################
# Construct Galois field array classes
###############################################################################

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
def Field(order, irreducible_poly=None, primitive_element=None, verify=True, compile=None, display=None):
    """
    Alias of :func:`galois.GF`.
    """
    # pylint: disable=redefined-outer-name,redefined-builtin
    return GF(order, irreducible_poly=irreducible_poly, primitive_element=primitive_element, verify=verify, compile=compile, display=display)


def GF_prime(characteristic, primitive_element_=None, verify=True, compile_=None, display=None):
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


def GF_extension(characteristic, degree, irreducible_poly_=None, primitive_element_=None, verify=True, compile_=None, display=None):
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
    elif isinstance(irreducible_poly_, int):
        irreducible_poly_ = Poly.Integer(irreducible_poly_, field=prime_subfield)
    elif isinstance(irreducible_poly_, str):
        irreducible_poly_ = Poly.String(irreducible_poly_, field=prime_subfield)
    elif isinstance(irreducible_poly_, (tuple, list, np.ndarray)):
        irreducible_poly_ = Poly(irreducible_poly_, field=prime_subfield)
    elif not isinstance(irreducible_poly_, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be an int, tuple, list, np.ndarray, or galois.Poly, not {type(irreducible_poly_)}.")

    # Get default primitive element
    if primitive_element_ is None:
        primitive_element_ = primitive_element(irreducible_poly_)
        verify_element = False
    elif isinstance(primitive_element_, int):
        primitive_element_ = Poly.Integer(primitive_element_, field=prime_subfield)
    elif isinstance(primitive_element_, str):
        primitive_element_ = Poly.String(primitive_element_, field=prime_subfield)
    elif isinstance(primitive_element_, (tuple, list, np.ndarray)):
        primitive_element_ = Poly(primitive_element_, field=prime_subfield)
    elif not isinstance(primitive_element_, Poly):
        raise TypeError(f"Argument `primitive_element` must be an int, tuple, list, np.ndarray, or galois.Poly, not {type(primitive_element_)}.")

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
    key = (order, primitive_element_.integer, irreducible_poly_.integer)
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
            "primitive_element": primitive_element_.integer,
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
            "primitive_element": primitive_element_.integer,
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
# Generate and test irreducible polynomials
###############################################################################

@set_module("galois")
def irreducible_poly(order, degree, method="min"):
    r"""
    Returns a monic irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order : int
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired irreducible polynomial.
    method : str, optional
        The search method for finding the irreducible polynomial.

        * `"min"` (default): Returns the lexicographically-minimal monic irreducible polynomial.
        * `"max"`: Returns the lexicographically-maximal monic irreducible polynomial.
        * `"random"`: Returns a randomly generated degree-:math:`m` monic irreducible polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` monic irreducible polynomial over :math:`\mathrm{GF}(q)`.

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible. In addition to other applications, :math:`f(x)` produces the field extension
    :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    The lexicographically-minimal, monic irreducible polynomial over :math:`\mathrm{GF}(7)` with degree :math:`5`.

    .. ipython:: python

        p = galois.irreducible_poly(7, 5); p
        galois.is_irreducible(p)

    Irreducible polynomials scaled by non-zero field elements are also irreducible.

    .. ipython:: python

        GF = galois.GF(7)
        galois.is_irreducible(p * GF(3))

    A random, monic irreducible polynomial over :math:`\mathrm{GF}(7^2)` with degree :math:`3`.

    .. ipython:: python

        p = galois.irreducible_poly(7**2, 3, method="random"); p
        galois.is_irreducible(p)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    field = GF(order)

    # Only search monic polynomials of degree m over GF(q)
    min_ = order**degree
    max_ = 2*order**degree

    if method == "random":
        while True:
            integer = random.randint(min_, max_ - 1)
            poly = Poly.Integer(integer, field=field)
            if is_irreducible(poly):
                break
    else:
        elements = range(min_, max_) if method == "min" else range(max_ - 1, min_ - 1, -1)
        for element in elements:
            poly = Poly.Integer(element, field=field)
            if is_irreducible(poly):
                break

    return poly


@set_module("galois")
def irreducible_polys(order, degree):
    r"""
    Returns all monic irreducible polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order : int
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired irreducible polynomial.

    Returns
    -------
    list
        All degree-:math:`m` monic irreducible polynomials over :math:`\mathrm{GF}(q)`.

    Notes
    -----
    If :math:`f(x)` is an irreducible polynomial over :math:`\mathrm{GF}(q)` and :math:`a \in \mathrm{GF}(q) \backslash \{0\}`,
    then :math:`a \cdot f(x)` is also irreducible. In addition to other applications, :math:`f(x)` produces the field extension
    :math:`\mathrm{GF}(q^m)` of :math:`\mathrm{GF}(q)`.

    Examples
    --------
    All monic irreducible polynomials over :math:`\mathrm{GF}(2)` with degree :math:`5`.

    .. ipython:: python

        galois.irreducible_polys(2, 5)

    All monic irreducible polynomials over :math:`\mathrm{GF}(3^2)` with degree :math:`2`.

    .. ipython:: python

        galois.irreducible_polys(3**2, 2)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

    field = GF(order)

    # Only search monic polynomials of degree m over GF(p)
    min_ = order**degree
    max_ = 2*order**degree

    polys = []
    for element in range(min_, max_):
        poly = Poly.Integer(element, field=field)
        if is_irreducible(poly):
            polys.append(poly)

    return polys


@set_module("galois")
def is_irreducible(poly):
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)` is irreducible.

    Parameters
    ----------
    poly : galois.Poly
        A polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    bool
        `True` if the polynomial is irreducible.

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
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")

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
    zero = Poly.Zero(field)
    one = Poly.One(field)
    x = Poly.Identity(field)

    primes, _ = factors(m)
    h0 = Poly.Identity(field)
    n0 = 0
    for ni in sorted([m // pi for pi in primes]):
        # The GCD of f(x) and (x^(q^(m/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of m
        hi = poly_math.pow(h0, q**(ni - n0), poly)
        g = poly_math.gcd(poly, hi - x)
        if g != one:
            return False
        h0, n0 = hi, ni

    # f(x) must divide (x^(q^m) - x) to be irreducible
    h = poly_math.pow(h0, q**(m - n0), poly)
    g = (h - x) % poly
    if g != zero:
        return False

    return True


###############################################################################
# Generate and test primitive polynomials
###############################################################################

@set_module("galois")
def primitive_poly(order, degree, method="min"):
    r"""
    Returns a monic primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order : int
        The prime power order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired primitive polynomial.
    method : str, optional
        The search method for finding the primitive polynomial.

        * `"min"` (default): Returns the lexicographically-minimal monic primitive polynomial.
        * `"max"`: Returns the lexicographically-maximal monic primitive polynomial.
        * `"random"`: Returns a randomly generated degree-:math:`m` monic primitive polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` monic primitive polynomial over :math:`\mathrm{GF}(q)`.

    Notes
    -----
    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    Notice :func:`galois.primitive_poly` returns the lexicographically-minimal primitive polynomial, whereas
    :func:`galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials, which is not *necessarily* the same.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

    .. ipython:: python

        galois.primitive_poly(7, 10)
        galois.conway_poly(7, 10)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument `method` must be in ['min', 'max', 'random'], not {method!r}.")

    field = GF(order)

    # Only search monic polynomials of degree m over GF(p)
    min_ = order**degree
    max_ = 2*order**degree

    if method == "random":
        while True:
            integer = random.randint(min_, max_ - 1)
            poly = Poly.Integer(integer, field=field)
            if is_primitive(poly):
                break
    else:
        elements = range(min_, max_) if method == "min" else range(max_ - 1, min_ - 1, -1)
        for element in elements:
            poly = Poly.Integer(element, field=field)
            if is_primitive(poly):
                break

    return poly


@set_module("galois")
def primitive_polys(order, degree):
    r"""
    Returns all monic primitive polynomials :math:`f(x)` over :math:`\mathrm{GF}(q)` with degree :math:`m`.

    Parameters
    ----------
    order : int
        The prime order :math:`q` of the field :math:`\mathrm{GF}(q)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired primitive polynomial.

    Returns
    -------
    list
        All degree-:math:`m` monic primitive polynomials over :math:`\mathrm{GF}(q)`.

    Notes
    -----
    In addition to other applications, :math:`f(x)` produces the field extension :math:`\mathrm{GF}(q^m)`
    of :math:`\mathrm{GF}(q)`. Since :math:`f(x)` is primitive, :math:`x` is a primitive element :math:`\alpha`
    of :math:`\mathrm{GF}(q^m)` such that :math:`\mathrm{GF}(q^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{q^m-2}\}`.

    Examples
    --------
    All monic primitive polynomials over :math:`\mathrm{GF}(2)` with degree :math:`5`.

    .. ipython:: python

        galois.primitive_polys(2, 5)

    All monic primitive polynomials over :math:`\mathrm{GF}(3^2)` with degree :math:`2`.

    .. ipython:: python

        galois.primitive_polys(3**2, 2)
    """
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"Argument `order` must be an integer, not {type(order)}.")
    if not isinstance(degree, (int, np.integer)):
        raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
    if not is_prime_power(order):
        raise ValueError(f"Argument `order` must be a prime power, not {order}.")
    if not degree >= 1:
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

    field = GF(order)

    # Only search monic polynomials of degree m over GF(p)
    min_ = order**degree
    max_ = 2*order**degree

    polys = []
    for element in range(min_, max_):
        poly = Poly.Integer(element, field=field)
        if is_primitive(poly):
            polys.append(poly)

    return polys


@set_module("galois")
def conway_poly(characteristic, degree):
    r"""
    Returns the Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the Conway polynomial.

    Returns
    -------
    galois.Poly
        The degree-:math:`m` Conway polynomial :math:`C_{p,m}(x)` over :math:`\mathrm{GF}(p)`.

    Raises
    ------
    LookupError
        If the Conway polynomial :math:`C_{p,m}(x)` is not found in Frank Luebeck's database.

    Notes
    -----
    A Conway polynomial is a an irreducible and primitive polynomial over :math:`\mathrm{GF}(p)` that provides a standard
    representation of :math:`\mathrm{GF}(p^m)` as a splitting field of :math:`C_{p,m}(x)`. Conway polynomials
    provide compatability between fields and their subfields, and hence are the common way to represent extension
    fields.

    The Conway polynomial :math:`C_{p,m}(x)` is defined as the lexicographically-minimal monic primitive polynomial
    of degree :math:`m` over :math:`\mathrm{GF}(p)` that is compatible with all :math:`C_{p,n}(x)` for :math:`n` dividing
    :math:`m`.

    This function uses `Frank Luebeck's Conway polynomial database <http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html>`_
    for fast lookup, not construction.

    Examples
    --------
    Notice :func:`galois.primitive_poly` returns the lexicographically-minimal primitive polynomial, where
    :func:`galois.conway_poly` returns the lexicographically-minimal primitive polynomial that is *consistent*
    with smaller Conway polynomials, which is not *necessarily* the same.

    .. ipython:: python

        galois.primitive_poly(2, 4)
        galois.conway_poly(2, 4)

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
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

    coeffs = ConwayPolyDatabase().fetch(characteristic, degree)
    field = GF_prime(characteristic)
    poly = Poly(coeffs, field=field)

    return poly


@set_module("galois")
def matlab_primitive_poly(characteristic, degree):
    r"""
    Returns Matlab's default primitive polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` with degree :math:`m`.

    Parameters
    ----------
    characteristic : int
        The prime characteristic :math:`p` of the field :math:`\mathrm{GF}(p)` that the polynomial is over.
    degree : int
        The degree :math:`m` of the desired primitive polynomial.

    Returns
    -------
    galois.Poly
        Matlab's default degree-:math:`m` primitive polynomial over :math:`\mathrm{GF}(p)`.

    Notes
    -----
    This function returns the same result as Matlab's `gfprimdf(m, p)`. Matlab uses the primitive polynomial with minimum terms
    (equivalent to `galois.primitive_poly(p, m, method="min-terms")`) as the default... *mostly*. There are three
    notable exceptions:

    1. :math:`\mathrm{GF}(2^7)` uses :math:`x^7 + x^3 + 1`, not :math:`x^7 + x + 1`.
    2. :math:`\mathrm{GF}(2^{14})` uses :math:`x^{14} + x^{10} + x^6 + x + 1`, not :math:`x^{14} + x^5 + x^3 + x + 1`.
    3. :math:`\mathrm{GF}(2^{16})` uses :math:`x^{16} + x^{12} + x^3 + x + 1`, not :math:`x^{16} + x^5 + x^3 + x^2 + 1`.

    References
    ----------
    * S. Lin and D. Costello. Error Control Coding. Table 2.7.

    Warning
    -------
    This has been tested for all the :math:`\mathrm{GF}(2^m)` fields for :math:`2 \le m \le 16` (Matlab doesn't support
    larger than 16). And it has been spot-checked for :math:`\mathrm{GF}(p^m)`. There may exist other exceptions. Please
    submit a GitHub issue if you discover one.

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
        raise ValueError(f"Argument `degree` must be at least 1, not {degree}.")

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


@set_module("galois")
def is_primitive(poly):
    r"""
    Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is primitive.

    A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is *primitive* if it is irreducible and
    :math:`f(x)\ |\ (x^k - 1)` for :math:`k = q^m - 1` and no :math:`k` less than :math:`q^m - 1`.

    Parameters
    ----------
    poly : galois.Poly
        A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)`.

    Returns
    -------
    bool
        `True` if the polynomial is primitive.

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

        f = galois.Poly.Degrees([8,4,3,1,0]); f
        galois.is_primitive(f)
    """
    if not isinstance(poly, Poly):
        raise TypeError(f"Argument `poly` must be a galois.Poly, not {type(poly)}.")
    if not poly.degree >= 1:
        raise ValueError(f"Argument `poly` must have degree at least 1, not {poly.degree}.")

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
    zero = Poly.Zero(field)
    one = Poly.One(field)

    primes, _ = factors(q**m - 1)
    x = Poly.Identity(field)
    for ki in sorted([(q**m - 1) // pi for pi in primes]):
        # f(x) must not divide (x^((q^m - 1)/pi) - 1) for f(x) to be primitive, where pi are the prime factors of q**m - 1
        h = poly_math.pow(x, ki, poly)
        g = (h - one) % poly
        if g == zero:
            return False

    return True


###############################################################################
# Generate and test primitive elements
###############################################################################

@set_module("galois")
def primitive_element(irreducible_poly, start=None, stop=None, reverse=False):  # pylint: disable=redefined-outer-name
    r"""
    Finds the smallest primitive element :math:`g(x)` of the Galois field :math:`\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    Parameters
    ----------
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` that defines the extension field :math:`\mathrm{GF}(p^m)`.
    start : int, optional
        Starting value (inclusive, integer representation of the polynomial) in the search for a primitive element :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p`, which corresponds to :math:`g(x) = x` over :math:`\mathrm{GF}(p)`.
    stop : int, optional
        Stopping value (exclusive, integer representation of the polynomial) in the search for a primitive element :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p^m`, which corresponds to :math:`g(x) = x^m` over :math:`\mathrm{GF}(p)`.
    reverse : bool, optional
        Search for a primitive element in reverse order, i.e. find the largest primitive element first. Default is `False`.

    Returns
    -------
    galois.Poly
        A primitive element of :math:`\mathrm{GF}(p^m)` with irreducible polynomial :math:`f(x)`. The primitive element :math:`g(x)` is
        a polynomial over :math:`\mathrm{GF}(p)` with degree less than :math:`m`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        galois.primitive_element(f)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        galois.primitive_element(f)
    """
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    if not isinstance(start, (type(None), int, np.integer)):
        raise TypeError(f"Argument `start` must be an integer, not {type(start)}.")
    if not isinstance(stop, (type(None), int, np.integer)):
        raise TypeError(f"Argument `stop` must be an integer, not {type(stop)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")
    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument `irreducible_poly` must have degree greater than 1, not {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")

    field = irreducible_poly.field
    q = irreducible_poly.field.order
    m = irreducible_poly.degree
    start = q if start is None else start
    stop = q**m if stop is None else stop
    if not 1 <= start < stop <= q**m:
        raise ValueError(f"Arguments must satisfy `1 <= start < stop <= q^m`, `1 <= {start} < {stop} <= {q**m}` doesn't.")

    possible_elements = range(start, stop)
    if reverse:
        possible_elements = reversed(possible_elements)

    for integer in possible_elements:
        element = Poly.Integer(integer, field=field)
        if is_primitive_element(element, irreducible_poly):
            return element

    return None


@set_module("galois")
def primitive_elements(irreducible_poly, start=None, stop=None, reverse=False):  # pylint: disable=redefined-outer-name
    r"""
    Finds all primitive elements :math:`g(x)` of the Galois field :math:`\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    The number of primitive elements of :math:`\mathrm{GF}(p^m)` is :math:`\phi(p^m - 1)`, where
    :math:`\phi(n)` is the Euler totient function. See :obj:galois.euler_phi`.

    Parameters
    ----------
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` that defines the extension field :math:`\mathrm{GF}(p^m)`.
    start : int, optional
        Starting value (inclusive, integer representation of the polynomial) in the search for primitive elements :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p`, which corresponds to :math:`g(x) = x` over :math:`\mathrm{GF}(p)`.
    stop : int, optional
        Stopping value (exclusive, integer representation of the polynomial) in the search for primitive elements :math:`g(x)` of :math:`\mathrm{GF}(p^m)`.
        The default is `None` which represents :math:`p^m`, which corresponds to :math:`g(x) = x^m` over :math:`\mathrm{GF}(p)`.
    reverse : bool, optional
        Search for primitive elements in reverse order, i.e. largest to smallest. Default is `False`.

    Returns
    -------
    list
        List of all primitive elements of :math:`\mathrm{GF}(p^m)` with irreducible polynomial :math:`f(x)`. Each primitive element :math:`g(x)` is
        a polynomial over :math:`\mathrm{GF}(p)` with degree less than :math:`m`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        g = galois.primitive_elements(f); g
        len(g) == galois.euler_phi(3**2 - 1)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)
        g = galois.primitive_elements(f); g
        len(g) == galois.euler_phi(3**2 - 1)
    """
    # NOTE: `irreducible_poly` will be verified in the call to `primitive_element()`
    if not isinstance(start, (type(None), int, np.integer)):
        raise TypeError(f"Argument `start` must be an integer, not {type(start)}.")
    if not isinstance(stop, (type(None), int, np.integer)):
        raise TypeError(f"Argument `stop` must be an integer, not {type(stop)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")

    element = primitive_element(irreducible_poly)

    q = irreducible_poly.field.order
    m = irreducible_poly.degree
    start = q if start is None else start
    stop = q**m if stop is None else stop
    if not 1 <= start < stop <= q**m:
        raise ValueError(f"Arguments must satisfy `1 <= start < stop <= q^m`, `1 <= {start} < {stop} <= {q**m}` doesn't.")

    elements = []
    for totative in totatives(q**m - 1):
        h = poly_math.pow(element, totative, irreducible_poly)
        elements.append(h)

    elements = [e for e in elements if start <= e.integer < stop]  # Only return elements in the search range
    elements = sorted(elements, key=lambda e: e.integer, reverse=reverse)  # Sort element lexicographically

    return elements


@set_module("galois")
def is_primitive_element(element, irreducible_poly):  # pylint: disable=redefined-outer-name
    r"""
    Determines if :math:`g(x)` is a primitive element of the Galois field :math:`\mathrm{GF}(p^m)` with
    degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)`.

    Parameters
    ----------
    element : galois.Poly
        An element :math:`g(x)` of :math:`\mathrm{GF}(p^m)` as a polynomial over :math:`\mathrm{GF}(p)` with degree
        less than :math:`m`.
    irreducible_poly : galois.Poly
        The degree-:math:`m` irreducible polynomial :math:`f(x)` over :math:`\mathrm{GF}(p)` that defines the extension field :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    bool
        `True` if :math:`g(x)` is a primitive element of :math:`\mathrm{GF}(p^m)` with irreducible polynomial
        :math:`f(x)`.

    Notes
    -----
    The number of primitive elements of :math:`\mathrm{GF}(p^m)` is :math:`\phi(p^m - 1)`, where :math:`\phi(n)` is the Euler totient function,
    see :func:`galois.euler_phi`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,1,2], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)

        g = galois.Poly.Identity(GF); g
        galois.is_primitive_element(g, f)

    .. ipython:: python

        GF = galois.GF(3)
        f = galois.Poly([1,0,1], field=GF); f
        galois.is_irreducible(f)
        galois.is_primitive(f)

        g = galois.Poly.Identity(GF); g
        galois.is_primitive_element(g, f)
    """
    if not isinstance(element, Poly):
        raise TypeError(f"Argument `element` must be a galois.Poly, not {type(element)}.")
    if not isinstance(irreducible_poly, Poly):
        raise TypeError(f"Argument `irreducible_poly` must be a galois.Poly, not {type(irreducible_poly)}.")
    if not element.field == irreducible_poly.field:
        raise ValueError(f"Arguments `element` and `irreducible_poly` must be over the same field, not {element.field} and {irreducible_poly.field}.")
    if not element.degree < irreducible_poly.degree:
        raise ValueError(f"Argument `element` must have degree less than `irreducible_poly`, not {element.degree} and {irreducible_poly.degree}.")
    if not is_irreducible(irreducible_poly):
        raise ValueError(f"Argument `irreducible_poly` must be irreducible, {irreducible_poly} is reducible over {irreducible_poly.field.name}.")

    field = irreducible_poly.field
    p = field.order
    m = irreducible_poly.degree
    one = Poly.One(field)

    order = p**m - 1  # Multiplicative order of GF(p^m)
    primes, _ = factors(order)

    for k in sorted([order // pi for pi in primes]):
        g = poly_math.pow(element, k, irreducible_poly)
        if g == one:
            return False

    g = poly_math.pow(element, order, irreducible_poly)
    if g != one:
        return False

    return True
