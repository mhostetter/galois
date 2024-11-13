"""
A module to implement the Galois field class factory `GF()`.
"""

from __future__ import annotations

import sys
import types
from typing import Type, overload

from typing_extensions import Literal

from .._helper import export, verify_isinstance
from .._modular import is_primitive_root, primitive_root
from .._polys import Poly, conway_poly
from .._prime import factors
from ..typing import PolyLike
from ._array import FieldArray
from ._gf2 import GF2
from ._primitive_element import is_primitive_element, primitive_element
from ._ufunc import UFuncMixin_2_m, UFuncMixin_p_1, UFuncMixin_p_m


@overload
def GF(
    order: int,
    *,
    irreducible_poly: PolyLike | None = None,
    primitive_element: int | PolyLike | None = None,
    verify: bool = True,
    compile: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"] | None = None,
    repr: Literal["int", "poly", "power"] | None = None,
) -> Type[FieldArray]: ...


@overload
def GF(
    characteristic: int,
    degree: int,
    *,
    irreducible_poly: PolyLike | None = None,
    primitive_element: int | PolyLike | None = None,
    verify: bool = True,
    compile: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"] | None = None,
    repr: Literal["int", "poly", "power"] | None = None,
) -> Type[FieldArray]: ...


@export
def GF(
    *args,
    irreducible_poly=None,
    primitive_element=None,
    verify=True,
    compile=None,
    repr=None,
):
    r"""
    Creates a :obj:`~galois.FieldArray` subclass for $\mathrm{GF}(p^m)$.

    Arguments:
        order: The order $p^m$ of the field $\mathrm{GF}(p^m)$. The order must be a prime power.
        characteristic: The characteristic $p$ of the field $\mathrm{GF}(p^m)$. The characteristic must
            be prime.
        degree: The degree $m$ of the field $\mathrm{GF}(p^m)$. The degree must be a positive integer.
        irreducible_poly: Optionally specify an irreducible polynomial of degree $m$ over $\mathrm{GF}(p)$
            that defines the finite field arithmetic. The default is `None` which uses the Conway polynomial
            $C_{p,m}$, see :func:`~galois.conway_poly`.
        primitive_element: Optionally specify a primitive element of the field. This value is used when building the
            exponential and logarithm lookup tables and as the base of :obj:`numpy.log`. A primitive element is a
            generator of the multiplicative group of the field.

            For prime fields $\mathrm{GF}(p)$, the primitive element must be an integer and is a primitive root
            modulo $p$. The default is `None` which uses :func:`~galois.primitive_root`.

            For extension fields $\mathrm{GF}(p^m)$, the primitive element is a polynomial of degree less than
            $m$ over $\mathrm{GF}(p)$. The default is `None` which uses :func:`~galois.primitive_element`.
        verify: Indicates whether to verify that the user-provided irreducible polynomial is in fact irreducible and
            that the user-provided primitive element is in fact a generator of the multiplicative group.
            The default is `True`.

            For large fields and irreducible polynomials that are already known to be irreducible (which may take a
            while to verify), this argument may be set to `False`.

            The default irreducible polynomial and primitive element are never verified because they are already known
            to be irreducible and a multiplicative generator, respectively.
        compile: The ufunc calculation mode. This can be modified after class construction with the
            :func:`~galois.FieldArray.compile` method. See :doc:`/basic-usage/compilation-modes` for a further
            discussion.

            - `None` (default): For a newly created :obj:`~galois.FieldArray` subclass, `None` corresponds to
              `"auto"`. If the :obj:`~galois.FieldArray` subclass already exists, `None` does not modify its current
              compilation mode.
            - `"auto"`: Selects `"jit-lookup"` for fields with order less than $2^{20}$, `"jit-calculate"` for
              larger fields, and `"python-calculate"` for fields whose elements cannot be represented with
              :obj:`numpy.int64`.
            - `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for
              efficient computation. In the few cases where explicit calculation is faster than table lookup, explicit
              calculation is used.
            - `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"`
              mode is designed for large fields that cannot or should not store lookup tables in RAM. Generally, the
              `"jit-calculate"` mode is slower than `"jit-lookup"`.
            - `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is intended for fields
              whose elements cannot be represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_`
              with Python :obj:`int` (which has arbitrary precision). However, this mode can be used for any
              field, enabling the code to run without Numba JIT compilation.

        repr: The field element representation. This can be modified after class construction with the
            :func:`~galois.FieldArray.repr` method. See :doc:`/basic-usage/element-representation` for a further
            discussion.

            - `None` (default): For a newly created :obj:`~galois.FieldArray` subclass, `None` corresponds to `"int"`.
              If the :obj:`~galois.FieldArray` subclass already exists, `None` does not modify its current element
              representation.
            - `"int"`: Sets the element representation to the :ref:`integer representation <int-repr>`.
            - `"poly"`: Sets the element representation to the :ref:`polynomial representation <poly-repr>`.
            - `"power"`: Sets the element representation to the :ref:`power representation <power-repr>`.

    Returns:
        A :obj:`~galois.FieldArray` subclass for $\mathrm{GF}(p^m)$.

    Notes:
        :obj:`~galois.FieldArray` subclasses of the same type (order, irreducible polynomial, and primitive element)
        are singletons. So, calling this class factory with arguments that correspond to the same subclass will return
        the same class object.

    Examples:
        Create a :obj:`~galois.FieldArray` subclass for each type of finite field.

        .. md-tab-set::

            .. md-tab-item:: GF(2)

                Construct the binary field.

                .. ipython:: python

                    GF = galois.GF(2)
                    print(GF.properties)

            .. md-tab-item:: GF(p)

                Construct a prime field.

                .. ipython:: python

                    GF = galois.GF(31)
                    print(GF.properties)

            .. md-tab-item:: GF(2^m)

                Construct a binary extension field. Notice the default irreducible polynomial is primitive and
                $x$ is a primitive element.

                .. ipython:: python

                    GF = galois.GF(2**8)
                    print(GF.properties)

            .. md-tab-item:: GF(p^m)

                Construct a prime extension field. Notice the default irreducible polynomial is primitive and $x$
                is a primitive element.

                .. ipython:: python

                    GF = galois.GF(3**5)
                    print(GF.properties)

        Create a :obj:`~galois.FieldArray` subclass for extension fields and specify their irreducible polynomials.

        .. md-tab-set::

            .. md-tab-item:: GF(2^m)

                Construct the $\mathrm{GF}(2^8)$ field that is used in AES. Notice the irreducible polynomial
                is not primitive and $x$ is not a primitive element.

                .. ipython:: python

                    GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
                    print(GF.properties)

            .. md-tab-item:: GF(p^m)

                Construct $\mathrm{GF}(3^5)$ with an irreducible, but not primitive, polynomial. Notice that
                $x$ is not a primitive element.

                .. ipython:: python

                    GF = galois.GF(3**5, irreducible_poly="x^5 + 2x + 2")
                    print(GF.properties)

        Finite fields with arbitrarily large orders are supported.

        .. md-tab-set::

            .. md-tab-item:: GF(p)

                Construct a large prime field.

                .. ipython:: python

                    GF = galois.GF(36893488147419103183)
                    print(GF.properties)

            .. md-tab-item:: GF(2^m)

                Construct a large binary extension field.

                .. ipython:: python

                    GF = galois.GF(2**100)
                    print(GF.properties)

                The construction of large fields can be sped up by explicitly specifying $p$ and $m$.
                This avoids the need to factor the order $p^m$.

                .. ipython:: python

                    GF = galois.GF(2, 100)
                    print(GF.properties)

            .. md-tab-item:: GF(p^m)

                Construct a large prime extension field.

                .. ipython:: python

                    GF = galois.GF(109987**4)
                    print(GF.properties)

                The construction of large fields can be sped up by explicitly specifying $p$ and $m$.
                This avoids the need to factor the order $p^m$.

                .. ipython:: python

                    GF = galois.GF(109987, 4)
                    print(GF.properties)

    Group:
        galois-fields
    """
    if len(args) == 1:
        order = args[0]
        verify_isinstance(order, int)
        p, e = factors(order)
        if not len(p) == len(e) == 1:
            s = " * ".join([f"{pi}^{ei}" for pi, ei in zip(p, e)])
            raise ValueError(f"Argument 'order' must be a prime power, not {order} = {s}.")
        characteristic, degree = p[0], e[0]
    elif len(args) == 2:
        characteristic, degree = args
        verify_isinstance(characteristic, int)
        verify_isinstance(degree, int)
    else:
        raise TypeError(
            "Only 'order' or 'characteristic' and 'degree' may be specified as positional arguments. "
            "Other arguments must be specified as keyword arguments."
        )

    verify_isinstance(verify, bool)
    verify_isinstance(compile, str, optional=True)
    verify_isinstance(repr, str, optional=True)

    if not compile in [None, "auto", "jit-lookup", "jit-calculate", "python-calculate"]:
        raise ValueError(
            f"Argument 'compile' must be in ['auto', 'jit-lookup', 'jit-calculate', 'python-calculate'], "
            f"not {compile!r}."
        )
    if not repr in [None, "int", "poly", "power"]:
        raise ValueError(f"Argument 'repr' must be in ['int', 'poly', 'power'], not {repr!r}.")

    if degree == 1:
        if not irreducible_poly is None:
            raise ValueError(
                f"Argument 'irreducible_poly' can only be specified for extension fields, "
                f"not the prime field GF({characteristic})."
            )
        field = _GF_prime(
            characteristic,
            alpha=primitive_element,
            verify=verify,
            compile=compile,
            repr=repr,
        )
    else:
        field = _GF_extension(
            characteristic,
            degree,
            irreducible_poly_=irreducible_poly,
            alpha=primitive_element,
            verify=verify,
            compile=compile,
            repr=repr,
        )

    return field


@overload
def Field(
    order: int,
    *,
    irreducible_poly: PolyLike | None = None,
    primitive_element: int | PolyLike | None = None,
    verify: bool = True,
    compile: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"] | None = None,
    repr: Literal["int", "poly", "power"] | None = None,
) -> Type[FieldArray]: ...


@overload
def Field(
    characteristic: int,
    degree: int,
    *,
    irreducible_poly: PolyLike | None = None,
    primitive_element: int | PolyLike | None = None,
    verify: bool = True,
    compile: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"] | None = None,
    repr: Literal["int", "poly", "power"] | None = None,
) -> Type[FieldArray]: ...


@export
def Field(
    *args,
    irreducible_poly=None,
    primitive_element=None,
    verify=True,
    compile=None,
    repr=None,
):
    """
    Alias of :func:`~galois.GF`.

    Group:
        galois-fields
    """
    return GF(
        *args,
        irreducible_poly=irreducible_poly,
        primitive_element=primitive_element,
        verify=verify,
        compile=compile,
        repr=repr,
    )


def _GF_prime(
    p: int,
    alpha: int | None = None,
    verify: bool = True,
    compile: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"] | None = None,
    repr: Literal["int", "poly", "power"] | None = None,
) -> Type[FieldArray]:
    """
    Class factory for prime fields GF(p).
    """
    # Get default primitive element
    if alpha is None:
        alpha = primitive_root(p)

    # Check primitive element range
    if not 0 < alpha < p:
        raise ValueError(f"Argument 'primitive_element' must be non-zero in the field 0 < x < {p}, not {alpha}.")

    # If the requested field has already been constructed, return it
    name = f"FieldArray_{p}_{alpha}"
    key = (p, alpha)
    if key in _GF_prime._classes:
        field = _GF_prime._classes[key]
        if compile is not None:
            field.compile(compile)
        if repr is not None:
            field.repr(repr)
        return field

    if verify and not is_primitive_root(alpha, p):
        raise ValueError(f"Argument 'primitive_element' must be a primitive root modulo {p}, {alpha} is not.")

    if p == 2:
        field = GF2
    else:
        field = types.new_class(
            name,
            bases=(FieldArray, UFuncMixin_p_1),
            kwds={
                "p": p,
                "m": 1,
                "characteristic": p,
                "degree": 1,
                "order": p,
                "irreducible_poly_int": 2 * p - alpha,  # f(x) = x - e
                "primitive_element": alpha,
            },
        )

    # Add the class to this module's namespace
    field.__module__ = __name__
    setattr(sys.modules[__name__], name, field)

    # Since this is a new class, compile the ufuncs and set the element representation
    field.compile("auto" if compile is None else compile)
    field.repr("int" if repr is None else repr)

    field._is_primitive_poly = field._irreducible_poly(field._primitive_element, field=field) == 0

    # Add class to dictionary of flyweights
    _GF_prime._classes[key] = field

    return field


_GF_prime._classes = {}


def _GF_extension(
    p: int,
    m: int,
    irreducible_poly_: PolyLike | None = None,
    alpha: PolyLike | None = None,
    verify: bool = True,
    compile: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"] | None = None,
    repr: Literal["int", "poly", "power"] | None = None,
) -> Type[FieldArray]:
    """
    Class factory for extension fields GF(p^m).
    """
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
            # We know `g(x) = x` is a primitive element of the Conway polynomial because Conway polynomials are
            # primitive polynomials.
            verify_element = False
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
        raise ValueError(
            f"Argument 'irreducible_poly' must be over {prime_subfield.name}, not {irreducible_poly_.field.name}."
        )
    if not irreducible_poly_.degree == m:
        raise ValueError(f"Argument 'irreducible_poly' must have degree equal to {m}, not {irreducible_poly_.degree}.")
    if not alpha.field.order == p:
        raise ValueError(
            f"Argument 'primitive_element' must be a polynomial over {prime_subfield.name}, not {alpha.field.name}."
        )
    if not alpha.degree < m:
        raise ValueError(f"Argument 'primitive_element' must have degree strictly less than {m}, not {alpha.degree}.")

    # If the requested field has already been constructed, return it
    name = f"FieldArray_{p}_{m}_{int(alpha)}_{int(irreducible_poly_)}"
    key = (p, m, int(alpha), int(irreducible_poly_))
    if key in _GF_extension._classes:
        field = _GF_extension._classes[key]
        if compile is not None:
            field.compile(compile)
        if repr is not None:
            field.repr(repr)
        return field

    if verify_poly and not irreducible_poly_.is_irreducible():
        raise ValueError(f"Argument 'irreducible_poly' must be irreducible, {irreducible_poly_} is not.")
    if verify_element and not is_primitive_element(alpha, irreducible_poly_):
        raise ValueError(f"Argument 'primitive_element' must be a multiplicative generator of {name}, {alpha} is not.")

    ufunc_mixin = UFuncMixin_2_m if p == 2 else UFuncMixin_p_m

    field = types.new_class(
        name,
        bases=(FieldArray, ufunc_mixin),
        kwds={
            "p": p,
            "m": m,
            "characteristic": p,
            "degree": m,
            "order": p**m,
            "irreducible_poly_int": int(irreducible_poly_),
            "primitive_element": int(alpha),
            "prime_subfield": prime_subfield,
        },
    )

    # Add the class to this module's namespace
    field.__module__ = __name__
    setattr(sys.modules[__name__], name, field)

    # Since this is a new class, compile the ufuncs and set the element representation
    field.compile("auto" if compile is None else compile)
    field.repr("int" if repr is None else repr)

    if is_primitive_poly is not None:
        field._is_primitive_poly = is_primitive_poly
    else:
        field._is_primitive_poly = bool(field._irreducible_poly(field._primitive_element, field=field) == 0)

    # Add class to dictionary of flyweights
    _GF_extension._classes[key] = field

    return field


_GF_extension._classes = {}
