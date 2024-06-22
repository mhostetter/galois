"""
A module that defines the metaclass for the abstract base class Array.
"""

from __future__ import annotations

import abc
import inspect
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Literal

# Obtain forward references
if TYPE_CHECKING:
    from .._polys import Poly
    from ._array import Array

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


class ArrayMeta(abc.ABCMeta):
    """
    A metaclass that provides class properties for `Array` subclasses.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._characteristic: int = kwargs.get("characteristic", 0)
        cls._degree: int = kwargs.get("degree", 1)
        cls._order: int = kwargs.get("order", 0)
        cls._irreducible_poly_int: int = kwargs.get("irreducible_poly_int", 0)
        cls._primitive_element: int = kwargs.get("primitive_element", 0)
        cls._dtypes = cls._determine_dtypes()

        if cls._dtypes == [np.object_]:
            cls._default_ufunc_mode = "python-calculate"
            cls._ufunc_modes = ["python-calculate"]
        elif cls._order <= 2**20:
            cls._default_ufunc_mode = "jit-lookup"
            cls._ufunc_modes = ["jit-lookup", "jit-calculate", "python-calculate"]
        else:
            cls._default_ufunc_mode = "jit-calculate"
            cls._ufunc_modes = ["jit-lookup", "jit-calculate", "python-calculate"]
        cls._ufunc_mode = None  # This is set in the first call to compile

        cls._name = "Undefined"  # Needs overridden
        cls._is_prime_field = False  # Defaults to False for Galois rings

        # A dictionary of ufuncs and LUTs
        cls._EXP = np.array([], dtype=cls._dtypes[-1])
        cls._LOG = np.array([], dtype=cls._dtypes[-1])
        cls._ZECH_LOG = np.array([], dtype=cls._dtypes[-1])
        cls._ZECH_E = 0

        # Class variables needed when displaying elements with fixed width
        cls._element_repr = kwargs.get("repr", "int")  # TODO: Do this here?
        cls._element_fixed_width = None
        cls._element_fixed_width_counter = 0

        # By default, verify array elements are within the valid range when `.view()` casting
        cls._verify_on_view = True

        cls._assign_ufuncs()

    def __repr__(cls) -> str:
        if cls.order == 0:
            # This is not a runtime-created subclass, so return the base class name.
            return f"<class 'galois.{cls.__name__}'>"

        # When FieldArray instances are created they are added to the `galois._fields._factory` module with a name
        # like `FieldArray_<p>_<primitive_element>` or `FieldArray_<p>_<m>_<primitive_element>_<irreducible_poly>`.
        # This is visually unappealing. So here we override the repr() to be more succinct and indicate how the class
        # was created. So galois._fields._factory.FieldArray_31_3 is converted to galois.GF(31).
        return f"<class 'galois.{cls.name}'>"

    def __dir__(cls) -> list[str]:
        """
        Add class properties from the metaclass onto the new Array class's dir().
        """
        metaclass = type(cls)
        class_properties = [
            item for item in dir(metaclass) if item[0] != "_" and inspect.isdatadescriptor(getattr(metaclass, item))
        ]
        return sorted(list(super().__dir__()) + class_properties)

    ###############################################################################
    # Helper methods
    ###############################################################################

    def _determine_dtypes(cls) -> list[np.dtype]:
        """
        Determine which NumPy integer data types are valid for this finite field. At a minimum, valid dtypes are ones
        that can hold x for x in [0, order).
        """
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls._order - 1]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    def _assign_ufuncs(cls):
        # This will be implemented in UFuncMixin and its children
        return

    ###############################################################################
    # Class properties
    ###############################################################################

    @property
    def name(cls) -> str:
        """
        The name of the Galois field or Galois ring.
        """
        return cls._name

    @property
    def characteristic(cls) -> int:
        r"""
        The characteristic $p$ of the Galois field $\mathrm{GF}(p^m)$ or $p^e$ of the Galois ring
        $\mathrm{GR}(p^e, m)$.
        """
        return cls._characteristic

    @property
    def degree(cls) -> int:
        r"""
        The degree $m$ of the Galois field $\mathrm{GF}(p^m)$ or Galois ring $\mathrm{GR}(p^e, m)$.
        """
        return cls._degree

    @property
    def order(cls) -> int:
        r"""
        The order $p^m$ of the Galois field $\mathrm{GF}(p^m)$ or $p^{em}$ of the Galois ring
        $\mathrm{GR}(p^e, m)$.
        """
        return cls._order

    @property
    def irreducible_poly(cls) -> Poly:
        """
        The irreducible polynomial of the Galois field or Galois ring.
        """
        return cls._irreducible_poly

    @property
    def elements(cls) -> Array:
        """
        All elements of the Galois field or Galois ring.
        """
        return cls.Range(0, cls.order, step=1)

    @property
    def units(cls) -> Array:
        """
        All units of the Galois field or Galois ring.
        """
        return cls.Range(1, cls.order, step=1)

    @property
    def primitive_element(cls) -> Array:
        """
        A primitive element of the Galois field or Galois ring.
        """
        return cls(cls._primitive_element)

    @property
    def dtypes(cls) -> list[np.dtype]:
        """
        List of valid integer :obj:`numpy.dtype` values that are compatible with this Galois field or Galois ring.
        """
        return cls._dtypes

    @property
    def element_repr(cls) -> Literal["int", "poly", "power"]:
        """
        The current element representation of the Galois field or Galois ring.
        """
        return cls._element_repr

    @property
    def ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The current compilation mode of the Galois field or Galois ring.
        """
        return cls._ufunc_mode

    @property
    def ufunc_modes(cls) -> list[str]:
        """
        All supported compilation modes of the Galois field or Galois ring.
        """
        return cls._ufunc_modes

    @property
    def default_ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The default compilation mode of the Galois field or Galois ring.
        """
        return cls._default_ufunc_mode
