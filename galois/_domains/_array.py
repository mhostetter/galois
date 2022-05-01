"""
A module that defines the abstract base class Array.
"""
from __future__ import annotations

import abc
import contextlib
import inspect
import random
from typing import List, Optional, Union, TYPE_CHECKING
from typing_extensions import Literal

import numpy as np

from .._overrides import set_module, SPHINX_BUILD
from ..typing import ElementLike, IterableLike, ArrayLike, ShapeLike, DTypeLike

# Obtain forward references
if TYPE_CHECKING:
    from .._polys import Poly

__all__ = ["Array"]

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


class ArrayMeta(abc.ABCMeta):
    """
    A metaclass that provides class properties for `Array` subclasses.
    """
    # pylint: disable=no-value-for-parameter

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

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
            cls._ufunc_modes = ["jit-lookup", "jit-calculate"]
        else:
            cls._default_ufunc_mode = "jit-calculate"
            cls._ufunc_modes = ["jit-lookup", "jit-calculate"]
        cls._ufunc_mode = None  # This is set in the first call to compile

        cls._name = "Undefined"  # Needs overridden
        cls._is_prime_field = False  # Defaults to False for Galois rings

        # A dictionary of ufuncs and LUTs
        cls._EXP = np.array([], dtype=cls._dtypes[-1])
        cls._LOG = np.array([], dtype=cls._dtypes[-1])
        cls._ZECH_LOG = np.array([], dtype=cls._dtypes[-1])
        cls._ZECH_E = 0

        # Class variables needed when displaying elements with fixed width
        cls._display_mode = kwargs.get("display", "int")
        cls._element_fixed_width = None
        cls._element_fixed_width_counter = 0

        # By default, verify array elements are within the valid range when `.view()` casting
        cls._verify_on_view = True

    def __dir__(cls) -> List[str]:
        """
        Add class properties from the metaclass onto the new Array class's dir().
        """
        metacls = type(cls)
        classproperties = [item for item in dir(metacls) if item[0] != "_" and inspect.isdatadescriptor(getattr(metacls, item))]
        return sorted(list(super().__dir__()) + classproperties)

    ###############################################################################
    # Helper methods
    ###############################################################################

    def _determine_dtypes(cls) -> List[np.dtype]:
        """
        Determine which NumPy integer data types are valid for this finite field. At a minimum, valid dtypes are ones that
        can hold x for x in [0, order).
        """
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls._order - 1]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    ###############################################################################
    # View methods
    ###############################################################################

    def _view(cls, array: np.ndarray) -> "Array":
        """
        View the input array to the Array subclass `A` using the `_view_without_verification()` context manager. This disables
        bounds checking on the array elements. Instead of `x.view(A)` use `A._view(x)`. For internal library use only.
        """
        with cls._view_without_verification():
            array = array.view(cls)
        return array

    @contextlib.contextmanager
    def _view_without_verification(cls):
        """
        A context manager to disable verifying array element values are within [0, order). For internal library use only.
        """
        prev_value = cls._verify_on_view
        cls._verify_on_view = False
        yield
        cls._verify_on_view = prev_value

    ###############################################################################
    # Class properties
    ###############################################################################

    @property
    def name(cls) -> str:
        return cls._name

    @property
    def characteristic(cls) -> int:
        return cls._characteristic

    @property
    def degree(cls) -> int:
        return cls._degree

    @property
    def order(cls) -> int:
        return cls._order

    @property
    def irreducible_poly(cls) -> "Poly":
        return cls._irreducible_poly

    @property
    def primitive_element(cls) -> "Array":
        return cls(cls._primitive_element)

    @property
    def dtypes(cls) -> List[np.dtype]:
        return cls._dtypes

    @property
    def display_mode(cls) -> Literal["int", "poly", "power"]:
        return cls._display_mode

    @property
    def ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        return cls._ufunc_mode

    @property
    def ufunc_modes(cls) -> List[str]:
        return cls._ufunc_modes

    @property
    def default_ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        return cls._default_ufunc_mode


@set_module("galois")
class Array(np.ndarray, metaclass=ArrayMeta):
    r"""
    A :obj:`~numpy.ndarray` subclass over a Galois field or Galois ring.

    Important
    ---------
    :obj:`~galois.Array` is an abstract base class for :obj:`~galois.FieldArray` and cannot be instantiated directly.
    """

    def __new__(
        cls,
        x: Union[ElementLike, ArrayLike],
        dtype: Optional[DTypeLike] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ) -> Array:
        if not SPHINX_BUILD and cls is Array:
            raise NotImplementedError("Array is an abstract base class that cannot be directly instantiated. Instead, create a Array subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` and instantiate an array using `x = GF(array_like)`.")

        dtype = cls._get_dtype(dtype)

        x = cls._verify_array_like_types_and_values(x)
        array = np.array(x, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

        # Perform view without verification since the elements were verified in _verify_array_like_types_and_values()
        return cls._view(array)

    @classmethod
    def _get_dtype(cls, dtype: DTypeLike) -> np.dtype:
        if dtype is None:
            return cls._dtypes[0]

        # Convert "dtype" to a NumPy dtype. This does platform specific conversion, if necessary.
        # For example, np.dtype(int) == np.int64 (on some systems).
        dtype = np.dtype(dtype)
        if dtype not in cls._dtypes:
            raise TypeError(f"{cls._name} arrays only support dtypes {[np.dtype(d).name for d in cls._dtypes]}, not {dtype.name!r}.")

        return dtype

    ###############################################################################
    # Verification routines
    ###############################################################################

    @classmethod
    @abc.abstractmethod
    def _verify_array_like_types_and_values(cls, x: Union[ElementLike, ArrayLike]) -> Union[ElementLike, ArrayLike]:
        """
        Verify the types of the array-like object. Also verify the values of the array are within the range [0, order).
        """

    @classmethod
    @abc.abstractmethod
    def _verify_element_types_and_convert(cls, array: np.ndarray, object_=False) -> np.ndarray:
        """
        Iterate across each element and verify it's a valid type. Also, convert strings to integers along the way.
        """

    @classmethod
    @abc.abstractmethod
    def _verify_scalar_value(cls, scalar: int):
        """
        Verify the single integer element is within the valid range [0, order).
        """

    @classmethod
    @abc.abstractmethod
    def _verify_array_values(cls, array: np.ndarray):
        """
        Verify all the elements of the integer array are within the valid range [0, order).
        """

    ###############################################################################
    # Element conversion routines
    ###############################################################################

    @classmethod
    @abc.abstractmethod
    def _convert_to_element(cls, element: ElementLike) -> int:
        """
        Convert any element-like value to an integer.
        """

    @classmethod
    @abc.abstractmethod
    def _convert_iterable_to_elements(cls, iterable: IterableLike) -> np.ndarray:
        """
        Convert an iterable (recursive) to a NumPy integer array. Convert any strings to integers along the way.
        """

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(cls, shape: ShapeLike, dtype: Optional[DTypeLike] = None) -> "Array":
        """
        Creates an array of all zeros.

        Parameters
        ----------
        shape
            A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.Array` subclass (the first element in :obj:`~galois.Array.dtypes`).

        Returns
        -------
        :
            An array of zeros.
        """
        dtype = cls._get_dtype(dtype)
        array = np.zeros(shape, dtype=dtype)
        return cls._view(array)

    @classmethod
    def Ones(cls, shape: ShapeLike, dtype: Optional[DTypeLike] = None) -> "Array":
        """
        Creates an array of all ones.

        Parameters
        ----------
        shape
            A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.Array` subclass (the first element in :obj:`~galois.Array.dtypes`).

        Returns
        -------
        :
            An array of ones.
        """
        dtype = cls._get_dtype(dtype)
        array = np.ones(shape, dtype=dtype)
        return cls._view(array)

    @classmethod
    def Range(
        cls,
        start: ElementLike,
        stop: ElementLike,
        step: int = 1,
        dtype: Optional[DTypeLike] = None
    ) -> "Array":
        """
        Creates a 1-D array with a range of elements.

        Parameters
        ----------
        start
            The starting element (inclusive).
        stop
            The stopping element (exclusive).
        step
            The increment between elements. The default is 1.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.Array` subclass (the first element in :obj:`~galois.Array.dtypes`).

        Returns
        -------
        :
            A 1-D array of a range of elements.
        """
        # Coerce element-like values to integers in [0, order)
        if start != cls._order:
            start = int(cls(start))
        if stop != cls._order:
            stop = int(cls(stop))
        dtype = cls._get_dtype(dtype)

        if not 0 <= start <= cls._order:
            raise ValueError(f"Argument `start` must be within the field's order {cls._order}, not {start}.")
        if not 0 <= stop <= cls._order:
            raise ValueError(f"Argument `stop` must be within the field's order {cls._order}, not {stop}.")

        array = np.arange(start, stop, step=step, dtype=dtype)

        return cls._view(array)

    @classmethod
    def Random(
        cls,
        shape: ShapeLike = (),
        low: ElementLike = 0,
        high: Optional[ElementLike] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: Optional[DTypeLike] = None
    ) -> "Array":
        """
        Creates an array with random elements.

        Parameters
        ----------
        shape
            A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple. The default is `()` which represents a scalar.
        low
            The smallest element (inclusive). The default is 0.
        high
            The largest element (exclusive). The default is `None` which represents :obj:`~galois.Array.order`.
        seed
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.Array` subclass (the first element in :obj:`~galois.Array.dtypes`).

        Returns
        -------
        :
            An array of random elements.
        """
        # Coerce element-like values to integers in [0, p^m)
        low = int(cls(low))
        if high is None:
            high = cls._order
        elif high != cls._order:
            high = int(cls(high))
        dtype = cls._get_dtype(dtype)

        if not 0 <= low < high <= cls._order:
            raise ValueError(f"Arguments must satisfy `0 <= low < high <= order`, not `0 <= {low} < {high} <= {cls._order}`.")

        if seed is not None:
            if not isinstance(seed, (int, np.integer, np.random.Generator)):
                raise ValueError("Seed must be an integer, a numpy.random.Generator or None.")
            if isinstance(seed, (int, np.integer)) and seed < 0:
                raise ValueError("Seed must be non-negative.")

        if dtype != np.object_:
            rng = np.random.default_rng(seed)
            array = rng.integers(low, high, shape, dtype=dtype)
        else:
            array = np.empty(shape, dtype=dtype)
            iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
            _seed = None
            if seed is not None:
                if isinstance(seed, np.integer):
                    # np.integers not supported by random and seeding based on hashing deprecated since Python 3.9
                    _seed = seed.item()
                elif isinstance(seed, np.random.Generator):
                    _seed = seed.bit_generator.state['state']['state']
                    seed.bit_generator.advance(1)
                else:  # int
                    _seed = seed
            random.seed(_seed)
            for _ in iterator:
                array[iterator.multi_index] = random.randint(low, high - 1)

        return cls._view(array)

    @classmethod
    def Elements(cls, dtype: Optional[DTypeLike] = None) -> "Array":
        return cls.Range(0, cls._order, step=1, dtype=dtype)

    @classmethod
    def Identity(cls, size: int, dtype: Optional[DTypeLike] = None) -> "Array":
        r"""
        Creates an :math:`n \times n` identity matrix.

        Parameters
        ----------
        size
            The size :math:`n` along one dimension of the identity matrix.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.Array` subclass (the first element in :obj:`~galois.Array.dtypes`).

        Returns
        -------
        :
            A 2-D identity matrix with shape `(size, size)`.
        """
        dtype = cls._get_dtype(dtype)
        array = np.identity(size, dtype=dtype)
        return cls._view(array)

    ###############################################################################
    # Ufunc compilation routines
    ###############################################################################

    @classmethod
    def compile(cls, mode: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]):
        """
        Recompile the just-in-time compiled ufuncs for a new calculation mode.

        This function updates :obj:`ufunc_mode`.

        Parameters
        ----------
        mode
            The ufunc calculation mode.

            - `"auto"`: Selects `"jit-lookup"` for fields with order less than :math:`2^{20}`, `"jit-calculate"` for larger fields, and `"python-calculate"`
              for fields whose elements cannot be represented with :obj:`numpy.int64`.
            - `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
              In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
            - `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"` mode is designed for large
              fields that cannot or should not store lookup tables in RAM. Generally, the `"jit-calculate"` mode is slower than `"jit-lookup"`.
            - `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
              represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_` with Python :obj:`int` (which has arbitrary precision).
        """
        if not isinstance(mode, str):
            raise TypeError(f"Argument `mode` must be a string, not {type(mode)}.")
        if not mode in ["auto", "jit-lookup", "jit-calculate", "python-calculate"]:
            raise ValueError(f"Argument `mode` must be in ['auto', 'jit-lookup', 'jit-calculate', 'python-calculate'], not {mode!r}.")
        mode = cls._default_ufunc_mode if mode == "auto" else mode
        if mode not in cls._ufunc_modes:
            raise ValueError(f"Argument `mode` must be in {cls._ufunc_modes} for {cls._name}, not {mode!r}.")

        if mode == cls._ufunc_mode:
            # Don't need to rebuild these ufuncs
            return

        cls._ufunc_mode = mode

        if cls._ufunc_mode == "jit-lookup" and cls._EXP.size == 0:
            cls._build_lookup_tables()

    @classmethod
    @abc.abstractmethod
    def _build_lookup_tables(cls):
        ...

    ###############################################################################
    # Element display methods
    ###############################################################################

    @classmethod
    def display(cls, mode: Literal["int", "poly", "power"] = "int") -> contextlib.AbstractContextManager:
        r"""
        Sets the display mode for all arrays from this :obj:`~galois.FieldArray` subclass.

        The display mode can be set to either the integer representation, polynomial representation, or power
        representation. See :doc:`/basic-usage/element-representation` for a further discussion.

        This function updates :obj:`~galois.FieldArray.display_mode`.

        Warning
        -------
        For the power representation, :func:`numpy.log` is computed on each element. So for large fields without lookup
        tables, displaying arrays in the power representation may take longer than expected.

        Parameters
        ----------
        mode
            The field element representation.

            - `"int"`: Sets the display mode to the :ref:`integer representation <int repr>`.
            - `"poly"`: Sets the display mode to the :ref:`polynomial representation <poly repr>`.
            - `"power"`: Sets the display mode to the :ref:`power representation <power repr>`.

        Returns
        -------
        :
            A context manager for use in a `with` statement. If permanently setting the display mode, disregard the
            return value.

        Examples
        --------
        The default display mode is the integer representation.

        .. ipython:: python

            GF = galois.GF(3**2)
            x = GF.Elements(); x

        Permanently set the display mode by calling :func:`display`.

        .. tab-set::

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF.display("poly");
                    x

            .. tab-item:: Power

                .. ipython:: python

                    GF.display("power");
                    x
                    @suppress
                    GF.display()

        Temporarily modify the display mode by using :func:`display` as a context manager.

        .. tab-set::

            .. tab-item:: Polynomial

                .. ipython:: python

                    print(x)
                    with GF.display("poly"):
                        print(x)
                    # Outside the context manager, the display mode reverts to its previous value
                    print(x)

            .. tab-item:: Power

                .. ipython:: python

                    print(x)
                    with GF.display("power"):
                        print(x)
                    # Outside the context manager, the display mode reverts to its previous value
                    print(x)
                    @suppress
                    GF.display()
        """
        if not isinstance(mode, (type(None), str)):
            raise TypeError(f"Argument `mode` must be a string, not {type(mode)}.")
        if mode not in ["int", "poly", "power"]:
            raise ValueError(f"Argument `mode` must be in ['int', 'poly', 'power'], not {mode!r}.")

        prev_mode = cls._display_mode
        cls._display_mode = mode

        @set_module("galois")
        class context(contextlib.AbstractContextManager):
            """Simple display_mode context manager."""
            def __init__(self, mode):
                self.mode = mode

            def __enter__(self):
                # Don't need to do anything, we already set the new mode in the display() method
                pass

            def __exit__(self, exc_type, exc_value, traceback):
                cls._display_mode = self.mode

        return context(prev_mode)

    ###############################################################################
    # NumPy getter/setter functions that need redefined
    ###############################################################################

    def __getitem__(self, key):
        """
        Ensure that slices that return a single value return a 0-D Galois field array and not a single integer. This
        ensures subsequent arithmetic with the finite field scalar works properly.
        """
        item = super().__getitem__(key)
        if np.isscalar(item):
            item = self.__class__(item, dtype=self.dtype)
        return item

    def __setitem__(self, key, value):
        """
        Before assigning new values to a Galois field array, ensure the values are valid finite field elements. That is,
        they are within [0, p^m).
        """
        value = self._verify_array_like_types_and_values(value)
        super().__setitem__(key, value)

    def __array_finalize__(self, obj):
        """
        A NumPy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        if obj is not None and not isinstance(obj, Array):
            # Only invoked on view casting
            if type(self)._verify_on_view:
                if obj.dtype not in type(self)._dtypes:
                    raise TypeError(f"{type(self)._name} can only have integer dtypes {type(self)._dtypes}, not {obj.dtype}.")
                self._verify_array_values(obj)

    def __pow__(self, other):
        # We call power here instead of `super().__pow__(other)` because when doing so `x ** GF(2)` will invoke `np.square(x)`
        # and not throw a TypeError. This way `np.power(x, GF(2))` is called which correctly checks whether the second argument
        # is an integer.
        return np.power(self, other)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        # Before changing the array's data type, ensure it is a supported data type for this Array class.
        if dtype not in type(self)._dtypes:
            raise TypeError(f"{type(self)._name} arrays can only be cast as integer dtypes in {type(self)._dtypes}, not {dtype}.")
        return super().astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
