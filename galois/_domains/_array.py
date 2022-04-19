"""
A module that defines the abstract base class Array.
"""
from __future__ import annotations

import contextlib
import inspect
import random
from typing import List, Sequence, Optional, Union, Type
from typing_extensions import Literal

import numpy as np

from .._overrides import set_module, SPHINX_BUILD

__all__ = ["Array"]


ElementLike = Union[int, str, "Array"]
ElementLike.__doc__ = """
A :obj:`~typing.Union` representing objects that can be coerced into a Galois field element.

Scalars are 0-D :obj:`~galois.Array` objects.

.. rubric:: Union

- :obj:`int`: A finite field element in its :ref:`integer representation <int repr>`.

.. ipython:: python

    GF = galois.GF(3**5)
    GF(17)

- :obj:`str`: A finite field element in its :ref:`polynomial representation <poly repr>`. Many string conventions are
  accepted, including: with/without `*`, with/without spaces, `^` or `**`, any indeterminate variable, increasing/decreasing
  degrees, etc. Or any combination of the above.

.. ipython:: python

    GF("x^2 + 2x + 2")
    # Add explicit * for multiplication
    GF("x^2 + 2*x + 2")
    # No spaces
    GF("x^2+2x+2")
    # ** instead of ^
    GF("x**2 + 2x + 2")
    # Different indeterminate
    GF("α^2 + 2α + 2")
    # Ascending degrees
    GF("2 + 2x + x^2")

- :obj:`~galois.Array`: A previously-created scalar :obj:`~galois.Array` object. No coercion is necessary.

.. rubric:: Alias
"""

IterableLike = Union[Sequence[ElementLike], Sequence["IterableLike"]]
IterableLike.__doc__ = """
A :obj:`~typing.Union` representing iterable objects that can be coerced into a Galois field array.

.. rubric:: Union

- :obj:`~typing.Sequence` [ :obj:`~galois.typing.ElementLike` ]: An iterable of elements.

.. ipython:: python

    GF = galois.GF(3**5)
    GF([17, 4])
    # Mix and match integers and strings
    GF([17, "x + 1"])

- :obj:`~typing.Sequence` [ :obj:`~galois.typing.IterableLike` ]: A recursive iterable of iterables of elements.

.. ipython:: python

    GF = galois.GF(3**5)
    GF([[17, 4], [148, 205]])
    # Mix and match integers and strings
    GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

.. rubric:: Alias
"""

ArrayLike = Union[IterableLike, np.ndarray, "Array"]
ArrayLike.__doc__ = """
A :obj:`~typing.Union` representing objects that can be coerced into a Galois field array.

.. rubric:: Union

- :obj:`~galois.typing.IterableLike`: A recursive iterable of iterables of elements.

.. ipython:: python

    GF = galois.GF(3**5)
    GF([[17, 4], [148, 205]])
    # Mix and match integers and strings
    GF([["x^2 + 2x + 2", 4], ["x^4 + 2x^3 + x^2 + x + 1", 205]])

- :obj:`~numpy.ndarray`: A NumPy array of integers, representing finite field elements in their :ref:`integer representation <int repr>`.

.. ipython:: python

    x = np.array([[17, 4], [148, 205]]); x
    GF(x)

- :obj:`~galois.Array`: A previously-created :obj:`~galois.Array` object. No coercion is necessary.

.. rubric:: Alias
"""

ShapeLike = Union[int, Sequence[int]]
ShapeLike.__doc__ = """
A :obj:`~typing.Union` representing objects that can be coerced into a NumPy :obj:`~numpy.ndarray.shape` tuple.

.. rubric:: Union

- :obj:`int`: The size of a 1-D array.

.. ipython:: python

    GF = galois.GF(3**5)
    x = GF.Random(4); x
    x.shape

- :obj:`~typing.Sequence` [ :obj:`int` ]: An iterable of integer dimensions. Tuples or lists are allowed. An empty iterable, `()` or `[]`,
  represents a 0-D array (scalar).

.. ipython:: python

    x = GF.Random((2, 3)); x
    x.shape
    x = GF.Random([2, 3, 4]); x
    x.shape
    x = GF.Random(()); x
    x.shape

.. rubric:: Alias
"""

DTypeLike = Union[np.integer, int, str, object]
DTypeLike.__doc__ = """
A :obj:`~typing.Union` representing objects that can be coerced into a NumPy data type.

.. rubric:: Union

- :obj:`numpy.integer`: A fixed-width NumPy integer data type.

.. ipython:: python

    GF = galois.GF(3**5)
    x = GF.Random(4, dtype=np.uint16); x.dtype
    x = GF.Random(4, dtype=np.int32); x.dtype

- :obj:`int`: The system default integer.

.. ipython:: python

    x = GF.Random(4, dtype=int); x.dtype

- :obj:`str`: The string that can be coerced with :obj:`numpy.dtype`.

.. ipython:: python

    x = GF.Random(4, dtype="uint16"); x.dtype
    x = GF.Random(4, dtype="int32"); x.dtype

- :obj:`object`: A Python object data type. This applies to non-compiled fields.

.. ipython:: python

    GF = galois.GF(2**100)
    x = GF.Random(4, dtype=object); x.dtype

.. rubric:: Alias
"""


DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


class Meta(type):
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

        cls._name = ""  # Needs overridden

        # A dictionary of ufuncs and LUTs
        cls._ufuncs = {}
        cls._EXP = np.array([], dtype=cls._dtypes[-1])
        cls._LOG = np.array([], dtype=cls._dtypes[-1])
        cls._ZECH_LOG = np.array([], dtype=cls._dtypes[-1])
        cls._ZECH_E = 0

        cls._functions = {}

        # Class variables needed when displaying elements with fixed width
        cls._display_mode = kwargs.get("display", "int")
        cls._element_fixed_width = None
        cls._element_fixed_width_counter = 0

        # By default, verify array elements are within the valid range when `.view()` casting
        cls._verify_on_view = True

    def __repr__(cls) -> str:
        return f"<class 'numpy.ndarray over {cls._name}'>"

    def __str__(cls) -> str:
        string = "Domain:"
        string += f"\n  name: {cls._name}"
        string += f"\n  characteristic: {cls._characteristic}"
        string += f"\n  degree: {cls._degree}"
        string += f"\n  order: {cls._order}"

        return string

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

    @contextlib.contextmanager
    def _view_without_verification(cls):
        """
        A context manager to disable verifying array element values are within [0, p^m). For internal library use only.
        """
        prev_value = cls._verify_on_view
        cls._verify_on_view = False
        yield
        cls._verify_on_view = prev_value

    def _view(cls, array: np.ndarray) -> Array:
        """
        View the input array to the FieldArray subclass using the `_view_without_verification()` context manager. This disables
        bounds checking on the array elements. Instead of `x.view(field)` use `field._view(x)`.
        """
        with cls._view_without_verification():
            array = array.view(cls)
        return array


@set_module("galois")
class Array(np.ndarray, metaclass=Meta):
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
    def _verify_array_like_types_and_values(cls, x: Union[ElementLike, ArrayLike]) -> Union[ElementLike, ArrayLike]:
        raise NotImplementedError

    @classmethod
    def _verify_element_types_and_convert(cls, array: np.ndarray, object_=False) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _verify_scalar_value(cls, scalar: int):
        raise NotImplementedError

    @classmethod
    def _verify_array_values(cls, array: np.ndarray):
        raise NotImplementedError

    ###############################################################################
    # Element conversion routines
    ###############################################################################

    @classmethod
    def _convert_to_element(cls, element: ElementLike) -> int:
        raise NotImplementedError

    @classmethod
    def _convert_iterable_to_elements(cls, iterable: IterableLike) -> np.ndarray:
        raise NotImplementedError

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
        cls._reset_ufuncs()

    @classmethod
    def _reset_ufuncs(cls):
        """
        Reset the dictionary of ufuncs, they will be re-compiled on demand.  If the ufunc mode uses lookup tables, build them if they
        haven't been built before.
        """
        cls._ufuncs = {}

        if cls._ufunc_mode == "jit-lookup" and cls._EXP.size == 0:
            cls._build_lookup_tables()

    @classmethod
    def _build_lookup_tables(cls):
        raise NotImplementedError

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


def FIELD_FACTORY(*args, **kwargs) -> Type[Array]:  # pylint: disable=unused-argument
    """
    This will be monkey-patched to be `galois.GF()` in __init__.py.
    """
    return Array


DEFAULT_FIELD_ARRAY = Array
"""
This will be monkey-patched to be `galois.GF2` in __init__.py.
"""
