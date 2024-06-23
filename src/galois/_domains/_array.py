"""
A module that defines the abstract base class Array.
"""

from __future__ import annotations

import abc
import contextlib
import random
from typing import Generator

import numpy as np
import numpy.typing as npt
from typing_extensions import Literal, Self

from .._helper import export, verify_isinstance, verify_literal
from ..typing import ArrayLike, DTypeLike, ElementLike, IterableLike, ShapeLike
from ._function import FunctionMixin
from ._linalg import LinalgFunctionMixin
from ._lookup import UFuncMixin
from ._meta import ArrayMeta


@export
class Array(LinalgFunctionMixin, FunctionMixin, UFuncMixin, np.ndarray, metaclass=ArrayMeta):
    r"""
    An abstract :obj:`~numpy.ndarray` subclass over a Galois field or Galois ring.

    .. abstract::

        :obj:`~galois.Array` is an abstract base class for :obj:`~galois.FieldArray` and cannot be
        instantiated directly.

    Group:
        arrays
    """

    def __new__(
        cls,
        x: ElementLike | ArrayLike,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0,
    ) -> Self:
        if cls is Array:
            raise NotImplementedError(
                "Array is an abstract base class that cannot be directly instantiated. "
                "Instead, create an Array subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` "
                "and instantiate an array using `x = GF(array_like)`."
            )

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
            raise TypeError(
                f"{cls._name} arrays only support dtypes {[np.dtype(d).name for d in cls._dtypes]}, not {dtype.name!r}."
            )

        return dtype

    ###############################################################################
    # Verification routines
    ###############################################################################

    @classmethod
    @abc.abstractmethod
    def _verify_array_like_types_and_values(cls, x: ElementLike | ArrayLike) -> ElementLike | ArrayLike:
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
    # View methods
    ###############################################################################

    @classmethod
    def _view(cls, array: np.ndarray) -> Self:
        """
        View the input array to the Array subclass `A` using the `_view_without_verification()` context manager.
        This disables bounds checking on the array elements. Instead of `x.view(A)` use `A._view(x)`.

        For internal library use only.
        """
        with cls._view_without_verification():
            array = array.view(cls)
        return array

    @classmethod
    @contextlib.contextmanager
    def _view_without_verification(cls):
        """
        A context manager to disable verifying array element values are within [0, order).

        For internal library use only.
        """
        prev_value = cls._verify_on_view
        cls._verify_on_view = False
        yield
        cls._verify_on_view = prev_value

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(cls, shape: ShapeLike, dtype: DTypeLike | None = None) -> Self:
        """
        Creates an array of all zeros.

        Arguments:
            shape: A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.Array` subclass (the first element in
                :obj:`~galois.Array.dtypes`).

        Returns:
            An array of zeros.
        """
        dtype = cls._get_dtype(dtype)
        array = np.zeros(shape, dtype=dtype)
        return cls._view(array)

    @classmethod
    def Ones(cls, shape: ShapeLike, dtype: DTypeLike | None = None) -> Self:
        """
        Creates an array of all ones.

        Arguments:
            shape: A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.Array` subclass (the first element in
                :obj:`~galois.Array.dtypes`).

        Returns:
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
        dtype: DTypeLike | None = None,
    ) -> Self:
        """
        Creates a 1-D array with a range of elements.

        Arguments:
            start: The starting element (inclusive).
            stop: The stopping element (exclusive).
            step: The increment between elements. The default is 1.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.Array` subclass (the first element in
                :obj:`~galois.Array.dtypes`).

        Returns:
            A 1-D array of a range of elements.
        """
        # Coerce element-like values to integers in [0, order)
        if start != cls.order:
            start = int(cls(start))
        if stop != cls.order:
            stop = int(cls(stop))
        dtype = cls._get_dtype(dtype)

        if not 0 <= start <= cls.order:
            raise ValueError(f"Argument 'start' must be within the field's order {cls.order}, not {start}.")
        if not 0 <= stop <= cls.order:
            raise ValueError(f"Argument 'stop' must be within the field's order {cls.order}, not {stop}.")

        array = np.arange(start, stop, step=step, dtype=dtype)

        return cls._view(array)

    @classmethod
    def Random(
        cls,
        shape: ShapeLike = (),
        low: ElementLike = 0,
        high: ElementLike | None = None,
        seed: int | np.random.Generator | None = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        """
        Creates an array with random elements.

        Arguments:
            shape: A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple. The default is `()` which represents a scalar.
            low: The smallest element (inclusive). The default is 0.
            high: The largest element (exclusive). The default is `None` which represents :obj:`~galois.Array.order`.
            seed: Non-negative integer used to initialize the PRNG. The default is `None` which means that
                unpredictable entropy will be pulled from the OS to be used as the seed.
                A :obj:`numpy.random.Generator` can also be passed.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.Array` subclass (the first element in
                :obj:`~galois.Array.dtypes`).

        Returns:
            An array of random elements.
        """
        # Coerce element-like values to integers in [0, p^m)
        low = int(cls(low))
        if high is None:
            high = cls.order
        elif high != cls.order:
            high = int(cls(high))
        dtype = cls._get_dtype(dtype)

        if not 0 <= low < high <= cls.order:
            raise ValueError(
                f"Arguments must satisfy `0 <= low < high <= order`, not `0 <= {low} < {high} <= {cls.order}`."
            )

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
                    _seed = seed.bit_generator.state["state"]["state"]
                    seed.bit_generator.advance(1)
                else:  # int
                    _seed = seed
            random.seed(_seed)
            for _ in iterator:
                array[iterator.multi_index] = random.randint(low, high - 1)

        return cls._view(array)

    @classmethod
    def Identity(cls, size: int, dtype: DTypeLike | None = None) -> Self:
        r"""
        Creates an $n \times n$ identity matrix.

        Arguments:
            size: The size $n$ along one dimension of the identity matrix.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.Array` subclass (the first element in
                :obj:`~galois.Array.dtypes`).

        Returns:
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

        Arguments:
            mode: The ufunc calculation mode.

                - `"auto"`: Selects `"jit-lookup"` for fields with order less than $2^{20}$, `"jit-calculate"`
                  for larger fields, and `"python-calculate"` for fields whose elements cannot be represented with
                  :obj:`numpy.int64`.
                - `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for
                  efficient computation. In the few cases where explicit calculation is faster than table lookup,
                  explicit calculation is used.
                - `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"`
                  mode is designed for large fields that cannot or should not store lookup tables in RAM. Generally,
                  the `"jit-calculate"` mode is slower than `"jit-lookup"`.
                - `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is intended for fields
                  whose elements cannot be represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_`
                  with Python :obj:`int` (which has arbitrary precision). However, this mode can be used for any
                  field, enabling the code to run without Numba JIT compilation.
        """
        verify_isinstance(mode, str)
        if not mode in ["auto", "jit-lookup", "jit-calculate", "python-calculate"]:
            raise ValueError(
                f"Argument 'mode' must be in ['auto', 'jit-lookup', 'jit-calculate', 'python-calculate'], not {mode!r}."
            )
        mode = cls.default_ufunc_mode if mode == "auto" else mode
        if mode not in cls.ufunc_modes:
            raise ValueError(f"Argument 'mode' must be in {cls.ufunc_modes} for {cls._name}, not {mode!r}.")

        if mode == cls.ufunc_mode:
            # Don't need to rebuild these ufuncs
            return

        cls._ufunc_mode = mode

        if cls.ufunc_mode == "jit-lookup" and cls._EXP.size == 0:
            cls._build_lookup_tables()

    ###############################################################################
    # Element display methods
    ###############################################################################

    @classmethod
    def repr(cls, element_repr: Literal["int", "poly", "power"] = "int") -> Generator[None, None, None]:
        r"""
        Sets the element representation for all arrays from this :obj:`~galois.FieldArray` subclass.

        Arguments:
            element_repr: The field element representation.

                - `"int"` (default): Sets the representation to the :ref:`integer representation <int-repr>`.
                - `"poly"`: Sets the representation to the :ref:`polynomial representation <poly-repr>`.
                - `"power"`: Sets the representation to the :ref:`power representation <power-repr>`.

                .. slow-performance::

                    To display elements in the power representation, :obj:`galois` must compute the discrete logarithm
                    of each element displayed. For large fields or fields using
                    :ref:`explicit calculation <explicit-calculation>`, this process can take a while. However, when
                    using :ref:`lookup tables <lookup-tables>` this representation is just as fast as the others.

        Returns:
            A context manager for use in a `with` statement. If permanently setting the element representation,
            disregard the return value.

        Notes:
            This function updates :obj:`~galois.FieldArray.element_repr`.
        """
        verify_literal(element_repr, ["int", "poly", "power"])

        prev_element_repr = cls.element_repr
        cls._element_repr = element_repr

        # Return a context manager for optional use in a `with` statement that will reset the element representation
        # to its original value
        return cls._repr_context_manager(prev_element_repr)

    @classmethod
    @contextlib.contextmanager
    def _repr_context_manager(cls, element_repr: Literal["int", "poly", "power"]):
        yield
        cls._element_repr = element_repr

    ###############################################################################
    # Override getters/setters and type conversion functions
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
        Before assigning new values to a Galois field array, ensure the values are valid finite field elements.
        That is, they are within [0, p^m).
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
                    raise TypeError(
                        f"{type(self)._name} can only have integer dtypes {type(self)._dtypes}, not {obj.dtype}."
                    )
                self._verify_array_values(obj)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        # Before changing the array's data type, ensure it is a supported data type for this Array class.
        if dtype not in type(self)._dtypes:
            raise TypeError(
                f"{type(self)._name} arrays can only be cast as integer dtypes in {type(self)._dtypes}, not {dtype}."
            )
        return super().astype(dtype, order=order, casting=casting, subok=subok, copy=copy)

    ###############################################################################
    # Override arithmetic operators so type checking is appeased
    ###############################################################################

    def __add__(self, other: npt.NDArray) -> Self:
        return super().__add__(other)

    def __iadd__(self, other: npt.NDArray) -> Self:
        return super().__iadd__(other)

    def __radd__(self, other: npt.NDArray) -> Self:
        return super().__radd__(other)

    def __sub__(self, other: npt.NDArray) -> Self:
        return super().__sub__(other)

    def __isub__(self, other: npt.NDArray) -> Self:
        return super().__isub__(other)

    def __rsub__(self, other: npt.NDArray) -> Self:
        return super().__rsub__(other)

    def __mul__(self, other: int | npt.NDArray) -> Self:
        return super().__mul__(other)

    def __imul__(self, other: int | npt.NDArray) -> Self:
        return super().__imul__(other)

    def __rmul__(self, other: int | npt.NDArray) -> Self:
        return super().__rmul__(other)

    def __truediv__(self, other: npt.NDArray) -> Self:
        return super().__truediv__(other)

    def __itruediv__(self, other: npt.NDArray) -> Self:
        return super().__itruediv__(other)

    def __rtruediv__(self, other: npt.NDArray) -> Self:
        return super().__rtruediv__(other)

    def __floordiv__(self, other: npt.NDArray) -> Self:
        return super().__floordiv__(other)

    def __ifloordiv__(self, other: npt.NDArray) -> Self:
        return super().__ifloordiv__(other)

    def __rfloordiv__(self, other: npt.NDArray) -> Self:
        return super().__rfloordiv__(other)

    def __neg__(self) -> Self:
        return super().__neg__()

    def __mod__(self, other: npt.NDArray) -> Self:
        return super().__mod__(other)

    def __imod__(self, other: npt.NDArray) -> Self:
        return super().__imod__(other)

    def __rmod__(self, other: npt.NDArray) -> Self:
        return super().__rmod__(other)

    def __pow__(self, other: int | npt.NDArray) -> Self:
        return super().__pow__(other)

    def __ipow__(self, other: int | npt.NDArray) -> Self:
        return super().__ipow__(other)

    # def __rpow__(self, other) -> Self:
    #     return super().__rpow__(other)

    def __matmul__(self, other: npt.NDArray) -> Self:
        return super().__matmul__(other)

    def __rmatmul__(self, other: npt.NDArray) -> Self:
        return super().__rmatmul__(other)

    def __lshift__(self, other: int | npt.NDArray) -> Self:
        return super().__lshift__(other)

    def __ilshift__(self, other: int | npt.NDArray) -> Self:
        return super().__ilshift__(other)

    # def __rlshift__(self, other) -> Self:
    #     return super().__rlshift__(other)

    def __rshift__(self, other: int | npt.NDArray) -> Self:
        return super().__rshift__(other)

    def __irshift__(self, other: int | npt.NDArray) -> Self:
        return super().__irshift__(other)

    # def __rrshift__(self, other) -> Self:
    #     return super().__rrshift__(other)
