"""
A module containing abstract base classes.
"""
import contextlib
from typing import List, Iterable, Optional, Union
from typing_extensions import Literal

import numpy as np

from ._overrides import set_module

__all__ = ["ArrayClass", "Array"]

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


@set_module("galois")
class ArrayClass(type):
    """
    Defines a metaclass for all :obj:`galois.Array` classes.

    Important
    ---------
    :obj:`galois.ArrayClass` is a metaclass for :obj:`galois.Array` subclasses. It cannot be instantiated directly.
    """
    # pylint: disable=no-value-for-parameter

    _name = ""
    _characteristic = 0
    _degree = 0
    _order = 0
    _dtypes: List[np.dtype] = []
    _verify_on_view = True

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._dtypes = cls._determine_dtypes()

    def __repr__(cls) -> str:
        return f"<class 'numpy.ndarray over {cls._name}'>"

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
    # Array helper methods
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

    def _view(cls, array: np.ndarray) -> "Array":
        """
        View the input array to the FieldArray subclass using the `_view_without_verification()` context manager. This disables
        bounds checking on the array elements. Instead of `x.view(field)` use `field._view(x)`.
        """
        with cls._view_without_verification():
            array = array.view(cls)
        return array


@set_module("galois")
class Array(np.ndarray, metaclass=ArrayClass):
    r"""
    An array over a finite field or ring.

    Important
    ---------
    :obj:`galois.Array` is an abstract base class for :obj:`galois.FieldArray` and cannot be instantiated directly.
    """
    # pylint: disable=no-value-for-parameter

    def __new__(
        cls,
        array_like: Union[int, str, Iterable, np.ndarray, "Array"],
        dtype: Optional[Union[np.dtype, int, object]] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ) -> "Array":
        if cls is Array:
            raise NotImplementedError("Array is an abstract base class that cannot be directly instantiated. Instead, create a Array subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` and instantiate an array using `x = GF(array_like)`.")

        dtype = cls._get_dtype(dtype)

        array_like = cls._verify_array_like_types_and_values(array_like)
        array = np.array(array_like, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

        # Perform view without verification since the elements were verified in _verify_array_like_types_and_values()
        return cls._view(array)

    @classmethod
    def _get_dtype(cls, dtype: Optional[Union[np.dtype, int, object]]) -> np.dtype:
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
    def _verify_array_like_types_and_values(cls, array_like: Union[int, str, Iterable, np.ndarray, "Array"]):
        raise NotImplementedError

    @classmethod
    def _verify_element_types_and_convert(cls, array: np.ndarray, object_=False) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def _verify_scalar_value(cls, scalar: np.ndarray):
        raise NotImplementedError

    @classmethod
    def _verify_array_values(cls, array: np.ndarray):
        raise NotImplementedError

    ###############################################################################
    # Element conversion routines
    ###############################################################################

    @classmethod
    def _convert_to_element(cls, element) -> int:
        raise NotImplementedError

    @classmethod
    def _convert_iterable_to_elements(cls, iterable: Iterable) -> np.ndarray:
        raise NotImplementedError

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


# This will be monkey-patched in __init__.py to be GF2. It can't be set to that currently due to
# circular dependencies.
DEFAULT_ARRAY = Array
