from __future__ import annotations

import contextlib
import inspect
from typing import List

import numpy as np

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


class ArrayMeta(type):
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

    def _view(cls, array: np.ndarray) -> "Array":
        """
        View the input array to the FieldArray subclass using the `_view_without_verification()` context manager. This disables
        bounds checking on the array elements. Instead of `x.view(field)` use `field._view(x)`.
        """
        with cls._view_without_verification():
            array = array.view(cls)
        return array
