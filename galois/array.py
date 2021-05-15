"""
A module that defines Array which is a base class for GroupArray, RingArray, and FieldArray.
"""
import numpy as np

from .meta import Meta


class Array(np.ndarray, metaclass=Meta):
    """
    A :obj:`numpy.ndarray` subclass and base class for :obj:`GroupArray`, :obj:`RingArray`, and :obj:`FieldArray`.
    """

    def __new__(cls, array, dtype=None, copy=True, order="K", ndmin=0):
        if cls is Array:
            raise NotImplementedError("Array is an abstract base class that cannot be directly instantiated. Instead, create a Array subclass for GF(p^m) arithmetic using `galois.GF(p**m)`.")
        return cls._array(array, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    @classmethod
    def _get_dtype(cls, dtype):
        if dtype is None:
            return cls.dtypes[0]

        # Convert "dtype" to a numpy dtype. This does platform specific conversion, if necessary.
        # For example, np.dtype(int) == np.int64 (on some systems).
        dtype = np.dtype(dtype)
        if dtype not in cls.dtypes:
            raise TypeError(f"{cls.name} arrays only support dtypes {[np.dtype(d).name for d in cls.dtypes]}, not '{dtype.name}'.")

        return dtype

    @classmethod
    def _array(cls, array_like, dtype=None, copy=True, order="K", ndmin=0):
        dtype = cls._get_dtype(dtype)
        array_like = cls._check_array_like_object(array_like)
        array = np.array(array_like, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
        return array.view(cls)

    @classmethod
    def _check_array_like_object(cls, array_like):
        if isinstance(array_like, str):
            # Convert the string to an integer
            array_like = cls._check_string_value(array_like)

        if isinstance(array_like, (int, np.integer)):
            # Just check that the single int is in range
            cls._check_array_values(array_like)

        elif isinstance(array_like, (list, tuple)):
            # Recursively check the items in the iterable to ensure they're of the correct type
            # and that their values are in range
            array_like = cls._check_iterable_types_and_values(array_like)

        elif isinstance(array_like, np.ndarray):
            if array_like.dtype == np.object_:
                array_like = cls._check_array_types_dtype_object(array_like)
            elif not np.issubdtype(array_like.dtype, np.integer):
                raise TypeError(f"{cls.name} arrays must have integer dtypes, not {array_like.dtype}.")
            cls._check_array_values(array_like)

        else:
            raise TypeError(f"{cls.name} arrays can be created with scalars of type int, not {type(array_like)}.")

        return array_like

    @classmethod
    def _check_iterable_types_and_values(cls, iterable):
        new_iterable = []
        for item in iterable:
            if isinstance(item, (list, tuple)):
                item = cls._check_iterable_types_and_values(item)
                new_iterable.append(item)
                continue

            if isinstance(item, str):
                item = cls._check_string_value(item)
            elif not isinstance(item, (int, np.integer, cls)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with an iterable, each element must be an integer. Found type {type(item)}.")

            cls._check_array_values(item)
            # if not 0 <= item < cls.order:
            #     raise ValueError(f"{cls.name} arrays must have elements in 0 <= x < {cls.order}, not {item}.")

            # Ensure the type is int so dtype=object classes don't get all mixed up
            new_iterable.append(int(item))

        return new_iterable

    @classmethod
    def _check_array_types_dtype_object(cls, array):
        if array.size == 0:
            return array
        if array.ndim == 0:
            if not isinstance(array[()], (int, np.integer, cls)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with dtype=object, each element must be an integer. Found type {type(array[()])}.")
            return int(array)

        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for _ in iterator:
            a = array[iterator.multi_index]
            if not isinstance(a, (int, np.integer, cls)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with dtype=object, each element must be an integer. Found type {type(a)}.")

            # Ensure the type is int so dtype=object classes don't get all mixed up
            array[iterator.multi_index] = int(a)

        return array

    @classmethod
    def _check_array_values(cls, array):
        raise NotImplementedError

    @classmethod
    def _check_string_value(cls, string):
        raise NotImplementedError

    ###############################################################################
    # Overridden numpy methods
    ###############################################################################

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        if dtype not in type(self).dtypes:
            raise TypeError(f"{type(self).name} arrays can only be cast as integer dtypes in {type(self).dtypes}, not {dtype}.")
        return super().astype(dtype, **kwargs)

    def __array_finalize__(self, obj):
        """
        A numpy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        if obj is not None and not isinstance(obj, Array):
            # Only invoked on view casting
            if obj.dtype not in type(self).dtypes:
                raise TypeError(f"{type(self).name} can only have integer dtypes {type(self).dtypes}, not {obj.dtype}.")
            self._check_array_values(obj)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if np.isscalar(item):
            # Return scalar array elements as 0-dimension Galois field arrays. This enables Galois field arithmetic
            # on scalars, which would otherwise be implemented using standard integer arithmetic.
            item = self.__class__(item, dtype=self.dtype)
        return item

    def __setitem__(self, key, value):
        # Verify the values to be written to the Galois field array are in the field
        value = self._check_array_like_object(value)
        super().__setitem__(key, value)

    def __array_function__(self, func, types, args, kwargs):
        if func in type(self)._overridden_functions:
            output = getattr(type(self), type(self)._overridden_functions[func])(*args, **kwargs)

        elif func in type(self)._overridden_linalg_functions:
            output = type(self)._overridden_linalg_functions[func](*args, **kwargs)

        elif func in type(self)._unsupported_functions:
            raise NotImplementedError(f"The numpy function '{func.__name__}' is not supported on Galois field arrays. If you believe this function should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\nIf you'd like to perform this operation on the data (but not necessarily a Galois field array), you should first call `array = array.view(np.ndarray)` and then call the function.")

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._check_array_like_object(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in type(self)._functions_requiring_view:
                output = output.view(type(self)) if not np.isscalar(output) else type(self)(output, dtype=self.dtype)

        return output

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # pylint: disable=too-many-branches
        meta = {}
        meta["types"] = [type(inputs[i]) for i in range(len(inputs))]
        meta["operands"] = list(range(len(inputs)))
        if method in ["at", "reduceat"]:
            # Remove the second argument for "at" ufuncs which is the indices list
            meta["operands"].pop(1)
        meta["field_operands"] = [i for i in meta["operands"] if isinstance(inputs[i], self.__class__)]
        meta["non_field_operands"] = [i for i in meta["operands"] if not isinstance(inputs[i], self.__class__)]
        meta["field"] = self.__class__
        meta["dtype"] = self.dtype
        # meta["ufuncs"] = self._ufuncs

        if ufunc in type(self)._overridden_ufuncs:
            # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
            # to integers. We know this is safe because we already verified the inputs.
            if method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            # Need to set the intermediate dtype for reduction operations or an error will be thrown. We
            # use the largest valid dtype for this field.
            if method in ["reduce"]:
                kwargs["dtype"] = type(self).dtypes[-1]

            return getattr(type(self), type(self)._overridden_ufuncs[ufunc])(ufunc, method, inputs, kwargs, meta)

        elif ufunc in type(self)._unsupported_ufuncs:
            raise NotImplementedError(f"The numpy ufunc '{ufunc.__name__}' is not supported on {type(self).name} arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            inputs, kwargs = type(self)._view_inputs_as_ndarray(inputs, kwargs)
            output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

            if ufunc in type(self)._ufuncs_requiring_view and output is not None:
                output = output.view(type(self)) if not np.isscalar(output) else type(self)(output, dtype=self.dtype)

            return output

    ###############################################################################
    # Display methods
    ###############################################################################

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        formatter = type(self)._formatter(self)

        cls = type(self)
        class_name = cls.__name__
        with np.printoptions(formatter=formatter):
            cls.__name__ = cls.short_name  # Rename the class so very large fields don't create large indenting
            string = super().__repr__()
        cls.__name__ = class_name

        # Remove the dtype from the repr and add the Galois field order
        dtype_idx = string.find("dtype")
        if dtype_idx == -1:
            string = string[:-1] + f", {cls._order_str})"
        else:
            string = string[:dtype_idx] + f"{cls._order_str})"

        return string


class GroupArrayBase(Array):  # pylint: disable=abstract-method
    """
    A class for `GroupArray` to inherit from.
    """


class RingArrayBase(Array):  # pylint: disable=abstract-method
    """
    A class for `RingArray` to inherit from.
    """


class FieldArrayBase(Array):  # pylint: disable=abstract-method
    """
    A class for `FieldArray` to inherit from.
    """
