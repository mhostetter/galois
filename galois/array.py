import random

import numpy as np

from .meta_gf import GFMeta
from .array_mixin_function import FunctionMixin
from .array_mixin_ufunc import UfuncMixin
from .array_mixin_linalg import LinearAlgebraMixin
from .poly_conversion import integer_to_poly, poly_to_str, str_to_integer


class GFArray(FunctionMixin, UfuncMixin, LinearAlgebraMixin, np.ndarray, metaclass=GFMeta):  # pylint: disable=too-many-ancestors
    """
    Create an array over :math:`\\mathrm{GF}(p^m)`.

    The :obj:`galois.GFArray` class is a parent class for all Galois field array classes. Any Galois field :math:`\\mathrm{GF}(p^m)`
    with prime characteristic :math:`p` and positive integer :math:`m`, can be constructed by calling the class factory
    `galois.GF(p**m)`.

    Warning
    -------
        This is an abstract base class for all Galois field array classes. :obj:`galois.GFArray` cannot be instantiated
        directly. Instead, Galois field array classes are created using :obj:`galois.GF`.

        For example, one can create the :math:`\\mathrm{GF}(7)` field array class as follows:

        .. ipython:: python

            GF7 = galois.GF(7)
            print(GF7)

        This subclass can then be used to instantiate arrays over :math:`\\mathrm{GF}(7)`.

        .. ipython:: python

            GF7([3,5,0,2,1])
            GF7.Random((2,5))

    :obj:`galois.GFArray` is a subclass of :obj:`numpy.ndarray`. The :obj:`galois.GFArray` constructor has the same syntax as
    :obj:`numpy.array`. The returned :obj:`galois.GFArray` object is an array that can be acted upon like any other
    numpy array.

    Parameters
    ----------
    array : array_like
        The input array to be converted to a Galois field array. The input array is copied, so the original array
        is unmodified by changes to the Galois field array. Valid input array types are :obj:`numpy.ndarray`,
        :obj:`list` or :obj:`tuple` of ints or strs, :obj:`int`, or :obj:`str`.
    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
        dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.
    copy : bool, optional
        The `copy` keyword argument from :obj:`numpy.array`. The default is `True` which makes a copy of the input
        object is it's an array.
    order : str, optional
        The `order` keyword argument from :obj:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
    ndmin : int, optional
        The `ndmin` keyword argument from :obj:`numpy.array`. The minimum number of dimensions of the output.
        The default is 0.

    Returns
    -------
    galois.GFArray
        The copied input array as a :math:`\\mathrm{GF}(p^m)` field array.

    Examples
    --------
    Construct various kinds of Galois fields using :obj:`galois.GF`.

    .. ipython:: python

        # Construct a GF(2^m) class
        GF256 = galois.GF(2**8); print(GF256)

        # Construct a GF(p) class
        GF571 = galois.GF(571); print(GF571)

        # Construct a very large GF(2^m) class
        GF2m = galois.GF(2**100); print(GF2m)

        # Construct a very large GF(p) class
        GFp = galois.GF(36893488147419103183); print(GFp)

    Depending on the field's order (size), only certain `dtype` values will be supported.

    .. ipython:: python

        GF256.dtypes
        GF571.dtypes

    Very large fields, which can't be represented using `np.int64`, can only be represented as `dtype=np.object_`.

    .. ipython:: python

        GF2m.dtypes
        GFp.dtypes

    Newly-created arrays will use the smallest, valid dtype.

    .. ipython:: python

        a = GF256.Random(10); a
        a.dtype

    This can be explicitly set by specifying the `dtype` keyword argument.

    .. ipython:: python

        a = GF256.Random(10, dtype=np.uint32); a
        a.dtype

    Arrays can also be created explicitly by converting an "array-like" object.

    .. ipython:: python

        # Construct a Galois field array from a list
        l = [142, 27, 92, 253, 103]; l
        GF256(l)

        # Construct a Galois field array from an existing numpy array
        x_np = np.array(l, dtype=np.int64); x_np
        GF256(l)

    Arrays can also be created by "view casting" from an existing numpy array. This avoids
    a copy operation, which is especially useful for large data already brought into memory.

    .. ipython:: python

        a = x_np.view(GF256); a

        # Changing `x_np` will change `a`
        x_np[0] = 0; x_np
        a
    """

    def __new__(cls, array, dtype=None, copy=True, order="K", ndmin=0):
        if cls is GFArray:
            raise NotImplementedError("GFArray is an abstract base class that cannot be directly instantiated. Instead, create a GFArray subclass using `galois.GF`.")
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
            array_like = str_to_integer(array_like, cls.ground_field)

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
                item = str_to_integer(item, cls.ground_field)
            elif not isinstance(item, (int, np.integer, cls)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with an iterable, each element must be an integer. Found type {type(item)}.")

            if not 0 <= item < cls.order:
                raise ValueError(f"{cls.name} arrays must have elements in 0 <= x < {cls.order}, not {item}.")

            # Ensure the type is int so dtype=object classes don't get all mixed up
            new_iterable.append(int(item))

        return new_iterable

    @classmethod
    def _check_array_types_dtype_object(cls, array):
        if array.size == 0:
            return array
        if array.ndim == 0:
            return int(array)

        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for _ in iterator:
            a = array[iterator.multi_index]
            if not isinstance(a, (int, cls)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with dtype=object, each element must be an integer. Found type {type(a)}.")

            # Ensure the type is int so dtype=object classes don't get all mixed up
            array[iterator.multi_index] = int(a)

        return array

    @classmethod
    def _check_array_values(cls, array):
        if not isinstance(array, np.ndarray):
            # Convert single integer to array so next step doesn't fail
            array = np.array(array)

        # Check the value of the "field elements" and make sure they are valid
        if np.any(array < 0) or np.any(array >= cls.order):
            idxs = np.logical_or(array < 0, array >= cls.order)
            raise ValueError(f"{cls.name} arrays must have elements in 0 <= x < {cls.order}, not {array[idxs]}.")

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(cls, shape, dtype=None):
        """
        Creates a Galois field array with all zeros.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.

        Returns
        -------
        galois.GFArray
            A Galois field array of zeros.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Zeros((2,5))
        """
        dtype = cls._get_dtype(dtype)
        array = np.zeros(shape, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Ones(cls, shape, dtype=None):
        """
        Creates a Galois field array with all ones.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.

        Returns
        -------
        galois.GFArray
            A Galois field array of ones.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Ones((2,5))
        """
        dtype = cls._get_dtype(dtype)
        array = np.ones(shape, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Eye(cls, size, dtype=None):
        """
        Creates a Galois field identity matrix.

        Parameters
        ----------
        size : int
            The size along one axis of the matrix. The resulting array has shape `(size,size)`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.

        Returns
        -------
        galois.GFArray
            A Galois field identity matrix of shape `(size, size)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Eye(4)
        """
        dtype = cls._get_dtype(dtype)
        array = np.eye(size, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Range(cls, start, stop, step=1, dtype=None):
        """
        Creates a Galois field array with a range of field elements.

        Parameters
        ----------
        start : int
            The starting value (inclusive).
        stop : int
            The stopping value (exclusive).
        step : int, optional
            The space between values. The default is 1.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.

        Returns
        -------
        galois.GFArray
            A Galois field array of a range of field elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Range(10,20)
        """
        dtype = cls._get_dtype(dtype)
        if not stop <= cls.order:
            raise ValueError(f"The stopping value must be less than the field order of {cls.order}, not {stop}.")

        if dtype != np.object_:
            array = np.arange(start, stop, step=step, dtype=dtype)
        else:
            array = np.array(range(start, stop, step), dtype=dtype)

        return array.view(cls)

    @classmethod
    def Random(cls, shape=(), low=0, high=None, dtype=None):
        """
        Creates a Galois field array with random field elements.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        low : int, optional
            The lowest value (inclusive) of a random field element. The default is 0.
        high : int, optional
            The highest value (exclusive) of a random field element. The default is `None` which represents the
            field's order :math:`p^m`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.

        Returns
        -------
        galois.GFArray
            A Galois field array of random field elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Random((2,5))
        """
        dtype = cls._get_dtype(dtype)
        high = cls.order if high is None else high
        if not 0 <= low < high <= cls.order:
            raise ValueError(f"Arguments must satisfy `0 <= low < high <= order`, not `0 <= {low} < {high} <= {cls.order}`.")

        if dtype != np.object_:
            array = np.random.randint(low, high, shape, dtype=dtype)
        else:
            array = np.empty(shape, dtype=dtype)
            iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
            for _ in iterator:
                array[iterator.multi_index] = random.randint(low, high - 1)

        return array.view(cls)

    @classmethod
    def Elements(cls, dtype=None):
        """
        Creates a Galois field array of the field's elements :math:`\\{0, \\dots, p^m-1\\}`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GFMeta.dtypes`.

        Returns
        -------
        galois.GFArray
            A Galois field array of all the field's elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Elements()
        """
        return cls.Range(0, cls.order, step=1, dtype=dtype)

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
        if obj is not None and not isinstance(obj, GFArray):
            # Only invoked on view casting
            if obj.dtype not in type(self).dtypes:
                raise TypeError(f"{type(self).name} can only have integer dtypes {type(self).dtypes}, not {obj.dtype}.")
            if np.any(obj < 0) or np.any(obj >= type(self).order):
                idxs = np.logical_or(obj < 0, obj >= type(self).order)
                raise ValueError(f"{type(self).name} arrays must have values in 0 <= x < {type(self).order}, not {obj[idxs]}.")

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

    ###############################################################################
    # Display methods
    ###############################################################################

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        formatter = {}
        if type(self).display_mode == "poly":
            formatter["int"] = self._print_poly
            formatter["object"] = self._print_poly
        elif self.dtype == np.object_:
            formatter["object"] = self._print_int

        cls = type(self)
        class_name = cls.__name__
        with np.printoptions(formatter=formatter):
            cls.__name__ = "GF"  # Rename the class so very large fields don't create large indenting
            string = super().__repr__()
        cls.__name__ = class_name

        if cls.degree == 1:
            order = "{}".format(cls.order)
        else:
            order = "{}^{}".format(cls.characteristic, cls.degree)

        # Remove the dtype from the repr and add the Galois field order
        dtype_idx = string.find("dtype")
        if dtype_idx == -1:
            string = string[:-1] + f", order={order})"
        else:
            string = string[:dtype_idx] + f"order={order})"

        return string

    @staticmethod
    def _print_int(decimal):
        return "{:d}".format(int(decimal))

    def _print_poly(self, decimal):
        poly = integer_to_poly(decimal, type(self).characteristic)
        return poly_to_str(poly, poly_var=type(self).display_poly_var)

    # TODO: Figure out where to put this

    @classmethod
    def _poly_eval(cls, coeffs, x):
        coeffs = cls(coeffs)  # Convert coefficient into the field
        coeffs = coeffs.view(np.ndarray)  # View cast to normal integers so ufunc_poly_eval call uses normal arithmetic
        coeffs = np.atleast_1d(coeffs)
        if coeffs.size == 1:
            # TODO: Why must coeffs have atleast 2 elements otherwise it will be converted to a scalar, not 1d array?
            coeffs = np.insert(coeffs, 0, 0)

        x = cls(x)  # Convert evaluation values into the field (checks that values are in the field)
        x = x.view(np.ndarray)  # View cast to normal integers so ufunc_poly_eval call uses normal arithmetic
        x = np.atleast_1d(x)

        if cls.dtypes[-1] == np.object_:
            # For object dtypes, call the vectorized classmethod
            y = cls._ufuncs["poly_eval"](coeffs=coeffs, values=x)  # pylint: disable=not-callable
        else:
            # For integer dtypes, call the JIT-compiled gufunc
            y = np.copy(x)
            cls._ufuncs["poly_eval"](coeffs, x, y, casting="unsafe")  # pylint: disable=not-callable

        y = cls(y)
        if y.size == 1:
            y = y[0]

        return y
