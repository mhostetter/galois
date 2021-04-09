import random

import numba
import numpy as np

from .conversion import integer_to_poly, poly_to_str

# Dictionary mapping numpy ufuncs to our implementation method
OVERRIDDEN_UFUNCS = {
    np.add: "_add",
    np.subtract: "_subtract",
    np.multiply: "_multiply",
    np.floor_divide: "_divide",
    np.true_divide: "_divide",
    np.negative: "_negative",
    np.power: "_power",
    np.square: "_square",
    np.log: "_log"
}

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field
ORDER = None  # The field's order `p^m`

# Lookup table globals
EXP = []  # EXP[i] = alpha^i
LOG = []  # LOG[i] = x, such that alpha^x = i
ZECH_LOG = []  # ZECH_LOG[i] = log(1 + alpha^i)
ZECH_E = None  # alpha^ZECH_E = -1, ZECH_LOG[ZECH_E] = -Inf

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y


class GFArrayMeta(type):
    """
    Defines a metaclass to give all GFArray classes a `__str__()` special method, not just their instances.
    """

    def __str__(cls):
        return f"<class 'numpy.ndarray' over {cls.name}>"

    @property
    def name(cls):
        if cls.degree == 1:
            return f"GF({cls.characteristic})"
        else:
            return f"GF({cls.characteristic}^{cls.degree})"


class DisplayContext:
    """
    Simple context manager for the :obj:`galois.GFArray.display` method.
    """

    def __init__(self, cls, mode, poly_var):
        self.cls = cls
        self.mode = mode
        self.poly_var = poly_var

    def __enter__(self):
        # Don't need to do anything, we already set the new mode and poly_var in the display() method
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Reset mode and poly_var upon exiting the context
        self.cls._display_mode = self.mode
        self.cls._display_poly_var = self.poly_var


class GFArray(np.ndarray, metaclass=GFArrayMeta):
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
        :obj:`list`, :obj:`tuple`, or :obj:`int`.
    dtype : numpy.dtype, optional
        The `copy` keyword argument from :obj:`numpy.array`. Setting `dtype` will explicitly set the data type of each
        element. The default is `None` which represents the smallest valid dtype for this field class, i.e. `cls.dtypes[0]`.
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

    # NOTE: These class attributes will be set in the subclasses of GFArray

    characteristic = None
    """
    int: The prime characteristic :math:`p` of the Galois field :math:`\\mathrm{GF}(p^m)`. Adding
    :math:`p` copies of any element will always result in :math:`0`.
    """

    degree = None
    """
    int: The prime characteristic's degree :math:`m` of the Galois field :math:`\\mathrm{GF}(p^m)`. The degree
    is a positive integer.
    """

    order = None
    """
    int: The order :math:`p^m` of the Galois field :math:`\\mathrm{GF}(p^m)`. The order of the field is also equal to
    the field's size.
    """

    prim_poly = None
    """
    galois.Poly: The primitive polynomial :math:`p(x)` of the Galois field :math:`\\mathrm{GF}(p^m)`. The primitive
    polynomial is of degree :math:`m` in :math:`\\mathrm{GF}(p)[x]`.
    """

    alpha = None
    """
    int: The primitive element :math:`\\alpha` of the Galois field :math:`\\mathrm{GF}(p^m)`. The primitive element is a root of the
    primitive polynomial :math:`p(x)`, such that :math:`p(\\alpha) = 0`. The primitive element is also a multiplicative
    generator of the field, such that :math:`\\mathrm{GF}(p^m) = \\{0, 1, \\alpha^1, \\alpha^2, \\dots, \\alpha^{p^m - 2}\\}`.
    """

    dtypes = []
    """
    list: List of valid integer :obj:`numpy.dtype` objects that are compatible with this Galois field. Valid data
    types are signed and unsinged integers that can represent decimal values in :math:`[0, p^m)`.
    """

    ufunc_mode = None
    """
    str: The mode for ufunc compilation, either `"lookup"`, `"calculate"`, `"object"`.
    """

    ufunc_target = None
    """
    str: The numba target for the JIT-compiled ufuncs, either `"cpu"`, `"parallel"`, `"cuda"`, or `None`.
    """

    _display_mode = "int"
    _display_poly_var = "x"

    # Integer representations of the field's primitive element and primitive polynomial to be used in the
    # pure python ufunc implementations for `ufunc_mode = "object"`
    _alpha_dec = None
    _prim_poly_dec = None

    # Stored lookup tables which are JIT compiled in `ufunc_mode = "lookup"``
    _EXP = None
    _LOG = None
    _ZECH_LOG = None

    # Set of ufuncs for overridden numpy arithmetic
    _ufunc_add = None
    _ufunc_subtract = None
    _ufunc_multiply = None
    _ufunc_divide = None
    _ufunc_negative = None
    _ufunc_multiple_add = None
    _ufunc_power = None
    _ufunc_log = None
    _ufunc_poly_eval = None

    def __new__(cls, array, dtype=None, copy=True, order="K", ndmin=0):
        if cls is GFArray:
            raise NotImplementedError("GFArray is an abstract base class that cannot be directly instantiated. Instead, create a GFArray subclass using `galois.GF`.")
        return cls._array(array, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    @classmethod
    def Zeros(cls, shape, dtype=None):
        """
        Create a Galois field array with all zeros.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this field class, i.e. `cls.dtypes[0]`.

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
        Create a Galois field array with all ones.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this field class, i.e. `cls.dtypes[0]`.

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
    def Range(cls, start, stop, step=1, dtype=None):
        """
        Create a Galois field array with a range of field elements.

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
            dtype for this field class, i.e. `cls.dtypes[0]`.

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
            raise ValueError(f"The stopping value must be less than the field order of {cls.order}, not {stop}")

        if dtype != np.object_:
            array = np.arange(start, stop, step=step, dtype=dtype)
        else:
            array = np.array(range(start, stop, step), dtype=dtype)

        return array.view(cls)

    @classmethod
    def Random(cls, shape=(), low=0, high=None, dtype=None):
        """
        Create a Galois field array with random field elements.

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
            dtype for this field class, i.e. `cls.dtypes[0]`.

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
        Create a Galois field array of the field's elements :math:`\\{0, \\dots, p^m-1\\}`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this field class, i.e. `cls.dtypes[0]`.

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

    @classmethod
    def _get_dtype(cls, dtype):
        if dtype is None:
            return cls.dtypes[0]

        # Convert "dtype" to a numpy dtype. This does platform specific conversion, if necessary.
        # For example, np.dtype(int) == np.int64 (on some systems).
        dtype = np.dtype(dtype)
        if dtype not in cls.dtypes:
            raise TypeError(f"GF({cls.characteristic}^{cls.degree}) arrays only support dtypes {cls.dtypes}, not {dtype}")

        return dtype

    @classmethod
    def _array(cls, array_like, dtype=None, copy=True, order="K", ndmin=0):
        dtype = cls._get_dtype(dtype)
        array_like = cls._check_array_like_object(array_like)

        # After confirming the input is of the correct type and that all the values are in the
        # field, we can convert to an array with the specified dtype. We need to check the values
        # before converting to an array because, for example, -1 will cast to 255 with dtype=np.uint8.
        # 255 is a valid field element in GF(2^8) but -1 isn't. We want to catch that condition, i.e. you
        # shouldn't be able to silently convert -1 to a field element in GF(2^8).
        array = np.array(array_like, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

        return array.view(cls)

    @classmethod
    def _check_array_like_object(cls, array_like):
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
                raise TypeError(f"GF({cls.characteristic}^{cls.degree}) arrays must have integer dtypes, not {array_like.dtype}")
            cls._check_array_values(array_like)

        else:
            raise TypeError(f"GF({cls.characteristic}^{cls.degree}) arrays can be created with scalars of type int, not {type(array_like)}")

        return array_like

    @classmethod
    def _check_iterable_types_and_values(cls, iterable):
        new_iterable = []
        for item in iterable:
            if isinstance(item, (list, tuple)):
                item = cls._check_iterable_types_and_values(item)
                new_iterable.append(item)
                continue

            if not isinstance(item, (int, np.integer, cls)):
                raise TypeError(f"When GF({cls.characteristic}^{cls.degree}) arrays are created/assigned with an iterable, each element must be an integer. Found type {type(item)}.")
            if not 0 <= item < cls.order:
                raise ValueError(f"GF({cls.characteristic}^{cls.degree}) arrays must have elements in 0 <= x < {cls.order}, not {item}")

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
                raise TypeError(f"When GF({cls.characteristic}^{cls.degree}) arrays are created/assigned with a numpy array with dtype=object, each element must be an integer. Found type {type(a)}.")

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
            raise ValueError(f"GF({cls.characteristic}^{cls.degree}) arrays must have elements in 0 <= x < {cls.order}, not {array[idxs]}")

    @classmethod
    def target(cls, target, mode, rebuild=False):  # pylint: disable=unused-argument
        """
        Retarget the just-in-time compiled numba ufuncs.
        """
        return

    @classmethod
    def display(cls, mode="int", poly_var="x"):
        """
        Sets the printing mode for arrays.

        Parameters
        ----------
        mode : str, optional
            The field element display mode, either `"int"` (default) or `"poly"`.
        poly_var : str, optional
            The polynomial representation's variable. The default is `"x"`.

        Examples
        --------
        Change the display mode by calling the :obj:`galois.GFArray.display` method.

        .. ipython:: python

            GF = galois.GF(2**3)
            a = GF.Random(4); a
            GF.display("poly"); a
            GF.display("poly", "r"); a

            # Reset the print mode
            GF.display(); a

        The :obj:`galois.GFArray.display` method can also be used as a context manager.

        .. ipython:: python

            # The original display mode
            print(a)

            # The new display context
            with GF.display("poly"):
                print(a)

            # Returns to the original display mode
            print(a)
        """
        if mode not in ["int", "poly"]:
            raise ValueError(f"Argument `mode` must be in ['int', 'poly'], not {mode}.")
        if not isinstance(poly_var, str):
            raise TypeError(f"Argument `poly_var` must be a string, not {type(poly_var)}.")

        context = DisplayContext(cls, cls._display_mode, cls._display_poly_var)

        # Set the new state
        cls._display_mode = mode
        cls._display_poly_var = poly_var

        return context

    @classmethod
    def _print_int(cls, decimal):
        return "{:d}".format(int(decimal))

    @classmethod
    def _print_poly(cls, decimal):
        poly = integer_to_poly(decimal, cls.characteristic)
        return poly_to_str(poly, poly_var=cls._display_poly_var)

    def __repr__(self):
        formatter = {}
        if self._display_mode == "poly":
            formatter["int"] = self._print_poly
            formatter["object"] = self._print_poly
        elif self.dtype == np.object_:
            formatter["object"] = self._print_int

        cls = self.__class__
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

    def __str__(self):
        return self.__repr__()

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        if dtype not in self.dtypes:
            raise TypeError(f"GF({self.characteristic}^{self.degree}) arrays can only be cast as integer dtypes in {self.dtypes}, not {dtype}")
        return super().astype(dtype, **kwargs)

    @classmethod
    def _link_python_calculate_ufuncs(cls):
        cls._ufunc_add = np.frompyfunc(cls._add_calculate, 2, 1)
        cls._ufunc_subtract = np.frompyfunc(cls._subtract_calculate, 2, 1)
        cls._ufunc_multiply = np.frompyfunc(cls._multiply_calculate, 2, 1)
        cls._ufunc_divide = np.frompyfunc(cls._divide_calculate, 2, 1)
        cls._ufunc_negative = np.frompyfunc(cls._additive_inverse_calculate, 1, 1)
        cls._ufunc_multiple_add = np.frompyfunc(cls._multiple_add_calculate, 2, 1)
        cls._ufunc_power = np.frompyfunc(cls._power_calculate, 2, 1)
        cls._ufunc_log = np.frompyfunc(cls._log_calculate, 1, 1)
        cls._ufunc_poly_eval = np.vectorize(cls._poly_eval_calculate, excluded=["coeffs"], otypes=[np.object_])

    @classmethod
    def _jit_compile_lookup_ufuncs(cls, target):
        # Export lookup tables to global variables so JIT compiling can cache the tables in the binaries
        global CHARACTERISTIC, ORDER, EXP, LOG, ZECH_LOG, ZECH_E, ADD_JIT, MULTIPLY_JIT
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        ZECH_LOG = cls._ZECH_LOG
        if cls.characteristic == 2:
            ZECH_E = 0
        else:
            ZECH_E = (cls.order - 1) // 2

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # JIT-compile add and multiply routines for reference in other routines
        ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(_add_lookup)
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_lookup)

        # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
        cls._ufunc_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_lookup)
        cls._ufunc_subtract = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_lookup)
        cls._ufunc_multiply = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_lookup)
        cls._ufunc_divide = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_lookup)
        cls._ufunc_negative = numba.vectorize(["int64(int64)"], **kwargs)(_additive_inverse_lookup)
        cls._ufunc_multiple_add = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiple_add_lookup)
        cls._ufunc_power = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_lookup)
        cls._ufunc_log = numba.vectorize(["int64(int64)"], **kwargs)(_log_lookup)
        cls._ufunc_poly_eval = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_lookup)

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
            y = cls._ufunc_poly_eval(coeffs=coeffs, values=x)  # pylint: disable=not-callable
        else:
            # For integer dtypes, call the JIT-compiled gufunc
            y = np.copy(x)
            cls._ufunc_poly_eval(coeffs, x, y, casting="unsafe")  # pylint: disable=not-callable

        y = cls(y)
        if y.size == 1:
            y = y[0]

        return y

    def __array_finalize__(self, obj):
        """
        A numpy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        if obj is not None and not isinstance(obj, GFArray):
            # Only invoked on view casting
            if obj.dtype not in self.dtypes:
                raise TypeError(f"GF({self.characteristic}^{self.degree}) can only have integer dtypes {self.dtypes}, not {obj.dtype}")
            if np.any(obj < 0) or np.any(obj >= self.order):
                idxs = np.logical_or(obj < 0, obj >= self.order)
                raise ValueError(f"GF({self.characteristic}^{self.degree}) arrays must have values in 0 <= x < {self.order}, not {obj[idxs]}")

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

    def _view_input_gf_as_ndarray(self, inputs, kwargs, meta):
        # View all input operands as np.ndarray to avoid infinite recursion
        v_inputs = list(inputs)
        # for i in meta["operands"]:
        for i in meta["gf_operands"]:
            if isinstance(inputs[i], self.__class__):
                v_inputs[i] = inputs[i].view(np.ndarray)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if isinstance(output, self.__class__):
                    o = output.view(np.ndarray)
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _view_input_int_as_ndarray(self, inputs, meta):  # pylint: disable=no-self-use
        v_inputs = list(inputs)
        for i in meta["operands"]:
            if isinstance(inputs[i], int):
                # Use the largest valid dtype for this field
                v_inputs[i] = np.array(inputs[i], dtype=self.dtypes[-1])

        return v_inputs

    def _view_output_ndarray_as_gf(self, ufunc, v_outputs):
        if v_outputs is NotImplemented:
            return v_outputs
        if ufunc.nout == 1:
            v_outputs = (v_outputs, )

        outputs = []
        for v_output in v_outputs:
            o = self.__class__(v_output, dtype=self.dtype)
            outputs.append(o)

        return outputs[0] if len(outputs) == 1 else outputs

    def _verify_inputs(self, ufunc, method, inputs, meta):  # pylint: disable=too-many-branches
        types = [meta["types"][i] for i in meta["operands"]]  # List of types of the "operands", excludes index lists, etc
        operands = [inputs[i] for i in meta["operands"]]

        if method == "reduceat":
            return

        # Verify input operand types
        if ufunc in [np.add, np.subtract, np.true_divide, np.floor_divide]:
            if not all(t is self.__class__ for t in types):
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the same field {self.__class__}, not {types}")
        if ufunc in [np.multiply, np.power, np.square]:
            if not all(np.issubdtype(o.dtype, np.integer) or o.dtype == np.object_ for o in operands):
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields must be performed against elements in the field {self.__class__} or integers, not {types}")
        if ufunc in [np.power, np.square]:
            if not types[0] is self.__class__:
                raise TypeError(f"Operation '{ufunc.__name__}' in Galois fields can only exponentiate elements in the same field {self.__class__}, not {types[0]}")

        # Verify no divide by zero or log(0) errors
        if ufunc in [np.true_divide, np.floor_divide] and np.count_nonzero(operands[-1]) != operands[-1].size:
            raise ZeroDivisionError("Divide by 0")
        if ufunc is np.power:
            if method == "outer" and (np.any(operands[0] == 0) and np.any(operands[1] < 0)):
                raise ZeroDivisionError("Divide by 0")
            if method == "__call__" and np.any(np.logical_and(operands[0] == 0, operands[1] < 0)):
                raise ZeroDivisionError("Divide by 0")
        if ufunc is np.log and np.count_nonzero(operands[0]) != operands[0].size:
            raise ArithmeticError("Log(0) error")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # pylint: disable=too-many-branches
        """
        Intercept various numpy ufuncs (triggered by operators like `+` , `-`, etc). Then determine
        which operations will result in the correct answer in the given Galois field. Wherever
        appropriate, use native numpy ufuncs for their efficiency and generality in supporting various array
        shapes, etc.
        """
        meta = {}
        meta["types"] = [type(inputs[i]) for i in range(len(inputs))]
        meta["operands"] = list(range(0, len(inputs)))
        if method in ["at", "reduceat"]:
            # Remove the second argument for "at" ufuncs which is the indices list
            meta["operands"].pop(1)
        meta["gf_operands"] = [i for i in meta["operands"] if isinstance(inputs[i], self.__class__)]
        meta["non_gf_operands"] = [i for i in meta["operands"] if not isinstance(inputs[i], self.__class__)]

        # View Galois field array inputs as np.ndarray so subsequent numpy ufunc calls go to numpy and don't
        # result in infinite recursion
        inputs, kwargs = self._view_input_gf_as_ndarray(inputs, kwargs, meta)

        # For ufuncs we are not overriding, call the parent implementation
        if ufunc not in OVERRIDDEN_UFUNCS.keys():
            return super().__array_ufunc__(ufunc, method, *inputs)  # pylint: disable=no-member

        inputs = self._view_input_int_as_ndarray(inputs, meta)

        self._verify_inputs(ufunc, method, inputs, meta)

        # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
        # to integers. We know this is safe because we already verified the inputs.
        if method not in ["reduce", "accumulate", "at", "reduceat"]:
            kwargs["casting"] = "unsafe"

        # Need to set the intermediate dtype for reduction operations or an error will be thrown. We
        # use the largest valid dtype for this field.
        if method in ["reduce"]:
            kwargs["dtype"] = self.dtypes[-1]

        # Call appropriate ufunc method (implemented in subclasses)
        if ufunc is np.add:
            outputs = getattr(self._ufunc_add, method)(*inputs, **kwargs)
        elif ufunc is np.subtract:
            outputs = getattr(self._ufunc_subtract, method)(*inputs, **kwargs)
        elif ufunc is np.multiply:
            if meta["gf_operands"] == meta["operands"]:
                # In-field multiplication
                outputs = getattr(self._ufunc_multiply, method)(*inputs, **kwargs)
            else:
                # In-field "multiple addition" by an integer, i.e. GF(x) * 3 = GF(x) + GF(x) + GF(x)
                if 0 not in meta["gf_operands"]:
                    # If the integer is the first argument and the field element is the second, switch them. This
                    # is done because the ufunc needs to know which input is not in the field (so it can perform a
                    # modulus operation).
                    i = meta["gf_operands"][0]
                    j = meta["non_gf_operands"][0]
                    inputs[j], inputs[i] = inputs[i], inputs[j]
                outputs = getattr(self._ufunc_multiple_add, method)(*inputs, **kwargs)
        elif ufunc in [np.true_divide, np.floor_divide]:
            outputs = getattr(self._ufunc_divide, method)(*inputs, **kwargs)
        elif ufunc is np.negative:
            outputs = getattr(self._ufunc_negative, method)(*inputs, **kwargs)
        elif ufunc is np.power:
            outputs = getattr(self._ufunc_power, method)(*inputs, **kwargs)
        elif ufunc is np.square:
            inputs.append(np.array(2, dtype=self.dtype))
            outputs = getattr(self._ufunc_power, method)(*inputs, **kwargs)
        elif ufunc is np.log:
            outputs = getattr(self._ufunc_log, method)(*inputs, **kwargs)

        if outputs is None or ufunc is np.log:
            return outputs
        else:
            outputs = self._view_output_ndarray_as_gf(ufunc, outputs)
            return outputs

    ###############################################################################
    # Galois field explicit arithmetic in pure python for extremely large fields
    ###############################################################################

    @classmethod
    def _add_calculate(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _subtract_calculate(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _multiply_calculate(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _divide_calculate(cls, a, b):
        if a == 0 or b == 0:
            # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
            return 0
        b_inv = cls._multiplicative_inverse_calculate(b)
        return cls._multiply_calculate(a, b_inv)

    @classmethod
    def _additive_inverse_calculate(cls, a):
        raise NotImplementedError

    @classmethod
    def _multiplicative_inverse_calculate(cls, a):
        raise NotImplementedError

    @classmethod
    def _multiple_add_calculate(cls, a, multiple):
        b = multiple % cls.characteristic
        return cls._multiply_calculate(a, b)

    @classmethod
    def _power_calculate(cls, a, power):
        """
        Square and Multiply Algorithm

        a^13 = (1) * (a)^13
            = (a) * (a)^12
            = (a) * (a^2)^6
            = (a) * (a^4)^3
            = (a * a^4) * (a^4)^2
            = (a * a^4) * (a^8)
            = result_m * result_s
        """
        # NOTE: The a == 0 and b < 0 condition will be caught outside of the the ufunc and raise ZeroDivisonError
        if power == 0:
            return 1
        elif power < 0:
            a = cls._multiplicative_inverse_calculate(a)
            power = abs(power)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while power > 1:
            if power % 2 == 0:
                result_s = cls._multiply_calculate(result_s, result_s)
                power //= 2
            else:
                result_m = cls._multiply_calculate(result_m, result_s)
                power -= 1

        result = cls._multiply_calculate(result_m, result_s)

        return result

    @classmethod
    def _log_calculate(cls, beta):
        """
        TODO: Replace this with more efficient algorithm

        alpha in GF(p^m) and generates field
        beta in GF(p^m)

        gamma = log_alpha(beta), such that: alpha^gamma = beta
        """
        # Naive algorithm
        result = 1
        for i in range(0, cls.order - 1):
            if result == beta:
                break
            result = cls._multiply_calculate(result, cls.alpha)
        return i

    @classmethod
    def _poly_eval_calculate(cls, coeffs, values):
        result = coeffs[0]
        for j in range(1, coeffs.size):
            p = cls._multiply_calculate(result, values)
            result = cls._add_calculate(coeffs[j], p)
        return result


###############################################################################
# Galois field arithmetic using EXP, LOG, and ZECH_LOG lookup tables
###############################################################################

def _add_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    a + b = alpha^m + alpha^n
          = alpha^m * (1 + alpha^(n - m))  # If n is larger, factor out alpha^m
          = alpha^m * alpha^ZECH_LOG(n - m)
          = alpha^(m + ZECH_LOG(n - m))
    """
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return b
    if b == 0:
        return a

    if m > n:
        # We want to factor out alpha^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    if n - m == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and alpha^(-Inf) = 0
        return 0

    return EXP[m + ZECH_LOG[n - m]]


def _subtract_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    a - b = alpha^m - alpha^n
          = alpha^m + (-alpha^n)
          = alpha^m + (-1 * alpha^n)
          = alpha^m + (alpha^e * alpha^n)
          = alpha^m + alpha^(e + n)
    """
    # Same as addition if n = LOG[b] + e
    m = LOG[a]
    n = LOG[b] + ZECH_E

    # LOG[0] = -Inf, so catch these conditions
    if b == 0:
        return a
    if a == 0:
        return EXP[n]

    if m > n:
        # We want to factor out alpha^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    z = n - m
    if z == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and alpha^(-Inf) = 0
        return 0
    if z >= ORDER - 1:
        # Reduce index of ZECH_LOG by the multiplicative order of the field, i.e. `order - 1`
        z -= ORDER - 1

    return EXP[m + ZECH_LOG[z]]


def _multiply_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    a * b = alpha^m * alpha^n
          = alpha^(m + n)
    """
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0 or b == 0:
        return 0

    return EXP[m + n]


def _divide_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    a / b = alpha^m / alpha^n
          = alpha^(m - n)
          = 1 * alpha^(m - n)
          = alpha^(ORDER - 1) * alpha^(m - n)
          = alpha^(ORDER - 1 + m - n)
    """
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0 or b == 0:
        # NOTE: The b == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0

    # We add `ORDER - 1` to guarantee the index is non-negative
    return EXP[(ORDER - 1) + m - n]


def _additive_inverse_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    -a = -alpha^n
       = -1 * alpha^n
       = alpha^e * alpha^n
       = alpha^(e + n)
    """
    n = LOG[a]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return 0

    return EXP[ZECH_E + n]


def _multiplicative_inverse_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    1 / a = 1 / alpha^m
          = alpha^(-m)
          = 1 * alpha^(-m)
          = alpha^(ORDER - 1) * alpha^(-m)
          = alpha^(ORDER - 1 - m)
    """
    m = LOG[a]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        # NOTE: The a == 0 condition will be caught outside of the ufunc and raise ZeroDivisonError
        return 0

    return EXP[(ORDER - 1) - m]


def _multiple_add_lookup(a, b_int):  # pragma: no cover
    """
    a in GF(p^m)
    b_int in Z
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}
    b in GF(p^m)

    a . b_int = a + a + ... + a = b_int additions of a
    a . p_int = 0, where p_int is the prime characteristic of the field

    a . b_int = a * ((b_int // p_int)*p_int + b_int % p_int)
              = a * ((b_int // p_int)*p_int) + a * (b_int % p_int)
              = 0 + a * (b_int % p_int)
              = a * (b_int % p_int)
              = a * b, field multiplication

    b = b_int % p_int
    """
    b = b_int % CHARACTERISTIC
    m = LOG[a]
    n = LOG[b]

    # LOG[0] = -Inf, so catch these conditions
    if a == 0 or b == 0:
        return 0

    return EXP[m + n]


def _power_lookup(a, b_int):  # pragma: no cover
    """
    a in GF(p^m)
    b_int in Z
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    a ** b_int = alpha^m ** b_int
               = alpha^(m * b_int)
               = alpha^(m * ((b_int // (ORDER - 1))*(ORDER - 1) + b_int % (ORDER - 1)))
               = alpha^(m * ((b_int // (ORDER - 1))*(ORDER - 1)) * alpha^(m * (b_int % (ORDER - 1)))
               = 1 * alpha^(m * (b_int % (ORDER - 1)))
               = alpha^(m * (b_int % (ORDER - 1)))
    """
    m = LOG[a]

    if b_int == 0:
        return 1

    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return 0

    return EXP[(m * b_int) % (ORDER - 1)]


def _log_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    alpha is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, alpha^1, ..., alpha^(p^m - 2)}

    log_alpha(a) = log_alpha(alpha^m)
                 = m
    """
    return LOG[a]


def _poly_eval_lookup(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
