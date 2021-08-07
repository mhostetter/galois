import random

import numpy as np

from .._overrides import set_module

from ._linalg import row_reduce, lu_decompose, lup_decompose
from ._meta_class import FieldClass
from ._poly_conversion import str_to_integer

__all__ = ["FieldArray"]


@set_module("galois")
class FieldArray(np.ndarray, metaclass=FieldClass):
    r"""
    Creates an array over :math:`\mathrm{GF}(p^m)`.

    Warning
    -------
    :obj:`galois.FieldArray` is an abstract base class for all Galois field array classes and cannot be instantiated
    directly. Instead, :obj:`galois.FieldArray` subclasses are created using the class factory :func:`galois.GF`.

    Parameters
    ----------
    array : int, str, tuple, list, numpy.ndarray, galois.FieldArray
        The input array-like object to be converted to a Galois field array. See the examples section for demonstations of array creation
        using each input type. See see :func:`galois.FieldClass.display` and :obj:`galois.FieldClass.display_mode` for a description of the
        "integer" and "polynomial" representation of Galois field elements.

        * :obj:`int`: A single integer, which is the "integer representation" of a Galois field element, creates a 0-D array.
        * :obj:`str`: A single string, which is the "polynomial representation" of a Galois field element, creates a 0-D array.
        * :obj:`tuple`, :obj:`list`: A list or tuple (or nested lists/tuples) of ints or strings (which can be mix-and-matched) creates an array of
          Galois field elements from their integer or polynomial representations.
        * :obj:`numpy.ndarray`, :obj:`galois.FieldArray`: An array of ints creates a copy of the array over this specific field.

    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
        dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.
    copy : bool, optional
        The `copy` keyword argument from :func:`numpy.array`. The default is `True` which makes a copy of the input array.
    order : str, optional
        The `order` keyword argument from :func:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
    ndmin : int, optional
        The `ndmin` keyword argument from :func:`numpy.array`. The minimum number of dimensions of the output.
        The default is 0.

    Returns
    -------
    galois.FieldArray
        The copied input array as a Galois field array over :math:`\mathrm{GF}(p^m)`.

    Notes
    -----
    :obj:`galois.FieldArray` is an abstract base class and cannot be instantiated directly. Instead, the user creates a :obj:`galois.FieldArray`
    subclass for the field :math:`\mathrm{GF}(p^m)` by calling the class factory :func:`galois.GF`, e.g. `GF = galois.GF(p**m)`. In this case,
    `GF` is a subclass of :obj:`galois.FieldArray` and an instance of :obj:`galois.FieldClass`, a metaclass that defines special methods and attributes
    related to the Galois field.

    :obj:`galois.FieldArray`, and `GF`, is a subclass of :obj:`numpy.ndarray` and its constructor `x = GF(array_like)` has the same syntax as
    :func:`numpy.array`. The returned :obj:`galois.FieldArray` instance `x` is a :obj:`numpy.ndarray` that is acted upon like any other
    numpy array, except all arithmetic is performed in :math:`\mathrm{GF}(p^m)` not in :math:`\mathbb{Z}` or :math:`\mathbb{R}`.

    Examples
    --------
    Construct the Galois field class for :math:`\mathrm{GF}(2^8)` using the class factory :func:`galois.GF` and then display
    some relevant properties of the field. See :obj:`galois.FieldClass` for a complete list of Galois field array class
    methods and attributes.

    .. ipython:: python

        GF256 = galois.GF(2**8)
        GF256
        print(GF256.properties)

    Depending on the field's order, only certain numpy dtypes are supported. See :obj:`galois.FieldClass.dtypes` for more details.

    .. ipython:: python

        GF256.dtypes

    Galois field arrays can be created from existing numpy arrays.

    .. ipython:: python

        x = np.array([155, 232, 162, 159,  63,  29, 247, 141,  75, 189], dtype=int)

        # Explicit Galois field array creation (a copy is performed)
        GF256(x)

        # Or view an existing numpy array as a Galois field array (no copy is performed)
        x.view(GF256)

    Galois field arrays can also be created explicitly by converting an "array-like" object.

    .. ipython:: python

        # A scalar GF(2^8) element from its integer representation
        GF256(37)

        # A scalar GF(2^8) element from its polynomial representation
        GF256("x^5 + x^2 + 1")

        # A GF(2^8) array from a list of elements in their integer representation
        GF256([[142, 27], [92, 253]])

        # A GF(2^8) array from a list of elements in their integer and polynomial representations
        GF256([[142, "x^5 + x^2 + 1"], [92, 253]])

    There's also an alternate constructor :func:`Vector` (and accompanying :func:`vector` method) to convert an array of coefficients
    over :math:`\mathrm{GF}(p)` with last dimension :math:`m` into Galois field elements in :math:`\mathrm{GF}(p^m)`.

    .. ipython:: python

        # A scalar GF(2^8) element from its vector representation
        GF256.Vector([0, 0, 1, 0, 0, 1, 0, 1])

        # A GF(2^8) array from a list of elements in their vector representation
        GF256.Vector([[[1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1, 1]], [[0, 1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 1]]])

    Newly-created arrays will use the smallest unsigned dtype, unless otherwise specified.

    .. ipython:: python

        a = GF256([66, 166, 27, 182, 125]); a
        a.dtype
        b = GF256([66, 166, 27, 182, 125], dtype=np.int64); b
        b.dtype
    """
    # pylint: disable=unsupported-membership-test,not-an-iterable

    def __new__(cls, array, dtype=None, copy=True, order="K", ndmin=0):
        if cls is FieldArray:
            raise NotImplementedError("FieldArray is an abstract base class that cannot be directly instantiated. Instead, create a FieldArray subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` and instantiate an array using `x = GF(array_like)`.")
        return cls._array(array, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    @classmethod
    def _get_dtype(cls, dtype):
        if dtype is None:
            return cls.dtypes[0]

        # Convert "dtype" to a numpy dtype. This does platform specific conversion, if necessary.
        # For example, np.dtype(int) == np.int64 (on some systems).
        dtype = np.dtype(dtype)
        if dtype not in cls.dtypes:
            raise TypeError(f"{cls.name} arrays only support dtypes {[np.dtype(d).name for d in cls.dtypes]}, not {dtype.name!r}.")

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
            elif not isinstance(item, (int, np.integer, FieldArray)):
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
            if not isinstance(array[()], (int, np.integer, FieldArray)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with `dtype=object`, each element must be an integer. Found type {type(array[()])}.")
            return int(array)

        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for _ in iterator:
            a = array[iterator.multi_index]
            if not isinstance(a, (int, np.integer, FieldArray)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with `dtype=object`, each element must be an integer. Found type {type(a)}.")

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
            values = array if array.ndim == 0 else array[idxs]
            raise ValueError(f"{cls.name} arrays must have elements in `0 <= x < {cls.order}`, not {values}.")

    @classmethod
    def _check_string_value(cls, string):
        return str_to_integer(string, cls.prime_subfield)

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(cls, shape, dtype=None):
        """
        Creates a Galois field array with all zeros.

        Parameters
        ----------
        shape : int, tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M,N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
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
        shape : int, tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M,N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
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
        Creates a 1-D Galois field array with a range of field elements.

        Parameters
        ----------
        start : int
            The starting Galois field value (inclusive) in its integer representation.
        stop : int
            The stopping Galois field value (exclusive) in its integer representation.
        step : int, optional
            The space between values. The default is 1.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A 1-D Galois field array of a range of field elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Range(10,20)
        """
        if not stop <= cls.order:
            raise ValueError(f"The stopping value must be less than the field order of {cls.order}, not {stop}.")
        dtype = cls._get_dtype(dtype)

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
        shape : int, tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M,N)`, represents a 2-D array with each element indicating the size in each dimension.
        low : int, optional
            The lowest value (inclusive) of a random field element in its integer representation. The default is 0.
        high : int, optional
            The highest value (exclusive) of a random field element in its integer representation. The default is `None`
            which represents the field's order :math:`p^m`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
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
        r"""
        Creates a 1-D Galois field array of the field's elements :math:`\{0, \dots, p^m-1\}`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A 1-D Galois field array of all the field's elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**4)
            GF.Elements()

        As usual, Galois field elements can be displayed in either the "integer" (default), "polynomial", or "power" representation.
        This can be changed by calling :func:`galois.FieldClass.display`.

        .. ipython:: python

            # Permanently set the display mode to "poly"
            GF.display("poly");
            GF.Elements()
            # Temporarily set the display mode to "power"
            with GF.display("power"):
                print(GF.Elements())
            # Reset the display mode to "int"
            GF.display();
        """
        return cls.Range(0, cls.order, step=1, dtype=dtype)

    @classmethod
    def Identity(cls, size, dtype=None):
        r"""
        Creates an :math:`n \times n` Galois field identity matrix.

        Parameters
        ----------
        size : int
            The size :math:`n` along one axis of the matrix. The resulting array has shape `(size, size)`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field identity matrix of shape `(size, size)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Identity(4)
        """
        dtype = cls._get_dtype(dtype)
        array = np.identity(size, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Vandermonde(cls, a, m, n, dtype=None):
        r"""
        Creates an :math:`m \times n` Vandermonde matrix of :math:`a \in \mathrm{GF}(p^m)`.

        Parameters
        ----------
        a : int, galois.FieldArray
            An element of :math:`\mathrm{GF}(p^m)`.
        m : int
            The number of rows in the Vandermonde matrix.
        n : int
            The number of columns in the Vandermonde matrix.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            The :math:`m \times n` Vandermonde matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**3)
            a = GF.primitive_element
            V = GF.Vandermonde(a, 7, 7)
            with GF.display("power"):
                print(V)
        """
        if not isinstance(a, (int, np.integer, cls)):
            raise TypeError(f"Argument `a` must be an integer or element of {cls.name}, not {type(a)}.")
        if not isinstance(m, (int, np.integer)):
            raise TypeError(f"Argument `m` must be an integer, not {type(m)}.")
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
        if not m > 0:
            raise ValueError(f"Argument `m` must be non-negative, not {m}.")
        if not n > 0:
            raise ValueError(f"Argument `n` must be non-negative, not {n}.")

        dtype = cls._get_dtype(dtype)
        a = cls(a, dtype=dtype)
        if not a.ndim == 0:
            raise ValueError(f"Argument `a` must be a scalar, not {a.ndim}-D.")

        v = a ** np.arange(0, m)
        V = np.power.outer(v, np.arange(0, n))

        return V

    @classmethod
    def Vector(cls, array, dtype=None):
        r"""
        Creates a Galois field array over :math:`\mathrm{GF}(p^m)` from length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`vector` method.

        Parameters
        ----------
        array : array_like
            The input array with field elements in :math:`\mathrm{GF}(p)` to be converted to a Galois field array in :math:`\mathrm{GF}(p^m)`.
            The last dimension of the input array must be :math:`m`. An input array with shape `(n1, n2, m)` has output shape `(n1, n2)`. By convention,
            the vectors are ordered from highest degree to 0-th degree.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array over :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**6)
            vec = galois.GF2.Random((3,6)); vec
            a = GF.Vector(vec); a
            with GF.display("poly"):
                print(a)
            a.vector()
        """
        order = cls.prime_subfield.order
        degree = cls.degree
        array = cls.prime_subfield(array).view(np.ndarray).astype(cls.dtypes[-1])  # Use the largest dtype so computation doesn't overflow
        if not array.shape[-1] == degree:
            raise ValueError(f"The last dimension of `array` must be the field extension dimension {cls.degree}, not {array.shape[-1]}.")
        degrees = np.arange(degree - 1, -1, -1, dtype=cls.dtypes[-1])
        array = np.sum(array * order**degrees, axis=-1)
        return cls(array, dtype=dtype)

    ###############################################################################
    # Instance methods
    ###############################################################################

    def vector(self, dtype=None):
        r"""
        Converts the Galois field array over :math:`\mathrm{GF}(p^m)` to length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`Vector` constructor. For an array with shape `(n1, n2)`, the output shape
        is `(n1, n2, m)`. By convention, the vectors are ordered from highest degree to 0-th degree.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of length-:math:`m` vectors over :math:`\mathrm{GF}(p)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**6)
            a = GF.Random(3); a
            with GF.display("poly"):
                print(a)
            vec = a.vector(); vec
            GF.Vector(vec)
        """
        order = type(self).prime_subfield.order
        degree = type(self).degree
        array = self.view(np.ndarray)
        array = np.repeat(array, degree).reshape(*array.shape, degree)
        x = 0
        for i in range(degree):
            q = (array[...,i] - x) // order**(degree - 1 - i)
            array[...,i] = q
            x += q*order**(degree - 1 - i)
        return type(self).prime_subfield(array, dtype=dtype)  # pylint: disable=unexpected-keyword-arg

    def row_reduce(self, ncols=None):
        r"""
        Performs Gaussian elimination on the matrix to achieve reduced row echelon form.

        **Row reduction operations**

        1. Swap the position of any two rows.
        2. Multiply a row by a non-zero scalar.
        3. Add one row to a scalar multiple of another row.

        Parameters
        ----------
        ncols : int, optional
            The number of columns to perform Gaussian elimination over. The default is `None` which represents
            the number of columns of the input array.

        Returns
        -------
        galois.FieldArray
            The reduced row echelon form of the input array.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        One column is a linear combination of another.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A[:,2] = A[:,1] * GF(17); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        One row is a linear combination of another.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A[3,:] = A[2,:] * GF(8); A
            A.row_reduce()
            np.linalg.matrix_rank(A)
        """
        return row_reduce(self, ncols=ncols)

    def lu_decompose(self):
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices.

        Returns
        -------
        galois.FieldArray
            The lower triangular matrix.
        galois.FieldArray
            The upper triangular matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(5)

            # Not every square matrix has an LU decomposition
            A = GF([[2, 4, 4, 1], [3, 3, 1, 4], [4, 3, 4, 2], [4, 4, 3, 1]])
            L, U = A.lu_decompose()
            L
            U

            # A = L U
            np.array_equal(A, L @ U)
        """
        return lu_decompose(self)

    def lup_decompose(self):
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices using partial pivoting.

        Returns
        -------
        galois.FieldArray
            The lower triangular matrix.
        galois.FieldArray
            The upper triangular matrix.
        galois.FieldArray
            The permutation matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(5)
            A = GF([[1, 3, 2, 0], [3, 4, 2, 3], [0, 2, 1, 4], [4, 3, 3, 1]])
            L, U, P = A.lup_decompose()
            L
            U
            P

            # P A = L U
            np.array_equal(P @ A, L @ U)
        """
        return lup_decompose(self)

    ###############################################################################
    # Special methods (redefined to add docstrings)
    ###############################################################################

    def __add__(self, other):  # pylint: disable=useless-super-delegation
        """
        Adds two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self + other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5); b
            a + b
        """
        return super().__add__(other)

    def __sub__(self, other):  # pylint: disable=useless-super-delegation
        """
        Subtracts two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self - other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5); b
            a - b
        """
        return super().__sub__(other)

    def __mul__(self, other):  # pylint: disable=useless-super-delegation
        """
        Multiplies two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field.

        Warning
        -------
        When both multiplicands are :obj:`galois.FieldArray`, that indicates a Galois field multiplication. When one
        multiplicand is an integer or integer :obj:`numpy.ndarray`, that indicates a scalar multiplication (repeated addition).
        Galois field multiplication and scalar multiplication are equivalent in prime fields, but not in extension fields.

        Parameters
        ----------
        other : numpy.ndarray, galois.FieldArray
            A :obj:`numpy.ndarray` of integers for scalar multiplication or a :obj:`galois.FieldArray` of Galois field elements
            for finite field multiplication.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self * other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5); b
            a * b

        When both multiplicands are Galois field elements, that indicates a Galois field multiplication.

        .. ipython:: python

            GF = galois.GF(2**4, display="poly")
            a = GF(7); a
            b = GF(2); b
            a * b

        When one multiplicand is an integer, that indicates a scalar multiplication (repeated addition).

        .. ipython:: python

            a * 2
            a + a
        """
        return super().__mul__(other)

    def __truediv__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self / other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            a / b
        """
        return super().__truediv__(other)

    def __floordiv__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self // other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            a // b
        """
        return super().__floordiv__(other)  # pylint: disable=too-many-function-args

    def __divmod__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise and returns the quotient and remainder.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent. In Galois fields, the remainder
        is always zero.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self // other`.
        galois.FieldArray
            The Galois field array `self % other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            q, r = divmod(a, b)
            q, r
            b*q + r
        """
        return super().__divmod__(other)

    def __mod__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise and returns the remainder.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent. In Galois fields, the remainder
        is always zero.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self % other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            a % b
        """
        return super().__mod__(other)

    def __pow__(self, other):
        """
        Exponentiates a Galois field array element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. The first array must be a
        Galois field array and the second must be an integer or integer array.

        Parameters
        ----------
        other : int, numpy.ndarray
            The exponent(s) as an integer or integer array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self ** other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = np.random.randint(0, 10, 5); b
            a ** b
        """
        # NOTE: Calling power here instead of `super().__pow__(other)` because when doing so `x ** GF(2)` will invoke `np.square(x)` and not throw
        # an error. This way `np.power(x, GF(2))` is called which correctly checks whether the second argument is an integer.
        return np.power(self, other)

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
        if obj is not None and not isinstance(obj, FieldArray):
            # Only invoked on view casting
            if obj.dtype not in type(self).dtypes:
                raise TypeError(f"{type(self).name} can only have integer dtypes {type(self).dtypes}, not {obj.dtype}.")
            self._check_array_values(obj)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if np.isscalar(item):
            # Return scalar array elements as 0-dimensional Galois field arrays. This enables Galois field arithmetic
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
            raise NotImplementedError(f"The numpy function {func.__name__!r} is not supported on Galois field arrays. If you believe this function should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\nIf you'd like to perform this operation on the data (but not necessarily a Galois field array), you should first call `array = array.view(np.ndarray)` and then call the function.")

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._check_array_like_object(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in type(self)._functions_requiring_view:
                output = output.view(type(self)) if not np.isscalar(output) else type(self)(output, dtype=self.dtype)

        return output

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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
            raise NotImplementedError(f"The numpy ufunc {ufunc.__name__!r} is not supported on {type(self).name} arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            if ufunc in [np.bitwise_and, np.bitwise_or, np.bitwise_xor] and method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

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
            cls.__name__ = "GF"  # Rename the class so very large fields don't create large indenting
            string = super().__repr__()
        cls.__name__ = class_name

        # Remove the dtype from the repr and add the Galois field order
        dtype_idx = string.find("dtype")
        if dtype_idx == -1:
            string = string[:-1] + f", {cls._order_str})"
        else:
            string = string[:dtype_idx] + f"{cls._order_str})"

        return string
