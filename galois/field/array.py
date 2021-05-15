import random

import numpy as np

from ..array import Array
from ..overrides import set_module

from .linalg import row_reduce, lu_decompose, lup_decompose
from .meta import FieldMeta
from .poly_conversion import str_to_integer

__all__ = ["FieldArray"]


@set_module("galois")
class FieldArray(Array, metaclass=FieldMeta):
    """
    Creates an array over :math:`\\mathrm{GF}(p^m)`.

    The :obj:`galois.FieldArray` class is a parent class for all Galois field array classes. Any Galois field :math:`\\mathrm{GF}(p^m)`
    with prime characteristic :math:`p` and positive integer :math:`m`, can be constructed by calling the class factory
    `galois.GF(p**m)`.

    Warning
    -------
        This is an abstract base class for all Galois field array classes. :obj:`galois.FieldArray` cannot be instantiated
        directly. Instead, Galois field array classes are created using :func:`galois.GF`.

        For example, one can create the :math:`\\mathrm{GF}(7)` field array class as follows:

        .. ipython:: python

            GF7 = galois.GF(7)
            print(GF7)

        This subclass can then be used to instantiate arrays over :math:`\\mathrm{GF}(7)`.

        .. ipython:: python

            GF7([3,5,0,2,1])
            GF7.Random((2,5))

    :obj:`galois.FieldArray` is a subclass of :obj:`numpy.ndarray`. The :obj:`galois.FieldArray` constructor has the same syntax as
    :func:`numpy.array`. The returned :obj:`galois.FieldArray` object is an array that can be acted upon like any other
    numpy array.

    Parameters
    ----------
    array : array_like
        The input array to be converted to a Galois field array. The input array is copied, so the original array
        is unmodified by changes to the Galois field array. Valid input array types are :obj:`numpy.ndarray`,
        :obj:`list` or :obj:`tuple` of int or str, :obj:`int`, or :obj:`str`.
    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
        dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.
    copy : bool, optional
        The `copy` keyword argument from :func:`numpy.array`. The default is `True` which makes a copy of the input
        object is it's an array.
    order : {`"K"`, `"A"`, `"C"`, `"F"`}, optional
        The `order` keyword argument from :func:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
    ndmin : int, optional
        The `ndmin` keyword argument from :func:`numpy.array`. The minimum number of dimensions of the output.
        The default is 0.

    Returns
    -------
    galois.FieldArray
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
        if cls is FieldArray:
            raise NotImplementedError("FieldArray is an abstract base class that cannot be directly instantiated. Instead, create a FieldArray subclass for GF(p^m) arithmetic using `galois.GF(p**m)`.")
        return cls._array(array, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    @classmethod
    def _check_array_values(cls, array):
        if not isinstance(array, np.ndarray):
            # Convert single integer to array so next step doesn't fail
            array = np.array(array)

        # Check the value of the "field elements" and make sure they are valid
        if np.any(array < 0) or np.any(array >= cls.order):
            idxs = np.logical_or(array < 0, array >= cls.order)
            values = array if array.ndim == 0 else array[idxs]
            raise ValueError(f"{cls.name} arrays must have elements in 0 <= x < {cls.order}, not {values}.")

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
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

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
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

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
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of a range of field elements.

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
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

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
        """
        Creates a Galois field array of the field's elements :math:`\\{0, \\dots, p^m-1\\}`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of all the field's elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Elements()
        """
        return cls.Range(0, cls.order, step=1, dtype=dtype)

    @classmethod
    def Identity(cls, size, dtype=None):
        """
        Creates an :math:`n \\times n` Galois field identity matrix.

        Parameters
        ----------
        size : int
            The size :math:`n` along one axis of the matrix. The resulting array has shape `(size,size)`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

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
        """
        Creates a :math:`m \\times n` Vandermonde matrix of :math:`a \\in \\mathrm{GF}(p^m)`.

        Parameters
        ----------
        a : int, galois.FieldArray
            An element of :math:`\\mathrm{GF}(p^m)`.
        m : int
            The number of rows in the Vandermonde matrix.
        n : int
            The number of columns in the Vandermonde matrix.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

        Returns
        -------
        galois.FieldArray
            The :math:`m \\times n` Vandermonde matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**3)
            a = GF.primitive_element
            V = GF.Vandermonde(a, 7, 7)
            with GF.display("power"):
                print(V)
        """
        if not isinstance(a, (int, np.integer,cls)):
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
        """
        Creates a Galois field array over :math:`\\mathrm{GF}(p^m)` from length-:math:`m` vectors over the prime subfield :math:`\\mathrm{GF}(p)`.

        Parameters
        ----------
        array : array_like
            The input array with field elements in :math:`\\mathrm{GF}(p)` to be converted to a Galois field array in :math:`\\mathrm{GF}(p^m)`.
            The last dimension of the input array must be :math:`m`. An input array with shape `(n1, n2, m)` has output shape `(n1, n2)`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array over :math:`\\mathrm{GF}(p^m)`.

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
    # Array methods
    ###############################################################################

    def vector(self, dtype=None):
        """
        Converts the Galois field array over :math:`\\mathrm{GF}(p^m)` to length-:math:`m` vectors over the prime subfield :math:`\\mathrm{GF}(p)`.

        For an input array with shape `(n1, n2)`, the output shape is `(n1, n2, m)`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.FieldMeta.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of length-:math:`m` vectors over :math:`\\mathrm{GF}(p)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**6)
            a = GF.Random(3); a
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
        return type(self).prime_subfield(array, dtype=dtype)

    def row_reduce(self, ncols=None):
        """
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
        """
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
        """
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
