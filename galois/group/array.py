import random

import numpy as np

from ..array import Array
from ..overrides import set_module

from .meta import GroupMeta

__all__ = ["GroupArray"]


@set_module("galois")
class GroupArray(Array, metaclass=GroupMeta):
    """
    Creates an array over :math:`(\\mathbb{Z}/n\\mathbb{Z}){^+}` or :math:`(\\mathbb{Z}/n\\mathbb{Z}){^\\times}`.

    The :obj:`galois.GroupArray` class is a parent class for all finite group array classes. Any finite group
    :math:`(\\mathbb{Z}/n\\mathbb{Z}){^+}` or :math:`(\\mathbb{Z}/n\\mathbb{Z}){^\\times}` can be constructed by calling the class factory
    `galois.Group(n, "+")` or `galois.Group(n, "*")`.

    Warning
    -------
        This is an abstract base class for all finite group array classes. :obj:`galois.GroupArray` cannot be instantiated
        directly. Instead, finite group array classes are created using :func:`galois.Group`.

        For example, one can create the :math:`(\\mathbb{Z}/16\\mathbb{Z}){^+}` finite additive group array class as follows:

        .. ipython:: python

            G = galois.Group(16, "+")
            print(G.properties)

        This subclass can then be used to instantiate arrays over :math:`(\\mathbb{Z}/16\\mathbb{Z}){^+}`.

        .. ipython:: python

            G([3,5,0,2,1])
            G.Random((2,5))

        Creating the :math:`(\\mathbb{Z}/16\\mathbb{Z}){^\\times}` finite multiplicative group array class is just as easy:

        .. ipython:: python

            G = galois.Group(16, "*")
            print(G.properties)
            G.Random((2,5))

    :obj:`galois.GroupArray` is a subclass of :obj:`numpy.ndarray`. The :obj:`galois.GroupArray` constructor has the same syntax as
    :func:`numpy.array`. The returned :obj:`galois.GroupArray` object is an array that can be acted upon like any other
    numpy array.

    Parameters
    ----------
    array : array_like
        The input array to be converted to a finite group array. The input array is copied, so the original array
        is unmodified by changes to the finite group array. Valid input array types are :obj:`numpy.ndarray`,
        :obj:`list` or :obj:`tuple` of int, or :obj:`int`.
    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
        dtype for this class, i.e. the first element in :obj:`galois.GroupMeta.dtypes`.
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
    galois.GroupArray
        The copied input array as a finite group array.
    """

    def __new__(cls, array, dtype=None, copy=True, order="K", ndmin=0):
        if cls is GroupArray:
            raise NotImplementedError("GroupArray is an abstract base class that cannot be directly instantiated. Instead, create a GroupArray subclass for ℤn+ arithmetic using `galois.Group(n, \"+\")` or for ℤn* using `galois.Group(n, \"*\")`.")
        return cls._array(array, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    @classmethod
    def _check_string_value(cls, string):
        raise ValueError(f"Cannot convert a string to an element of {cls.name}.")

    @classmethod
    def _check_array_values(cls, array):
        if not isinstance(array, np.ndarray):
            # Convert single integer to array so next step doesn't fail
            array = np.array(array)

        if cls.operator == "+":
            # Check the value of the "field elements" and make sure they are valid
            if np.any(array < 0) or np.any(array >= cls.modulus):
                idxs = np.logical_or(array < 0, array >= cls.modulus)
                values = array if array.ndim == 0 else array[idxs]
                raise ValueError(f"{cls.name} arrays must have elements in 0 <= x < {cls.modulus}, not {values}.")
        else:
            # Check that each element is coprime with n
            if not np.all(np.gcd(array, cls.modulus) == 1):
                idxs = np.where(np.gcd(array, cls.modulus) != 1)
                values = array if array.ndim == 0 else array[idxs]
                raise ValueError(f"{cls.name} arrays must have elements coprime to {cls.modulus}, not {values}.")

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(cls, shape, dtype=None):
        """
        Creates a finite group array with all zeros.

        This constructor is only valid for additive groups, since 0 is not an element of multiplicative groups.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GroupMeta.dtypes`.

        Returns
        -------
        galois.GroupArray
            A finite group array of zeros.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+")
            G.Zeros((2,5))
        """
        if cls.operator == "+":
            dtype = cls._get_dtype(dtype)
            array = np.zeros(shape, dtype=dtype)
            return array.view(cls)
        else:
            raise ValueError(f"0 is not a valid element of {cls.name}.")

    @classmethod
    def Ones(cls, shape, dtype=None):
        """
        Creates a finite group array with all ones.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GroupMeta.dtypes`.

        Returns
        -------
        galois.GroupArray
            A finite group array of ones.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "*")
            G.Ones((2,5))
        """
        dtype = cls._get_dtype(dtype)
        array = np.ones(shape, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Range(cls, start, stop, step=1, dtype=None):
        """
        Creates a finite group array with a range of group elements.

        This constructor is only valid for additive groups since multiplicative groups don't have equally-spaced
        elements.

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
            dtype for this class, i.e. the first element in :obj:`galois.GroupMeta.dtypes`.

        Returns
        -------
        galois.GroupArray
            A finite group array of a range of group elements.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(36, "+")
            G.Range(10, 20)
        """
        if not stop <= cls.modulus:
            raise ValueError(f"The stopping value must be less than the group modulus of {cls.modulus}, not {stop}.")
        if cls.operator == "*":
            raise NotImplementedError("Creating a range of elements from a multiplicative group is not supported because multiplicative groups don't have equally-spaced elements.")
        dtype = cls._get_dtype(dtype)

        if dtype != np.object_:
            array = np.arange(start, stop, step=step, dtype=dtype)
        else:
            array = np.array(range(start, stop, step), dtype=dtype)

        return array.view(cls)

    @classmethod
    def Random(cls, shape=(), low=0, high=None, dtype=None):
        """
        Creates a finite group array with random group elements.

        Parameters
        ----------
        shape : tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-dim array. An n-tuple, e.g.
            `(M,N)`, represents an n-dim array with each element indicating the size in each dimension.
        low : int, optional
            The lowest value (inclusive) of a random group element. The default is 0.
        high : int, optional
            The highest value (exclusive) of a random group element. The default is `None` which represents the group's
            modulus :math:`n`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GroupMeta.dtypes`.

        Returns
        -------
        galois.GroupArray
            A finite group array of random group elements.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "*")
            G.Random((2,5))
        """
        # pylint: disable=too-many-branches
        dtype = cls._get_dtype(dtype)
        high = cls.order if high is None else high
        if not 0 <= low < high <= cls.modulus:
            raise ValueError(f"Arguments must satisfy `0 <= low < high <= modulus`, not `0 <= {low} < {high} <= {cls.modulus}`.")

        if dtype != np.object_:
            if cls.operator == "+":
                array = np.random.randint(low, high, shape, dtype=dtype)
            elif cls.modulus <= 100_000:
                # The modulus is small enough to generate the entire set
                return cls(np.random.choice(list(cls.set), size=shape), dtype=dtype)
            else:
                # The modulus may be very large and we don't want to generate the whole set. Instead, we'll iteratively
                # generate random numbers until they're all coprime to n
                array = np.random.randint(low, high, shape, dtype=dtype)

                array = np.atleast_1d(array)
                while True:
                    idxs = np.where(np.gcd(array, cls.modulus) != 1)
                    if idxs[0].size == 0:
                        break
                    array[idxs] = np.random.randint(low, high, idxs[0].size, dtype=dtype)
                if shape == ():
                    array = array[0]
        else:
            array = np.empty(shape, dtype=dtype)
            iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
            for _ in iterator:
                array[iterator.multi_index] = random.randint(low, high - 1)

            if cls.operator == "*":
                array = np.atleast_1d(array)
                while True:
                    idxs = np.where(np.gcd(array, cls.modulus) != 1)
                    if idxs[0].size == 0:
                        break
                    array[idxs] = [random.randint(low, high - 1) for _ in range(idxs[0].size)]
                if shape == ():
                    array = np.array(array[0], dtype=object)

        return array.view(cls)

    @classmethod
    def Elements(cls, dtype=None):
        """
        Creates a finite group array of the group's elements.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
            dtype for this class, i.e. the first element in :obj:`galois.GroupMeta.dtypes`.

        Returns
        -------
        galois.GroupArray
            A finite group array of all the group's elements.

        Examples
        --------
        .. ipython:: python

            G = galois.Group(16, "+")
            G.Elements()

        .. ipython:: python

            G = galois.Group(16, "*")
            G.Elements()
        """
        dtype = cls._get_dtype(dtype)

        if dtype != np.object_:
            array = np.arange(0, cls.modulus, dtype=dtype)
        else:
            array = np.array(range(0, cls.modulus), dtype=dtype)

        if cls.operator == "*":
            idxs = np.where(np.gcd(array, cls.modulus) == 1)
            array = array[idxs]

        return array.view(cls)
