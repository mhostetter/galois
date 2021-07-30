from .._overrides import set_module

from ._array import FieldArray
from ._meta_gf2 import GF2Meta

__all__ = ["GF2"]


@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, compile="jit-calculate"):
    r"""
    Creates an array over :math:`\mathrm{GF}(2)`.

    This class is a pre-generated :obj:`galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
    for convenience. See :obj:`galois.FieldArray` and :obj:`galois.FieldClass` for more complete documentation and examples.

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
        The copied input array as a Galois field array over :math:`\mathrm{GF}(2)`.

    Examples
    --------
    This class is equivalent (and, in fact, identical) to the class returned from the Galois field class constructor.

    .. ipython:: python

        print(galois.GF2)
        GF2 = galois.GF(2); print(GF2)
        GF2 is galois.GF2

    The Galois field properties can be viewed by class attributes, see :obj:`galois.FieldClass`.

    .. ipython:: python

        # View a summary of the field's properties
        print(galois.GF2.properties)

        # Or access each attribute individually
        galois.GF2.irreducible_poly
        galois.GF2.is_prime_field

    The class's constructor mimics the call signature of :func:`numpy.array`.

    .. ipython:: python

        # Construct a Galois field array from an iterable
        galois.GF2([1,0,1,1,0,0,0,1])

        # Or an iterable of iterables
        galois.GF2([[1,0], [1,1]])

        # Or a single integer
        galois.GF2(1)
    """
