from ..overrides import set_module

from .array import FieldArray
from .meta_gf2 import GF2Meta

__all__ = ["GF2"]


@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, mode="jit-calculate"):
    """
    Creates an array over :math:`\\mathrm{GF}(2)`.

    This class is a subclass of :obj:`galois.FieldArray` and instance of :obj:`galois.FieldClass`.

    Parameters
    ----------
    array : array_like
        The input array to be converted to a Galois field array. The input array is copied, so the original array
        is unmodified by changes to the Galois field array. Valid input array types are :obj:`numpy.ndarray`,
        :obj:`list` or :obj:`tuple` of int or str, :obj:`int`, or :obj:`str`.
    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest valid
        dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.
    copy : bool, optional
        The `copy` keyword argument from :func:`numpy.array`. The default is `True` which makes a copy of the input
        object is it's an array.
    order : {`"K"`, `"A"`, `"C"`, `"F"`}, optional
        The `order` keyword argument from :func:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
    ndmin : int, optional
        The `ndmin` keyword argument from :func:`numpy.array`. The minimum number of dimensions of the output.
        The default is 0.

    Examples
    --------
    This class is equivalent (and, in fact, identical) to the class returned from the Galois field
    array class constructor.

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
        galois.GF2([[1,0],[1,1]])

        # Or a single integer
        galois.GF2(1)
    """
