from .array import GFArray
from .meta_gf2 import GF2Meta
from .overrides import set_module

__all__ = ["GF2"]


@set_module("galois")
class GF2(GFArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, mode="jit-calculate", target="cpu"):
    """
    A pre-created Galois field array class for :math:`\\mathrm{GF}(2)`.

    This class is a subclass of :obj:`galois.GFArray` and has metaclass :obj:`galois.GFMeta`.

    Examples
    --------
    This class is equivalent (and, in fact, identical) to the class returned from the Galois field
    array class constructor.

    .. ipython:: python

        print(galois.GF2)
        GF2 = galois.GF(2); print(GF2)
        GF2 is galois.GF2

    The Galois field properties can be viewed by class attributes, see :obj:`galois.GFMeta`.

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
