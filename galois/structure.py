"""
A module to determine the algebraic structure of an array or array class.
"""
import numpy as np

from .field.array import FieldArray

__all__ = ["is_field", "is_prime_field", "is_extension_field"]


###############################################################################
# Fields
###############################################################################

def is_field(obj):
    """
    Determines if the object is a Galois field array class created from :func:`galois.GF` (or :func:`galois.Field`)
    of one of its instances.

    Parameters
    ----------
    obj : type
        Any object.

    Returns
    -------
    bool
        `True` if `obj` is a Galois field array class generated from :func:`galois.GF` (or :func:`galois.Field`) or one of its instances.
    """
    if isinstance(obj, np.ndarray):
        obj = type(obj)
    return issubclass(obj, FieldArray) and obj is not FieldArray


def is_prime_field(obj):
    """
    Determines if the object is a Galois field array class of type :math:`\\mathrm{GF}(p)` created from :func:`galois.GF` (or :func:`galois.Field`)
    of one of its instances.

    Parameters
    ----------
    obj : type
        Any object.

    Returns
    -------
    bool
        `True` if `obj` is a Galois field array class of type :math:`\\mathrm{GF}(p)` generated from :func:`galois.GF` (or :func:`galois.Field`) or one of its instances.
    """
    if isinstance(obj, np.ndarray):
        obj = type(obj)
    return issubclass(obj, FieldArray) and obj is not FieldArray and obj.degree == 1


def is_extension_field(obj):
    """
    Determines if the object is a Galois field array class of type :math:`\\mathrm{GF}(p^m)` created from :func:`galois.GF` (or :func:`galois.Field`)
    of one of its instances.

    Parameters
    ----------
    obj : type
        Any object.

    Returns
    -------
    bool
        `True` if `obj` is a Galois field array class of type :math:`\\mathrm{GF}(p^m)` generated from :func:`galois.GF` (or :func:`galois.Field`) or one of its instances.
    """
    if isinstance(obj, np.ndarray):
        obj = type(obj)
    return issubclass(obj, FieldArray) and obj is not FieldArray and obj.degree > 1
