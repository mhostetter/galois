"""
A module that defines the GF(2) array class.
"""
from __future__ import annotations

from .._overrides import set_module

from ._array import FieldArray
from ._ufunc import FieldUFuncs_2_1

__all__ = ["GF2"]


# NOTE: There is a "verbatim" block in the docstring because we were not able to monkey-patch GF2 like the
# other classes in docs/conf.py. So, technically, at doc-build-time issubclass(galois.GF2, galois.FieldArray) == False
# because galois.FieldArray is monkey-patched and GF2 is not. This all stems from an inability of Sphinx to
# document class properties... :(


@set_module("galois")
class GF2(FieldArray, FieldUFuncs_2_1, characteristic=2, degree=1, order=2, irreducible_poly_int=3, is_primitive_poly=True, primitive_element=1, compile="jit-calculate"):
    r"""
    A :obj:`~numpy.ndarray` subclass over :math:`\mathrm{GF}(2)`.

    Important
    ---------
    This class is a pre-generated :obj:`~galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
    for convenience.

    Examples
    --------
    This class is equivalent, and in fact identical, to the :obj:`~galois.FieldArray` subclass returned from the class factory
    :func:`~galois.GF`.

    .. ipython::

        In [2]: galois.GF2 is galois.GF(2)

        @verbatim
        In [3]: issubclass(galois.GF2, galois.FieldArray)
        Out[3]: True

        In [4]: print(galois.GF2.properties)

    Create a :obj:`~galois.FieldArray` instance using :obj:`~galois.GF2`'s constructor.

    .. ipython:: python

        x = galois.GF2([1, 0, 1, 1]); x
        isinstance(x, galois.GF2)

    Note
    ----
    Only the class docstring is documented on this page. See :obj:`~galois.FieldArray` for all other classmethods, class properties,
    and methods inherited by :obj:`~galois.GF2`.
    """

    # @property
    # def ufunc_modes(cls):
    #     return ["jit-calculate"]
    _ufunc_modes = ["jit-calculate"]

    # @property
    # def default_ufunc_mode(cls):
    #     return "jit-calculate"
    _default_ufunc_mode = "jit-calculate"
