import numba
import numpy as np

from ..dtypes import DTYPES
from .meta import FieldMeta


class GF2Meta(FieldMeta):
    """
    Create an array over :math:`\\mathrm{GF}(2)`.

    Note
    ----
        This Galois field class is a pre-made subclass of :obj:`galois.FieldArray`. It is included in the package
        because of the ubiquity of :math:`\\mathrm{GF}(2)` fields.

        .. ipython:: python

            # The pre-made GF(2) class
            print(galois.GF2)

            # The GF class factory for an order of 2 returns `galois.GF2`
            GF2 = galois.GF(2); print(GF2)
            GF2 is galois.GF2

    Parameters
    ----------
    array : array_like
        The input array to be converted to a Galois field array. The input array is copied, so the original array
        is unmodified by changes to the Galois field array. Valid input array types are :obj:`numpy.ndarray`,
        :obj:`list`, :obj:`tuple`, or :obj:`int`.
    dtype : numpy.dtype, optional
        The :obj:`numpy.dtype` of the array elements. The default is :obj:`numpy.uint8`.

    Returns
    -------
    galois.GF2
        The copied input array as a :math:`\\mathrm{GF}(2)` field array.

    Examples
    --------

    Various Galois field properties are accessible as class attributes.

    .. ipython:: python

        print(galois.GF2)
        galois.GF2.characteristic
        galois.GF2.degree
        galois.GF2.order
        galois.GF2.irreducible_poly

    Construct arrays over :math:`\\mathrm{GF}(2)`.

    .. ipython:: python

        a = galois.GF2([1,0,1,1]); a
        b = galois.GF2([1,1,1,1]); b

    Perform array arithmetic over :math:`\\mathrm{GF}(2)`.

    .. ipython:: python

        # Element-wise addition
        a + b

        # Element-wise subtraction
        a - b

        # Element-wise multiplication
        a * b

        # Element-wise division
        a / b
    """
    # pylint: disable=abstract-method,no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly = None  # Will be set in __init__.py to avoid circular import with Poly
        cls._prime_subfield = cls
        cls._is_primitive_poly = True

        cls.compile(kwargs["mode"], kwargs["target"])

    @property
    def dtypes(cls):
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(d) == 0:
            d = [np.object_]
        return d

    @property
    def ufunc_modes(cls):
        return ["jit-calculate"]

    @property
    def default_ufunc_mode(cls):
        return "jit-calculate"

    def _compile_ufuncs(cls, target):
        assert cls._ufunc_mode == "jit-calculate"
        assert target == "cpu"

        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor
        cls._ufuncs["multiply"] = np.bitwise_and
        # NOTE: Don't need a ufunc for "reciprocal", already overrode _ufunc_reciprocal()
        cls._ufuncs["divide"] = np.bitwise_and
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
        # NOTE: Don't need a ufunc for "log", already overrode _ufunc_log()

    ###############################################################################
    # Override ufunc routines to use native numpy bitwise ufuncs for GF(2)
    # arithmetic, which is faster than custom ufuncs
    ###############################################################################

    def _ufunc_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, b in GF(2)
        b = 1 / a, a = 1 is the only valid element with a multiplicative inverse, which is 1
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        if np.count_nonzero(inputs[0]) != inputs[0].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        return inputs[0]

    def _ufunc_divide(cls, ufunc, method, inputs, kwargs, meta):
        """
        Need to re-implement this to manually throw ZeroDivisionError if necessary
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        if np.count_nonzero(inputs[meta["operands"][-1]]) != inputs[meta["operands"][-1]].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufuncs["divide"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, c in GF(2)
        c = a ** 2
          = a * a
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        return inputs[0]

    def _ufunc_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a in GF(2)
        b in Z
        b = log_α(a), a = 1 is the only valid element with a logarithm base α, which is 0
          = 0
        """
        cls._verify_method_only_call(ufunc, method)
        if np.count_nonzero(inputs[meta["operands"][0]]) != inputs[meta["operands"][0]].size:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(np.bitwise_and, method)(*inputs, 0, **kwargs)
        return output


###############################################################################
# Galois field arithmetic, explicitly calculated without lookup tables
###############################################################################

def _power_calculate(a, power):  # pragma: no cover
    if a == 0 and power < 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if power == 0:
        return 1
    elif a == 0:
        return 0
    else:
        return a
