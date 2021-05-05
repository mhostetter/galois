import numba
import numpy as np

from .dtypes import DTYPES
from .meta_gf import GFMeta

# Field attribute globals
CHARACTERISTIC = None  # The prime characteristic `p` of the Galois field

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y


class GF2Meta(GFMeta):
    """
    Create an array over :math:`\\mathrm{GF}(2)`.

    Note
    ----
        This Galois field class is a pre-made subclass of :obj:`galois.GFArray`. It is included in the package
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
        The :obj:`numpy.dtype` of the array elements. The default is :obj:`numpy.int64`.

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
        cls._primitive_element_int = int(cls._primitive_element)
        cls._prime_subfield = cls

        # Use cached ufuncs only where necessary
        kwargs = {"nopython": True, "cache": True}
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_calculate)
        cls._ufuncs["poly_eval"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_calculate)

        cls._primitive_element = cls(cls.primitive_element)
        cls._is_primitive_poly = True

    def compile(cls, mode, target="cpu"):
        """
        Error
        -----
        The Galois field array class for GF(2) cannot be recompiled. It is pre-compiled using
        native numpy bitwise ufuncs.
        """
        raise RuntimeError("Cannot recompile GF(2) Galois field arrays. They are pre-compiled using numpy bitwise operations.")

    @property
    def dtypes(cls):
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(d) == 0:
            d = [np.object_]
        return d

    @property
    def ufunc_mode(cls):
        return "jit-calculate"

    @property
    def ufunc_modes(cls):
        return ["jit-calculate"]

    @property
    def default_ufunc_mode(cls):
        return "jit-calculate"

    @property
    def ufunc_target(cls):
        return "cpu"

    ###############################################################################
    # Override ufunc routines to use native numpy bitwise ufuncs for GF(2)
    ###############################################################################

    def _ufunc_add(cls, ufunc, method, inputs, kwargs, meta):
        """
        a, b, c in GF(2)
        c = a + b
          = a ^ b
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(np.bitwise_xor, method)(*inputs, **kwargs)
        if np.isscalar(output):
            output = meta["field"](output)
        return output

    def _ufunc_subtract(cls, ufunc, method, inputs, kwargs, meta):
        """
        a, b, c in GF(2)
        c = a - b
          = a ^ b
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(np.bitwise_xor, method)(*inputs, **kwargs)
        if np.isscalar(output):
            output = meta["field"](output)
        return output

    def _ufunc_multiply(cls, ufunc, method, inputs, kwargs, meta):
        """
        In-field multiplication:
        a, b, c in GF(2)
        c = a * b
          = a & b

        Scalar multiplication:
        a, c in GF(2)
        b in Z
        c = a * b
          = a * (b % 2)
          = a * (b & 0b1)
          = a & b & 0b1
        """
        if len(meta["non_field_operands"]) == 0:
            # In-field multiplication
            output = getattr(np.bitwise_and, method)(*inputs, **kwargs)
        else:
            # Scalar multiplication
            inputs = cls._verify_and_flip_operands_first_field_second_int(ufunc, method, inputs, meta)
            inputs[meta["operands"][1]] = np.bitwise_and(inputs[meta["operands"][1]], 0b1, dtype=inputs[meta["operands"][0]].dtype, casting="unsafe")
            output = getattr(np.bitwise_and, method)(*inputs, **kwargs)
        if np.isscalar(output):
            output = meta["field"](output)
        return output

    def _ufunc_divide(cls, ufunc, method, inputs, kwargs, meta):
        """
        In-field multiplication:
        a, b, c in GF(2)
        c = a / b, b = 1 is the only valid element with a multiplicative inverse, which is 1
          = a & b
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        if np.count_nonzero(inputs[meta["operands"][-1]]) != inputs[meta["operands"][-1]].size:
            raise ZeroDivisionError("Cannot divide by 0 in Galois fields.")
        output = getattr(np.bitwise_and, method)(*inputs, **kwargs)
        if np.isscalar(output):
            output = meta["field"](output)
        return output

    def _ufunc_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, b in GF(2)
        b = -a
          = a
        """
        return inputs[0]

    def _ufunc_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, b in GF(2)
        b = 1 / a, a = 1 is the only valid element with a multiplicative inverse, which is 1
          = a
        """
        if np.count_nonzero(inputs[0]) != inputs[0].size:
            raise ZeroDivisionError("Cannot divide by 0 in Galois fields.")
        return inputs[0]

    def _ufunc_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, c in GF(2)
        c = a ** 2
          = a * a
          = a
        """
        return inputs[0]

    def _ufunc_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a in GF(2)
        b in Z
        b = log_α(a), a = 1 is the only valid element with a logarithm base α, which is 0
          = 0
        """
        if np.count_nonzero(inputs[meta["operands"][0]]) != inputs[meta["operands"][0]].size:
            raise ArithmeticError("Cannot take the logarithm of 0 in Galois fields.")
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        inputs = list(inputs) + [np.int(0)]
        output = getattr(np.bitwise_and, method)(*inputs, **kwargs)
        return output


###############################################################################
# Galois field arithmetic, explicitly calculated without lookup tables
###############################################################################

def _power_calculate(a, power):  # pragma: no cover
    # NOTE: The a == 0 and b < 0 condition will be caught outside of the the ufunc and raise ZeroDivisonError
    if power == 0:
        return 1
    elif a == 0:
        return 0
    else:
        return a


def _poly_eval_calculate(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = coeffs[j] ^ (results[i] & values[i])
