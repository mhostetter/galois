import numba
import numpy as np

from ..meta_ufunc import Ufunc

CHARACTERISTIC = None  # The field's prime characteristic `p`
ORDER = None  # The field's order `p^m`

EXP = []  # EXP[i] = α^i
LOG = []  # LOG[i] = x, such that α^x = i
ZECH_LOG = []  # ZECH_LOG[i] = log(1 + α^i)
ZECH_E = None  # α^ZECH_E = -1, ZECH_LOG[ZECH_E] = -Inf


class FieldUfunc(Ufunc):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    _overridden_ufuncs = {
        np.add: "_ufunc_add",
        np.negative: "_ufunc_negative",
        np.subtract: "_ufunc_subtract",
        np.multiply: "_ufunc_multiply",
        np.reciprocal: "_ufunc_reciprocal",
        np.floor_divide: "_ufunc_divide",
        np.true_divide: "_ufunc_divide",
        np.power: "_ufunc_power",
        np.square: "_ufunc_square",
        np.log: "_ufunc_log",
        np.matmul: "_ufunc_matmul",
    }

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._EXP = None
        cls._LOG = None
        cls._ZECH_LOG = None
        cls._ZECH_E = None

    def _build_lookup_tables(cls):
        if cls._EXP is not None:
            return

        order = cls.order
        primitive_element = int(cls.primitive_element)
        dtype = np.int64
        if order > np.iinfo(dtype).max:
            raise RuntimeError(f"Cannot build lookup tables for {cls.name} since the elements cannot be represented with dtype {dtype}.")

        cls._EXP = np.zeros(2*order, dtype=dtype)
        cls._LOG = np.zeros(order, dtype=dtype)
        cls._ZECH_LOG = np.zeros(order, dtype=dtype)
        if cls.characteristic == 2:
            cls._ZECH_E = 0
        else:
            cls._ZECH_E = (cls.order - 1) // 2

        element = 1
        cls._EXP[0] = element
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element = cls._multiply_python(element, primitive_element)
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, order):
            one_plus_element = cls._add_python(1, cls._EXP[i])
            cls._ZECH_LOG[i] = cls._LOG[one_plus_element]

        if not cls._EXP[order - 1] == 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not cyclic with size {order - 1}, which means the primitive element {cls.primitive_element} does not have multiplicative order {order - 1} and therefore isn't a multiplicative generator for {cls.name}.")
        if not len(set(cls._EXP[0:order - 1])) == order - 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not unique, which means the primitive element {cls.primitive_element} has order less than {order - 1} and is not a multiplicative generator of {cls.name}.")
        if not len(set(cls._LOG[1:order])) == order - 1:
            raise RuntimeError(f"The log lookup table for {cls.name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[order:2*order] = cls._EXP[1:1 + order]

    ###############################################################################
    # Compile general-purpose lookup functions
    ###############################################################################

    def _compile_add_lookup(cls, target):
        global EXP, LOG, ZECH_LOG, ZECH_E
        EXP = cls._EXP
        LOG = cls._LOG
        ZECH_LOG = cls._ZECH_LOG
        ZECH_E = cls._ZECH_E
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_lookup)

    def _compile_negative_lookup(cls, target):
        global EXP, LOG, ZECH_E
        EXP = cls._EXP
        LOG = cls._LOG
        ZECH_E = cls._ZECH_E
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64)"], **kwargs)(_negative_lookup)

    def _compile_subtract_lookup(cls, target):
        global ORDER, EXP, LOG, ZECH_LOG, ZECH_E
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        ZECH_LOG = cls._ZECH_LOG
        ZECH_E = cls._ZECH_E
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_lookup)

    def _compile_multiply_lookup(cls, target):
        global EXP, LOG
        EXP = cls._EXP
        LOG = cls._LOG
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_lookup)

    def _compile_reciprocal_lookup(cls, target):
        global ORDER, EXP, LOG
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64)"], **kwargs)(_reciprocal_lookup)

    def _compile_divide_lookup(cls, target):
        global ORDER, EXP, LOG
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_lookup)

    def _compile_power_lookup(cls, target):
        global ORDER, EXP, LOG
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_lookup)

    def _compile_log_lookup(cls, target):
        global LOG
        LOG = cls._LOG
        kwargs = {"nopython": True, "target": target} if target != "cuda" else {"target": target}
        return numba.vectorize(["int64(int64)"], **kwargs)(_log_lookup)

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _ufunc_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["add"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        output = getattr(cls._ufuncs["negative"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_subtract(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["subtract"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_multiply(cls, ufunc, method, inputs, kwargs, meta):
        if len(meta["non_field_operands"]) > 0:
            # Scalar multiplication
            cls._verify_operands_in_field_or_int(ufunc, inputs, meta)
            inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
            inputs[meta["non_field_operands"][0]] = np.mod(inputs[meta["non_field_operands"][0]], cls.characteristic)
        output = getattr(cls._ufuncs["multiply"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        output = getattr(cls._ufuncs["reciprocal"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_divide(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["divide"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_binary_method_not_reduction(ufunc, method)
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["power"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        output = getattr(cls._ufuncs["power"], method)(*inputs, 2, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        output = getattr(cls._ufuncs["log"], method)(*inputs, **kwargs)
        return output

    def _ufunc_matmul(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        return cls._matmul(*inputs, **kwargs)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _negative_python(cls, a):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _subtract_python(cls, a, b):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _multiply_python(cls, a, b):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _reciprocal_python(cls, a):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _divide_python(cls, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            b_inv = cls._reciprocal_python(b)
            return cls._multiply_python(a, b_inv)

    def _power_python(cls, a, power):
        """
        Square and Multiply Algorithm

        a^13 = (1) * (a)^13
             = (a) * (a)^12
             = (a) * (a^2)^6
             = (a) * (a^4)^3
             = (a * a^4) * (a^4)^2
             = (a * a^4) * (a^8)
             = result_m * result_s
        """
        if a == 0 and power < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if power == 0:
            return 1
        elif power < 0:
            a = cls._reciprocal_python(a)
            power = abs(power)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while power > 1:
            if power % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                power //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                power -= 1

        result = cls._multiply_python(result_m, result_s)

        return result

    def _log_python(cls, beta):
        """
        TODO: Replace this with a more efficient algorithm

        α in GF(p^m) and is a multiplicative generator
        β in GF(p^m)

        ɣ = log_α(β), such that: α^ɣ = β
        """
        if beta == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        # Naive algorithm
        result = 1
        for i in range(0, cls.order - 1):
            if result == beta:
                break
            result = cls._multiply_python(result, int(cls.primitive_element))

        return i

        # N = cls.order
        # alpha = cls._primitive_element_int
        # n = N - 1  # Multiplicative order of the group
        # x, a, b = 1, 0, 0
        # X, A, B = x, a, b

        # def update(x, a, b):
        #     if x % 3 == 0:
        #         return (x*x) % N, (a*2) % n, (b*2) % n
        #     elif x % 3 == 1:
        #         return (x*alpha) % N, (a + 1) % n, b
        #     else:
        #         return (x*beta) % N, a, (b + 1) % n

        # for i in range(1, n):
        #     x, a, b = update(x, a, b)
        #     X, A, B = update(X, A, B)
        #     X, A, B = update(X, A, B)
        #     if x == X:
        #         break

        # return cls(a - A) / cls(B - b)


###################################################################################
# Galois field arithmetic for any field using EXP, LOG, and ZECH_LOG lookup tables
###################################################################################

def _add_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a + b = α^m + α^n
          = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
          = α^m * α^ZECH_LOG(n - m)
          = α^(m + ZECH_LOG(n - m))
    """
    # LOG[0] = -Inf, so catch these conditions
    if a == 0:
        return b
    elif b == 0:
        return a

    m = LOG[a]
    n = LOG[b]

    if m > n:
        # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    if n - m == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and α^(-Inf) = 0
        return 0
    else:
        return EXP[m + ZECH_LOG[n - m]]


def _negative_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    -a = -α^n
       = -1 * α^n
       = α^e * α^n
       = α^(e + n)
    """
    if a == 0:  # LOG[0] = -Inf, so catch this condition
        return 0
    else:
        n = LOG[a]
        return EXP[ZECH_E + n]


def _subtract_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a - b = α^m - α^n
          = α^m + (-α^n)
          = α^m + (-1 * α^n)
          = α^m + (α^e * α^n)
          = α^m + α^(e + n)
    """
    # Same as addition if n = LOG[b] + e
    m = LOG[a]
    n = LOG[b] + ZECH_E

    # LOG[0] = -Inf, so catch these conditions
    if b == 0:
        return a
    elif a == 0:
        return EXP[n]

    if m > n:
        # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If
        # m is larger than n, switch a and b in the addition.
        m, n = n, m

    z = n - m
    if z == ZECH_E:
        # ZECH_LOG[ZECH_E] = -Inf and α^(-Inf) = 0
        return 0
    if z >= ORDER - 1:
        # Reduce index of ZECH_LOG by the multiplicative order of the field, i.e. `order - 1`
        z -= ORDER - 1

    return EXP[m + ZECH_LOG[z]]


def _multiply_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a * b = α^m * α^n
          = α^(m + n)
    """
    if a == 0 or b == 0:  # LOG[0] = -Inf, so catch these conditions
        return 0
    else:
        m = LOG[a]
        n = LOG[b]
        return EXP[m + n]


def _reciprocal_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    1 / a = 1 / α^m
          = α^(-m)
          = 1 * α^(-m)
          = α^(ORDER - 1) * α^(-m)
          = α^(ORDER - 1 - m)
    """
    if a == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    m = LOG[a]
    return EXP[(ORDER - 1) - m]


def _divide_lookup(a, b):  # pragma: no cover
    """
    a in GF(p^m)
    b in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a / b = α^m / α^n
          = α^(m - n)
          = 1 * α^(m - n)
          = α^(ORDER - 1) * α^(m - n)
          = α^(ORDER - 1 + m - n)
    """
    if b == 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if a == 0:  # LOG[0] = -Inf, so catch this condition
        return 0
    else:
        m = LOG[a]
        n = LOG[b]
        return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative


def _power_lookup(a, b_int):  # pragma: no cover
    """
    a in GF(p^m)
    b_int in Z
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    a ** b_int = α^m ** b_int
               = α^(m * b_int)
               = α^(m * ((b_int // (ORDER - 1))*(ORDER - 1) + b_int % (ORDER - 1)))
               = α^(m * ((b_int // (ORDER - 1))*(ORDER - 1)) * α^(m * (b_int % (ORDER - 1)))
               = 1 * α^(m * (b_int % (ORDER - 1)))
               = α^(m * (b_int % (ORDER - 1)))
    """
    if a == 0 and b_int < 0:
        raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

    if b_int == 0:
        return 1
    elif a == 0:  # LOG[0] = -Inf, so catch this condition
        return 0
    else:
        m = LOG[a]
        return EXP[(m * b_int) % (ORDER - 1)]


def _log_lookup(a):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    log(a, α) = log(α^m, α)
              = m
    """
    if a == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

    return LOG[a]
