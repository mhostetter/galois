import numba
import numpy as np

from .linalg import matmul

# Placeholder globals that will be set in _compile_jit_lookup()
CHARACTERISTIC = None  # The field's prime characteristic `p`
ORDER = None  # The field's order `p^m`

EXP = []  # EXP[i] = α^i
LOG = []  # LOG[i] = x, such that α^x = i
ZECH_LOG = []  # ZECH_LOG[i] = log(1 + α^i)
ZECH_E = None  # α^ZECH_E = -1, ZECH_LOG[ZECH_E] = -Inf

ADD_JIT = lambda x, y: x + y
MULTIPLY_JIT = lambda x, y: x * y


class UfuncMixin(type):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._EXP = None
        cls._LOG = None
        cls._ZECH_LOG = None
        cls._ZECH_E = None
        cls._ufuncs = {}

        # Integer representations of the field's primitive element and primitive polynomial to be used in the
        # pure python ufunc implementations for `ufunc_mode = "python-calculate"`
        cls._primitive_element_int = None
        cls._irreducible_poly_int = None

    def compile(cls, mode, target="cpu"):
        """
        Recompile the just-in-time compiled numba ufuncs with a new calculation mode or target.

        Parameters
        ----------
        mode : str
            The method of field computation, either `"jit-lookup"`, `"jit-calculate"`, `"python-calculate"`. The "jit-lookup" mode will
            use Zech log, log, and anti-log lookup tables for speed. The "jit-calculate" mode will not store any lookup tables, but perform field
            arithmetic on the fly. The "jit-calculate" mode is designed for large fields that cannot store lookup tables in RAM.
            Generally, "jit-calculate" is slower than "jit-lookup". The "python-calculate" mode is reserved for extremely large fields. In
            this mode the ufuncs are not JIT-compiled, but are pur python functions operating on python ints. The list of valid
            modes for this field is in :obj:`galois.GFMeta.ufunc_modes`.
        target : str, optional
            The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. The default
            is `"cpu"`. For extremely large fields the only supported target is `"cpu"` (which doesn't use numba it uses pure python to
            calculate the field arithmetic). The list of valid targets for this field is in :obj:`galois.GFMeta.ufunc_targets`.
        """
        mode = cls.default_ufunc_mode if mode == "auto" else mode
        if mode not in cls.ufunc_modes:
            raise ValueError(f"Argument `mode` must be in {cls.ufunc_modes} for {cls.name}, not {mode}.")
        if target not in cls.ufunc_targets:
            raise ValueError(f"Argument `target` must be in {cls.ufunc_targets} for {cls.name}, not {target}.")

        if mode == cls.ufunc_mode and target == cls.ufunc_target:
            # Don't need to rebuild these ufuncs
            return

        cls._ufunc_mode = mode
        cls._ufunc_target = target

        if cls.ufunc_mode == "jit-lookup":
            cls._compile_jit_lookup(target)
        elif cls.ufunc_mode == "jit-calculate":
            cls._compile_jit_calculate(target)
        elif cls.ufunc_mode == "python-calculate":
            cls._compile_python_calculate()
        else:
            raise RuntimeError(f"Attribute `ufunc_mode` was not processed, {cls._ufunc_mode}. Please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

    def _build_lookup_tables(cls):
        order = cls.order
        primitive_element = cls._primitive_element_int
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

    def _compile_jit_lookup(cls, target):
        """
        A method to JIT-compile the standard lookup arithmetic for any field. The functions that are
        JIT compiled are at the bottom of this file.
        """
        global CHARACTERISTIC, ORDER, EXP, LOG, ZECH_LOG, ZECH_E, ADD_JIT, MULTIPLY_JIT

        # Build the lookup tables if they don't exist
        if cls._EXP is None:
            cls._build_lookup_tables()

        # Export lookup tables to global variables so JIT compiling can cache the tables in the binaries
        CHARACTERISTIC = cls.characteristic
        ORDER = cls.order
        EXP = cls._EXP
        LOG = cls._LOG
        ZECH_LOG = cls._ZECH_LOG
        ZECH_E = cls._ZECH_E

        # JIT-compile add and multiply routines for reference in other routines
        ADD_JIT = numba.jit("int64(int64, int64)", nopython=True)(_add_lookup)
        MULTIPLY_JIT = numba.jit("int64(int64, int64)", nopython=True)(_multiply_lookup)

        kwargs = {"nopython": True, "target": target}
        if target == "cuda":
            kwargs.pop("nopython")

        # TODO: Use smallest possible dtype for ufuncs
        # d = np.dtype(self.dtypes[0]).name

        # Create numba JIT-compiled ufuncs using the *current* EXP, LOG, and MUL_INV lookup tables
        cls._ufuncs["add"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_add_lookup)
        cls._ufuncs["subtract"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_subtract_lookup)
        cls._ufuncs["multiply"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_multiply_lookup)
        cls._ufuncs["divide"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_divide_lookup)
        cls._ufuncs["negative"] = numba.vectorize(["int64(int64)"], **kwargs)(_additive_inverse_lookup)
        cls._ufuncs["reciprocal"] = numba.vectorize(["int64(int64)"], **kwargs)(_multiplicative_inverse_lookup)
        cls._ufuncs["power"] = numba.vectorize(["int64(int64, int64)"], **kwargs)(_power_lookup)
        cls._ufuncs["log"] = numba.vectorize(["int64(int64)"], **kwargs)(_log_lookup)
        cls._ufuncs["poly_eval"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_eval_lookup)

    def _compile_jit_calculate(cls, target):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta. The functions that will
        be JIT-compiled will be located at the bottom of those files.
        """
        raise NotImplementedError

    def _compile_python_calculate(cls):
        cls._ufuncs["add"] = np.frompyfunc(cls._add_python, 2, 1)
        cls._ufuncs["subtract"] = np.frompyfunc(cls._subtract_python, 2, 1)
        cls._ufuncs["multiply"] = np.frompyfunc(cls._multiply_python, 2, 1)
        cls._ufuncs["divide"] = np.frompyfunc(cls._divide_python, 2, 1)
        cls._ufuncs["negative"] = np.frompyfunc(cls._additive_inverse_python, 1, 1)
        cls._ufuncs["reciprocal"] = np.frompyfunc(cls._multiplicative_inverse_python, 1, 1)
        cls._ufuncs["power"] = np.frompyfunc(cls._power_python, 2, 1)
        cls._ufuncs["log"] = np.frompyfunc(cls._log_python, 1, 1)
        cls._ufuncs["poly_eval"] = np.vectorize(cls._poly_eval_python, excluded=["coeffs"], otypes=[np.object_])

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _verify_operands_in_same_field(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(f"Operation '{ufunc.__name__}' requires both operands to be Galois field arrays over the same field, not {[inputs[i] for i in meta['operands']]}.")

    def _verify_operands_in_field_or_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if meta["field"].dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{inputs[i].dtype}'.")
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{inputs[i].dtype}'.")
            else:
                raise TypeError(f"Operation '{ufunc.__name__}' requires operands that are not Galois field arrays to be an integers or integer np.ndarrays, not {type(inputs[i])}.")

    def _verify_operands_first_field_second_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(f"Operation '{ufunc.__name__}' requires the first operand to be a Galois field array, not {meta['types'][meta['operands'][0]]}.")
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(f"Operation '{ufunc.__name__}' requires the second operand to be an integer array, not {meta['types'][meta['operands'][1]]}.")

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return
        # elif type(second) is np.ndarray:
        #     if not np.issubdtype(second.dtype, np.integer):
        #         raise ValueError(f"Operation '{ufunc.__name__}' requires the second operand with type np.ndarray to have integer dtype, not '{second.dtype}'.")
        elif isinstance(second, np.ndarray):
            if meta["field"].dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{second.dtype}'.")
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(f"Operation '{ufunc.__name__}' requires operands with type np.ndarray to have integer dtype, not '{second.dtype}'.")
        else:
            raise TypeError(f"Operation '{ufunc.__name__}' requires the second operand to be an integer or integer np.ndarray, not {type(second)}.")

    def _view_inputs_as_ndarray(cls, inputs, kwargs, dtype=None):  # pylint: disable=no-self-use
        # View all inputs that are Galois field arrays as np.ndarray to avoid infinite recursion
        v_inputs = list(inputs)
        for i in range(len(inputs)):
            if issubclass(type(inputs[i]), cls):
                v_inputs[i] = inputs[i].view(np.ndarray) if dtype is None else inputs[i].view(np.ndarray).astype(dtype)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if issubclass(type(output), cls):
                    o = output.view(np.ndarray) if dtype is None else output.view(np.ndarray).astype(dtype)
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _view_output_as_field(cls, output, field, dtype):  # pylint: disable=no-self-use
        if isinstance(type(output), field):
            return output
        elif isinstance(output, np.ndarray):
            return output.astype(dtype).view(field)
        elif output is None:
            return None
        else:
            return field(output, dtype=dtype)

    def _ufunc_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["add"], method)(*inputs, **kwargs)
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

    def _ufunc_divide(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["divide"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        output = getattr(cls._ufuncs["negative"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        output = getattr(cls._ufuncs["reciprocal"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        output = getattr(cls._ufuncs["power"], method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        output = getattr(cls._ufuncs["power"], method)(*inputs, 2, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        output = getattr(cls._ufuncs["log"], method)(*inputs, **kwargs)
        return output

    def _ufunc_matmul(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument,no-self-use
        assert method == "__call__"
        return matmul(*inputs, **kwargs)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
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

    def _divide_python(cls, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            b_inv = cls._multiplicative_inverse_python(b)
            return cls._multiply_python(a, b_inv)

    def _additive_inverse_python(cls, a):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

    def _multiplicative_inverse_python(cls, a):
        """
        To be implemented in GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta.
        """
        raise NotImplementedError

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
            a = cls._multiplicative_inverse_python(a)
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
            result = cls._multiply_python(result, cls._primitive_element_int)

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

    def _poly_eval_python(cls, coeffs, values):
        result = coeffs[0]
        for j in range(1, coeffs.size):
            p = cls._multiply_python(result, values)
            result = cls._add_python(coeffs[j], p)
        return result


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


def _additive_inverse_lookup(a):  # pragma: no cover
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


def _multiplicative_inverse_lookup(a):  # pragma: no cover
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


def _poly_eval_lookup(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
