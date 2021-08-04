import numba
import numpy as np

from ._meta_properties import PropertiesMeta

_FUNCTION_TYPE = {
    "add": "binary",
    "negative": "unary",
    "subtract": "binary",
    "multiply": "binary",
    "reciprocal": "unary",
    "divide": "binary",
    "power": "binary",
    "log": "binary",
}

# List of ufuncs that are not valid on arrays over finite groups, rings, and fields
UNSUPPORTED_UFUNCS_UNARY = [
    np.invert, np.sqrt,
    np.log2, np.log10,
    np.exp, np.expm1, np.exp2,
    np.sin, np.cos, np.tan,
    np.sinh, np.cosh, np.tanh,
    np.arcsin, np.arccos, np.arctan,
    np.arcsinh, np.arccosh, np.arctanh,
    np.degrees, np.radians,
    np.deg2rad, np.rad2deg,
    np.floor, np.ceil, np.trunc, np.rint,
]

UNSUPPORTED_UFUNCS_BINARY = [
    np.hypot, np.arctan2,
    np.logaddexp, np.logaddexp2,
    np.fmod, np.modf,
    np.fmin, np.fmax,
]

UFUNCS_REQUIRING_VIEW = [
    np.bitwise_and, np.bitwise_or, np.bitwise_xor,
    np.left_shift, np.right_shift,
    np.positive,
]


class UfuncMeta(PropertiesMeta):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    _unsupported_ufuncs = UNSUPPORTED_UFUNCS_UNARY + UNSUPPORTED_UFUNCS_BINARY
    _ufuncs_requiring_view = UFUNCS_REQUIRING_VIEW

    _overridden_ufuncs = {
        np.add: "_ufunc_routine_add",
        np.negative: "_ufunc_routine_negative",
        np.subtract: "_ufunc_routine_subtract",
        np.multiply: "_ufunc_routine_multiply",
        np.reciprocal: "_ufunc_routine_reciprocal",
        np.floor_divide: "_ufunc_routine_divide",
        np.true_divide: "_ufunc_routine_divide",
        np.divmod: "_ufunc_routine_divmod",
        np.remainder: "_ufunc_routine_remainder",
        np.power: "_ufunc_routine_power",
        np.square: "_ufunc_routine_square",
        np.log: "_ufunc_routine_log",
        np.matmul: "_ufunc_routine_matmul",
    }

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._EXP = np.array([], dtype=np.int64)
        cls._LOG = np.array([], dtype=np.int64)
        cls._ZECH_LOG = np.array([], dtype=np.int64)
        cls._ZECH_E = 0

        cls._ufuncs = {}

    def _compile_ufuncs(cls):
        cls._ufuncs = {}  # Reset the dictionary so each ufunc will get recompiled

        if cls.ufunc_mode == "jit-lookup":
            cls._build_lookup_tables()

    def _build_lookup_tables(cls):
        # TODO: JIT compile this

        if cls._EXP.size > 0:
            return

        order = cls.order
        primitive_element = int(cls.primitive_element)

        cls._EXP = np.zeros(2*order, dtype=np.int64)
        cls._LOG = np.zeros(order, dtype=np.int64)
        cls._ZECH_LOG = np.zeros(order, dtype=np.int64)
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
    # Individual JIT arithmetic functions, pre-compiled (cached)
    ###############################################################################

    def _lookup_jit(cls, name):  # pylint: disable=no-self-use
        return compile_jit(name)

    def _calculate_jit(cls, name):
        raise NotImplementedError

    def _python_func(cls, name):
        raise NotImplementedError

    ###############################################################################
    # Individual ufuncs, compiled on-demand
    ###############################################################################

    def _ufunc(cls, name):
        if name not in cls._ufuncs:
            if cls.ufunc_mode == "jit-lookup":
                cls._ufuncs[name] = cls._lookup_ufunc(name)
            elif cls.ufunc_mode == "jit-calculate":
                cls._ufuncs[name] = cls._calculate_ufunc(name)
            else:
                cls._ufuncs[name] = cls._python_ufunc(name)
        return cls._ufuncs[name]

    def _lookup_ufunc(cls, name):
        return compile_ufunc(name, cls.characteristic, cls.degree, cls._irreducible_poly_int, cls._EXP, cls._LOG, cls._ZECH_LOG, cls._ZECH_E)

    def _calculate_ufunc(cls, name):
        raise NotImplementedError

    def _python_ufunc(cls, name):  # pylint: disable=no-self-use
        function = eval(f"cls._{name}_python")
        if _FUNCTION_TYPE[name] == "unary":
            return np.frompyfunc(function, 1, 1)
        else:
            return np.frompyfunc(function, 2, 1)

    ###############################################################################
    # Input/output conversion functions
    ###############################################################################

    def _verify_unary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat", "outer"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Reduction methods are only supported on binary functions.")

    def _verify_binary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r} because it takes inputs with type {cls.name} array and integer array. Different types do not support reduction.")

    def _verify_method_only_call(cls, ufunc, method):  # pylint: disable=no-self-use
        if not method == "__call__":
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Only '__call__' is supported.")

    def _verify_operands_in_same_field(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(f"Operation {ufunc.__name__!r} requires both operands to be {cls.name} arrays, not {[inputs[i] for i in meta['operands']]}.")

    def _verify_operands_in_field_or_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if meta["field"].dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
            else:
                raise TypeError(f"Operation {ufunc.__name__!r} requires operands that are not {cls.name} arrays to be integers or an integer np.ndarray, not {type(inputs[i])}.")

    def _verify_operands_first_field_second_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the first operand to be a {cls.name} array, not {meta['types'][meta['operands'][0]]}.")
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer array, not {meta['types'][meta['operands'][1]]}.")

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return
        # elif type(second) is np.ndarray:
        #     if not np.issubdtype(second.dtype, np.integer):
        #         raise ValueError(f"Operation {ufunc.__name__!r} requires the second operand with type np.ndarray to have integer dtype, not {second.dtype}.")
        elif isinstance(second, np.ndarray):
            if meta["field"].dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
        else:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer or integer np.ndarray, not {type(second)}.")

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

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    def _ufunc_routine_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("add"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("negative"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_subtract(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("subtract"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_multiply(cls, ufunc, method, inputs, kwargs, meta):
        if len(meta["non_field_operands"]) > 0:
            # Scalar multiplication
            cls._verify_operands_in_field_or_int(ufunc, inputs, meta)
            inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
            inputs[meta["non_field_operands"][0]] = np.mod(inputs[meta["non_field_operands"][0]], cls.characteristic)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("multiply"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("reciprocal"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_divide(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("divide"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_divmod(cls, ufunc, method, inputs, kwargs, meta):
        q = cls._ufunc_routine_divide(ufunc, method, inputs, kwargs, meta)
        r = cls.Zeros(q.shape)
        output = q, r
        return output

    def _ufunc_routine_remainder(cls, ufunc, method, inputs, kwargs, meta):
        # Perform dummy addition operation to get shape of output zeros
        x = cls._ufunc_routine_add(ufunc, method, inputs, kwargs, meta)
        output = cls.Zeros(x.shape)
        return output

    def _ufunc_routine_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_binary_method_not_reduction(ufunc, method)
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("power"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs = list(inputs) + [2]
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("power"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        inputs = list(inputs) + [int(cls.primitive_element)]
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("log"), method)(*inputs, **kwargs)
        return output

    def _ufunc_routine_matmul(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        return cls._matmul(*inputs, **kwargs)

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _add_python(cls, a, b):
        raise NotImplementedError

    def _negative_python(cls, a):
        raise NotImplementedError

    def _subtract_python(cls, a, b):
        raise NotImplementedError

    def _multiply_python(cls, a, b):
        raise NotImplementedError

    def _reciprocal_python(cls, a):
        raise NotImplementedError

    def _divide_python(cls, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            b_inv = cls._reciprocal_python(b)
            return cls._multiply_python(a, b_inv)

    def _power_python(cls, a, b):
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif b < 0:
            a = cls._reciprocal_python(a)
            b = abs(b)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                result_s = cls._multiply_python(result_s, result_s)
                b //= 2
            else:
                result_m = cls._multiply_python(result_m, result_s)
                b -= 1

        result = cls._multiply_python(result_m, result_s)

        return result

    def _log_python(cls, a, b):
        """
        TODO: Replace this with a more efficient algorithm
        """
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        # Naive algorithm
        result = 1
        for i in range(0, cls.order - 1):
            if result == a:
                break
            result = cls._multiply_python(result, b)

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


###############################################################################
# Compile functions
###############################################################################

EXP = []  # EXP[i] = α^i
LOG = []  # LOG[i] = x, such that α^x = i
ZECH_LOG = []  # ZECH_LOG[i] = log(1 + α^i)
ZECH_E = 0  # α^ZECH_E = -1, ZECH_LOG[ZECH_E] = -Inf

# pylint: disable=redefined-outer-name,unused-argument


def compile_jit(name):
    """
    Compile a JIT arithmetic function. These can be cached.
    """
    if name not in compile_jit.cache:
        function = eval(f"{name}")
        if _FUNCTION_TYPE[name] == "unary":
            compile_jit.cache[name] = numba.jit("int64(int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)
        else:
            compile_jit.cache[name] = numba.jit("int64(int64, int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)

    return compile_jit.cache[name]

compile_jit.cache = {}


def compile_ufunc(name, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY, EXP_, LOG_, ZECH_LOG_, ZECH_E_):
    """
    Compile an arithmetic ufunc. These cannot be cached as the lookup tables are compiled into the binary.
    """
    key = (name, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
    if key not in compile_ufunc.cache:
        global EXP, LOG, ZECH_LOG, ZECH_E
        EXP = EXP_
        LOG = LOG_
        ZECH_LOG = ZECH_LOG_
        ZECH_E = ZECH_E_

        function = eval(f"{name}_ufunc")
        if _FUNCTION_TYPE[name] == "unary":
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64)"], nopython=True)(function)
        else:
            compile_ufunc.cache[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(function)

    return compile_ufunc.cache[key]

compile_ufunc.cache = {}


###############################################################################
# Arithmetic using lookup tables
###############################################################################

@numba.extending.register_jitable(inline="always")
def add(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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


def add_ufunc(a, b):  # pragma: no cover
    return add(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def negative(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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


def negative_ufunc(a):  # pragma: no cover
    return negative(a, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def subtract(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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
    ORDER = LOG.size

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


def subtract_ufunc(a, b):  # pragma: no cover
    return subtract(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def multiply(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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


def multiply_ufunc(a, b):  # pragma: no cover
    return multiply(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def reciprocal(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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

    ORDER = LOG.size
    m = LOG[a]
    return EXP[(ORDER - 1) - m]


def reciprocal_ufunc(a):  # pragma: no cover
    return reciprocal(a, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def divide(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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
        ORDER = LOG.size
        m = LOG[a]
        n = LOG[b]
        return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative


def divide_ufunc(a, b):  # pragma: no cover
    return divide(a, b, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def power(a, b_int, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
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
        ORDER = LOG.size
        m = LOG[a]
        return EXP[(m * b_int) % (ORDER - 1)]


def power_ufunc(a, b_int):  # pragma: no cover
    return power(a, b_int, EXP, LOG, ZECH_LOG, ZECH_E)


@numba.extending.register_jitable(inline="always")
def log(beta, alpha, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
    """
    a in GF(p^m)
    α is a primitive element of GF(p^m), such that GF(p^m) = {0, 1, α^1, ..., α^(p^m - 2)}

    log(beta, α) = log(α^m, α)
                 = m
    """
    if beta == 0:
        raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

    return LOG[beta]


def log_ufunc(beta, alpha):  # pragma: no cover
    return log(beta, alpha, EXP, LOG, ZECH_LOG, ZECH_E)
