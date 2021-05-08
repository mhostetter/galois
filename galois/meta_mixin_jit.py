import numba
import numpy as np

# Placeholder functions to be replaced by JIT-compiled function
ADD_JIT = lambda x, y: x + y
SUBTRACT_JIT = lambda x, y: x - y
MULTIPLY_JIT = lambda x, y: x * y
DIVIDE_JIT = lambda x, y: x // y


class JITMixin(type):
    """
    A mixin class that JIT compiles general purpose functions for polynomial arithmetic and convolution.
    """
    # pylint: disable=no-value-for-parameter

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._ADD_JIT = None
        cls._SUBTRACT_JIT = None
        cls._MULTIPLY_JIT = None
        cls._DIVIDE_JIT = None

        cls._funcs = {}

    def _compile_special_functions(cls, target):
        global ADD_JIT, SUBTRACT_JIT, MULTIPLY_JIT, DIVIDE_JIT

        if cls.ufunc_mode == "python-calculate":
            # NOTE: Don't need to vectorize cls._poly_multiply or cls._poly_divmod
            cls._funcs["poly_evaluate"] = np.vectorize(cls._poly_evaluate_python, excluded=["coeffs"], otypes=[np.object_])

        else:
            kwargs = {"nopython": True}
            if target == "cuda":
                kwargs.pop("nopython")

            # Create numba JIT-compiled functions using the already JIT-compiled basic arithmetic operators
            ADD_JIT = cls._ADD_JIT
            SUBTRACT_JIT = cls._SUBTRACT_JIT
            MULTIPLY_JIT = cls._MULTIPLY_JIT
            DIVIDE_JIT = cls._DIVIDE_JIT
            cls._funcs["poly_multiply"] = numba.jit("int64[:](int64[:], int64[:])", **kwargs)(_poly_multiply_calculate)
            cls._funcs["poly_divmod"] = numba.jit("int64[:](int64[:], int64[:])", **kwargs)(_poly_divmod_calculate)
            cls._funcs["poly_evaluate"] = numba.guvectorize([(numba.int64[:], numba.int64[:], numba.int64[:])], "(n),(m)->(m)", **kwargs)(_poly_evaluate_calculate)

    # def _convolve(cls, a, b):
    #     if type(a) is not type(b):
    #         raise TypeError(f"Convolve requires both operands to be Galois field arrays over the same field, not {type(a)} and {type(b)}.")
    #     field = type(a)
    #     dtype = a.dtype
    #     if cls.ufunc_mode == "python-calculate":
    #         raise NotImplementedError
    #     else:
    #         c = cls._funcs["convolve"](a.astype(np.int64), b.astype(np.int64))
    #         c = c.astype(dtype).view(field)
    #     return c

    def _poly_multiply(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        field = type(a)
        dtype = a.dtype

        if cls.ufunc_mode == "python-calculate":
            c = cls._poly_multiply_python(a, b)
        else:
            c = cls._funcs["poly_multiply"](a.astype(np.int64), b.astype(np.int64))
            c = c.astype(dtype).view(field)

        return c

    def _poly_divmod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        field = type(a)
        dtype = a.dtype
        q_degree = a.size - b.size
        r_degree = b.size - 1

        if cls.ufunc_mode == "python-calculate":
            qr = cls._poly_divmod_python(a, b)
        else:
            qr = cls._funcs["poly_divmod"](a.astype(np.int64), b.astype(np.int64))
            qr = qr.astype(dtype).view(field)

        return qr[0:q_degree + 1], qr[q_degree + 1:q_degree + 1 + r_degree + 1]

    def _poly_evaluate(cls, coeffs, x):
        assert isinstance(coeffs, cls) and isinstance(x, cls)
        assert coeffs.ndim == 1
        field = cls
        dtype = x.dtype
        x = np.atleast_1d(x)

        if cls.ufunc_mode == "python-calculate":
            # For object dtypes, call the vectorized classmethod
            y = cls._funcs["poly_evaluate"](coeffs=coeffs.view(np.ndarray), values=x.view(np.ndarray))  # pylint: disable=not-callable
        else:
            # For integer dtypes, call the JIT-compiled gufunc
            y = cls._funcs["poly_evaluate"](coeffs, x, field.Zeros(x.shape), casting="unsafe")  # pylint: disable=not-callable
            y = y.astype(dtype)
        y = y.view(field)

        if y.size == 1:
            y = y[0]

        return y

    ###############################################################################
    # Pure python arithmetic methods
    ###############################################################################

    def _poly_multiply_python(cls, a, b):
        c = cls.Zeros(a.size + b.size - 1, dtype=a.dtype)

        # Want a to be the shorter sequence
        if b.size < a.size:
            a, b = b, a

        for i in range(a.size):
            c[i:i + b.size] += a[i] * b

        return c

    def _poly_divmod_python(cls, a, b):
        assert a.size >= b.size
        q_degree = a.size - b.size
        qr = cls(a)

        for i in range(0, q_degree + 1):
            if qr[i] > 0:
                q = qr[i] / b[0]
                qr[i:i + b.size] -= q*b
                qr[i] = q

        return qr

    def _poly_evaluate_python(cls, coeffs, values):
        result = coeffs[0]
        for j in range(1, coeffs.size):
            result = cls._add_python(coeffs[j], cls._multiply_python(result, values))
        return result


###############################################################################
# Galois field arithmetic, explicitly calculated without lookup tables
###############################################################################

def _poly_multiply_calculate(a, b):  # pragma: no cover
    c = np.zeros(a.size + b.size - 1, dtype=a.dtype)

    for i in range(a.size):
        for j in range(b.size - 1, -1, -1):
            c[i + j] = ADD_JIT(c[i + j], MULTIPLY_JIT(a[i], b[j]))

    return c


def _poly_divmod_calculate(a, b):  # pragma: no cover
    assert a.size >= b.size
    q_degree = a.size - b.size
    qr = np.copy(a)

    for i in range(0, q_degree + 1):
        if qr[i] > 0:
            q = DIVIDE_JIT(qr[i], b[0])
            for j in range(0, b.size):
                qr[i + j] = SUBTRACT_JIT(qr[i + j], MULTIPLY_JIT(q, b[j]))
            qr[i] = q

    return qr


def _poly_evaluate_calculate(coeffs, values, results):  # pragma: no cover
    for i in range(values.size):
        results[i] = coeffs[0]
        for j in range(1, coeffs.size):
            results[i] = ADD_JIT(coeffs[j], MULTIPLY_JIT(results[i], values[i]))
