"""
A module that contains a metaclass mixin that provides NumPy function overriding for an ndarray subclass. Additionally, other
JIT functions are created for use in polynomials and error-correcting codes, such as _poly_evaluate() or _poly_divmod().
"""
import numba
from numba import int64, uint64
import numpy as np

from . import _linalg
from ._ufuncs import FieldUfunc


class FieldFunction(FieldUfunc):
    """
    A mixin class that JIT compiles general-purpose functions on Galois field arrays.
    """
    # pylint: disable=abstract-method,no-member

    _UNSUPPORTED_FUNCTIONS_UNARY = [
        np.packbits, np.unpackbits,
        np.unwrap,
        np.around, np.round_, np.fix,
        np.gradient, np.trapz,
        np.i0, np.sinc,
        np.angle, np.real, np.imag, np.conj, np.conjugate,
    ]

    _UNSUPPORTED_FUNCTIONS_BINARY = [
        np.lib.scimath.logn,
        np.cross,
    ]

    _UNSUPPORTED_FUNCTIONS = _UNSUPPORTED_FUNCTIONS_UNARY + _UNSUPPORTED_FUNCTIONS_BINARY

    _FUNCTIONS_REQUIRING_VIEW = [
        np.concatenate,
        np.broadcast_to,
        np.trace,
    ]

    _OVERRIDDEN_FUNCTIONS = {
        np.convolve: "_convolve",
        np.fft.fft: "_fft",
        np.fft.ifft: "_ifft",
    }

    _OVERRIDDEN_LINALG_FUNCTIONS = {
        np.dot: _linalg.dot,
        np.vdot: _linalg.vdot,
        np.inner: _linalg.inner,
        np.outer: _linalg.outer,
        # np.tensordot: _linalg."tensordot",
        np.linalg.det: _linalg.det,
        np.linalg.matrix_rank: _linalg.matrix_rank,
        np.linalg.solve: _linalg.solve,
        np.linalg.inv: _linalg.inv,
    }

    _FUNCTION_CACHE_CALCULATE = {}

    def __array_function__(self, func, types, args, kwargs):
        """
        Override the standard NumPy function calls with the new finite field functions.
        """
        field = type(self)

        if func in field._OVERRIDDEN_FUNCTIONS:
            output = getattr(field, field._OVERRIDDEN_FUNCTIONS[func])(*args, **kwargs)

        elif func in field._OVERRIDDEN_LINALG_FUNCTIONS:
            output = field._OVERRIDDEN_LINALG_FUNCTIONS[func](*args, **kwargs)

        elif func in field._UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(f"The NumPy function {func.__name__!r} is not supported on FieldArray. If you believe this function should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\nIf you'd like to perform this operation on the data, you should first call `array = array.view(np.ndarray)` and then call the function.")

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._verify_array_like_types_and_values(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in field._FUNCTIONS_REQUIRING_VIEW:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

        return output

    def dot(self, b, out=None):
        # The `np.dot(a, b)` ufunc is also available as `a.dot(b)`. Need to override this method for consistent results.
        return _linalg.dot(self, b, out=out)

    ###############################################################################
    # Individual functions, pre-compiled (cached)
    ###############################################################################

    @classmethod
    def _function(cls, name):
        """
        Returns the function for the specific routine. The function compilation is based on `ufunc_mode`.
        """
        if name not in cls._functions:
            if cls._ufunc_mode != "python-calculate":
                cls._functions[name] = cls._function_calculate(name)
            else:
                cls._functions[name] = cls._function_python(name)
        return cls._functions[name]

    @classmethod
    def _function_calculate(cls, name):
        """
        Returns a JIT-compiled function using explicit calculation. These functions are once-compiled and shared for all
        Galois fields. The only difference between Galois fields are the arithmetic funcs, characteristic, degree, and
        irreducible polynomial that are passed in as inputs.
        """
        key = (name,)

        if key not in cls._FUNCTION_CACHE_CALCULATE:
            function = getattr(cls, f"_{name}_calculate")
            sig = getattr(cls, f"_{name.upper()}_CALCULATE_SIG")
            cls._FUNCTION_CACHE_CALCULATE[key] = numba.jit(sig.signature, nopython=True, cache=True)(function)

        return cls._FUNCTION_CACHE_CALCULATE[key]

    @classmethod
    def _function_python(cls, name):
        """
        Returns a pure-Python function using explicit calculation.
        """
        return getattr(cls, f"_{name}_calculate")

    ###############################################################################
    # Function routines
    ###############################################################################

    @classmethod
    def _fft(cls, x, n=None, axis=-1, norm=None, forward=True, scaled=True):
        if not axis == -1:
            raise ValueError("The FFT is only implemented on 1-D arrays.")
        if not norm in [None, "backward"]:
            raise ValueError("The FFT normalization is only applied to the backward transform.")

        field = cls
        dtype = x.dtype

        if n is None:
            n = x.size
        x = np.append(x, np.zeros(n - x.size, dtype=x.dtype))

        omega = field.primitive_root_of_unity(x.size)
        if not forward:
            omega = omega ** -1

        if cls._ufunc_mode != "python-calculate":
            x = x.astype(np.int64)
            omega = np.int64(omega)
            add = cls._func_calculate("add")
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            y = cls._function("dft_jit")(x, omega, add, subtract, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            y = y.astype(dtype)
        else:
            x = x.view(np.ndarray)
            omega = int(omega)
            add = cls._func_python("add")
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            y = cls._function("dft_python")(x, omega, add, subtract, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        y = field._view(y)

        # Scale the inverse NTT such that x = INTT(NTT(x))
        if not forward and scaled:
            y /= field(n % field.characteristic)

        return y

    @classmethod
    def _ifft(cls, x, n=None, axis=-1, norm=None, scaled=True):
        return cls._fft(x, n=n, axis=axis, norm=norm, forward=False, scaled=scaled)

    @classmethod
    def _matmul(cls, A, B, out=None, **kwargs):  # pylint: disable=unused-argument
        if not type(A) is type(B):
            raise TypeError(f"Operation 'matmul' requires both arrays be in the same Galois field, not {type(A)} and {type(B)}.")
        if not (A.ndim >= 1 and B.ndim >= 1):
            raise ValueError(f"Operation 'matmul' requires both arrays have dimension at least 1, not {A.ndim}-D and {B.ndim}-D.")
        if not (A.ndim <= 2 and B.ndim <= 2):
            raise ValueError("Operation 'matmul' currently only supports matrix multiplication up to 2-D. If you would like matrix multiplication of N-D arrays, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")
        field = type(A)
        dtype = A.dtype

        if field.is_prime_field:
            return _linalg._lapack_linalg(A, B, np.matmul, out=out)

        prepend, append = False, False
        if A.ndim == 1:
            A = A.reshape((1,A.size))
            prepend = True
        if B.ndim == 1:
            B = B.reshape((B.size,1))
            append = True

        if not A.shape[-1] == B.shape[-2]:
            raise ValueError(f"Operation 'matmul' requires the last dimension of A to match the second-to-last dimension of B, not {A.shape} and {B.shape}.")

        # if A.ndim > 2 and B.ndim == 2:
        #     new_shape = list(A.shape[:-2]) + list(B.shape)
        #     B = np.broadcast_to(B, new_shape)
        # if B.ndim > 2 and A.ndim == 2:
        #     new_shape = list(B.shape[:-2]) + list(A.shape)
        #     A = np.broadcast_to(A, new_shape)

        if cls._ufunc_mode != "python-calculate":
            A = A.astype(np.int64)
            B = B.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            C = cls._function("matmul")(A, B, add, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            C = C.astype(dtype)
        else:
            A = A.view(np.ndarray)
            B = B.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            C = cls._function("matmul")(A, B, add, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        C = field._view(C)

        shape = list(C.shape)
        if prepend:
            shape = shape[1:]
        if append:
            shape = shape[:-1]
        C = C.reshape(shape)

        # TODO: Determine a better way to do this
        if out is not None:
            assert isinstance(out, tuple) and len(out) == 1  # TODO: Why is `out` getting populated as tuple?
            out = out[0]
            out[:] = C[:]

        return C

    @classmethod
    def _convolve(cls, a, b, mode="full"):
        if not type(a) is type(b):
            raise TypeError(f"Arguments `a` and `b` must be of the same FieldArray subclass, not {type(a)} and {type(b)}.")
        if not mode == "full":
            raise ValueError(f"Operation 'convolve' currently only supports mode of 'full', not {mode!r}.")
        field = type(a)
        dtype = a.dtype

        if cls._ufunc_mode == "python-calculate":
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            c = cls._function("convolve")(a, b, add, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        elif field.is_prime_field:
            # Determine the minimum dtype to hold the entire product and summation without overflowing
            n_sum = min(a.size, b.size)
            max_value = n_sum * (field.characteristic - 1)**2
            dtypes = [dtype for dtype in cls._dtypes if np.iinfo(dtype).max >= max_value]
            dtype = np.object_ if len(dtypes) == 0 else dtypes[0]
            return_dtype = a.dtype
            a = a.view(np.ndarray).astype(dtype)
            b = b.view(np.ndarray).astype(dtype)
            c = np.convolve(a, b)  # Compute result using native numpy LAPACK/BLAS implementation
            c = c % field.characteristic  # Reduce the result mod p
            c = field._view(c.astype(return_dtype)) if not np.isscalar(c) else field(c, dtype=return_dtype)
        else:
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            c = cls._function("convolve")(a, b, add, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            c = c.astype(dtype)
        c = field._view(c)

        return c

    @classmethod
    def _poly_evaluate(cls, coeffs, x):
        field = cls
        dtype = x.dtype
        shape = x.shape
        x = np.atleast_1d(x.flatten())

        if cls._ufunc_mode != "python-calculate":
            coeffs = coeffs.astype(np.int64)
            x = x.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            results = cls._function("poly_evaluate")(coeffs, x, add, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            results = results.astype(dtype)
        else:
            coeffs = coeffs.view(np.ndarray)
            x = x.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            results = cls._function("poly_evaluate")(coeffs, x, add, multiply, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        results = field._view(results)
        results = results.reshape(shape)

        return results

    @classmethod
    def _poly_evaluate_matrix(cls, coeffs, X):
        field = cls
        assert X.ndim == 2 and X.shape[0] == X.shape[1]
        I = field.Identity(X.shape[0])

        results = coeffs[0]*I
        for j in range(1, coeffs.size):
            results = coeffs[j]*I + results @ X

        return results

    @classmethod
    def _poly_divmod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert 1 <= a.ndim <= 2 and b.ndim == 1
        field = type(a)
        dtype = a.dtype
        a_1d = a.ndim == 1
        a = np.atleast_2d(a)

        q_degree = a.shape[-1] - b.shape[-1]
        r_degree = b.shape[-1] - 1

        if cls._ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            qr = qr.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            qr = cls._function("poly_divmod")(a, b, subtract, multiply, divide, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        qr = field._view(qr)

        q = qr[:, 0:q_degree + 1]
        r = qr[:, q_degree + 1:q_degree + 1 + r_degree + 1]

        if a_1d:
            q = q.reshape(q.size)
            r = r.reshape(r.size)

        return q, r

    @classmethod
    def _poly_floordiv(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert a.ndim == 1 and b.ndim == 1
        field = type(a)
        dtype = a.dtype

        if cls._ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            q = cls._function("poly_floordiv")(a, b, subtract, multiply, divide, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            q = q.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            q = cls._function("poly_floordiv")(a, b, subtract, multiply, divide, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        q = field._view(q)

        return q

    @classmethod
    def _poly_mod(cls, a, b):
        assert isinstance(a, cls) and isinstance(b, cls)
        assert a.ndim == 1 and b.ndim == 1
        field = type(a)
        dtype = a.dtype

        if cls._ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            r = cls._function("poly_mod")(a, b, subtract, multiply, divide, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            r = r.astype(dtype)
        else:
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            r = cls._function("poly_mod")(a, b, subtract, multiply, divide, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        r = field._view(r)

        return r

    @classmethod
    def _poly_pow(cls, a, b, c=None):
        assert isinstance(a, cls) and isinstance(b, (int, np.integer)) and isinstance(c, (type(None), cls))
        assert a.ndim == 1 and c.ndim == 1 if c is not None else True
        field = type(a)
        dtype = a.dtype

        # Convert the integer b into a vector of uint64 [MSWord, ..., LSWord] so arbitrarily-large exponents may be
        # passed into the JIT-compiled version
        b_vec = []  # Pop on LSWord -> MSWord
        while b > 2**64:
            q, r = divmod(b, 2**64)
            b_vec.append(r)
            b = q
        b_vec.append(b)
        b_vec = np.array(b_vec[::-1], dtype=np.uint64)  # Make vector MSWord -> LSWord

        if cls._ufunc_mode != "python-calculate":
            a = a.astype(np.int64)
            c = np.array([], dtype=np.int64) if c is None else c.astype(np.int64)
            add = cls._func_calculate("add")
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            divide = cls._func_calculate("divide")
            convolve = cls._function("convolve")
            poly_mod = cls._function("poly_mod")
            z = cls._function("poly_pow")(a, b_vec, c, add, subtract, multiply, divide, convolve, poly_mod, cls._characteristic, cls._degree, cls._irreducible_poly_int)
            z = z.astype(dtype)
        else:
            a = a.view(np.ndarray)
            c = np.array([], dtype=dtype) if c is None else c.view(np.ndarray)
            add = cls._func_python("add")
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            divide = cls._func_python("divide")
            convolve = cls._function("convolve")
            poly_mod = cls._function("poly_mod")
            z = cls._function("poly_pow")(a, b_vec, c, add, subtract, multiply, divide, convolve, poly_mod, cls._characteristic, cls._degree, cls._irreducible_poly_int)
        z = field._view(z)

        return z

    @classmethod
    def _poly_roots(cls, nonzero_degrees, nonzero_coeffs):
        assert isinstance(nonzero_coeffs, cls)
        field = cls
        dtype = nonzero_coeffs.dtype

        if cls._ufunc_mode != "python-calculate":
            nonzero_degrees = nonzero_degrees.astype(np.int64)
            nonzero_coeffs = nonzero_coeffs.astype(np.int64)
            add = cls._func_calculate("add")
            multiply = cls._func_calculate("multiply")
            power = cls._func_calculate("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, np.int64(cls._primitive_element), add, multiply, power, cls._characteristic, cls._degree, cls._irreducible_poly_int)[0,:]
            roots = roots.astype(dtype)
        else:
            nonzero_degrees = nonzero_degrees.view(np.ndarray)
            nonzero_coeffs = nonzero_coeffs.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            power = cls._func_python("power")
            roots = cls._function("poly_roots")(nonzero_degrees, nonzero_coeffs, int(cls._primitive_element), add, multiply, power, cls._characteristic, cls._degree, cls._irreducible_poly_int)[0,:]
        roots = field._view(roots)

        idxs = np.argsort(roots)
        return roots[idxs]

    ###############################################################################
    # Function implementations using explicit calculation
    ###############################################################################

    _DFT_JIT_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    # TODO: Determine how to handle recursion with a single JIT-compiled/pure-Python function

    @staticmethod
    @numba.extending.register_jitable
    def _dft_jit_calculate(x, omega, ADD, SUBTRACT, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Need a separate function to invoke _dft_jit_calculate() for recursion in the JIT compilation.
        """
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = x.dtype
        N = x.size
        X = np.zeros(N, dtype=dtype)

        if N == 1:
            X[0] = x[0]
        elif N % 2 == 0:
            # Radix-2 Cooley-Tukey FFT
            omega2 = MULTIPLY(omega, omega, *args)
            EVEN = _dft_jit_calculate(x[0::2], omega2, ADD, SUBTRACT, MULTIPLY, *args)  # pylint: disable=undefined-variable
            ODD = _dft_jit_calculate(x[1::2], omega2, ADD, SUBTRACT, MULTIPLY, *args)  # pylint: disable=undefined-variable

            twiddle = 1
            for k in range(0, N//2):
                ODD[k] = MULTIPLY(ODD[k], twiddle, *args)
                twiddle = MULTIPLY(twiddle, omega, *args)  # Twiddle is omega^k

            for k in range(0, N//2):
                X[k] = ADD(EVEN[k], ODD[k], *args)
                X[k + N//2] = SUBTRACT(EVEN[k], ODD[k], *args)
        else:
            # DFT with O(N^2) complexity
            twiddle = 1
            for k in range(0, N):
                factor = 1
                for j in range(0, N):
                    X[k] = ADD(X[k], MULTIPLY(x[j], factor, *args), *args)
                    factor = MULTIPLY(factor, twiddle, *args)  # Factor is omega^(j*k)
                twiddle = MULTIPLY(twiddle, omega, *args)  # Twiddle is omega^k

        return X

    @staticmethod
    def _dft_python_calculate(x, omega, ADD, SUBTRACT, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Need a separate function to invoke FunctionMeta._dft_python_calculate() for recursion.
        """
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = x.dtype
        N = x.size
        X = np.zeros(N, dtype=dtype)

        if N == 1:
            X[0] = x[0]
        elif N % 2 == 0:
            # Radix-2 Cooley-Tukey FFT
            omega2 = MULTIPLY(omega, omega, *args)
            EVEN = FieldFunction._dft_python_calculate(x[0::2], omega2, ADD, SUBTRACT, MULTIPLY, *args)
            ODD = FieldFunction._dft_python_calculate(x[1::2], omega2, ADD, SUBTRACT, MULTIPLY, *args)

            twiddle = 1
            for k in range(0, N//2):
                ODD[k] = MULTIPLY(ODD[k], twiddle, *args)
                twiddle = MULTIPLY(twiddle, omega, *args)  # Twiddle is omega^k

            for k in range(0, N//2):
                X[k] = ADD(EVEN[k], ODD[k], *args)
                X[k + N//2] = SUBTRACT(EVEN[k], ODD[k], *args)
        else:
            # DFT with O(N^2) complexity
            twiddle = 1
            for k in range(0, N):
                factor = 1
                for j in range(0, N):
                    X[k] = ADD(X[k], MULTIPLY(x[j], factor, *args), *args)
                    factor = MULTIPLY(factor, twiddle, *args)  # Factor is omega^(j*k)
                twiddle = MULTIPLY(twiddle, omega, *args)  # Twiddle is omega^k

        return X

    _MATMUL_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:,:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _matmul_calculate(A, B, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = A.dtype

        assert A.ndim == 2 and B.ndim == 2
        assert A.shape[-1] == B.shape[-2]

        M, K = A.shape
        K, N = B.shape
        C = np.zeros((M, N), dtype=dtype)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C[i,j] = ADD(C[i,j], MULTIPLY(A[i,k], B[k,j], *args), *args)

        return C

    _CONVOLVE_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _convolve_calculate(a, b, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        c = np.zeros(a.size + b.size - 1, dtype=dtype)
        for i in range(a.size):
            for j in range(b.size - 1, -1, -1):
                c[i + j] = ADD(c[i + j], MULTIPLY(a[i], b[j], *args), *args)

        return c

    _POLY_EVALUATE_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_evaluate_calculate(coeffs, values, ADD, MULTIPLY, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):  # pragma: no cover
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = values.dtype

        results = np.zeros(values.size, dtype=dtype)
        for i in range(values.size):
            results[i] = coeffs[0]
            for j in range(1, coeffs.size):
                results[i] = ADD(coeffs[j], MULTIPLY(results[i], values[i], *args), *args)

        return results

    _POLY_DIVMOD_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:,:], int64[:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_divmod_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY

        assert a.ndim == 2 and b.ndim == 1
        assert a.shape[-1] >= b.shape[-1]

        q_degree = a.shape[1] - b.shape[-1]
        qr = a.copy()

        for k in range(a.shape[0]):
            for i in range(q_degree + 1):
                if qr[k,i] > 0:
                    q = DIVIDE(qr[k,i], b[0], *args)
                    for j in range(1, b.size):
                        qr[k, i + j] = SUBTRACT(qr[k, i + j], MULTIPLY(q, b[j], *args), *args)
                    qr[k,i] = q

        return qr

    _POLY_FLOORDIV_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_floordiv_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return np.array([0], dtype=dtype)

        q_degree = a.size - b.size
        q = np.zeros(q_degree + 1, dtype=dtype)
        aa = a[0:q_degree + 1].copy()

        for i in range(q_degree + 1):
            if aa[i] > 0:
                q[i] = DIVIDE(aa[i], b[0], *args)
                N = min(b.size, q_degree + 1 - i)  # We don't need to subtract in the "remainder" range
                for j in range(1, N):
                    aa[i + j] = SUBTRACT(aa[i + j], MULTIPLY(q[i], b[j], *args), *args)

        return q

    _POLY_MOD_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_mod_calculate(a, b, SUBTRACT, MULTIPLY, DIVIDE, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        if b.size == 1 and b[0] == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")

        if a.size < b.size:
            return a.copy()

        if b.size == 1:
            return np.array([0], dtype=dtype)

        q_degree = a.size - b.size
        r_degree = b.size - 1
        r = np.zeros(r_degree + 1, dtype=dtype)
        r[1:] = a[0:r_degree]

        for i in range(q_degree + 1):
            r = np.roll(r, -1)
            r[-1] = a[i + r_degree]

            if r[0] > 0:
                q = DIVIDE(r[0], b[0], *args)
                for j in range(1, b.size):
                    r[j] = SUBTRACT(r[j], MULTIPLY(q, b[j], *args), *args)

        r = r[1:]

        # Trim leading zeros to reduce computations in future calls
        if r.size > 1:
            idxs = np.nonzero(r)[0]
            if idxs.size > 0:
                r = r[idxs[0]:]
            else:
                r = r[-1:]

        return r

    _POLY_POW_CALCULATE_SIG = numba.types.FunctionType(int64[:](int64[:], uint64[:], int64[:], FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, _CONVOLVE_CALCULATE_SIG, _POLY_MOD_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_pow_calculate(a, b_vec, c, ADD, SUBTRACT, MULTIPLY, DIVIDE, POLY_MULTIPLY, POLY_MOD, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        b is a vector of uint64 [MSWord, ..., LSWord] so that arbitrarily-large exponents may be passed
        """
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = a.dtype

        if b_vec.size == 1 and b_vec[0] == 0:
            return np.array([1], dtype=dtype)

        result_s = a.copy()  # The "squaring" part
        result_m = np.array([1], dtype=dtype)  # The "multiplicative" part

        # Loop from LSWord to MSWord
        for i in range(b_vec.size - 1, -1, -1):
            j = 0  # Bit counter -- make sure we interate through 64 bits on all but the most-significant word
            while j < 64:
                if i == 0 and b_vec[i] <= 1:
                    # This is the MSB and we already accounted for the most-significant bit -- can exit now
                    break

                if b_vec[i] % 2 == 0:
                    result_s = POLY_MULTIPLY(result_s, result_s, ADD, MULTIPLY, *args)
                    if c.size > 0:
                        result_s = POLY_MOD(result_s, c, SUBTRACT, MULTIPLY, DIVIDE, *args)
                    b_vec[i] //= 2
                    j += 1
                else:
                    result_m = POLY_MULTIPLY(result_m, result_s, ADD, MULTIPLY, *args)
                    if c.size > 0:
                        result_m = POLY_MOD(result_m, c, SUBTRACT, MULTIPLY, DIVIDE, *args)
                    b_vec[i] -= 1

        result = POLY_MULTIPLY(result_s, result_m, ADD, MULTIPLY, *args)
        if c.size > 0:
            result = POLY_MOD(result, c, SUBTRACT, MULTIPLY, DIVIDE, *args)

        return result

    _POLY_ROOTS_CALCULATE_SIG = numba.types.FunctionType(int64[:,:](int64[:], int64[:], int64, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, FieldUfunc._BINARY_CALCULATE_SIG, int64, int64, int64))

    @staticmethod
    @numba.extending.register_jitable
    def _poly_roots_calculate(nonzero_degrees, nonzero_coeffs, primitive_element, ADD, MULTIPLY, POWER, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        args = CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY
        dtype = nonzero_coeffs.dtype
        ORDER = CHARACTERISTIC**DEGREE

        N = nonzero_degrees.size
        lambda_vector = nonzero_coeffs.copy()
        alpha_vector = np.zeros(N, dtype=dtype)
        for i in range(N):
            alpha_vector[i] = POWER(primitive_element, nonzero_degrees[i], *args)
        degree = np.max(nonzero_degrees)
        roots = []
        powers = []

        # Test if 0 is a root
        if nonzero_degrees[-1] != 0:
            roots.append(0)
            powers.append(-1)

        # Test if 1 is a root
        _sum = 0
        for i in range(N):
            _sum = ADD(_sum, lambda_vector[i], *args)
        if _sum == 0:
            roots.append(1)
            powers.append(0)

        # Test if the powers of alpha are roots
        for i in range(1, ORDER - 1):
            _sum = 0
            for j in range(N):
                lambda_vector[j] = MULTIPLY(lambda_vector[j], alpha_vector[j], *args)
                _sum = ADD(_sum, lambda_vector[j], *args)
            if _sum == 0:
                root = POWER(primitive_element, i, *args)
                roots.append(root)
                powers.append(i)
            if len(roots) == degree:
                break

        return np.array([roots, powers], dtype=dtype)
