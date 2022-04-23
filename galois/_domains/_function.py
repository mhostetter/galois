"""
A module that contains Array mixin classes that override NumPy functions.
"""
import abc

import numba
from numba import int64
import numpy as np

from ._ufunc import RingUfuncs, FieldUfuncs


class RingFunctions(RingUfuncs, abc.ABC):
    """
    A mixin base class that overrides NumPy functions to perform ring arithmetic (+, -, *), using *only* explicit
    calculation. It was determined that explicit calculation is always faster than lookup tables. For some reason,
    passing large LUTs to JIT functions is slow.
    """

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

    _FUNCTION_CACHE_CALCULATE = {}

    def __array_function__(self, func, types, args, kwargs):
        """
        Override the standard NumPy function calls with the new finite field functions.
        """
        field = type(self)

        if func in field._OVERRIDDEN_FUNCTIONS:
            output = getattr(field, field._OVERRIDDEN_FUNCTIONS[func])(*args, **kwargs)

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

    ###############################################################################
    # Individual functions, pre-compiled (cached)
    ###############################################################################

    @classmethod
    def _function(cls, name):
        """
        Returns the function for the specific routine. The function compilation is based on `ufunc_mode`.
        """
        if name not in cls._functions:
            if cls.ufunc_mode != "python-calculate":
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
    # Convolution
    ###############################################################################

    @classmethod
    def _convolve(cls, a, b, mode="full"):
        if not type(a) is type(b):
            raise TypeError(f"Arguments `a` and `b` must be of the same FieldArray subclass, not {type(a)} and {type(b)}.")
        if not mode == "full":
            raise ValueError(f"Operation 'convolve' currently only supports mode of 'full', not {mode!r}.")
        field = type(a)
        dtype = a.dtype

        if cls.ufunc_mode == "python-calculate":
            a = a.view(np.ndarray)
            b = b.view(np.ndarray)
            add = cls._func_python("add")
            multiply = cls._func_python("multiply")
            c = cls._function("convolve")(a, b, add, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        elif field.is_prime_field:
            # Determine the minimum dtype to hold the entire product and summation without overflowing
            n_sum = min(a.size, b.size)
            max_value = n_sum * (field.characteristic - 1)**2
            dtypes = [dtype for dtype in cls.dtypes if np.iinfo(dtype).max >= max_value]
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
            c = cls._function("convolve")(a, b, add, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            c = c.astype(dtype)
        c = field._view(c)

        return c

    _CONVOLVE_CALCULATE_SIG = numba.types.FunctionType(int64[:](
        int64[:],
        int64[:],
        RingUfuncs._BINARY_CALCULATE_SIG,
        RingUfuncs._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

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

    ###############################################################################
    # FFT and IFFT
    # TODO: Determine how to handle recursion with a single JIT-compiled or
    # pure-Python function
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

        omega = field.primitive_root_of_unity(x.size)  # pylint: disable=no-member
        if not forward:
            omega = omega ** -1

        if cls.ufunc_mode != "python-calculate":
            x = x.astype(np.int64)
            omega = np.int64(omega)
            add = cls._func_calculate("add")
            subtract = cls._func_calculate("subtract")
            multiply = cls._func_calculate("multiply")
            y = cls._function("dft_jit")(x, omega, add, subtract, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
            y = y.astype(dtype)
        else:
            x = x.view(np.ndarray)
            omega = int(omega)
            add = cls._func_python("add")
            subtract = cls._func_python("subtract")
            multiply = cls._func_python("multiply")
            y = cls._function("dft_python")(x, omega, add, subtract, multiply, cls.characteristic, cls.degree, int(cls.irreducible_poly))
        y = field._view(y)

        # Scale the inverse NTT such that x = INTT(NTT(x))
        if not forward and scaled:
            y /= field(n % field.characteristic)

        return y

    @classmethod
    def _ifft(cls, x, n=None, axis=-1, norm=None, scaled=True):
        return cls._fft(x, n=n, axis=axis, norm=norm, forward=False, scaled=scaled)

    _DFT_JIT_CALCULATE_SIG = numba.types.FunctionType(int64[:](
        int64[:],
        int64,
        RingUfuncs._BINARY_CALCULATE_SIG,
        RingUfuncs._BINARY_CALCULATE_SIG,
        RingUfuncs._BINARY_CALCULATE_SIG,
        int64,
        int64,
        int64
    ))

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
            EVEN = RingFunctions._dft_python_calculate(x[0::2], omega2, ADD, SUBTRACT, MULTIPLY, *args)
            ODD = RingFunctions._dft_python_calculate(x[1::2], omega2, ADD, SUBTRACT, MULTIPLY, *args)

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


class FieldFunctions(RingFunctions, FieldUfuncs):
    """
    A mixin base class that overrides NumPy functions to perform field arithmetic (+, -, *, /), using *only* explicit
    calculation.
    """
