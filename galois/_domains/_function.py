"""
A module that contains Array mixin classes that override NumPy functions.
"""
import abc
from typing import Type, Callable

import numba
from numba import int64
import numpy as np

from ._array import Array
from ._ufunc import RingUFuncs, FieldUFuncs

ADD = np.add
SUBTRACT = np.subtract
MULTIPLY = np.multiply


class JITFunction:
    """
    Wrapper class for optionally JIT-compiled functions.
    """
    _CACHE = {}  # A cache of compiled functions. Should be cleared for each derived class.

    call: Callable
    """Call the function, invoking either the JIT-compiled or pure-Python version."""

    @classmethod
    def set_globals(cls, field: Type[Array]):
        """
        Set the global variables used in `implementation()` before JIT compiling it or before invoking it in pure Python.
        """
        # pylint: disable=unused-argument
        return

    _SIGNATURE: numba.types.FunctionType
    """The function's Numba signature."""

    implementation: Callable
    """The function implementation in Python."""

    @classmethod
    def function(cls, field: Type[Array]):
        """
        Returns a JIT-compiled or pure-Python function based on field size.
        """
        if field.ufunc_mode != "python-calculate":
            return cls.jit(field)
        else:
            return cls.python(field)

    @classmethod
    def jit(cls, field: Type[Array]) -> numba.types.FunctionType:
        """
        Returns a JIT-compiled function implemented over the given field.
        """
        key = (field.characteristic, field.degree, int(field.irreducible_poly), int(field.primitive_element))
        if key not in cls._CACHE:
            # Set the globals once before JIT compiling the function
            cls.set_globals(field)
            cls._CACHE[key] = numba.jit(cls._SIGNATURE.signature, nopython=True)(cls.implementation)

        return cls._CACHE[key]

    @classmethod
    def python(cls, field: Type[Array]) -> Callable:
        """
        Returns the pure-Python function implemented over the given field.
        """
        # Set the globals each time before invoking the pure-Python function
        cls.set_globals(field)
        return cls.implementation


class RingFunctions(RingUFuncs, abc.ABC):
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

    _FUNCTION_CACHE = {}
    _FUNCTION_CACHE_PYTHON = {}

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
    # Individual functions compiled on-demand
    ###############################################################################

    @classmethod
    def _function(cls, name):
        """
        Returns the function for the specific routine. The function compilation is based on `ufunc_mode`.
        """
        if cls.ufunc_mode != "python-calculate":
            return cls._function_jit(name)
        else:
            return cls._function_python(name)

    @classmethod
    def _function_jit(cls, name):
        """
        Returns a JIT-compiled function.
        """
        key = (name, cls.characteristic, cls.degree, int(cls.irreducible_poly), int(cls.primitive_element))
        if key not in cls._FUNCTION_CACHE:
            # Set the globals once before JIT compiling the function
            getattr(cls, f"_set_{name}_jit_globals")()

            function = getattr(cls, f"_{name}_jit")
            sig = getattr(cls, f"_{name.upper()}_SIG")
            cls._FUNCTION_CACHE[key] = numba.jit(sig.signature, nopython=True)(function)

        return cls._FUNCTION_CACHE[key]

    @classmethod
    def _function_python(cls, name):
        """
        Returns a pure-Python arithmetic function using explicit calculation.
        """
        # Set the globals each time before invoking the pure-Python ufunc
        getattr(cls, f"_set_{name}_jit_globals")()
        return getattr(cls, f"_{name}_jit")

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
            c = cls._function("convolve")(a, b)
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
            c = cls._function("convolve")(a, b)
            c = c.astype(dtype)
        c = field._view(c)

        return c

    @classmethod
    def _set_convolve_jit_globals(cls):
        global ADD, MULTIPLY
        ADD = cls._ufunc("add")
        MULTIPLY = cls._ufunc("multiply")

    _CONVOLVE_SIG = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    @numba.extending.register_jitable
    def _convolve_jit(a, b):
        c = np.zeros(a.size + b.size - 1, dtype=a.dtype)
        for i in range(a.size):
            for j in range(b.size - 1, -1, -1):
                c[i + j] = ADD(c[i + j], MULTIPLY(a[i], b[j]))

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
            y = cls._function("dft1")(x, omega)
            y = y.astype(dtype)
        else:
            x = x.view(np.ndarray)
            omega = int(omega)
            y = cls._function("dft2")(x, omega)
        y = field._view(y)

        # Scale the inverse NTT such that x = INTT(NTT(x))
        if not forward and scaled:
            y /= field(n % field.characteristic)

        return y

    @classmethod
    def _ifft(cls, x, n=None, axis=-1, norm=None, scaled=True):
        return cls._fft(x, n=n, axis=axis, norm=norm, forward=False, scaled=scaled)

    _DFT1_SIG = numba.types.FunctionType(int64[:](int64[:], int64))

    @classmethod
    def _set_dft1_jit_globals(cls):
        global ADD, SUBTRACT, MULTIPLY
        ADD = cls._ufunc("add")
        SUBTRACT = cls._ufunc("subtract")
        MULTIPLY = cls._ufunc("multiply")

    @staticmethod
    @numba.extending.register_jitable
    def _dft1_jit(x, omega):
        """
        Need a separate function to invoke _dft_jit_jit() for recursion in the JIT compilation.
        """
        N = x.size
        X = np.zeros(N, dtype=x.dtype)

        if N == 1:
            X[0] = x[0]
        elif N % 2 == 0:
            # Radix-2 Cooley-Tukey FFT
            omega2 = MULTIPLY(omega, omega)
            EVEN = _dft1_jit(x[0::2], omega2)  # pylint: disable=undefined-variable
            ODD = _dft1_jit(x[1::2], omega2)  # pylint: disable=undefined-variable

            twiddle = 1
            for k in range(0, N//2):
                ODD[k] = MULTIPLY(ODD[k], twiddle)
                twiddle = MULTIPLY(twiddle, omega)  # Twiddle is omega^k

            for k in range(0, N//2):
                X[k] = ADD(EVEN[k], ODD[k])
                X[k + N//2] = SUBTRACT(EVEN[k], ODD[k])
        else:
            # DFT with O(N^2) complexity
            twiddle = 1
            for k in range(0, N):
                factor = 1
                for j in range(0, N):
                    X[k] = ADD(X[k], MULTIPLY(x[j], factor))
                    factor = MULTIPLY(factor, twiddle)  # Factor is omega^(j*k)
                twiddle = MULTIPLY(twiddle, omega)  # Twiddle is omega^k

        return X

    @classmethod
    def _set_dft2_jit_globals(cls):
        cls._set_dft1_jit_globals()

    @staticmethod
    def _dft2_jit(x, omega):
        """
        Need a separate function to invoke FunctionMeta._dft_python_jit() for recursion.
        """
        N = x.size
        X = np.zeros(N, dtype=x.dtype)

        if N == 1:
            X[0] = x[0]
        elif N % 2 == 0:
            # Radix-2 Cooley-Tukey FFT
            omega2 = MULTIPLY(omega, omega)
            EVEN = RingFunctions._dft2_jit(x[0::2], omega2)  # pylint: disable=undefined-variable
            ODD = RingFunctions._dft2_jit(x[1::2], omega2)  # pylint: disable=undefined-variable

            twiddle = 1
            for k in range(0, N//2):
                ODD[k] = MULTIPLY(ODD[k], twiddle)
                twiddle = MULTIPLY(twiddle, omega)  # Twiddle is omega^k

            for k in range(0, N//2):
                X[k] = ADD(EVEN[k], ODD[k])
                X[k + N//2] = SUBTRACT(EVEN[k], ODD[k])
        else:
            # DFT with O(N^2) complexity
            twiddle = 1
            for k in range(0, N):
                factor = 1
                for j in range(0, N):
                    X[k] = ADD(X[k], MULTIPLY(x[j], factor))
                    factor = MULTIPLY(factor, twiddle)  # Factor is omega^(j*k)
                twiddle = MULTIPLY(twiddle, omega)  # Twiddle is omega^k

        return X


class FieldFunctions(RingFunctions, FieldUFuncs):
    """
    A mixin base class that overrides NumPy functions to perform field arithmetic (+, -, *, /), using *only* explicit
    calculation.
    """
