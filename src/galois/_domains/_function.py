"""
A module that contains a NumPy function dispatcher and an Array mixin class that override NumPy functions. The function
dispatcher classes have snake_case naming because they are act like functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

import numba
import numpy as np
from numba import int64

from .._helper import verify_isinstance
from ._meta import ArrayMeta

if TYPE_CHECKING:
    from ._array import Array


class Function:
    """
    A function dispatcher for Array objects. The dispatcher will invoke a JIT-compiled or pure-Python function
    depending on the size of the Galois field or Galois ring.
    """

    _CACHE = {}  # A cache of compiled functions

    def __init__(self, field: Type[Array]):
        self.field = field

    def __call__(self):
        """
        Invokes the function, either JIT-compiled or pure-Python, performing necessary input/output conversion.
        """
        raise NotImplementedError

    def set_globals(self):
        """
        Sets the global variables used in `implementation()` before JIT compiling it or before invoking it in
        pure Python.
        """
        return

    _SIGNATURE: numba.types.FunctionType
    """The function's Numba signature."""

    _PARALLEL = False
    """Indicates if parallel processing should be performed."""

    implementation: Callable
    """The function's implementation in pure Python."""

    ###############################################################################
    # Various ufuncs based on implementation and compilation
    ###############################################################################

    @property
    def key_1(self):
        return (self.field.characteristic, self.field.degree, int(self.field.irreducible_poly))

    @property
    def key_2(self):
        if self.field.ufunc_mode == "jit-lookup":
            key = (str(self.__class__), self.field.ufunc_mode, int(self.field.primitive_element))
        else:
            key = (str(self.__class__), self.field.ufunc_mode)
        return key

    @property
    def function(self):
        """
        Returns a JIT-compiled or pure-Python function based on field size.
        """
        if self.field.ufunc_mode == "python-calculate":
            return self.python
        return self.jit

    @property
    def jit(self) -> numba.types.FunctionType:
        """
        Returns a JIT-compiled function implemented over the given field.
        """
        assert self.field.ufunc_mode in ["jit-lookup", "jit-calculate"]

        self._CACHE.setdefault(self.key_1, {})
        if self.key_2 not in self._CACHE[self.key_1]:
            self.set_globals()  # Set the globals once before JIT compiling the function
            func = numba.jit(self._SIGNATURE.signature, parallel=self._PARALLEL, nopython=True)(self.implementation)
            self._CACHE[self.key_1][self.key_2] = func

        return self._CACHE[self.key_1][self.key_2]

    @property
    def python(self) -> Callable:
        """
        Returns the pure-Python function implemented over the given field.
        """
        self.set_globals()  # Set the globals each time before invoking the pure-Python function
        return self.implementation


###############################################################################
# Ndarray function wrappers
###############################################################################


class convolve_jit(Function):
    """
    Function dispatcher to convolve two 1-D arrays.
    """

    def __call__(self, a: Array, b: Array, mode="full") -> Array:
        verify_isinstance(a, self.field)
        verify_isinstance(b, self.field)
        if not mode == "full":
            raise ValueError(f"Operation 'convolve' currently only supports mode of 'full', not {mode!r}.")
        dtype = a.dtype

        if self.field.ufunc_mode != "python-calculate":
            c = self.jit(a.astype(np.int64), b.astype(np.int64))
            c = c.astype(dtype)
        else:
            c = self.python(a.view(np.ndarray), b.view(np.ndarray))
        c = self.field._view(c)

        return c

    def set_globals(self):
        global IS_PRIME_FIELD, CHARACTERISTIC, ADD, MULTIPLY
        IS_PRIME_FIELD = self.field._is_prime_field
        CHARACTERISTIC = self.field.characteristic
        ADD = self.field._add.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(a, b):
        dtype = a.dtype

        if IS_PRIME_FIELD:
            try:
                max_sum = np.iinfo(dtype).max // (CHARACTERISTIC - 1) ** 2
                n_sum = min(a.size, b.size)
                overflow = n_sum > max_sum
            except:  # noqa: E722
                # This happens when the dtype is np.object_
                overflow = False

            if not overflow:
                # Compute the result using native NumPy LAPACK/BLAS implementation since it is guaranteed to not
                # overflow. Then reduce the result mod p.
                c = np.convolve(a, b)
                c = c % CHARACTERISTIC
                return c

        # Fall-back brute force method
        c = np.zeros(a.size + b.size - 1, dtype=dtype)
        for i in range(a.size):
            for j in range(b.size - 1, -1, -1):
                c[i + j] = ADD(c[i + j], MULTIPLY(a[i], b[j]))

        return c


class fft_jit(Function):
    """
    Function dispatcher to compute the Discrete Fourier Transform of the input array.
    """

    _direction = "forward"

    def __call__(self, x: Array, n=None, axis=-1, norm=None) -> Array:
        verify_isinstance(x, self.field)
        norm = "backward" if norm is None else norm
        if not axis == -1:
            raise ValueError("The FFT is only implemented on 1-D arrays.")
        if not norm in ["forward", "backward"]:
            raise ValueError("DFT normalization can only be applied to the forward or backward transform, not 'ortho'.")
        dtype = x.dtype

        if n is None:
            n = x.size
        x = np.append(x, np.zeros(n - x.size, dtype=x.dtype))

        omega = self.field.primitive_root_of_unity(x.size)
        if self._direction == "backward":
            omega = omega**-1

        if self.field.ufunc_mode != "python-calculate":
            y = self.jit(x.astype(np.int64), np.int64(omega))
            y = y.astype(dtype)
        else:
            y = self.python(x.view(np.ndarray), int(omega))
        y = self.field._view(y)

        # Scale the transform such that x = IDFT(DFT(x))
        if self._direction == norm:
            y /= self.field(n % self.field.characteristic)

        return y

    def set_globals(self):
        global ADD, SUBTRACT, MULTIPLY
        ADD = self.field._add.ufunc_call_only
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64))

    @staticmethod
    def implementation(x, omega):  # pragma: no cover
        N = x.size
        X = np.zeros(N, dtype=x.dtype)

        if N == 1:
            X[0] = x[0]
        elif N % 2 == 0:
            # Radix-2 Cooley-Tukey FFT
            omega2 = MULTIPLY(omega, omega)
            EVEN = implementation(x[0::2], omega2)  # noqa: F821
            ODD = implementation(x[1::2], omega2)  # noqa: F821

            twiddle = 1
            for k in range(0, N // 2):
                ODD[k] = MULTIPLY(ODD[k], twiddle)
                twiddle = MULTIPLY(twiddle, omega)  # Twiddle is omega^k

            for k in range(0, N // 2):
                X[k] = ADD(EVEN[k], ODD[k])
                X[k + N // 2] = SUBTRACT(EVEN[k], ODD[k])
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

    # Need a separate implementation for pure-Python to call the static method. It would be nice to avoid the
    # need for this.

    @staticmethod
    def implementation_2(x, omega):
        N = x.size
        X = np.zeros(N, dtype=x.dtype)

        if N == 1:
            X[0] = x[0]
        elif N % 2 == 0:
            # Radix-2 Cooley-Tukey FFT
            omega2 = MULTIPLY(omega, omega)
            EVEN = fft_jit.implementation_2(x[0::2], omega2)
            ODD = fft_jit.implementation_2(x[1::2], omega2)

            twiddle = 1
            for k in range(0, N // 2):
                ODD[k] = MULTIPLY(ODD[k], twiddle)
                twiddle = MULTIPLY(twiddle, omega)  # Twiddle is omega^k

            for k in range(0, N // 2):
                X[k] = ADD(EVEN[k], ODD[k])
                X[k + N // 2] = SUBTRACT(EVEN[k], ODD[k])
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

    @property
    def python(self):
        self.set_globals()
        return self.implementation_2


class ifft_jit(fft_jit):
    """
    Function dispatcher to compute the Inverse Discrete Fourier Transform of the input array.
    """

    _direction = "backward"


###############################################################################
# Array mixin class
###############################################################################


class FunctionMixin(np.ndarray, metaclass=ArrayMeta):
    """
    An Array mixin class that overrides the invocation of NumPy functions on Array objects.
    """

    _UNSUPPORTED_FUNCTIONS = [
        # Unary
        np.packbits,
        np.unpackbits,
        np.unwrap,
        np.around,
        np.round,
        np.fix,
        np.gradient,
        np.trapz,
        np.i0,
        np.sinc,
        np.angle,
        np.real,
        np.imag,
        np.conj,
        np.conjugate,
        # Binary
        np.lib.scimath.logn,
        np.cross,
    ]

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

    _convolve: Function
    _fft: Function
    _ifft: Function

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._convolve = convolve_jit(cls)
        cls._fft = fft_jit(cls)
        cls._ifft = ifft_jit(cls)

    def __array_function__(self, func, types, args, kwargs):
        """
        Override the standard NumPy function calls with the new finite field functions.
        """
        field = type(self)

        if func in field._OVERRIDDEN_FUNCTIONS:
            output = getattr(field, field._OVERRIDDEN_FUNCTIONS[func])(*args, **kwargs)

        elif func in field._UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(
                f"The NumPy function {func.__name__!r} is not supported on FieldArray. "
                "If you believe this function should be supported, "
                "please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\n"
                "If you'd like to perform this operation on the data, you should first call "
                "`array = array.view(np.ndarray)` and then call the function."
            )

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._verify_array_like_types_and_values(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)

            if func in field._FUNCTIONS_REQUIRING_VIEW:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

        return output

    def dot(self, b, out=None):
        # The `np.dot(a, b)` ufunc is also available as `a.dot(b)`. Need to override this method for
        # consistent results.
        return np.dot(self, b, out=out)
