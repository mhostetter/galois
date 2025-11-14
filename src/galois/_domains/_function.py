"""
A module that contains a NumPy function dispatcher and an Array mixin class that override NumPy functions. The function
dispatcher classes have snake_case naming because they are act like functions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Type

import numba
import numpy as np
from numba import int64, types

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
        factors = self._get_prime_factors(n)

        if self.field.ufunc_mode != "python-calculate":
            y = self.jit(x.astype(np.int64), np.int64(omega), factors)
            y = y.astype(dtype)
        else:
            y = self.python(x.view(np.ndarray), int(omega), factors)
        y = self.field._view(y)

        # Scale the transform such that x = IDFT(DFT(x))
        if self._direction == norm:
            y /= self.field(n % self.field.characteristic)

        return y

    @staticmethod
    @functools.cache
    def _get_prime_factors(N) -> np.ndarray:
        "returns the prime factors of N as an int64 numpy array."
        import galois  # noqa

        if N == 1:
            return np.int64([])
        primes, counts = galois.factors(N)
        array = np.int64([prime for prime, count in zip(primes, counts) for _ in range(count)])
        array.flags.writeable = False  # Make it read-only for safety, since we will reuse it.
        return array

    def set_globals(self):
        global ADD, SUBTRACT, MULTIPLY, POWER
        ADD = self.field._add.ufunc_call_only
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        POWER = self.field._power.ufunc_call_only

    # Make sure that numba knows that the third argument is read-only
    _SIGNATURE = numba.types.FunctionType(
        int64[:](
            int64[:],
            int64,
            # Tell numba that the third argument is a read-only array
            types.Array(types.int64, 1, "C", readonly=True),
        )
    )

    @staticmethod
    def implementation(array, omega, factors):
        """Adapted from https://dsp-book.narod.ru/FFTBB/0270_PDF_C15.pdf"""
        B = 1
        in_array = np.ascontiguousarray(array)
        out_array = np.empty_like(in_array)
        for index in range(len(factors)):
            if out_array is array:
                # don't overwrite the initial input array.
                out_array = np.empty_like(in_array)
            F = factors[~index]
            Q = array.size // (B * F)
            omega_bf = POWER(omega, Q)  # omega ** (B * F)
            z = 1
            # View the flat arrays as a 3-day array.
            in_array_3d = in_array.reshape((F, Q, B))
            out_array_3d = out_array.reshape((Q, F, B))
            if F == 2:
                for b in range(B):
                    for q in range(Q):
                        left = in_array_3d[0, q, b]
                        right = MULTIPLY(in_array_3d[1, q, b], z)
                        out_array_3d[q, 0, b] = ADD(left, right)
                        out_array_3d[q, 1, b] = SUBTRACT(left, right)
                    z = MULTIPLY(z, omega_bf)
            else:
                out_array_3d[:, :, :] = in_array_3d[F - 1, :, None, :]
                for f in range(F):
                    for b in range(B):
                        for q in range(Q):
                            for fx in range(1, F):
                                # We use Horner's rule to evaluate the polynomial.
                                out_array_3d[q, f, b] = ADD(MULTIPLY(out_array_3d[q, f, b], z), in_array_3d[~fx, q, b])
                        z = MULTIPLY(z, omega_bf)
            B = B * F
            in_array, out_array = out_array, in_array
        return in_array.ravel()


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

    if np.lib.NumpyVersion(np.__version__) < "2.4.0":
        _UNSUPPORTED_FUNCTIONS.append(np.trapz)
    else:
        _UNSUPPORTED_FUNCTIONS.append(np.trapezoid)

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
