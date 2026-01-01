"""
A module that contains a NumPy function dispatcher and an Array mixin class that override NumPy functions. The function
dispatcher classes have snake_case naming because they are act like functions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Type

import numba
import numpy as np
from numba import int64

from .._helper import verify_isinstance
from .._prime import factors as _factors
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
        elif n < x.size:
            x = x[:n]
        elif n > x.size:
            x = np.append(x, self.field.Zeros(n - x.size))

        omega = self.field.primitive_root_of_unity(n)
        if self._direction == "backward":
            omega = omega**-1
        factors = self._prime_factors(n)

        if self.field.ufunc_mode != "python-calculate":
            # NOTE: Performing x.astype() returns a copy of x, which is necessary to prevent modifying the original
            #       array in-place
            y = self.jit(x.astype(np.int64), np.int64(omega), np.array(factors, dtype=np.int64))
            y = y.astype(dtype)
        else:
            # NOTE: Make a copy of x to prevent modifying the original array in-place
            y = self.python(x.view(np.ndarray).copy(), int(omega), factors)
        y = self.field._view(y)

        # Scale the transform such that x = IDFT(DFT(x))
        if self._direction == norm:
            y /= self.field(n % self.field.characteristic)

        return y

    @staticmethod
    @functools.lru_cache(None)
    def _prime_factors(length: int) -> tuple[int, ...]:
        """
        Returns the prime factors of `length` with multiplicity, e.g. 176 → (2,2,2,2,11).
        """
        if length == 1:
            return ()
        else:
            primes, multiplicities = _factors(length)
            factors = [prime for prime, multiplicity in zip(primes, multiplicities) for _ in range(multiplicity)]
            return tuple(factors)

    def set_globals(self):
        global ADD, SUBTRACT, MULTIPLY, POWER
        ADD = self.field._add.ufunc_call_only
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        POWER = self.field._power.ufunc_call_only

    _SIGNATURE = numba.types.FunctionType(
        int64[:](
            int64[:],
            int64,
            int64[:],
        )
    )

    @staticmethod
    def implementation(array, omega, factors):
        """
        Compute the (mixed-radix) FFT of `array` using an iterative Cooley-Tukey style algorithm.

        This routine is written to operate over an abstract algebraic domain (finite fields, rings, etc.)
        by using the primitive operations `ADD/SUBTRACT/MULTIPLY/POWER` instead of Python arithmetic.

        Arguments:
            array:
                1-D input of length N.
            omega:
                A primitive N-th root of unity in the domain (so omega**N = 1, and no smaller positive
                power equals 1).
            factors:
                A factorization of N into small radices (e.g., for N = 2**k, factors = [2, 2, ..., 2]).
                The algorithm consumes these radices from the end (reverse order), which matches the
                indexing/layout used in the referenced implementation.

        Returns:
            The FFT of the input, same shape/dtype as `array` (1-D).

        Notes:
            The algorithm proceeds in stages. At each stage with radix `r`:

                - We assume we already have many independent FFTs of length `m` (initially m = 1).
                - We "stitch" groups of `r` such FFTs together to form FFTs of length `m*r`.

            Let:
                N = len(array)
                r = radix for this stage
                m = current block length (FFT size already computed per block)
                q = N / (m*r) = number of blocks after grouping

            We view the data as 3-D to make the grouping explicit:

                in_view  has shape (r, q, m)
                out_view has shape (q, r, m)

            For each fixed (qi, b) pair we combine `r` values:

                x_k = in_view[k, qi, b],   k = 0..r-1

            into `r` outputs stored as:

                out_view[qi, f, b],        f = 0..r-1

            The combination uses twiddle factors derived from omega. In this implementation,
            the twiddle “step” for the stage is:

                twiddle_step = omega^( N / (m*r) ) = omega^q

            and a running twiddle value `twiddle` is updated by multiplying by `twiddle_step` in the same
            order as the original reference implementation.
        """
        # Ensure a contiguous working buffer (important for predictable reshapes)
        in_buffer = np.ascontiguousarray(array)

        # Allocate the second buffer once and ping-pong between them each stage
        out_buffer = np.empty_like(in_buffer)

        N = in_buffer.size  # Total FFT size
        m = 1  # Size of FFT blocks already computed

        # The reference implementation consumes radices from the end.
        # (This affects how reshapes map onto the conceptual decomposition.)
        for r in factors[::-1]:
            q = N // (m * r)  # Number of blocks at this stage
            # Invariant: N == m * r * q

            twiddle = 1  # Running twiddle factor (advances by omega^q each step)
            twiddle_step = POWER(omega, q)  # Twiddle step for this stage: omega^(N / (m*r)) = omega^q

            # Reinterpret the flat buffers as 3-D views to express the Cooley–Tukey grouping:
            #   in_view[k, qi, b]  : k-th subblock (0..r-1), within block qi, at offset b
            #   out_view[qi, f, b] : output f (0..r-1) for block qi, at offset b
            in_view = in_buffer.reshape((r, q, m))
            out_view = out_buffer.reshape((q, r, m))

            # Note:
            #     Although the twiddle factor mathematically depends only on the output index f
            #     (twiddle = omega^(q * f)), we advance it imperatively here to match the memory
            #     layout induced by the reshape (r, q, m) → (q, r, m). The update schedule is
            #     therefore tied to the loop nesting and must not be reordered.

            if r == 2:
                # Radix-2 "butterfly":
                #   y0 = x0 + twiddle * x1
                #   y1 = x0 - twiddle * x1
                #
                # The running `twiddle` is advanced by `twiddle_step` in the same nested-loop
                # order as the reference implementation.
                for b in range(m):
                    for qi in range(q):
                        x0 = in_view[0, qi, b]
                        x1 = MULTIPLY(in_view[1, qi, b], twiddle)

                        out_view[qi, 0, b] = ADD(x0, x1)
                        out_view[qi, 1, b] = SUBTRACT(x0, x1)

                    twiddle = MULTIPLY(twiddle, twiddle_step)

            else:
                # General radix-r combine.
                #
                # For each (qi, b) we have r inputs: x_0..x_{r-1}.
                # Each output is computed by evaluating a polynomial in twiddle:
                #
                #   y = x_0 + x_1 * twiddle + x_2 * twiddle^2 + ... + x_{r-1} * twiddle^{r-1}
                #
                # This is evaluated using Horner's rule (minimizes multiplications):
                #
                #   (((x_{r-1} * twiddle + x_{r-2}) * twiddle + x_{r-3}) ... * twiddle + x_0)
                #
                # The running `twiddle` is advanced by `twiddle_step` in the same nested-loop
                # order as the reference implementation.
                for f in range(r):
                    for b in range(m):
                        for qi in range(q):
                            acc = in_view[r - 1, qi, b]
                            for k in range(r - 2, -1, -1):
                                acc = ADD(MULTIPLY(acc, twiddle), in_view[k, qi, b])
                            out_view[qi, f, b] = acc

                        twiddle = MULTIPLY(twiddle, twiddle_step)

            # After this stage, block FFT size grows by `r`
            m *= r

            # Ping-pong buffers: next stage reads from what we just wrote
            in_buffer, out_buffer = out_buffer, in_buffer

        return in_buffer.ravel()


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
