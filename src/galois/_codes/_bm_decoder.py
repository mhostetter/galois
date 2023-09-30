"""
A module containing a general Berlekamp-Massey decoder for BCH and Reed-Solomon codes.
"""
from __future__ import annotations

from typing import Callable, Hashable, Type

import numba
import numpy as np
import numpy.typing as npt
from numba import int64

from .._domains._function import Function
from .._fields import FieldArray
from .._lfsr import berlekamp_massey_jit
from .._polys._dense import evaluate_elementwise_jit, roots_jit

CHARACTERISTIC: int
SUBTRACT: Callable[[int, int], int]
MULTIPLY: Callable[[int, int], int]
RECIPROCAL: Callable[[int], int]
POWER: Callable[[int, int], int]
CONVOLVE: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
POLY_ROOTS: Callable[[npt.NDArray, npt.NDArray, int], npt.NDArray]
POLY_EVALUATE: Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
BERLEKAMP_MASSEY: Callable[[npt.NDArray], npt.NDArray]


class berlekamp_decode_jit(Function):
    """
    Performs general BCH and Reed-Solomon decoding.

    References:
        - Lin, S. and Costello, D. Error Control Coding. Section 7.4.
    """

    def __init__(self, field: Type[FieldArray], extension_field: Type[FieldArray]):
        super().__init__(field)
        self.extension_field = extension_field

    @property
    def key_1(self) -> Hashable:
        # Make the key in the cache lookup table specific to both the base field and extension field
        return (
            self.field.characteristic,
            self.field.degree,
            int(self.field.irreducible_poly),
            self.extension_field.characteristic,
            self.extension_field.degree,
            int(self.extension_field.irreducible_poly),
        )

    def __call__(
        self, codeword: FieldArray, design_n: int, alpha: int, c: int, roots: FieldArray
    ) -> tuple[FieldArray, npt.NDArray]:
        if self.extension_field.ufunc_mode != "python-calculate":
            output = self.jit(codeword.astype(np.int64), design_n, alpha, c, roots.astype(np.int64))
        else:
            output = self.python(codeword.view(np.ndarray), design_n, alpha, c, roots.view(np.ndarray))

        dec_codeword, N_errors = output[:, 0:-1], output[:, -1]
        dec_codeword = dec_codeword.astype(codeword.dtype)
        dec_codeword = dec_codeword.view(self.field)

        return dec_codeword, N_errors

    def set_globals(self) -> None:
        global CHARACTERISTIC, SUBTRACT, MULTIPLY, RECIPROCAL, POWER
        global CONVOLVE, POLY_ROOTS, POLY_EVALUATE, BERLEKAMP_MASSEY

        SUBTRACT = self.field._subtract.ufunc_call_only

        CHARACTERISTIC = self.extension_field.characteristic
        MULTIPLY = self.extension_field._multiply.ufunc_call_only
        RECIPROCAL = self.extension_field._reciprocal.ufunc_call_only
        POWER = self.extension_field._power.ufunc_call_only
        CONVOLVE = self.extension_field._convolve.function
        POLY_ROOTS = roots_jit(self.extension_field).function
        POLY_EVALUATE = evaluate_elementwise_jit(self.extension_field).function
        BERLEKAMP_MASSEY = berlekamp_massey_jit(self.extension_field).function

    _SIGNATURE = numba.types.FunctionType(int64[:, :](int64[:, :], int64, int64, int64, int64[:]))

    @staticmethod
    def implementation(
        codewords: npt.NDArray, design_n: int, alpha: int, c: int, roots: npt.NDArray
    ) -> npt.NDArray:  # pragma: no cover
        dtype = codewords.dtype
        N = codewords.shape[0]  # The number of codewords
        n = codewords.shape[1]  # The codeword size (could be less than the design n for shortened codes)
        d = roots.size + 1
        t = (d - 1) // 2  # Number of correctable errors

        # The last column of the returned decoded codeword is the number of corrected errors
        dec_codewords = np.zeros((N, n + 1), dtype=dtype)
        dec_codewords[:, 0:n] = codewords[:, :]

        for i in range(N):
            # Compute the syndrome by evaluating each codeword at the roots of the generator polynomial.
            # The syndrome vector is S = [S_1, S_2, ..., S_2t]
            syndrome = POLY_EVALUATE(codewords[i, :], roots)

            # If the syndrome is zero, then the codeword is a valid codeword and no errors need to be corrected.
            if np.all(syndrome == 0):
                continue

            # The error pattern is defined as the polynomial e(x) = e_j1*x^j1 + e_j2*x^j2 + ... for j1 to jν,
            # implying there are ν errors. δi = e_ji is the i-th error value, βi = α^ji is the i-th
            # error-location number, and ji is the error location.

            # The error-location polynomial σ(x) = (1 - β1*x)(1 - β2*x)...(1 - βν*x) where βi are the inverse of the
            # roots of σ(x).

            # Compute the error-location polynomial σ(x)
            # TODO: Re-evaluate these equations since changing BMA to return the characteristic polynomial,
            #       not the feedback polynomial
            sigma = BERLEKAMP_MASSEY(syndrome)[::-1]
            v = sigma.size - 1  # The number of errors, which is the degree of the error-locator polynomial

            if v > t:
                dec_codewords[i, -1] = -1
                continue

            # Compute βi^-1, the roots of σ(x)
            degrees = np.arange(sigma.size - 1, -1, -1)
            results = POLY_ROOTS(degrees, sigma, alpha)
            beta_inv = results[0, :]  # The roots βi^-1 of σ(x)
            error_locations_inv = results[1, :]  # The roots βi^-1 as powers of the primitive element α
            error_locations = -error_locations_inv % design_n  # The error locations as degrees of c(x)

            if np.any(error_locations > n - 1):
                # Indicates there are "errors" in the zero-ed portion of a shortened code, which indicates there are
                # actually more errors than alleged. Return failure to decode.
                dec_codewords[i, -1] = -1
                continue

            if beta_inv.size != v:
                dec_codewords[i, -1] = -1
                continue

            # Compute σ'(x)
            sigma_prime = np.zeros(v, dtype=dtype)
            for j in range(v):
                degree = v - j
                sigma_prime[j] = MULTIPLY(degree % CHARACTERISTIC, sigma[j])  # Scalar multiplication

            # The error-value evaluator polynomial Z0(x) = S0*σ0 + (S1*σ0 + S0*σ1)*x + (S2*σ0 + S1*σ1 + S0*σ2)*x^2 + ...
            # with degree v-1
            Z0 = CONVOLVE(sigma[-v:], syndrome[0:v][::-1])[-v:]

            # The error value δi = -1 * βi^(1-c) * Z0(βi^-1) / σ'(βi^-1)
            for j in range(v):
                beta_i = POWER(beta_inv[j], c - 1)
                # NOTE: poly_eval() expects a 1-D array of values
                Z0_i = POLY_EVALUATE(Z0, np.array([beta_inv[j]], dtype=dtype))[0]
                # NOTE: poly_eval() expects a 1-D array of values
                sigma_prime_i = POLY_EVALUATE(sigma_prime, np.array([beta_inv[j]], dtype=dtype))[0]
                delta_i = MULTIPLY(beta_i, Z0_i)
                delta_i = MULTIPLY(delta_i, RECIPROCAL(sigma_prime_i))
                delta_i = SUBTRACT(0, delta_i)
                dec_codewords[i, n - 1 - error_locations[j]] = SUBTRACT(
                    dec_codewords[i, n - 1 - error_locations[j]], delta_i
                )

            dec_codewords[i, -1] = v  # The number of corrected errors

        return dec_codewords
