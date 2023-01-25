"""
A module containing functions for interpolation polynomials over finite fields.
"""
from __future__ import annotations

import numba
import numpy as np
from numba import int64

from .._domains import Array
from .._domains._function import Function
from .._helper import export, verify_isinstance
from ._dense import add_jit, floordiv_jit
from ._poly import Poly


@export
def lagrange_poly(x: Array, y: Array) -> Poly:
    r"""
    Computes the Lagrange interpolating polynomial :math:`L(x)` such that :math:`L(x_i) = y_i`.

    Arguments:
        x: An array of :math:`x_i` values for the coordinates :math:`(x_i, y_i)`. Must be 1-D. Must have no
            duplicate entries.
        y: An array of :math:`y_i` values for the coordinates :math:`(x_i, y_i)`. Must be 1-D. Must be the same
            size as :math:`x`.

    Returns:
        The Lagrange polynomial :math:`L(x)`.

    Notes:
        The Lagrange interpolating polynomial is defined as

        .. math::
            L(x) = \sum_{j=0}^{k-1} y_j \ell_j(x)

        .. math::
            \ell_j(x) = \prod_{\substack{0 \le m < k \\ m \ne j}} \frac{x - x_m}{x_j - x_m} .

        It is the polynomial of minimal degree that satisfies :math:`L(x_i) = y_i`.

    References:
        - https://en.wikipedia.org/wiki/Lagrange_polynomial

    Examples:
        Create random :math:`(x, y)` pairs in :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2)
            x = GF.elements; x
            y = GF.Random(x.size); y

        Find the Lagrange polynomial that interpolates the coordinates.

        .. ipython:: python

            L = galois.lagrange_poly(x, y); L

        Show that the polynomial evaluated at :math:`x` is :math:`y`.

        .. ipython:: python

            np.array_equal(L(x), y)

    Group:
        polys-interpolating
    """
    verify_isinstance(x, Array)
    verify_isinstance(y, Array)

    coeffs = lagrange_poly_jit(type(x))(x, y)
    poly = Poly(coeffs)

    return poly


class lagrange_poly_jit(Function):
    """
    Finds the roots of the polynomial f(x).
    """

    def __call__(self, x: Array, y: Array) -> Array:
        verify_isinstance(x, Array)
        verify_isinstance(y, Array)
        if not type(x) == type(y):  # pylint: disable=unidiomatic-typecheck
            raise TypeError(f"Arguments 'x' and 'y' must be over the same Galois field, not {type(x)} and {type(y)}.")
        if not x.ndim == 1:
            raise ValueError(f"Argument 'x' must be 1-D, not have shape {x.shape}.")
        if not y.ndim == 1:
            raise ValueError(f"Argument 'y' must be 1-D, not have shape {y.shape}.")
        if not x.size == y.size:
            raise ValueError(f"Arguments 'x' and 'y' must be the same size, not {x.size} and {y.size}.")
        if not x.size == np.unique(x).size:
            raise ValueError(f"Argument 'x' must have unique entries, not {x}.")
        dtype = x.dtype

        if self.field.ufunc_mode != "python-calculate":
            coeffs = self.jit(x.astype(np.int64), y.astype(np.int64))
            coeffs = coeffs.astype(dtype)
        else:
            coeffs = self.python(x.view(np.ndarray), y.view(np.ndarray))
        coeffs = self.field._view(coeffs)

        return coeffs

    def set_globals(self):
        # pylint: disable=global-variable-undefined
        global NEGATIVE, SUBTRACT, MULTIPLY, RECIPROCAL, POLY_ADD, POLY_FLOORDIV, POLY_MULTIPLY
        NEGATIVE = self.field._negative.ufunc_call_only
        SUBTRACT = self.field._subtract.ufunc_call_only
        MULTIPLY = self.field._multiply.ufunc_call_only
        RECIPROCAL = self.field._reciprocal.ufunc_call_only
        POLY_ADD = add_jit(self.field).function
        POLY_FLOORDIV = floordiv_jit(self.field).function
        POLY_MULTIPLY = self.field._convolve.function

    _SIGNATURE = numba.types.FunctionType(int64[:](int64[:], int64[:]))

    @staticmethod
    def implementation(x, y):  # pragma: no cover
        dtype = x.dtype
        L = np.array([0], dtype=dtype)  # The Lagrange polynomial L(x) = 0
        k = x.size  # The number of coordinates

        for j in range(k):
            lj = np.array([y[j]], dtype=dtype)
            for m in range(k):
                if m == j:
                    continue
                ljm = MULTIPLY(np.array([1, NEGATIVE(x[m])], dtype=dtype), RECIPROCAL(SUBTRACT(x[j], x[m])))
                lj = POLY_MULTIPLY(lj, ljm)
            L = POLY_ADD(L, lj)

        return L
