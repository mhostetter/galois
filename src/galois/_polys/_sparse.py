"""
A module containing polynomial arithmetic for polynomials with sparse coefficients.
"""

from __future__ import annotations

import numpy as np

from .._domains import Array


def add(
    a_degrees: np.ndarray,
    a_coeffs: Array,
    b_degrees: np.ndarray,
    b_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = a(x) + b(x)
    """
    field = type(a_coeffs)

    c = dict(zip(a_degrees, a_coeffs))
    for b_degree, b_coeff in zip(b_degrees, b_coeffs):
        c[b_degree] = c.get(b_degree, field(0)) + b_coeff

    return np.array(list(c.keys())), field(list(c.values()))


def negative(
    a_degrees: np.ndarray,
    a_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    return a_degrees, -a_coeffs


def subtract(
    a_degrees: np.ndarray,
    a_coeffs: Array,
    b_degrees: np.ndarray,
    b_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = a(x) - b(x)
    """
    field = type(a_coeffs)

    # c(x) = a(x) - b(x)
    c = dict(zip(a_degrees, a_coeffs))
    for b_degree, b_coeff in zip(b_degrees, b_coeffs):
        c[b_degree] = c.get(b_degree, field(0)) - b_coeff

    return np.array(list(c.keys())), field(list(c.values()))


def multiply(
    a_degrees: np.ndarray,
    a_coeffs: Array,
    b_degrees: np.ndarray,
    b_coeffs: Array,
) -> tuple[np.ndarray, Array]:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    # c(x) = a(x) * b(x)
    field = type(a_coeffs)

    c = {}
    for a_degree, a_coeff in zip(a_degrees, a_coeffs):
        for b_degree, b_coeff in zip(b_degrees, b_coeffs):
            c[a_degree + b_degree] = c.get(a_degree + b_degree, field(0)) + a_coeff * b_coeff

    return np.array(list(c.keys())), field(list(c.values()))
