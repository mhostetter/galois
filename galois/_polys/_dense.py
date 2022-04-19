"""
A module containing polynomial arithmetic for polynomials with dense coefficients.
"""
from typing import Tuple, Optional

import numpy as np

from .._domains import Array


def add(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) + b(x)
    """
    field = type(a)

    c = field.Zeros(max(a.size, b.size))
    c[-a.size:] = a
    c[-b.size:] += b

    return c


def neg(a: Array) -> Array:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    return -a


def sub(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) - b(x)
    """
    field = type(a)

    # c(x) = a(x) - b(x)
    c = field.Zeros(max(a.size, b.size))
    c[-a.size:] = a
    c[-b.size:] -= b

    return c


def mul(a: Array, b: Array) -> Array:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    # c(x) = a(x) * b(x)
    if a.ndim == 0 or b.ndim == 0:
        return a * b
    else:
        return np.convolve(a, b)


def divmod(a: Array, b: Array) -> Tuple[Array, Array]:  # pylint: disable=redefined-builtin
    """
    a(x) = q(x)*b(x) + r(x)
    """
    field = type(a)

    a_degree = a.size - 1
    b_degree = b.size - 1

    if b_degree == 0:
        q, r = a // b, field([0])
    elif a_degree == 0 and a[0] == 0:
        q, r = field([0]), field([0])
    elif a_degree < b_degree:
        q, r = field([0]), a
    else:
        q, r = field._poly_divmod(a, b)

    return q, r


def floordiv(a: Array, b: Array) -> Array:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    field = type(a)
    q = field._poly_floordiv(a, b)
    return q


def mod(a: Array, b: Array) -> Array:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    field = type(a)
    r = field._poly_mod(a, b)
    return r


def pow(a: Array, b: int, c: Optional[Array] = None) -> Array:  # pylint: disable=redefined-builtin
    """
    d(x) = a(x)^b % c(x)
    """
    field = type(a)
    d = field._poly_pow(a, b, c)
    return d
