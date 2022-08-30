"""
A module with functions for polynomials over Galois fields.
"""
from typing import Tuple

import numpy as np

from .._domains import Array
from .._helper import export, verify_isinstance

from ._poly import Poly


###############################################################################
# Divisibility
###############################################################################

def gcd(a: Poly, b: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.gcd()`.
    """
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    r2, r1 = a, b
    while r1 != 0:
        r2, r1 = r1, r2 % r1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 //= c

    return r2


def egcd(a: Poly, b: Poly) -> Tuple[Poly, Poly, Poly]:
    """
    This function is wrapped and documented in `_polymorphic.egcd()`.
    """
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    field = a.field
    zero = Poly.Zero(field)
    one = Poly.One(field)

    r2, r1 = a, b
    s2, s1 = one, zero
    t2, t1 = zero, one

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q*r1
        s2, s1 = s1, s2 - q*s1
        t2, t1 = t1, t2 - q*t1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 //= c
        s2 //= c
        t2 //= c

    return r2, s2, t2


def lcm(*args: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.lcm()`.
    """
    field = args[0].field

    lcm_  = Poly.One(field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}.")
        lcm_ = (lcm_ * arg) // gcd(lcm_, arg)

    # Make the LCM monic
    lcm_ //= lcm_.coeffs[0]

    return lcm_


def prod(*args: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.prod()`.
    """
    field = args[0].field

    prod_  = Poly.One(field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}.")
        prod_ *= arg

    return prod_


###############################################################################
# Special polynomials
###############################################################################

@export
def lagrange_poly(x: Array, y: Array) -> Poly:
    r"""
    Computes the Lagrange interpolating polynomial :math:`L(x)` such that :math:`L(x_i) = y_i`.

    :group: polys-interpolating

    Parameters
    ----------
    x
        An array of :math:`x_i` values for the coordinates :math:`(x_i, y_i)`. Must be 1-D. Must have no
        duplicate entries.
    y
        An array of :math:`y_i` values for the coordinates :math:`(x_i, y_i)`. Must be 1-D. Must be the same
        size as :math:`x`.

    Returns
    -------
    :
        The Lagrange polynomial :math:`L(x)`.

    Notes
    -----
    The Lagrange interpolating polynomial is defined as

    .. math::
        L(x) = \sum_{j=0}^{k-1} y_j \ell_j(x)

    .. math::
        \ell_j(x) = \prod_{\substack{0 \le m < k \\ m \ne j}} \frac{x - x_m}{x_j - x_m} .

    It is the polynomial of minimal degree that satisfies :math:`L(x_i) = y_i`.

    References
    ----------
    * https://en.wikipedia.org/wiki/Lagrange_polynomial

    Examples
    --------
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
    """
    verify_isinstance(x, Array)
    verify_isinstance(y, Array)
    if not type(x) == type(y):  # pylint: disable=unidiomatic-typecheck
        raise TypeError(f"Arguments `x` and `y` must be over the same Galois field, not {type(x)} and {type(y)}.")
    if not x.ndim == 1:
        raise ValueError(f"Argument `x` must be 1-D, not have shape {x.shape}.")
    if not y.ndim == 1:
        raise ValueError(f"Argument `y` must be 1-D, not have shape {y.shape}.")
    if not x.size == y.size:
        raise ValueError(f"Arguments `x` and `y` must be the same size, not {x.size} and {y.size}.")
    if not x.size == np.unique(x).size:
        raise ValueError(f"Argument `x` must have unique entries, not {x}.")

    field = type(x)
    L = Poly.Zero(field)  # The Lagrange polynomial L(x)
    k = x.size  # The number of coordinates

    for j in range(k):
        lj = Poly.One(field)
        for m in range(k):
            if m == j:
                continue
            lj *= Poly([1, -x[m]], field=field) // (x[j] - x[m])
        L += y[j] * lj

    return L
