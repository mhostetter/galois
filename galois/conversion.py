import math

import numpy as np


def integer_to_poly(decimal, order):
    """
    Convert decimal value into polynomial representation.

    Parameters
    ----------
    decimal : int
        Any non-negative integer.
    order : int
        The order of coefficient field.

    Returns
    -------
    list
        List of polynomial coefficients in ascending order.
    """
    decimal = int(decimal)
    if decimal > 0:
        degree = int(math.ceil(math.log(decimal, order)))
    else:
        degree = 0

    c = []  # Coefficients in descending order
    for d in range(degree, -1, -1):
        c += [decimal // order**d]
        decimal = decimal % order**d

    c = c[::-1] # Coefficients in ascending order

    return c


def poly_to_integer(coeffs, order):
    """
    Converts polynomial to decimal representation.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients in ascending order.
    order : int
        The coefficient's field order.

    Returns
    -------
    int
        The decimal representation.
    """
    decimal = 0
    for i in range(coeffs.size):
        decimal += int(coeffs[i]) * order**i
    return decimal


def poly_to_str(coeffs, poly_var="x"):
    """
    Convert list of polynomial coefficients into polynomial string representation.

    Parameters
    ----------
    coeffs : array_like
        List of exponent-ascending coefficients.
    poly_var : str, optional
        The variable to use in the polynomial string. The default is `"x"`.

    Returns
    -------
    str
        The polynomial string representation.
    """
    idxs = np.nonzero(coeffs)[0]
    if idxs.size == 0:
        degree = 0
    else:
        degree = idxs[-1]

    x = []
    if degree >= 0 and coeffs[0] != 0:
        x += ["{}".format(coeffs[0])]
    if degree >= 1 and coeffs[1] != 0:
        x += ["{}{}".format(coeffs[1] if coeffs[1] != 1 else "", poly_var)]
    if degree >= 2:
        idxs = np.nonzero(coeffs[2:])[0]  # Indices with non-zeros coefficients
        x += ["{}{}^{}".format(coeffs[2+i] if coeffs[2+i] != 1 else "", poly_var, 2+i) for i in idxs]

    poly_str = " + ".join(x[::-1]) if x else "0"

    return poly_str
