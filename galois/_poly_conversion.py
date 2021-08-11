"""
A module that contains various functions to convert between polynomial strings, coefficients, and integer representations.
These functions are shared between the _field and _poly subpackages.
"""
import math

import numpy as np


def integer_to_poly(decimal, order, degree=None):
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
        List of polynomial coefficients in descending order.
    """
    decimal = int(decimal)
    if degree is None:
        if decimal > 0:
            degree = int(math.floor(math.log(decimal, order)))
            # math.log() is notoriously wrong, need to manually check that it isn't wrong
            if decimal < order**degree:
                degree -= 1
            if decimal >= order**(degree + 1):
                degree += 1
        else:
            degree = 0

    c = []  # Coefficients in descending order
    for d in range(degree, -1, -1):
        c += [decimal // order**d]
        decimal = decimal % order**d

    return c


def poly_to_integer(coeffs, order):
    """
    Converts polynomial to decimal representation.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients in descending order.
    order : int
        The coefficient's field order.

    Returns
    -------
    int
        The decimal representation.
    """
    decimal = 0
    coeffs = coeffs[::-1]  # Coefficients in ascending order
    for i in range(len(coeffs)):
        decimal += int(coeffs[i]) * order**i
    return decimal


def sparse_poly_to_integer(degrees, coeffs, order):
    """
    Converts polynomial to decimal representation.

    Parameters
    ----------
    degrees : array_like
        List of degrees of non-zero coefficients.
    coeffs : array_like
        List of non-zero coefficients.
    order : int
        The coefficient's field order.

    Returns
    -------
    int
        The decimal representation.
    """
    assert len(degrees) == len(coeffs)
    order = int(order)
    decimal = 0
    for d, c in zip(degrees, coeffs):
        decimal += int(c) * order**int(d)
    return decimal


def poly_to_str(coeffs, poly_var="x"):
    """
    Convert list of polynomial coefficients into polynomial string representation.

    Parameters
    ----------
    coeffs : array_like
        List of exponent-descending coefficients.
    poly_var : str, optional
        The variable to use in the polynomial string. The default is `"x"`.

    Returns
    -------
    str
        The polynomial string representation.
    """
    coeffs = coeffs[::-1]  # Coefficients in ascending order

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


def sparse_poly_to_str(degrees, coeffs, poly_var="x"):
    """
    Convert list of polynomial degrees and coefficients into polynomial string representation.

    Parameters
    ----------
    degrees : array_like
        List of degrees.
    coeffs : array_like
        List of coefficients.
    poly_var : str, optional
        The variable to use in the polynomial string. The default is `"x"`.

    Returns
    -------
    str
        The polynomial string representation.
    """
    x = []
    for degree, coeff in zip(degrees, coeffs):
        if hasattr(coeff, "_display_mode") and getattr(coeff, "_display_mode") in ["poly", "power"]:
            # This is a Galois field array coefficient using the polynomial or power representation
            coeff_repr = repr(coeff)
            start = coeff_repr.find("(")
            stop = coeff_repr.find(",")
            coeff_repr = coeff_repr[start:stop] + ")"
        else:
            coeff_repr = coeff

        if degree > 1:
            s = "{}{}^{}".format(coeff_repr if coeff != 1 else "", poly_var, degree)
        elif degree == 1:
            s = "{}{}".format(coeff_repr if coeff != 1 else "", poly_var)
        elif coeff != 0:
            s = "{}".format(coeff_repr)
        else:
            continue
        x.append(s)

    poly_str = " + ".join(x) if x else "0"

    return poly_str


def str_to_sparse_poly(poly_str):
    """
    Convert a polynomial string to its non-zero degrees and coefficients.

    Parameters
    ----------
    poly_str : str
        A polynomial representation of the string.

    Returns
    -------
    list
        The polynomial non-zero degrees.
    list
        The polynomial non-zero coefficients.
    """
    s = poly_str.replace(" ", "")  # Strip whitespace for easier processing
    s = s.replace("-", "+-")  # Convert `x^2 - x` into `x^2 + -x`
    s = s.replace("++", "+")  # Places that were `x^2 + - x` before the previous line are now `x^2 + + -x`, fix them
    s = s[1:] if s[0] == "+" else s  # Remove added `+` to a negative leading coefficient

    # Find the unique polynomial indeterminate
    indeterminates = set(c for c in s if c.isalpha())
    if len(indeterminates) == 0:
        var = "x"
    elif len(indeterminates) == 1:
        var = list(indeterminates)[0]
    else:
        raise ValueError(f"Found multiple polynomial indeterminates {vars} in string {poly_str!r}.")

    degrees = []
    coeffs = []
    for element in s.split("+"):
        if var not in element:
            degree = 0
            coeff = element
        elif "^" not in element and "**" not in element:
            degree = 1
            coeff = element.split(var, 1)[0]
        elif "^" in element:
            coeff, degree = element.split(var + "^", 1)
        elif "**" in element:
            coeff, degree = element.split(var + "**", 1)
        else:
            raise ValueError(f"Could not parse polynomial degree in {element}.")

        # If the degree was negative `3*x^-2`, it was converted to `3*x^+-2` previously. When split
        # by "+" it will leave the degree empty.
        if degree == "":
            raise ValueError(f"Cannot parse polynomials with negative exponents, {poly_str!r}.")
        degree = int(degree)

        coeff = coeff.replace("*", "")  # Remove multiplication sign for elements like `3*x^2`
        coeff = int(coeff) if coeff != "" else 1

        degrees.append(degree)
        coeffs.append(coeff)

    return degrees, coeffs


def str_to_integer(poly_str, prime_subfield):
    """
    Convert a polynomial string to its integer representation.

    Parameters
    ----------
    poly_str : str
        A polynomial representation of the string.
    prime_subfield : galois.FieldClass
        The Galois field the polynomial is over.

    Returns
    -------
    int
        The polynomial integer representation.
    """
    degrees, coeffs = str_to_sparse_poly(poly_str)

    integer = 0
    for degree, coeff in zip(degrees, coeffs):
        coeff = coeff if coeff >= 0 else int(-prime_subfield(abs(coeff)))
        integer += coeff * prime_subfield.order**degree

    return integer
