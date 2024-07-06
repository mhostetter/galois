"""
A module that contains various functions to convert between polynomial strings, coefficients, and integer
representations.
"""

from __future__ import annotations

import numpy as np

from .._math import ilog
from .._options import get_printoptions


def integer_to_degree(integer: int, order: int) -> int:
    """
    Converts the integer representation of the polynomial to its degree.
    """
    if order == 2:
        return max(integer.bit_length() - 1, 0)

    return ilog(integer, order)


def integer_to_poly(integer: int, order: int, degree: int | None = None) -> list[int]:
    """
    Converts the integer representation of the polynomial to its coefficients in descending order.
    """
    if order == 2:
        c = [int(bit) for bit in bin(integer)[2:]]
    else:
        c = []  # Coefficients in ascending order
        while integer > 0:
            q, r = divmod(integer, order)
            c.append(r)
            integer = q

        # Ensure the coefficient list is not empty
        if not c:
            c = [0]

        c = c[::-1]  # Coefficients in descending order

    # Set to a fixed degree if requested
    if degree is not None:
        assert degree >= len(c) - 1
        c = [0] * (degree - len(c) + 1) + c

    return c


def poly_to_integer(coeffs: list[int], order: int) -> int:
    """
    Converts the polynomial coefficients (descending order) to its integer representation.
    """
    integer = 0
    coeffs = coeffs[::-1]  # Coefficients in ascending order
    factor = 1

    for coeff in coeffs:
        integer += int(coeff) * factor
        factor *= order

    return integer


def sparse_poly_to_integer(degrees: list[int], coeffs: list[int], order: int) -> int:
    """
    Converts the polynomial non-zero degrees and coefficients to its integer representation.
    """
    assert len(degrees) == len(coeffs)

    integer = 0
    for d, c in zip(degrees, coeffs):
        integer += int(c) * order ** int(d)

    return integer


def poly_to_str(coeffs: list[int], poly_var: str = "x") -> str:
    """
    Converts the polynomial coefficients (descending order) into its string representation.
    """
    degrees = np.arange(len(coeffs) - 1, -1, -1)

    return sparse_poly_to_str(degrees, coeffs, poly_var=poly_var)


def sparse_poly_to_str(degrees: list[int], coeffs: list[int], poly_var: str = "x") -> str:
    """
    Converts the polynomial non-zero degrees and coefficients into its string representation.
    """
    x = []

    # Use brackets around coefficients only when using the "poly" or "power" element representation
    brackets = hasattr(type(coeffs), "_element_repr") and type(coeffs)._element_repr in ["poly", "power"]

    # If the element representation is polynomial and the irreducible polynomial is not primitive, convert x to z
    # for the coefficients
    convert_x_to_z = (
        hasattr(type(coeffs), "_element_repr")
        and type(coeffs)._element_repr == "poly"
        and not type(coeffs).is_primitive_poly
    )

    for degree, coeff in zip(degrees, coeffs):
        if coeff == 0:
            continue

        if convert_x_to_z and coeff != 1:
            coeff = str(coeff).replace("x", "z")

        if degree > 1:
            if coeff == 1:
                c = ""
            elif brackets:
                c = f"({coeff!s})"
            else:
                c = f"{coeff!s}"
            s = f"{c}{poly_var}^{degree}"
        elif degree == 1:
            if coeff == 1:
                c = ""
            elif brackets:
                c = f"({coeff!s})"
            else:
                c = f"{coeff!s}"
            s = f"{c}{poly_var}"
        else:
            # Use () around 0-degree term only if it has a + separator in it ("poly" element representation)
            if "+" in str(coeff):
                s = f"({coeff!s})"
            else:
                s = f"{coeff!s}"

        x.append(s)

    if get_printoptions()["coeffs"] == "asc":
        x = x[::-1]

    return " + ".join(x) if x else "0"


def str_to_sparse_poly(poly_str: str) -> tuple[list[int], list[int]]:
    """
    Converts the polynomial string into its non-zero degrees and coefficients.
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
        if coeff == "-":
            coeff = -1
        elif coeff != "":
            coeff = int(coeff)
        else:
            coeff = 1

        degrees.append(degree)
        coeffs.append(coeff)

    return degrees, coeffs


def str_to_integer(poly_str: str, prime_subfield) -> int:
    """
    Converts the polynomial string to its integer representation.
    """
    degrees, coeffs = str_to_sparse_poly(poly_str)

    integer = 0
    for degree, coeff in zip(degrees, coeffs):
        coeff = coeff if coeff >= 0 else int(-prime_subfield(abs(coeff)))
        integer += coeff * prime_subfield.order**degree

    return integer
