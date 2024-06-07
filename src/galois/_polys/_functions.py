"""
A module with functions for polynomials over Galois fields.
"""

from __future__ import annotations

from ._poly import Poly


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


def egcd(a: Poly, b: Poly) -> tuple[Poly, Poly, Poly]:
    """
    This function is wrapped and documented in `_polymorphic.egcd()`.
    """
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    field = a.field
    zero = Poly([0], field=field)
    one = Poly([1], field=field)

    r2, r1 = a, b
    s2, s1 = one, zero
    t2, t1 = zero, one

    while r1 != 0:
        q = r2 // r1
        r2, r1 = r1, r2 - q * r1
        s2, s1 = s1, s2 - q * s1
        t2, t1 = t1, t2 - q * t1

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

    lcm_ = Poly([1], field=field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(
                f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}."
            )
        lcm_ = (lcm_ * arg) // gcd(lcm_, arg)

    # Make the LCM monic
    lcm_ //= lcm_.coeffs[0]

    return lcm_


def prod(*args: Poly) -> Poly:
    """
    This function is wrapped and documented in `_polymorphic.prod()`.
    """
    field = args[0].field

    prod_ = Poly([1], field=field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(
                f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}."
            )
        prod_ *= arg

    return prod_
