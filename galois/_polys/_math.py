"""
A module of various functions on polynomials, such as GCD and modular exponentiation.
"""
from ._poly import Poly


###############################################################################
# Divisibility
###############################################################################

def gcd(a, b):
    """
    This function is wrapped and documented in `_polymorphic.gcd()`.
    """
    if not a.field is b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {a.field} and {b.field}.")

    field = a.field
    zero = Poly.Zero(field)

    r2, r1 = a, b

    while r1 != zero:
        r2, r1 = r1, r2 % r1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 /= c

    return r2


def egcd(a, b):
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

    while r1 != zero:
        q = r2 / r1
        r2, r1 = r1, r2 - q*r1
        s2, s1 = s1, s2 - q*s1
        t2, t1 = t1, t2 - q*t1

    # Make the GCD polynomial monic
    c = r2.coeffs[0]  # The leading coefficient
    if c > 1:
        r2 /= c
        s2 /= c
        t2 /= c

    return r2, s2, t2


def lcm(*args):
    """
    This function is wrapped and documented in `_polymorphic.lcm()`.
    """
    field = args[0].field
    lcm_  = Poly.One(field)
    for arg in args:
        if not arg.field == field:
            raise ValueError(f"All polynomial arguments must be over the same field, not {[arg.field for arg in args]}.")
        lcm_ = (lcm_ * arg) // gcd(lcm_, arg)
    return lcm_


def prod(*args):
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
# Congruences
###############################################################################

def pow(base, exponent, modulus):  # pylint: disable=redefined-builtin
    """
    This function is wrapped and documented in `_polymorphic.pow()`.
    """
    if not base.field is modulus.field:
        raise ValueError(f"Arguments `base` and `modulus` must be polynomials over the same Galois field, not {base.field} and {modulus.field}.")

    if exponent == 0:
        return Poly.One(base.field)

    result_s = base  # The "squaring" part
    result_m = Poly.One(base.field)  # The "multiplicative" part

    while exponent > 1:
        if exponent % 2 == 0:
            result_s = (result_s * result_s) % modulus
            exponent //= 2
        else:
            result_m = (result_m * result_s) % modulus
            exponent -= 1

    result = (result_s * result_m) % modulus

    return result
