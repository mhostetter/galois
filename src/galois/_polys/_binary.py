"""
A module containing polynomial arithmetic for polynomials over GF(2).
"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """
    c(x) = a(x) + b(x)
    """
    return a ^ b


def negative(a: int) -> int:
    """
    c(x) = -a(x)
    a(x) + -a(x) = 0
    """
    return a


def subtract(a: int, b: int) -> int:
    """
    c(x) = a(x) - b(x)
    """
    return a ^ b


def multiply(a: int, b: int) -> int:
    """
    c(x) = a(x) * b(x)
    c(x) = a(x) * b = a(x) + ... + a(x)
    """
    # Re-order operands such that a > b so the while loop has less loops
    if b > a:
        a, b = b, a

    c = 0
    while b > 0:
        if b & 0b1:
            c ^= a  # Add a(x) to c(x)
        b >>= 1  # Divide b(x) by x
        a <<= 1  # Multiply a(x) by x

    return c


def divmod(a: int, b: int) -> tuple[int, int]:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    deg_a = max(a.bit_length() - 1, 0)
    deg_b = max(b.bit_length() - 1, 0)
    deg_q = deg_a - deg_b
    deg_r = deg_b - 1

    q = 0
    mask = 1 << deg_a
    for i in range(deg_q, -1, -1):
        q <<= 1
        if a & mask:
            a ^= b << i
            q ^= 1  # Set the LSB then left shift
        assert a & mask == 0
        mask >>= 1

    # q = a >> deg_r
    mask = (1 << (deg_r + 1)) - 1  # The last deg_r + 1 bits of a
    r = a & mask

    return q, r


def floordiv(a: int, b: int) -> int:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    # TODO: Make more efficient?
    return divmod(a, b)[0]


def mod(a: int, b: int) -> int:
    """
    a(x) = q(x)*b(x) + r(x)
    """
    # TODO: Make more efficient?
    return divmod(a, b)[1]


def pow(a: int, b: int, c: int | None = None) -> int:
    """
    d(x) = a(x)^b % c(x)
    """
    if b == 0:
        return 1

    result_s = a  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    if c:
        while b > 1:
            if b % 2 == 0:
                result_s = mod(multiply(result_s, result_s), c)
                b //= 2
            else:
                result_m = mod(multiply(result_m, result_s), c)
                b -= 1

        result = mod(multiply(result_s, result_m), c)
    else:
        while b > 1:
            if b % 2 == 0:
                result_s = multiply(result_s, result_s)
                b //= 2
            else:
                result_m = multiply(result_m, result_s)
                b -= 1

        result = multiply(result_s, result_m)

    return result
