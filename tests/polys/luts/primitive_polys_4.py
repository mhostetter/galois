"""
A module containing LUTs for primitive polynomials over GF(2^2).

Sage:
    def integer(coeffs, order):
        i = 0
        for d, c in enumerate(coeffs[::-1]):
            i += (c.integer_representation() * order**d)
        return i

    order = 2**2
    degree = 1
    list_ = []
    R = GF(order, repr="int")["x"]
    for f in R.polynomials(degree):
        # For some reason `is_primitive()` crashes on f(x) = x
        if f.coefficients(sparse=False) == [0, 1]:
            continue
        if f.is_monic() and f.is_primitive():
            list_.append(f.coefficients(sparse=False)[::-1])

    # Sort in lexicographical order
    if not is_prime(order):
        list_ = sorted(list_, key=lambda item: integer(item, order))

    print(f"PRIMITIVE_POLYS_{order}_{degree} = {list_}")
"""

# LUT items are poly coefficients in degree-descending order

PRIMITIVE_POLYS_4_1 = [
    [1, 2],
    [1, 3],
]

PRIMITIVE_POLYS_4_2 = [
    [1, 1, 2],
    [1, 1, 3],
    [1, 2, 2],
    [1, 3, 3],
]

PRIMITIVE_POLYS_4_3 = [
    [1, 1, 1, 2],
    [1, 1, 1, 3],
    [1, 1, 2, 3],
    [1, 1, 3, 2],
    [1, 2, 1, 3],
    [1, 2, 2, 2],
    [1, 2, 3, 2],
    [1, 2, 3, 3],
    [1, 3, 1, 2],
    [1, 3, 2, 2],
    [1, 3, 2, 3],
    [1, 3, 3, 3],
]
