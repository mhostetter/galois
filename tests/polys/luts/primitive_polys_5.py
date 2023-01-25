"""
A module containing LUTs for primitive polynomials over GF(5).

Sage:
    def integer(coeffs, order):
        i = 0
        for d, c in enumerate(coeffs[::-1]):
            i += (c.integer_representation() * order**d)
        return i

    order = 5
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

PRIMITIVE_POLYS_5_1 = [
    [1, 2],
    [1, 3],
]

PRIMITIVE_POLYS_5_2 = [
    [1, 1, 2],
    [1, 2, 3],
    [1, 3, 3],
    [1, 4, 2],
]

PRIMITIVE_POLYS_5_3 = [
    [1, 0, 3, 2],
    [1, 0, 3, 3],
    [1, 0, 4, 2],
    [1, 0, 4, 3],
    [1, 1, 0, 2],
    [1, 1, 1, 3],
    [1, 1, 4, 3],
    [1, 2, 0, 3],
    [1, 2, 1, 3],
    [1, 2, 2, 2],
    [1, 2, 2, 3],
    [1, 2, 4, 2],
    [1, 3, 0, 2],
    [1, 3, 1, 2],
    [1, 3, 2, 2],
    [1, 3, 2, 3],
    [1, 3, 4, 3],
    [1, 4, 0, 3],
    [1, 4, 1, 2],
    [1, 4, 4, 2],
]

PRIMITIVE_POLYS_5_4 = [
    [1, 0, 1, 2, 2],
    [1, 0, 1, 2, 3],
    [1, 0, 1, 3, 2],
    [1, 0, 1, 3, 3],
    [1, 0, 4, 1, 2],
    [1, 0, 4, 1, 3],
    [1, 0, 4, 4, 2],
    [1, 0, 4, 4, 3],
    [1, 1, 0, 1, 3],
    [1, 1, 0, 2, 3],
    [1, 1, 0, 3, 2],
    [1, 1, 0, 4, 2],
    [1, 1, 1, 1, 3],
    [1, 1, 2, 0, 2],
    [1, 1, 2, 1, 2],
    [1, 1, 3, 0, 3],
    [1, 1, 3, 4, 2],
    [1, 1, 4, 4, 3],
    [1, 2, 0, 1, 3],
    [1, 2, 0, 2, 2],
    [1, 2, 0, 3, 3],
    [1, 2, 0, 4, 2],
    [1, 2, 1, 2, 3],
    [1, 2, 2, 0, 3],
    [1, 2, 2, 2, 2],
    [1, 2, 3, 0, 2],
    [1, 2, 3, 3, 2],
    [1, 2, 4, 3, 3],
    [1, 3, 0, 1, 2],
    [1, 3, 0, 2, 3],
    [1, 3, 0, 3, 2],
    [1, 3, 0, 4, 3],
    [1, 3, 1, 3, 3],
    [1, 3, 2, 0, 3],
    [1, 3, 2, 3, 2],
    [1, 3, 3, 0, 2],
    [1, 3, 3, 2, 2],
    [1, 3, 4, 2, 3],
    [1, 4, 0, 1, 2],
    [1, 4, 0, 2, 2],
    [1, 4, 0, 3, 3],
    [1, 4, 0, 4, 3],
    [1, 4, 1, 4, 3],
    [1, 4, 2, 0, 2],
    [1, 4, 2, 4, 2],
    [1, 4, 3, 0, 3],
    [1, 4, 3, 1, 2],
    [1, 4, 4, 1, 3],
]
