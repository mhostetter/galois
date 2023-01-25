"""
A module containing LUTs for primitive polynomials over GF(5^2).

Sage:
    def integer(coeffs, order):
        i = 0
        for d, c in enumerate(coeffs[::-1]):
            i += (c.integer_representation() * order**d)
        return i

    order = 5**2
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

PRIMITIVE_POLYS_25_1 = [
    [1, 5],
    [1, 9],
    [1, 10],
    [1, 13],
    [1, 15],
    [1, 17],
    [1, 20],
    [1, 21],
]

PRIMITIVE_POLYS_25_2 = [
    [1, 1, 9],
    [1, 1, 13],
    [1, 1, 15],
    [1, 1, 20],
    [1, 2, 5],
    [1, 2, 10],
    [1, 2, 17],
    [1, 2, 21],
    [1, 3, 5],
    [1, 3, 10],
    [1, 3, 17],
    [1, 3, 21],
    [1, 4, 9],
    [1, 4, 13],
    [1, 4, 15],
    [1, 4, 20],
    [1, 5, 5],
    [1, 5, 13],
    [1, 5, 15],
    [1, 5, 17],
    [1, 6, 9],
    [1, 6, 15],
    [1, 6, 20],
    [1, 6, 21],
    [1, 7, 5],
    [1, 7, 13],
    [1, 7, 15],
    [1, 7, 21],
    [1, 8, 5],
    [1, 8, 9],
    [1, 8, 13],
    [1, 8, 20],
    [1, 9, 10],
    [1, 9, 13],
    [1, 9, 15],
    [1, 9, 21],
    [1, 10, 10],
    [1, 10, 13],
    [1, 10, 17],
    [1, 10, 20],
    [1, 11, 5],
    [1, 11, 17],
    [1, 11, 20],
    [1, 11, 21],
    [1, 12, 5],
    [1, 12, 9],
    [1, 12, 10],
    [1, 12, 21],
    [1, 13, 9],
    [1, 13, 10],
    [1, 13, 15],
    [1, 13, 17],
    [1, 14, 9],
    [1, 14, 10],
    [1, 14, 17],
    [1, 14, 20],
    [1, 15, 10],
    [1, 15, 13],
    [1, 15, 17],
    [1, 15, 20],
    [1, 16, 9],
    [1, 16, 10],
    [1, 16, 17],
    [1, 16, 20],
    [1, 17, 9],
    [1, 17, 10],
    [1, 17, 15],
    [1, 17, 17],
    [1, 18, 5],
    [1, 18, 9],
    [1, 18, 10],
    [1, 18, 21],
    [1, 19, 5],
    [1, 19, 17],
    [1, 19, 20],
    [1, 19, 21],
    [1, 20, 5],
    [1, 20, 13],
    [1, 20, 15],
    [1, 20, 17],
    [1, 21, 10],
    [1, 21, 13],
    [1, 21, 15],
    [1, 21, 21],
    [1, 22, 5],
    [1, 22, 9],
    [1, 22, 13],
    [1, 22, 20],
    [1, 23, 5],
    [1, 23, 13],
    [1, 23, 15],
    [1, 23, 21],
    [1, 24, 9],
    [1, 24, 15],
    [1, 24, 20],
    [1, 24, 21],
]
