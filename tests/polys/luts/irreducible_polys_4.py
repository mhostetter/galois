"""
A module containing LUTs for irreducible polynomials over GF(2^2).

Sage:
    def integer(coeffs, order):
        i = 0
        for d, c in enumerate(coeffs[::-1]):
            i += (c.integer_representation() * order**d)
        return i

    order = 2**2
    degree =1
    list_ = []
    R = GF(order, repr="int")["x"]
    for f in R.polynomials(degree):
        if f.is_monic() and f.is_irreducible():
            list_.append(f.coefficients(sparse=False)[::-1])

    # Sort in lexicographical order
    if not is_prime(order):
        list_ = sorted(list_, key=lambda item: integer(item, order))

    print(f"IRREDUCIBLE_POLYS_{order}_{degree} = {list_}")
"""

# LUT items are poly coefficients in degree-descending order

IRREDUCIBLE_POLYS_4_1 = [
    [1, 0],
    [1, 1],
    [1, 2],
    [1, 3],
]

IRREDUCIBLE_POLYS_4_2 = [
    [1, 1, 2],
    [1, 1, 3],
    [1, 2, 1],
    [1, 2, 2],
    [1, 3, 1],
    [1, 3, 3],
]

IRREDUCIBLE_POLYS_4_3 = [
    [1, 0, 0, 2],
    [1, 0, 0, 3],
    [1, 0, 1, 1],
    [1, 0, 2, 1],
    [1, 0, 3, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 2],
    [1, 1, 1, 3],
    [1, 1, 2, 3],
    [1, 1, 3, 2],
    [1, 2, 0, 1],
    [1, 2, 1, 3],
    [1, 2, 2, 2],
    [1, 2, 3, 2],
    [1, 2, 3, 3],
    [1, 3, 0, 1],
    [1, 3, 1, 2],
    [1, 3, 2, 2],
    [1, 3, 2, 3],
    [1, 3, 3, 3],
]
