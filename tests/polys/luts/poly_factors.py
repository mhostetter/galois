"""
A module containing LUTs for factored polynomials over finite fields.

Sage:
    to_coeffs = lambda poly: poly.coefficients(sparse=False)[::-1] if poly != 0 else [0]

    order = 2
    degree = 1
    N = 20
    print(f"POLY_FACTORS_{p}_{m} = [")
    R = GF(p**m, repr="int")["x"]
    for _ in range(N):
        a = R.random_element(randint(10, 20))
        a = a / a.coefficients()[-1]  # Make monic
        polys = []
        exponents = []
        for item in factor(a):
            polys.append(to_coeffs(item[0]))
            exponents.append(item[1])
        print(f"    ({to_coeffs(a)}, {polys}, {exponents}),")
    print("]\n")
"""

# LUT items are (a(x), factors(x), multiplicities). All coefficients in degree-descending order.

POLY_FACTORS_2_1 = [
    (
        [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        [[1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]],
        [1],
    ),
    (
        [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [[1, 1], [1, 0], [1, 1, 1, 1, 1, 0, 1, 0, 1]],
        [1, 3, 1],
    ),
    (
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],
        [[1, 1, 1, 0, 1, 1], [1, 1, 0, 0, 0, 0, 1]],
        [1, 1],
    ),
    (
        [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [[1, 0], [1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]],
        [1, 1, 1, 1],
    ),
    (
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [[1, 0], [1, 1, 0, 1], [1, 0, 0, 0, 1, 1, 1, 1]],
        [5, 1, 1],
    ),
    (
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [[1, 1], [1, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1]],
        [3, 1, 1],
    ),
    (
        [1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [[1, 1, 0, 1], [1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 1]],
        [1, 1, 1],
    ),
    (
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [[1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 0, 1]],
        [1, 1],
    ),
    (
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        [[1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 1, 0, 1]],
        [1, 1],
    ),
    (
        [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
        [[1, 1], [1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1]],
        [3, 1],
    ),
    (
        [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        [[1, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]],
        [1, 1],
    ),
    (
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [[1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]],
        [2, 1, 1],
    ),
    (
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [[1, 1, 0, 0, 1], [1, 1, 0, 1, 1, 0, 1]],
        [1, 1],
    ),
    (
        [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
        [[1, 1], [1, 1, 1], [1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 1, 1, 1]],
        [1, 1, 1, 1],
    ),
    (
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        [[1, 1], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]],
        [2, 1, 1],
    ),
    (
        [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [[1, 0], [1, 1], [1, 1, 0, 1], [1, 0, 1, 1, 1, 1]],
        [2, 4, 1, 1],
    ),
    (
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
        [[1, 1], [1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1]],
        [1, 1, 1],
    ),
    (
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [[1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, 0, 0, 1], [1, 1, 1, 1, 1]],
        [2, 1, 1, 1, 1, 1],
    ),
    (
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [[1, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1]],
        [2, 1],
    ),
    (
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [[1, 0], [1, 0, 1, 1, 1, 1]],
        [5, 1],
    ),
]

POLY_FACTORS_3_1 = [
    (
        [1, 0, 1, 0, 2, 2, 1, 1, 1, 2, 0, 0, 2, 0],
        [[1, 0], [1, 1], [1, 2, 2, 1, 1, 1, 0, 1, 0, 2, 1, 2]],
        [1, 1, 1],
    ),
    (
        [1, 0, 2, 0, 1, 2, 2, 0, 1, 0, 0, 2, 1, 1, 2, 2, 0],
        [[1, 0], [1, 1], [1, 1, 1, 2], [1, 1, 0, 0, 2], [1, 2, 1, 2, 1]],
        [1, 1, 2, 1, 1],
    ),
    (
        [1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2],
        [[1, 1], [1, 0, 2, 2, 0, 2], [1, 0, 0, 0, 2, 0, 2, 1, 2, 1, 0, 1]],
        [1, 1, 1],
    ),
    (
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 2, 0, 1, 1, 0, 0, 2, 2, 1, 2, 0, 1],
        [[1, 0, 1], [1, 1, 1, 2], [1, 2, 2, 0, 1, 2, 0, 2], [1, 0, 1, 1, 1, 0, 2, 1, 1]],
        [1, 1, 1, 1],
    ),
    (
        [1, 2, 2, 0, 2, 0, 1, 2, 1, 2, 0, 0, 0, 2, 2, 0, 1, 1, 0],
        [[1, 0], [1, 1, 1, 2, 1, 2], [1, 1, 0, 0, 2, 1, 2, 1, 0, 1, 2, 1, 2]],
        [1, 1, 1],
    ),
    (
        [1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, 0, 1],
        [[1, 1, 2], [1, 2, 0, 1], [1, 0, 2, 1, 0, 0, 0, 2, 2]],
        [1, 1, 1],
    ),
    (
        [1, 2, 1, 2, 0, 0, 2, 1, 2, 1, 0, 1, 0, 0, 2],
        [[1, 2], [1, 2, 1, 0, 1], [1, 0, 0, 0, 1, 2, 0, 1]],
        [3, 1, 1],
    ),
    (
        [1, 1, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 0, 0],
        [[1, 0], [1, 0, 1], [1, 0, 0, 1, 2], [1, 1, 0, 2, 1]],
        [3, 1, 1, 1],
    ),
    (
        [1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 0, 2, 2, 1, 1, 1, 1, 0],
        [[1, 0], [1, 1], [1, 2], [1, 2, 2, 1, 0, 2, 2, 2, 1, 2, 2, 0, 2]],
        [1, 1, 3, 1],
    ),
    (
        [1, 0, 2, 1, 0, 0, 1, 0, 2, 0, 2, 2, 2],
        [[1, 0, 1], [1, 1, 2, 0, 0, 1], [1, 2, 0, 0, 2, 2]],
        [1, 1, 1],
    ),
    (
        [1, 1, 1, 0, 2, 0, 2, 0, 2, 1, 0, 0, 0, 1, 1],
        [[1, 1], [1, 2], [1, 2, 1, 1, 2], [1, 0, 0, 1, 2, 2, 0, 2]],
        [1, 2, 1, 1],
    ),
    (
        [1, 1, 0, 2, 2, 0, 0, 2, 1, 2, 2, 2, 2, 1, 0, 0],
        [[1, 2], [1, 0], [1, 1, 2, 2, 2], [1, 1, 2, 1, 0, 1, 2, 2, 1]],
        [1, 2, 1, 1],
    ),
    (
        [1, 1, 0, 1, 2, 2, 0, 2, 2, 1, 1, 0, 0, 0],
        [[1, 0], [1, 2, 2], [1, 2, 1, 1], [1, 0, 2, 1, 1, 2]],
        [3, 1, 1, 1],
    ),
    (
        [1, 1, 1, 1, 0, 1, 1, 2, 0, 2, 2, 1, 0, 2, 1, 1, 0],
        [[1, 0], [1, 2, 2], [1, 2, 1, 1, 2, 1, 1, 1, 2, 2, 0, 0, 0, 2]],
        [1, 1, 1],
    ),
    (
        [1, 0, 1, 0, 2, 0, 2, 1, 0, 0, 0, 1, 1, 2, 1, 0, 2, 0, 1, 2, 1],
        [[1, 2], [1, 1], [1, 2, 0, 0, 1, 2, 1, 0, 0, 1, 2, 0, 1, 0, 2, 2, 2, 2]],
        [1, 2, 1],
    ),
    (
        [1, 2, 2, 1, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 0, 2],
        [[1, 2, 2, 1, 1, 1, 1, 0, 1, 2, 2, 2, 0, 2, 2, 2, 0, 1, 0, 2]],
        [1],
    ),
    (
        [1, 0, 2, 0, 2, 1, 0, 1, 1, 1, 1, 2, 0, 2],
        [[1, 1], [1, 1, 0, 0, 2], [1, 2, 0, 2, 1, 0, 1]],
        [3, 1, 1],
    ),
    (
        [1, 1, 0, 1, 1, 1, 2, 0, 0, 2, 0, 2, 1, 1, 1],
        [[1, 1, 0, 1, 1, 1, 2, 0, 0, 2, 0, 2, 1, 1, 1]],
        [1],
    ),
    (
        [1, 2, 1, 0, 1, 1, 2, 0, 2, 1, 2, 0, 2, 2, 2, 0, 2, 0, 1, 0],
        [[1, 0], [1, 0, 1], [1, 0, 1, 0, 2, 1, 2], [1, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2]],
        [1, 1, 1, 1],
    ),
    (
        [1, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 2],
        [[1, 2, 1, 1, 0, 0, 0, 2, 1], [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 2, 2]],
        [1, 1],
    ),
]

POLY_FACTORS_5_1 = [
    (
        [1, 0, 1, 1, 2, 4, 1, 4, 4, 4, 1, 1, 0, 0],
        [[1, 3], [1, 0], [1, 2, 0, 1, 4, 2, 0, 4, 2, 3, 2]],
        [1, 2, 1],
    ),
    (
        [1, 4, 3, 2, 2, 1, 2, 4, 3, 4, 2, 4, 1, 3, 4, 2],
        [[1, 2, 2, 3], [1, 0, 2, 4, 3, 4, 4], [1, 2, 0, 3, 4, 2, 1]],
        [1, 1, 1],
    ),
    (
        [1, 0, 1, 4, 1, 0, 2, 1, 2, 4, 2, 0, 4, 4, 0, 2, 1, 2, 2],
        [[1, 1, 2], [1, 1, 1, 3], [1, 4, 0, 4], [1, 4, 0, 0, 0, 0, 1, 3, 3, 3, 3]],
        [1, 1, 1, 1],
    ),
    (
        [1, 2, 0, 1, 2, 4, 0, 0, 0, 2, 0, 0, 2, 3, 2, 1, 4, 4, 4],
        [[1, 3, 1, 2], [1, 4, 4, 1, 0, 4], [1, 0, 3, 1, 4, 4, 1, 1, 2, 4, 3]],
        [1, 1, 1],
    ),
    (
        [1, 2, 4, 2, 3, 1, 1, 0, 3, 1, 1, 0, 3, 3, 1, 1, 0, 1, 2, 1, 3],
        [[1, 1], [1, 1, 3, 4, 4, 2, 4, 1, 2, 4, 2, 3, 0, 3, 3, 3, 2, 4, 3, 3]],
        [1, 1],
    ),
    (
        [1, 4, 4, 4, 3, 0, 4, 2, 0, 4, 2, 2, 1, 1, 2, 2],
        [[1, 3], [1, 1, 1, 1, 0, 0, 4, 0, 0, 4, 0, 2, 0, 1, 4]],
        [1, 1],
    ),
    (
        [1, 0, 4, 0, 2, 1, 4, 3, 0, 4, 0, 4, 1, 3, 2, 1],
        [[1, 4], [1, 1, 2], [1, 4, 1], [1, 4, 3, 4], [1, 0, 4, 4, 2]],
        [4, 1, 1, 1, 1],
    ),
    (
        [1, 2, 0, 2, 0, 0, 1, 3, 3, 3, 1, 1, 1, 3, 3, 4, 0],
        [[1, 0], [1, 0, 2, 3, 3], [1, 0, 4, 4, 2], [1, 2, 4, 3, 4, 3, 1, 4]],
        [1, 1, 1, 1],
    ),
    (
        [1, 3, 0, 0, 2, 4, 1, 2, 0, 0, 4, 2, 3, 2, 4],
        [[1, 3, 2, 2], [1, 0, 3, 4, 4, 3, 1, 0, 2, 2, 4, 2]],
        [1, 1],
    ),
    (
        [1, 1, 1, 1, 2, 1, 3, 2, 1, 4, 1],
        [[1, 1], [1, 2], [1, 3, 0, 0, 2, 0, 4, 0, 3]],
        [1, 1, 1],
    ),
    (
        [1, 3, 1, 2, 4, 2, 0, 3, 0, 2, 3, 2, 1, 0, 2, 0, 1, 1, 1, 3, 0],
        [[1, 0], [1, 3, 3], [1, 2, 4, 4], [1, 3, 3, 1, 0, 4, 0, 1, 0, 3, 3, 4, 4, 0, 4]],
        [1, 1, 1, 1],
    ),
    (
        [1, 1, 2, 1, 0, 2, 3, 2, 2, 2, 2, 0, 3, 2, 2, 2, 0],
        [[1, 0], [1, 2, 3], [1, 2, 2, 1, 4, 1], [1, 2, 0, 2, 3, 1, 2, 2, 4]],
        [1, 1, 1, 1],
    ),
    (
        [1, 1, 0, 0, 3, 1, 1, 4, 2, 3, 4, 0, 1, 0, 2, 0, 4],
        [[1, 0, 2], [1, 3, 4], [1, 3, 0, 4], [1, 0, 0, 2, 3, 0, 1, 1, 1, 2]],
        [1, 1, 1, 1],
    ),
    (
        [1, 0, 2, 3, 4, 0, 3, 0, 3, 2, 0, 1, 4, 2, 4, 4],
        [[1, 2], [1, 1, 3, 2, 3, 0, 2, 4], [1, 2, 1, 2, 0, 2, 0, 3]],
        [1, 1, 1],
    ),
    (
        [1, 3, 1, 4, 1, 1, 4, 2, 1, 4, 3, 0, 2],
        [[1, 1, 1, 0, 1], [1, 2, 3, 4, 3, 2, 1, 0, 2]],
        [1, 1],
    ),
    (
        [1, 3, 2, 0, 3, 1, 3, 2, 4, 3, 3, 3, 0],
        [[1, 0], [1, 3, 2, 0, 3, 1, 3, 2, 4, 3, 3, 3]],
        [1, 1],
    ),
    (
        [1, 1, 1, 1, 3, 2, 1, 2, 1, 0, 4, 4, 3],
        [[1, 3], [1, 1, 1, 3, 3, 4], [1, 2, 4, 1, 4, 1, 4]],
        [1, 1, 1],
    ),
    (
        [1, 3, 3, 0, 1, 3, 3, 4, 2, 4, 2, 1, 3, 4, 0, 2, 1, 2],
        [[1, 4, 1, 4, 0, 3], [1, 4, 1, 3, 2, 0, 2, 0, 1, 1, 2, 2, 4]],
        [1, 1],
    ),
    (
        [1, 1, 4, 2, 2, 2, 4, 4, 3, 3, 0, 3, 4, 4, 2, 3],
        [[1, 3, 3], [1, 1, 0, 1], [1, 2, 0, 1, 2, 4, 1, 0, 2, 3, 1]],
        [1, 1, 1],
    ),
    (
        [1, 1, 1, 0, 1, 4, 0, 1, 4, 2, 4, 2, 4, 3, 2, 0],
        [[1, 0], [1, 4], [1, 2, 2, 3], [1, 0, 4, 2, 1], [1, 0, 2, 1, 2, 4, 1]],
        [1, 1, 1, 1, 1],
    ),
]

POLY_FACTORS_2_2 = [
    (
        [1, 1, 1, 2, 0, 1, 3, 3, 2, 0, 3, 2, 2, 0, 1, 3, 2],
        [[1, 2, 2, 1, 1, 2], [1, 3, 2, 1, 3, 1, 2, 3, 0, 3, 1, 1]],
        [1, 1],
    ),
    (
        [1, 0, 1, 2, 0, 0, 2, 3, 2, 0, 3, 3, 2, 2],
        [[1, 3], [1, 1, 2], [1, 2, 2], [1, 1, 2, 0, 3], [1, 1, 2, 2, 2]],
        [1, 1, 1, 1, 1],
    ),
    (
        [1, 0, 3, 2, 0, 1, 2, 0, 1, 1, 2],
        [[1, 2], [1, 1, 3], [1, 2, 1, 0, 0, 3]],
        [1, 2, 1],
    ),
    (
        [1, 0, 1, 1, 1, 1, 3, 0, 2, 3, 2, 3, 0, 0, 3, 1, 2, 2, 1, 0, 2],
        [[1, 2], [1, 1, 2], [1, 3, 1], [1, 0, 2, 1, 2, 1, 1, 0, 1, 0, 0, 0, 2, 3, 2, 3]],
        [1, 1, 1, 1],
    ),
    (
        [1, 1, 3, 3, 1, 1, 3, 2, 3, 0, 1, 1, 0, 3, 1, 1, 2, 3, 0, 1, 0],
        [[1, 0], [1, 3], [1, 2, 2], [1, 0, 0, 2, 3, 3, 0, 3, 3, 1, 1, 0, 2, 0, 0, 3, 1]],
        [1, 1, 1, 1],
    ),
    (
        [1, 2, 2, 1, 1, 0, 3, 2, 3, 1, 1, 1, 3, 2, 1, 3, 2, 2, 1, 0, 1],
        [[1, 2], [1, 3, 3, 3], [1, 0, 0, 2, 1, 1, 1], [1, 3, 3, 3, 1, 2, 1, 3, 2, 3, 1]],
        [1, 1, 1, 1],
    ),
    (
        [1, 0, 2, 1, 3, 2, 2, 0, 3, 3, 2, 1, 2],
        [[1, 1], [1, 3], [1, 1, 1, 0, 3, 3, 0, 0, 2, 1]],
        [1, 2, 1],
    ),
    (
        [1, 1, 3, 1, 1, 3, 1, 1, 3, 0, 0, 2, 2, 1, 0, 2, 1],
        [[1, 2], [1, 3], [1, 1, 3, 3, 2], [1, 1, 0, 3, 2, 2, 0, 1, 1, 3, 3]],
        [1, 1, 1, 1],
    ),
    (
        [1, 0, 2, 0, 1, 2, 1, 3, 1, 0, 0, 3, 3, 3, 1, 0, 3, 3, 2, 3],
        [[1, 1], [1, 2, 3, 1, 0, 3], [1, 3, 1, 2, 1, 1, 0, 3, 2, 1, 0, 1, 2, 1]],
        [1, 1, 1],
    ),
    (
        [1, 2, 3, 2, 1, 3, 1, 0, 1, 3, 0, 3, 3],
        [[1, 0, 2, 1, 2, 0, 3], [1, 2, 1, 0, 3, 1, 1]],
        [1, 1],
    ),
    (
        [1, 1, 3, 1, 1, 3, 1, 2, 1, 0, 2, 1, 2, 3],
        [[1, 1], [1, 2, 1], [1, 3, 1, 0, 1], [1, 1, 3, 1, 0, 0, 3]],
        [1, 1, 1, 1],
    ),
    (
        [1, 2, 1, 3, 2, 1, 2, 0, 2, 0, 0, 3, 1, 3, 1, 0],
        [[1, 0], [1, 3], [1, 2, 3, 0, 3, 1, 3, 2, 3, 3, 1, 2, 3]],
        [1, 2, 1],
    ),
    (
        [1, 3, 1, 1, 3, 3, 1, 1, 2, 2, 3],
        [[1, 3], [1, 0, 1, 2, 2, 2, 0, 1, 1, 1]],
        [1, 1],
    ),
    (
        [1, 2, 0, 2, 3, 3, 1, 0, 2, 1, 2, 3, 1],
        [[1, 1, 2, 3], [1, 3, 1, 1, 2, 0, 1, 0, 0, 2]],
        [1, 1],
    ),
    (
        [1, 0, 3, 0, 1, 3, 3, 1, 2, 1, 3, 3, 3, 0, 3, 0],
        [[1, 0], [1, 3], [1, 3, 1, 3, 3, 1, 0, 1, 1, 2, 2, 2, 2, 1]],
        [1, 1, 1],
    ),
    (
        [1, 1, 3, 0, 1, 0, 3, 1, 0, 0, 0, 3, 3, 0, 3, 1, 2, 1],
        [[1, 3], [1, 1, 2], [1, 2, 2], [1, 2, 0, 3, 2], [1, 0, 2, 1, 2, 1, 1, 3]],
        [2, 1, 1, 1, 1],
    ),
    (
        [1, 3, 1, 3, 2, 0, 3, 3, 0, 1, 3],
        [[1, 1], [1, 3, 1], [1, 1, 1, 2, 2, 1, 0, 3]],
        [1, 1, 1],
    ),
    (
        [1, 1, 1, 1, 1, 1, 0, 1, 3, 3, 1],
        [[1, 1], [1, 2, 0, 2, 2], [1, 2, 2, 1, 2, 3]],
        [1, 1, 1],
    ),
    (
        [1, 0, 2, 3, 3, 0, 0, 0, 2, 2, 0, 3, 0, 3, 3, 3, 2, 1, 0],
        [[1, 0], [1, 1], [1, 2], [1, 3, 3, 3], [1, 0, 2, 3, 3], [1, 0, 3, 0, 1, 1, 1, 0, 2]],
        [1, 1, 1, 1, 1, 1],
    ),
    (
        [1, 2, 2, 3, 3, 1, 2, 1, 2, 3, 3],
        [[1, 2, 2], [1, 0, 0, 3, 2, 3, 0, 0, 2]],
        [1, 1],
    ),
]

POLY_FACTORS_3_2 = [
    (
        [1, 0, 6, 7, 6, 3, 1, 8, 3, 7, 5, 7, 6, 6, 5, 7, 8],
        [[1, 0, 4, 4, 0, 0, 7, 5], [1, 0, 5, 3, 3, 5, 1, 5, 4, 5]],
        [1, 1],
    ),
    (
        [1, 4, 5, 6, 3, 7, 7, 0, 1, 3, 6, 6, 0, 4, 5, 6, 1, 2, 4, 8, 8],
        [[1, 3], [1, 6, 2], [1, 1, 4, 8, 0, 7, 8, 0, 4, 3, 6, 1, 5, 8, 5, 0, 1]],
        [2, 1, 1],
    ),
    (
        [1, 3, 2, 3, 3, 3, 2, 5, 5, 4, 4],
        [[1, 2, 0, 4], [1, 4, 3, 5, 6, 5, 1, 1]],
        [1, 1],
    ),
    (
        [1, 7, 6, 8, 1, 5, 6, 1, 7, 5, 1, 8],
        [[1, 7], [1, 0, 6, 7, 5, 1, 2, 8, 4, 8, 7]],
        [1, 1],
    ),
    (
        [1, 1, 5, 7, 4, 1, 4, 6, 2, 0, 7, 8, 8, 7, 2, 5],
        [[1, 7], [1, 2, 4], [1, 4, 3, 0, 3, 6, 4, 4, 0, 5, 3, 1, 4]],
        [1, 1, 1],
    ),
    (
        [1, 8, 1, 4, 4, 3, 7, 6, 8, 4, 3, 0, 1, 5, 1, 0, 2],
        [[1, 5, 4, 3, 1, 3], [1, 3, 8, 6, 2, 4, 0, 2, 8, 1, 8, 7]],
        [1, 1],
    ),
    (
        [1, 1, 0, 6, 7, 0, 8, 7, 1, 2, 3],
        [[1, 2], [1, 3, 1], [1, 8, 8, 7, 8, 3, 2, 6]],
        [1, 1, 1],
    ),
    (
        [1, 8, 8, 2, 4, 5, 8, 8, 4, 6, 7, 2, 8, 5, 6, 1, 6],
        [[1, 1], [1, 8], [1, 5, 1], [1, 7, 4, 4, 7, 2], [1, 2, 0, 7, 1, 0, 8, 7]],
        [1, 1, 1, 1, 1],
    ),
    (
        [1, 4, 6, 1, 3, 6, 8, 8, 3, 5, 0, 7, 3],
        [[1, 8], [1, 7, 3], [1, 1, 6, 3, 5, 1, 5, 5, 1, 4]],
        [1, 1, 1],
    ),
    (
        [1, 2, 3, 8, 8, 1, 0, 3, 3, 0, 8],
        [[1, 1], [1, 4, 6], [1, 6, 3, 4, 7, 6, 1, 3]],
        [1, 1, 1],
    ),
    (
        [1, 3, 7, 8, 8, 8, 2, 0, 6, 8, 8, 4, 5, 1, 1, 6],
        [[1, 4, 5], [1, 2, 1, 8], [1, 0, 8, 8, 8, 2, 5, 7, 7, 8, 1]],
        [1, 1, 1],
    ),
    (
        [1, 3, 5, 0, 2, 3, 4, 0, 7, 3, 2, 2, 3, 4, 5, 5, 0, 6, 7],
        [[1, 7], [1, 8], [1, 0, 2, 3, 1, 8, 8, 5, 4, 7, 5, 2, 8, 5, 2, 7, 4]],
        [1, 1, 1],
    ),
    (
        [1, 4, 1, 7, 4, 2, 6, 6, 3, 7, 2, 4, 6, 7, 3, 4, 2, 8, 1, 8],
        [[1, 5, 6, 5, 7, 8, 6, 4, 2, 1], [1, 2, 6, 0, 7, 7, 5, 3, 4, 6, 8]],
        [1, 1],
    ),
    (
        [1, 6, 5, 6, 2, 4, 0, 2, 2, 3, 8, 5, 5, 2, 1, 7, 2, 5],
        [[1, 7, 3, 3, 1], [1, 2, 6, 8, 5, 0, 0, 5, 5, 1, 2, 3, 1, 5]],
        [1, 1],
    ),
    (
        [1, 1, 6, 7, 1, 7, 4, 7, 6, 8, 8, 6, 2, 6, 4, 7, 3, 4, 7],
        [[1, 1, 4], [1, 0, 5, 5, 5, 2, 2, 0, 1, 7, 6, 3, 3, 8, 1, 8, 3]],
        [1, 1],
    ),
    (
        [1, 7, 6, 6, 0, 5, 5, 5, 7, 0, 1, 3, 0, 5, 2],
        [[1, 1], [1, 3], [1, 4, 7, 8, 3, 7], [1, 2, 5, 0, 6, 3, 8, 1]],
        [1, 1, 1, 1],
    ),
    (
        [1, 5, 7, 6, 4, 0, 7, 4, 8, 1, 3, 5, 2],
        [[1, 3, 6], [1, 2, 4, 8, 6, 6, 7, 1, 4, 6, 5]],
        [1, 1],
    ),
    (
        [1, 3, 7, 5, 6, 6, 4, 6, 3, 8, 4, 2, 5, 4, 6, 2, 6],
        [[1, 1], [1, 2], [1, 4], [1, 7], [1, 8], [1, 2, 3], [1, 6, 6, 5], [1, 0, 1, 8, 8, 6, 8]],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ),
    (
        [1, 0, 5, 1, 4, 7, 6, 0, 2, 2, 7, 8, 6, 7, 3, 0],
        [[1, 0], [1, 1], [1, 7, 2], [1, 8, 7, 2, 3], [1, 7, 3, 2, 4, 7, 2]],
        [1, 2, 1, 1, 1],
    ),
    (
        [1, 2, 5, 1, 0, 5, 2, 8, 6, 0, 3, 3, 7, 0, 3],
        [[1, 3, 2], [1, 8, 1, 3, 6, 0, 8, 3, 1, 0, 4, 8, 6]],
        [1, 1],
    ),
]

POLY_FACTORS_5_2 = [
    (
        [1, 21, 21, 12, 4, 2, 5, 24, 16, 7, 8, 24],
        [[1, 3, 9, 19], [1, 16, 16, 18], [1, 7, 16, 3, 0, 6]],
        [1, 1, 1],
    ),
    (
        [1, 0, 12, 16, 0, 1, 8, 5, 6, 13, 8, 5, 4],
        [[1, 5], [1, 20, 15, 2, 15, 12, 12, 14, 0, 13, 7, 17]],
        [1, 1],
    ),
    (
        [1, 0, 3, 10, 24, 11, 1, 24, 22, 22, 20, 14, 0, 4, 18],
        [[1, 9], [1, 8, 18, 0, 18, 22], [1, 18, 6, 18, 11, 5, 22, 16, 16]],
        [1, 1, 1],
    ),
    (
        [1, 9, 19, 11, 14, 1, 1, 11, 22, 17, 4],
        [[1, 14], [1, 19, 24, 1], [1, 6, 3, 13, 20, 19, 7]],
        [1, 1, 1],
    ),
    (
        [1, 20, 9, 0, 23, 17, 23, 1, 16, 11, 14, 23, 6, 19, 10, 23, 2, 20, 4],
        [[1, 21, 24, 2, 23, 16, 14, 4, 21], [1, 4, 6, 4, 12, 15, 4, 0, 4, 19, 10]],
        [1, 1],
    ),
    (
        [1, 23, 18, 8, 17, 5, 12, 19, 21, 7, 6, 24, 23, 22, 8, 15],
        [[1, 20], [1, 0, 5], [1, 11, 20], [1, 21, 6, 10], [1, 1, 23, 9, 7, 9, 19]],
        [2, 1, 1, 1, 1],
    ),
    (
        [1, 17, 2, 8, 12, 10, 3, 4, 21, 16, 8, 9, 16, 22, 22, 9, 23, 18, 3, 22],
        [[1, 17, 23, 20, 17, 6, 5, 6], [1, 0, 9, 1, 19, 4, 20, 5, 10, 5, 8, 4, 18]],
        [1, 1],
    ),
    (
        [1, 20, 19, 23, 7, 22, 24, 22, 6, 9, 15, 2, 2, 16, 19, 22, 11, 3, 12],
        [[1, 11], [1, 14, 23, 6, 5, 6, 22, 16, 2, 12, 16, 23, 10, 14, 23, 5, 20, 19]],
        [1, 1],
    ),
    (
        [1, 23, 3, 12, 2, 5, 3, 1, 0, 17, 16, 12, 10, 13, 14, 15],
        [[1, 17], [1, 21], [1, 18, 0, 7], [1, 14, 9, 7, 21, 0, 18, 15, 19]],
        [2, 2, 1, 1],
    ),
    (
        [1, 5, 3, 7, 9, 10, 12, 7, 13, 21, 13, 17, 3, 21, 13, 19, 7, 0, 19, 13],
        [[1, 3, 18, 17], [1, 7, 24, 23, 14, 9, 8, 16, 5, 6, 2, 20, 21, 13, 24, 12, 4]],
        [1, 1],
    ),
    (
        [1, 6, 19, 14, 1, 5, 6, 23, 19, 24, 4, 16, 4, 3, 12, 10, 11, 24, 8],
        [[1, 14], [1, 14, 12], [1, 13, 1, 14, 19, 5, 20, 10, 3, 21, 2, 11, 15, 5, 6, 5]],
        [1, 1, 1],
    ),
    (
        [1, 3, 9, 23, 15, 7, 24, 23, 1, 22, 22, 4, 19, 18],
        [[1, 13, 22, 1], [1, 6, 16, 5, 13], [1, 14, 8, 14, 23, 14, 9]],
        [1, 1, 1],
    ),
    (
        [1, 16, 15, 13, 11, 6, 16, 1, 10, 3, 18, 0, 9, 0, 8, 18, 19],
        [[1, 22], [1, 22, 19, 7, 2, 16], [1, 2, 11, 3, 13, 11, 13, 21, 8, 21, 14]],
        [1, 1, 1],
    ),
    (
        [1, 4, 1, 10, 8, 1, 9, 6, 13, 19, 5, 12, 12, 15, 18, 15, 8, 15, 0, 1],
        [[1, 2, 7, 9, 14, 4, 6, 14], [1, 2, 20, 7, 15, 0, 1, 5, 0, 23, 4, 18, 23]],
        [1, 1],
    ),
    (
        [1, 15, 6, 7, 21, 0, 23, 13, 22, 4, 9, 6, 8, 4, 10, 11, 14, 1, 1, 4],
        [[1, 1], [1, 19, 17, 15, 6, 24, 4, 14, 13, 16, 18, 18, 15, 14, 1, 10, 4, 2, 4]],
        [1, 1],
    ),
    (
        [1, 4, 12, 9, 22, 13, 2, 22, 5, 6, 8, 10, 16],
        [[1, 6], [1, 7], [1, 24, 1], [1, 16, 1, 4], [1, 6, 23, 20, 12, 11]],
        [1, 1, 1, 1, 1],
    ),
    (
        [1, 16, 22, 12, 13, 4, 3, 2, 24, 20, 12, 22, 7, 14, 22, 16, 18],
        [[1, 3], [1, 23, 1, 12], [1, 12, 5, 1, 24], [1, 13, 11, 9, 2, 19, 1, 20, 11]],
        [1, 1, 1, 1],
    ),
    (
        [1, 0, 5, 8, 9, 20, 1, 0, 16, 22, 24, 12, 10, 21, 2, 15, 18, 24, 3],
        [[1, 9, 13], [1, 14, 6, 5, 23, 7], [1, 12, 5, 4, 19, 14, 0, 20, 12, 6, 20, 12]],
        [1, 1, 1],
    ),
    (
        [1, 13, 20, 13, 9, 20, 5, 1, 17, 3, 14, 23, 21, 19],
        [[1, 1], [1, 12, 13, 0, 9, 16, 19, 12, 5, 23, 16, 7, 19]],
        [1, 1],
    ),
    (
        [1, 11, 16, 6, 6, 5, 8, 23, 5, 17, 19, 17, 21, 17, 9, 8],
        [[1, 5, 15], [1, 23, 24, 23], [1, 6, 14, 3, 17], [1, 7, 12, 5, 24, 1, 9]],
        [1, 1, 1, 1],
    ),
]


POLY_FACTORS = [
    (2, 1, POLY_FACTORS_2_1),
    (3, 1, POLY_FACTORS_3_1),
    (5, 1, POLY_FACTORS_5_1),
    (2, 2, POLY_FACTORS_2_2),
    (3, 2, POLY_FACTORS_3_2),
    (5, 2, POLY_FACTORS_5_2),
]
