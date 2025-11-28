"""
A module containing LUTs for primitive elements in GF(3^m).

Sage:
    order = 3
    degree = 1
    F.<x> = GF(order**degree, repr="int")
    print(f"PRIMITIVE_ELEMENTS_{order}_{degree} = [")
    for e in range(1, F.order()):
        e = F.fetch_int(e)
        if e.multiplicative_order() == F.order() - 1:
            c = e.polynomial().coefficients(sparse=False)[::-1]
            print(f"    {c},")
    print("]\n")
"""

# LUT items are poly coefficients in degree-descending order

PRIMITIVE_ELEMENTS_3_2 = [
    [1, 0],
    [1, 2],
    [2, 0],
    [2, 1],
]

PRIMITIVE_ELEMENTS_3_3 = [
    [1, 0],
    [1, 1],
    [1, 2],
    [1, 0, 1],
    [1, 1, 2],
    [1, 2, 2],
    [2, 0, 0],
    [2, 0, 1],
    [2, 1, 0],
    [2, 1, 2],
    [2, 2, 0],
    [2, 2, 2],
]

PRIMITIVE_ELEMENTS_3_4 = [
    [1, 0],
    [1, 2],
    [2, 0],
    [2, 1],
    [1, 1, 0],
    [1, 2, 2],
    [2, 1, 1],
    [2, 2, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 2],
    [1, 0, 1, 0],
    [1, 0, 1, 2],
    [1, 0, 2, 2],
    [1, 1, 2, 0],
    [1, 1, 2, 2],
    [1, 2, 0, 0],
    [1, 2, 0, 1],
    [1, 2, 1, 0],
    [1, 2, 1, 1],
    [1, 2, 2, 2],
    [2, 0, 0, 0],
    [2, 0, 0, 1],
    [2, 0, 1, 1],
    [2, 0, 2, 0],
    [2, 0, 2, 1],
    [2, 1, 0, 0],
    [2, 1, 0, 2],
    [2, 1, 1, 1],
    [2, 1, 2, 0],
    [2, 1, 2, 2],
    [2, 2, 1, 0],
    [2, 2, 1, 1],
]
