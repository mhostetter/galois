"""
A module containing LUTs for primitive elements in GF(3^m).

Sage:
    p = 3
    for m in range(1, 5):
        F.<x> = GF(p**m, repr="int")
        print(f"NORMAL_ELEMENTS_{p}_{m} = [")

        for e_int in range(1, F.order()):
            if m == 1:
                a = F(e_int)  # generic integer -> field element
            else:
                a = F.fetch_int(e_int)  # fetch from LUT
            conj = [a^(p^i) for i in range(m)]

            cols = []
            for y in conj:
                coeffs = y.polynomial().coefficients(sparse=False)
                coeffs += [0] * (m - len(coeffs))  # pad to length m (ascending degree)
                cols.append(coeffs)

            M = matrix(GF(p), m, m, list(zip(*cols)))  # columns are conjugates
            if M.rank() == m:
                # store degree-descending coeffs like your LUTs
                coeffs = a.polynomial().coefficients(sparse=False)[::-1]
                print(f"    {coeffs},")
        print("]\n")
"""

# LUT items are poly coefficients in degree-descending order

NORMAL_ELEMENTS_3_1 = [
    [1],
    [2],
]

NORMAL_ELEMENTS_3_2 = [
    [1, 0],
    [1, 2],
    [2, 0],
    [2, 1],
]

NORMAL_ELEMENTS_3_3 = [
    [1, 0, 0],
    [1, 0, 1],
    [1, 0, 2],
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 2],
    [1, 2, 0],
    [1, 2, 1],
    [1, 2, 2],
    [2, 0, 0],
    [2, 0, 1],
    [2, 0, 2],
    [2, 1, 0],
    [2, 1, 1],
    [2, 1, 2],
    [2, 2, 0],
    [2, 2, 1],
    [2, 2, 2],
]

NORMAL_ELEMENTS_3_4 = [
    [1, 0],
    [1, 1],
    [2, 0],
    [2, 2],
    [1, 1, 0],
    [1, 1, 2],
    [1, 2, 1],
    [1, 2, 2],
    [2, 1, 1],
    [2, 1, 2],
    [2, 2, 0],
    [2, 2, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 2, 1],
    [1, 0, 2, 2],
    [1, 1, 2, 0],
    [1, 1, 2, 1],
    [1, 2, 0, 1],
    [1, 2, 0, 2],
    [1, 2, 2, 0],
    [1, 2, 2, 2],
    [2, 0, 0, 0],
    [2, 0, 0, 2],
    [2, 0, 1, 1],
    [2, 0, 1, 2],
    [2, 1, 0, 1],
    [2, 1, 0, 2],
    [2, 1, 1, 0],
    [2, 1, 1, 1],
    [2, 2, 1, 0],
    [2, 2, 1, 2],
]
