"""
A module containing LUTs for primitive elements in GF(2^m).

Sage:
    p = 2
    for m in range(1, 7):
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

NORMAL_ELEMENTS_2_1 = [
    [1],
]

NORMAL_ELEMENTS_2_2 = [
    [1, 0],
    [1, 1],
]

NORMAL_ELEMENTS_2_3 = [
    [1, 1],
    [1, 0, 1],
    [1, 1, 1],
]

NORMAL_ELEMENTS_2_4 = [
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]

NORMAL_ELEMENTS_2_5 = [
    [1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
]

NORMAL_ELEMENTS_2_6 = [
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
]
