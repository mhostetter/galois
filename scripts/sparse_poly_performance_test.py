import numpy as np
from IPython import get_ipython
ipython = get_ipython()
assert ipython is not None, "Must run this script with ipython3"

import galois


def get_coeffs(degree, N, field):
    degrees = np.random.randint(0, degree + 1, N, dtype=int)
    degrees[0] = degree
    coeffs = field.Random(N, low=1)
    return degrees, coeffs


if True:
    print("BinaryPoly vs SparsePoly".center(80, "-"))
    GF = galois.GF2
    N = 3  # Number of non-zero coefficients
    degree = 500
    while degree <= 25_000:
        print(f"Nonzero: {N} / {degree}, {N/degree*100} %")
        sp1 = galois.poly.SparsePoly(*get_coeffs(degree, N, GF))
        sp2 = galois.poly.SparsePoly(*get_coeffs(degree, N, GF))
        bp1 = galois.poly.BinaryPoly(sp1.integer)
        bp2 = galois.poly.BinaryPoly(sp2.integer)

        print("  BinaryPoly:\t", end="")
        ipython.magic("timeit bp1 * bp2")

        print("  SparsePoly:\t", end="")
        ipython.magic("timeit sp1 * sp2")

        degree *= 2


if True:
    print("DensePoly vs SparsePoly".center(80, "-"))
    GF = galois.GF(31)
    N = 3  # Number of non-zero coefficients
    degree = 500
    while degree <= 25_000:
        print(f"Nonzero: {N} / {degree}, {N/degree*100} %")
        sp1 = galois.poly.SparsePoly(*get_coeffs(degree, N, GF))
        sp2 = galois.poly.SparsePoly(*get_coeffs(degree, N, GF))
        dp1 = galois.poly.BinaryPoly(sp1.integer)
        dp2 = galois.poly.BinaryPoly(sp2.integer)

        print("  DensePoly:\t", end="")
        ipython.magic("timeit dp1 * dp2")

        print("  SparsePoly:\t", end="")
        ipython.magic("timeit sp1 * sp2")

        degree *= 2
