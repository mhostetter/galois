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


print("BinaryPoly vs SparsePoly".center(80, "-"))
GF = galois.GF2
N = 3  # Number of non-zero coefficients
degree = 10
while degree <= 100_000:
    p1 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))
    p2 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))
    print(f"Nonzero: {N} / {degree}, {N/degree*100} %")

    print("  BinaryPoly:\t", end="")
    ipython.magic("timeit galois.poly.BinaryPoly._mul(p1, p2)")

    print("  SparsePoly:\t", end="")
    ipython.magic("timeit galois.poly.SparsePoly._mul(p1, p2)")

    degree *= 2


print("DensePoly vs SparsePoly".center(80, "-"))
GF = galois.GF(31)
N = 3  # Number of non-zero coefficients
degree = 10
while degree <= 1_000:
    p1 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))
    p2 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))
    print(f"Nonzero: {N} / {degree}, {N/degree*100} %")

    print("  DensePoly:\t", end="")
    ipython.magic("timeit galois.poly.DensePoly._mul(p1, p2)")

    print("  SparsePoly:\t", end="")
    ipython.magic("timeit galois.poly.SparsePoly._mul(p1, p2)")

    degree *= 2
