import numpy as np
from IPython import get_ipython

import galois

ipython = get_ipython()
assert ipython is not None, "Must run this script with ipython3"


def get_coeffs(degree, N, field):
    while True:
        nonzero_degrees = np.random.randint(0, degree + 1, N, dtype=int)
        nonzero_degrees[0] = degree
        nonzero_coeffs = field.Random(N, low=1)
        if nonzero_degrees.size == np.unique(nonzero_degrees).size:
            break

    return nonzero_degrees, nonzero_coeffs


GF = galois.GF(2**8)


# print("DensePoly vs SparsePoly Addition".center(80, "-"))
# N = 3  # Number of non-zero coefficients
# degree = 2000
# while degree <= 32000:
#     print(f"Nonzero: {N} / {degree}, {N/degree*100} %")
#     p1 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))
#     p2 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))

#     print("  SparsePoly:\t", end="")
#     p1._type = "sparse"
#     p2._type = "sparse"
#     p1.nonzero_degrees, p1.nonzero_coeffs, p2.nonzero_degrees, p2.nonzero_coeffs
#     ipython.run_line_magic("timeit", "p1 + p2")

#     print("  DensePoly:\t", end="")
#     p1._type = "dense"
#     p2._type = "dense"
#     p1.coeffs, p2.coeffs  # Ensure _coeffs is created for arithmetic
#     ipython.run_line_magic("timeit", "p1 + p2")

#     degree *= 2


print("DensePoly vs SparsePoly Multiplication".center(80, "-"))
N = 3  # Number of non-zero coefficients
degree = 100
while degree <= 1000:
    print(f"Nonzero: {N} / {degree}, {N / degree * 100} %")
    p1 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))
    p2 = galois.Poly.Degrees(*get_coeffs(degree, N, GF))

    print("  SparsePoly:\t", end="")
    p1._type = "sparse"
    p2._type = "sparse"
    print(p1.nonzero_degrees, p1.nonzero_coeffs, p2.nonzero_degrees, p2.nonzero_coeffs)
    ipython.run_line_magic("timeit", "p1 * p2")

    print("  DensePoly:\t", end="")
    p1._type = "dense"
    p2._type = "dense"
    print(p1.coeffs, p2.coeffs)  # Ensure _coeffs is created for arithmetic
    ipython.run_line_magic("timeit", "p1 * p2")

    degree += 100
