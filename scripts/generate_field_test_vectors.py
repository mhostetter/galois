"""
Script to generate unit test vectors for the galois package using SageMath.

Install SageMath with:
```
sudo apt install sagemath
```
"""
import json
import os
import pickle
import random
import shutil

import sage
import numpy as np
from sage.all import GF, PolynomialRing, log, matrix

FIELD = None
SPARSE_SIZE = 20
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")


def set_seed(seed):
    """Update the RNG seeds so the LUT is reproducible"""
    np.random.seed(seed)
    random.seed(seed)


def I(element):
    """Convert from various finite field elements to an integer"""
    if isinstance(element, sage.rings.finite_rings.element_pari_ffelt.FiniteFieldElement_pari_ffelt):
        coeffs = element._vector_()
        characteristic = int(FIELD.characteristic())
        return sum(int(c)*characteristic**i for i, c in enumerate(coeffs))
    try:
        return int(element)
    except TypeError:
        return element.integer_representation()


def F(integer):
    """Convert from an integer to various finite field elements"""
    if isinstance(FIELD, sage.rings.finite_rings.finite_field_pari_ffelt.FiniteField_pari_ffelt):
        l = []
        characteristic = int(FIELD.characteristic())
        degree = int(FIELD.degree())
        for d in range(degree - 1, -1, -1):
            q = integer // characteristic**d
            l += [f"{q}*x^{d}"]
            integer -= q*characteristic**d
        return FIELD(" + ".join(l))
    try:
        return FIELD.fetch_int(int(integer))
    except:
        return FIELD(integer)


def arange(low, high, sparse=False):
    if sparse:
        if high <= np.iinfo(np.int64).max:
            X = np.random.randint(low, high, SPARSE_SIZE, dtype=np.int64)
        else:
            X = np.empty(SPARSE_SIZE, dtype=object)
            iterator = np.nditer(X, flags=["multi_index", "refs_ok"])
            for i in iterator:
                X[iterator.multi_index] = random.randint(low, high - 1)
    else:
        X = np.arange(low, high, dtype=np.int64)

    if sparse:
        # Set a random element to the max (needed for testing for overflows)
        idx = random.randint(0, X.size - 2)
        X[idx] = high - 1

    return X


def randint_matrix(low, high, shape):
    if high <= np.iinfo(np.int64).max:
        X = np.random.randint(low, high, shape, dtype=np.int64)
    else:
        X = np.empty(shape, dtype=object)
        iterator = np.nditer(X, flags=["multi_index", "refs_ok"])
        for i in iterator:
            X[iterator.multi_index] = random.randint(low, high - 1)

    return X


def io_1d(low, high, sparse=False):
    X = arange(low, high, sparse=sparse)

    if high <= np.iinfo(np.int64).max:
        Z = np.zeros(X.shape, dtype=np.int64)
    else:
        Z = np.array(np.zeros(X.shape), dtype=object)

    return X, Z


def io_2d(x_low, x_high, y_low, y_high, sparse=False):
    X = arange(x_low, x_high, sparse=sparse)
    Y = arange(y_low, y_high, sparse=sparse)

    if sparse:
        # Set both random elements to the max (needed for testing for overflows)
        X[-1] = x_high - 1
        Y[-1] = y_high - 1

    XX, YY = np.meshgrid(X, Y, indexing="ij")

    if x_high <= np.iinfo(np.int64).max and y_high <= np.iinfo(np.int64).max:
        ZZ = np.zeros(XX.shape, dtype=np.int64)
    else:
        ZZ = np.array(np.zeros(XX.shape), dtype=object)

    return X, Y, XX, YY, ZZ


def random_coeffs(low, high, size_low, size_high):
    size = random.randint(size_low, size_high - 1)
    coeffs = [random.randint(low, high - 1) for _ in range(size)]
    if low == 0:
        # Ensure the leading coefficient is non-zero
        coeffs[0] = random.randint(low + 1, high - 1)
    return coeffs


def save_pickle(d, folder, name):
    with open(os.path.join(folder, name), "wb") as f:
        pickle.dump(d, f)


def save_json(d, folder, name, indent=False):
    indent = 4 if indent else None
    with open(os.path.join(folder, name), "w") as f:
        json.dump(d, f, indent=indent)


def make_luts(field, sub_folder, seed, sparse=False):
    global FIELD
    print(f"Making LUTs for {field}")

    ###############################################################################
    # Finite field arithmetic
    ###############################################################################
    folder = os.path.join(PATH, "fields", "data", sub_folder)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    FIELD = field
    characteristic = int(field.characteristic())
    order = int(field.order())
    dtype = np.int64 if order <= np.iinfo(np.int64).max else object
    alpha = field.primitive_element()
    ring = PolynomialRing(field, names="x")
    # assert field.gen() == field.multiplicative_generator()

    d = {
        "characteristic": int(field.characteristic()),
        "degree": int(field.degree()),
        "order": int(field.order()),
        "primitive_element": I(field.primitive_element()),
        "irreducible_poly": [int(c) for c in field.modulus().list()[::-1]]
    }
    save_json(d, folder, "properties.json", indent=True)

    set_seed(seed + 1)
    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) + F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "add.pkl")

    set_seed(seed + 2)
    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) -  F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "subtract.pkl")

    set_seed(seed + 3)
    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) * F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "multiply.pkl")

    set_seed(seed + 4)
    X, Y, XX, YY, ZZ = io_2d(0, order, -order-2, order+3, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) * YY[i,j])
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "scalar_multiply.pkl")

    set_seed(seed + 5)
    X, Y, XX, YY, ZZ = io_2d(0, order, 1, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) /  F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "divide.pkl")

    set_seed(seed + 6)
    X, Z = io_1d(0, order, sparse=sparse)
    for i in range(X.shape[0]):
        Z[i] = I(-F(X[i]))
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "additive_inverse.pkl")

    set_seed(seed + 7)
    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(X.shape[0]):
        Z[i] = I(1 / F(X[i]))
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "multiplicative_inverse.pkl")

    set_seed(seed + 8)
    X, Y, XX, YY, ZZ = io_2d(1, order, -order-2, order+3, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) **  YY[i,j])
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "power.pkl")

    set_seed(seed + 9)
    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(Z.shape[0]):
        try:
            Z[i] = I(field.fetch_int(X[i]).log(alpha))
        except:
            Z[i] = I(log(F(X[i]), alpha))
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "log.pkl")

    set_seed(seed + 10)
    X, Z = io_1d(0, order, sparse=sparse)
    for i in range(X.shape[0]):
        Z[i] = int(F(X[i]).additive_order())
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "additive_order.pkl")

    set_seed(seed + 11)
    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(X.shape[0]):
        Z[i] = int(F(X[i]).multiplicative_order())
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "multiplicative_order.pkl")

    set_seed(seed + 12)
    X, _ = io_1d(0, order, sparse=sparse)
    Z = []
    for i in range(len(X)):
        x = F(X[i])
        p = x.charpoly()
        z = np.array([I(e) for e in p.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "characteristic_poly_element.pkl")

    set_seed(seed + 13)
    shapes = [(2,2), (3,3), (4,4), (5,5), (6,6)]
    X = []
    Z = []
    for i in range(len(shapes)):
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        p = x.charpoly()
        z = np.array([I(e) for e in p.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "characteristic_poly_matrix.pkl")

    set_seed(seed + 14)
    X, _ = io_1d(0, order, sparse=sparse)
    Z = []
    for i in range(len(X)):
        x = F(X[i])
        p = x.minpoly()
        z = np.array([I(e) for e in p.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "minimal_poly_element.pkl")

    # set_seed(seed + 15)
    # shapes = [(2,2), (3,3), (4,4), (5,5), (6,6)]
    # X = []
    # Z = []
    # for i in range(len(shapes)):
    #     x = randint_matrix(0, order, shapes[i])
    #     X.append(x)
    #     x = matrix(FIELD, [[F(e) for e in row] for row in x])
    #     p = x.minpoly()
    #     z = np.array([I(e) for e in p.list()[::-1]], dtype=dtype).tolist()
    #     z = z if z != [] else [0]
    #     Z.append(z)
    # d = {"X": X, "Z": Z}
    # save_pickle(d, folder, "minimal_poly_matrix.pkl")

    set_seed(seed + 16)
    X, Z = io_1d(0, order, sparse=sparse)
    for i in range(X.shape[0]):
        z = F(X[i]).trace()
        Z[i] = int(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "field_trace.pkl")

    set_seed(seed + 17)
    X, Z = io_1d(0, order, sparse=sparse)
    for i in range(X.shape[0]):
        z = F(X[i]).norm()
        Z[i] = int(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "field_norm.pkl")

    ###############################################################################
    # Polynomial arithmetic
    ###############################################################################
    folder = os.path.join(PATH, "polys", "data", sub_folder)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    MIN_COEFFS = 1
    MAX_COEFFS = 12

    set_seed(seed + 101)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        y = ring([F(e) for e in Y[i][::-1]])
        z = x + y
        z = np.array([I(e) for e in z.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "add.pkl")

    set_seed(seed + 102)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        y = ring([F(e) for e in Y[i][::-1]])
        z = x - y
        z = np.array([I(e) for e in z.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "subtract.pkl")

    set_seed(seed + 103)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        y = ring([F(e) for e in Y[i][::-1]])
        z = x * y
        z = np.array([I(e) for e in z.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "multiply.pkl")

    set_seed(seed + 104)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random.randint(1, 2*characteristic) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        y = Y[i]
        z = x * y
        z = np.array([I(e) for e in z.list()[::-1]], dtype=dtype).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "scalar_multiply.pkl")

    set_seed(seed + 105)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    # Add some specific polynomial types
    X.append([0]), Y.append(random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS))  # 0 / y
    X.append(random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS//2)), Y.append(random_coeffs(0, order, MAX_COEFFS//2, MAX_COEFFS))  # x / y with x.degree < y.degree
    X.append(random_coeffs(0, order, 2, MAX_COEFFS)), Y.append(random_coeffs(0, order, 1, 2))  # x / y with y.degree = 0
    Q = []
    R = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        y = ring([F(e) for e in Y[i][::-1]])
        q = x // y
        r = x % y
        q = np.array([I(e) for e in q.list()[::-1]], dtype=dtype).tolist()
        q = q if q != [] else [0]
        Q.append(q)
        r = np.array([I(e) for e in r.list()[::-1]], dtype=dtype).tolist()
        r = r if r != [] else [0]
        R.append(r)
    d = {"X": X, "Y": Y, "Q": Q, "R": R}
    save_pickle(d, folder, "divmod.pkl")

    set_seed(seed + 106)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(4)]
    X.append(random_coeffs(0, order, 1, 2))
    Y = [0, 1, 2, 3]
    Z = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        ZZ = []
        for j in range(len(Y)):
            y = Y[j]
            z = x ** y
            z = np.array([I(e) for e in z.list()[::-1]], dtype=dtype).tolist()
            z = z if z != [] else [0]
            ZZ.append(z)
        Z.append(ZZ)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "power.pkl")

    set_seed(seed + 107)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = arange(0, order, sparse=sparse)
    Z = np.array(np.zeros((len(X),len(Y))), dtype=dtype)
    for i in range(len(X)):
        for j in range(len(Y)):
            x = ring([F(e) for e in X[i][::-1]])
            y = F(Y[j])
            z = x(y)
            Z[i,j] = I(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "evaluate.pkl")


if __name__ == "__main__":
    field = GF(2, modulus="primitive", repr="int")
    sub_folder = "GF(2)"
    seed = 123456789 + 1000
    make_luts(field, sub_folder, seed)

    field = GF(5, modulus="primitive", repr="int")
    sub_folder = "GF(5)"
    seed = 123456789 + 2000
    make_luts(field, sub_folder, seed)

    field = GF(7, modulus="primitive", repr="int")
    sub_folder = "GF(7)"
    seed = 123456789 + 3000
    make_luts(field, sub_folder, seed)

    field = GF(31, modulus="primitive", repr="int")
    sub_folder = "GF(31)"
    seed = 123456789 + 4000
    make_luts(field, sub_folder, seed)

    field = GF(3191, modulus="primitive", repr="int")
    sub_folder = "GF(3191)"
    seed = 123456789 + 5000
    make_luts(field, sub_folder, seed, sparse=True)

    # prime = 2**31 - 1, small enough to fit in np.int64
    field = GF(2147483647, modulus="primitive", repr="int")
    sub_folder = "GF(2147483647)"
    seed = 123456789 + 6000
    make_luts(field, sub_folder, seed, sparse=True)

    # prime = 2**65 - 49, large enough to not fit in np.int64 and require np.object_
    field = GF(36893488147419103183, modulus="primitive", repr="int")
    sub_folder = "GF(36893488147419103183)"
    seed = 123456789 + 7000
    make_luts(field, sub_folder, seed, sparse=True)

    field = GF(2**2, "x", modulus="primitive", repr="int")
    sub_folder = "GF(2^2)"
    seed = 123456789 + 8000
    make_luts(field, sub_folder, seed)

    field = GF(2**3, "x", modulus="primitive", repr="int")
    sub_folder = "GF(2^3)"
    seed = 123456789 + 9000
    make_luts(field, sub_folder, seed)

    field = GF(2**8, "x", modulus="primitive", repr="int")
    sub_folder = "GF(2^8)"
    seed = 123456789 + 10000
    make_luts(field, sub_folder, seed)

    field = GF(2**8, "x", modulus=[1,1,0,1,1,0,0,0,1], repr="int")
    sub_folder = "GF(2^8, 283, 19)"
    seed = 123456789 + 11000
    make_luts(field, sub_folder, seed)

    field = GF(2**32, "x", modulus="primitive", repr="int")
    sub_folder = "GF(2^32)"
    seed = 123456789 + 12000
    make_luts(field, sub_folder, seed, sparse=True)

    field = GF(2**100, "x", modulus="primitive", repr="int")
    sub_folder = "GF(2^100)"
    seed = 123456789 + 13000
    make_luts(field, sub_folder, seed, sparse=True)

    field = GF(7**3, "x", modulus="primitive", repr="int")
    sub_folder = "GF(7^3)"
    seed = 123456789 + 14000
    make_luts(field, sub_folder, seed)

    field = GF(7**3, "x", modulus=[6,0,6,1], repr="int")
    sub_folder = "GF(7^3, 643, 244)"
    seed = 123456789 + 15000
    make_luts(field, sub_folder, seed)

    field = GF(109987**4, "x", modulus="primitive", repr="int")
    sub_folder = "GF(109987^4)"
    seed = 123456789 + 16000
    make_luts(field, sub_folder, seed, sparse=True)
