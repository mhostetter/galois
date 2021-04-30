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

import numpy as np
from sage.all import GF, PolynomialRing, log

FIELD = None
SPARSE_SIZE = 20


def I(element):
    """Convert from various finite field elements to an integer"""
    try:
        return int(element)
    except TypeError:
        return element.integer_representation()


def F(integer):
    """Convert from an integer to various finite field elements"""
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


def make_luts(field, folder, sparse=False):
    global FIELD

    print(f"Making LUTs for {field}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    FIELD = field
    order = int(field.order())
    dtype = np.int64 if order <= np.iinfo(np.int64).max else object
    alpha = field.primitive_element()
    ring = PolynomialRing(field, names="x")
    # assert field.gen() == field.multiplicative_generator()

    d = {
        "characteristic": int(field.characteristic()),
        "degree": int(field.degree()),
        "order": int(field.order()),
        "alpha": I(field.primitive_element()),
        "prim_poly": [int(c) for c in field.modulus().list()[::-1]]
    }
    save_json(d, folder, "properties.json", indent=True)

    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) + F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "add.pkl")

    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) -  F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "subtract.pkl")

    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) * F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "multiply.pkl")

    X, Y, XX, YY, ZZ = io_2d(0, order, -order-2, order+3, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) * YY[i,j])
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "scalar_multiply.pkl")

    X, Y, XX, YY, ZZ = io_2d(0, order, 1, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) /  F(YY[i,j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "divide.pkl")

    X, Z = io_1d(0, order, sparse=sparse)
    for i in range(X.shape[0]):
        Z[i] = I(-F(X[i]))
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "additive_inverse.pkl")

    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(X.shape[0]):
        Z[i] = I(1 / F(X[i]))
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "multiplicative_inverse.pkl")

    X, Y, XX, YY, ZZ = io_2d(1, order, -order-2, order+3, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i,j] = I(F(XX[i,j]) **  YY[i,j])
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "power.pkl")

    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(Z.shape[0]):
        try:
            Z[i] = I(field.fetch_int(X[i]).log(alpha))
        except:
            Z[i] = I(log(F(X[i]), alpha))
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "log.pkl")

    MIN_COEFFS = 1
    MAX_COEFFS = 12

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
    save_pickle(d, folder, "poly_add.pkl")

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
    save_pickle(d, folder, "poly_subtract.pkl")

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
    save_pickle(d, folder, "poly_multiply.pkl")

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
    save_pickle(d, folder, "poly_divmod.pkl")

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
    save_pickle(d, folder, "poly_power.pkl")

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
    save_pickle(d, folder, "poly_evaluate.pkl")


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data")

    np.random.seed(123456789 + 1), random.seed(123456789 + 1)
    field = GF(2, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2)")
    make_luts(field, folder)

    np.random.seed(123456789 + 2), random.seed(123456789 + 2)
    field = GF(5, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(5)")
    make_luts(field, folder)

    np.random.seed(123456789 + 3), random.seed(123456789 + 3)
    field = GF(7, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(7)")
    make_luts(field, folder)

    np.random.seed(123456789 + 4), random.seed(123456789 + 4)
    field = GF(31, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(31)")
    make_luts(field, folder)

    np.random.seed(123456789 + 5), random.seed(123456789 + 5)
    field = GF(3191, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(3191)")
    make_luts(field, folder, sparse=True)

    np.random.seed(123456789 + 6), random.seed(123456789 + 6)
    # prime = 2**31 - 1, small enough to fit in np.int64
    field = GF(2147483647, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2147483647)")
    make_luts(field, folder, sparse=True)

    np.random.seed(123456789 + 7), random.seed(123456789 + 7)
    # prime = 2**65 - 49, large enough to not fit in np.int64 and require np.object_
    field = GF(36893488147419103183, modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(36893488147419103183)")
    make_luts(field, folder, sparse=True)

    np.random.seed(123456789 + 8), random.seed(123456789 + 8)
    field = GF(2**2, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2^2)")
    make_luts(field, folder)

    np.random.seed(123456789 + 9), random.seed(123456789 + 9)
    field = GF(2**3, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2^3)")
    make_luts(field, folder)

    np.random.seed(123456789 + 10), random.seed(123456789 + 10)
    field = GF(2**8, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2^8)")
    make_luts(field, folder)

    np.random.seed(123456789 + 100), random.seed(123456789 + 100)
    field = GF(2**8, "x", modulus=[1,1,0,1,1,0,0,0,1], repr="int")
    folder = os.path.join(path, "GF(2^8, 0x11b, 19)")
    make_luts(field, folder)

    np.random.seed(123456789 + 11), random.seed(123456789 + 11)
    field = GF(2**32, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2^32)")
    make_luts(field, folder, sparse=True)

    np.random.seed(123456789 + 12), random.seed(123456789 + 12)
    field = GF(2**100, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "GF(2^100)")
    make_luts(field, folder, sparse=True)
