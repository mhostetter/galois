"""
Script to generate unit test vectors for the galois package using Sage.

Install Sage with:
```
sudo apt install sagemath
```
"""
import json
import os
import pickle
import shutil
import numpy as np
from sage.all import *

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
        return FIELD.fetch_int(integer)
    except:
        return FIELD(integer)


def arange(x_low, x_high, sparse=False):
    if sparse:
        X = np.random.randint(x_low, x_high, SPARSE_SIZE, dtype=int)
    else:
        X = np.arange(x_low, x_high, dtype=int)
    return X


def io_1d(x_low, x_high, sparse=False):
    X = arange(x_low, x_high, sparse=sparse)
    Z = np.zeros(X.shape, dtype=int)
    return X, Z


def io_2d(x_low, x_high, y_low, y_high, sparse=False):
    X, Y = np.meshgrid(arange(x_low, x_high, sparse=sparse), arange(y_low, y_high, sparse=sparse), indexing="ij")
    Z = np.zeros(X.shape, dtype=int)
    return X, Y, Z


def random_coeffs(low, high, size_low, size_high):
    size = np.random.randint(size_low, size_high)
    return [np.random.randint(low, high) for i in range(size)]


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
    order = field.order()
    ring = PolynomialRing(field, names="x")
    assert field.gen() == field.multiplicative_generator()

    d = {
        "characteristic": int(field.characteristic()),
        "degree": int(field.degree()),
        "order": int(field.order()),
        "alpha": I(field.primitive_element()),
        "prim_poly": np.flip(np.array(field.modulus().list(), dtype=int)).tolist()
    }
    save_json(d, folder, "properties.json", indent=True)

    X, Y, Z = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = I(F(X[i,j]) + F(Y[i,j]))
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "add.pkl")

    X, Y, Z = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = I(F(X[i,j]) -  F(Y[i,j]))
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "subtract.pkl")

    X, Y, Z = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = I(F(X[i,j]) * F(Y[i,j]))
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "multiply.pkl")

    X, Y, Z = io_2d(0, order, -order-2, order+3, sparse=sparse)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = I(F(X[i,j]) * Y[i,j])
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "multiple_add.pkl")

    X, Y, Z = io_2d(0, order, 1, order, sparse=sparse)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = I(F(X[i,j]) /  F(Y[i,j]))
    d = {"X": X, "Y": Y, "Z": Z}
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

    X, Y, Z = io_2d(1, order, -order-2, order+3, sparse=sparse)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = I(F(X[i,j]) **  Y[i,j])
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "power.pkl")

    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(Z.shape[0]):
        try:
            # TODO: Figure out why we need to mod by (order - 1)
            Z[i] = field.int_to_log(X[i]) % (order - 1)
        except:
            Z[i] = I(log(F(X[i])))
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
        z = np.array([I(e) for e in z.list()[::-1]], dtype=int).tolist()
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
        z = np.array([I(e) for e in z.list()[::-1]], dtype=int).tolist()
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
        z = np.array([I(e) for e in z.list()[::-1]], dtype=int).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "poly_multiply.pkl")

    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    # Ensure no zero polynomials in Y for division
    for i in range(len(Y)):
        while np.array(Y[i]).sum() == 0:
            Y[i] = random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS)
    Q = []
    R = []
    for i in range(len(X)):
        x = ring([F(e) for e in X[i][::-1]])
        y = ring([F(e) for e in Y[i][::-1]])
        q = x // y
        r = x % y
        q = np.array([I(e) for e in q.list()[::-1]], dtype=int).tolist()
        q = q if q != [] else [0]
        Q.append(q)
        r = np.array([I(e) for e in r.list()[::-1]], dtype=int).tolist()
        r = r if r != [] else [0]
        R.append(r)
    d = {"X": X, "Y": Y, "Q": Q, "R": R}
    save_pickle(d, folder, "poly_divmod.pkl")

    X = random_coeffs(0, order, 1, 6)
    Y = np.arange(0, 5+1, dtype=int)
    Z = []
    x = ring([F(e) for e in X[::-1]])
    for j in range(len(Y)):
        y = Y[j]
        z = x ** y
        z = np.array([I(e) for e in z.list()[::-1]], dtype=int).tolist()
        z = z if z != [] else [0]
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "poly_power.pkl")

    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = arange(0, order, sparse=sparse)
    Z = np.zeros((len(X),len(Y)), dtype=int)
    for i in range(len(X)):
        for j in range(len(Y)):
            x = ring([F(e) for e in X[i][::-1]])
            y = F(Y[j])
            z = x(y)
            Z[i,j] = I(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "poly_evaluate.pkl")


if __name__ == "__main__":
    # Seed the RNG so the outputs are the same for subsequent runs
    np.random.seed(123456789)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    field = GF(2, modulus="primitive", repr="int")
    folder = os.path.join(path, "gf2")
    make_luts(field, folder)

    field = GF(5, modulus="primitive", repr="int")
    folder = os.path.join(path, "gf5")
    make_luts(field, folder)

    field = GF(7, modulus="primitive", repr="int")
    folder = os.path.join(path, "gf7")
    make_luts(field, folder)

    field = GF(31, modulus="primitive", repr="int")
    folder = os.path.join(path, "gf31")
    make_luts(field, folder)

    field = GF(3191, modulus="primitive", repr="int")
    folder = os.path.join(path, "gf3191")
    make_luts(field, folder, sparse=True)

    field = GF(2**2, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "gf4")
    make_luts(field, folder)

    field = GF(2**3, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "gf8")
    make_luts(field, folder)

    field = GF(2**8, "x", modulus="primitive", repr="int")
    folder = os.path.join(path, "gf256")
    make_luts(field, folder)
