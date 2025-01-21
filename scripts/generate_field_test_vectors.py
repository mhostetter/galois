"""
Script to generate unit test vectors for finite field arithmetic.

Install SageMath:
* `sudo apt install sagemath`
"""

import json
import os
import pickle
import random
import shutil

import numpy as np
import sage
from sage.all import (
    GF,
    PolynomialRing,
    convolution,
    copy,
    crt,
    lcm,
    log,
    matrix,
    prod,
    vector,
    xgcd,
)

FIELD = None
RING = None
SPARSE_SIZE = 20
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")


def set_seed(seed):
    """Update the RNG seeds so the LUT is reproducible"""
    np.random.seed(seed)
    random.seed(seed)


def I(element):  # noqa: E743
    """Convert from various finite field elements to an integer"""
    if isinstance(element, sage.rings.finite_rings.element_pari_ffelt.FiniteFieldElement_pari_ffelt):
        coeffs = element._vector_()
        characteristic = int(FIELD.characteristic())
        return sum(int(c) * characteristic**i for i, c in enumerate(coeffs))
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
            integer -= q * characteristic**d
        return FIELD(" + ".join(l))
    try:
        return FIELD.fetch_int(int(integer))
    except:  # noqa: E722
        return FIELD(integer)


def arange(low, high, sparse=False):
    if sparse:
        if high <= np.iinfo(np.int64).max:
            X = np.random.randint(low, high, SPARSE_SIZE, dtype=np.int64)
        else:
            X = np.empty(SPARSE_SIZE, dtype=object)
            iterator = np.nditer(X, flags=["multi_index", "refs_ok"])
            for _ in iterator:
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
        for _ in iterator:
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


def list_to_poly(coeffs):
    """
    Converts a Python coefficient list (degree-descending order) into a Sage polynomial object.
    """
    return RING([F(e) for e in coeffs[::-1]])


def poly_to_list(poly):
    """
    Converts a Sage polynomial object into a Python list (degree-descending order).
    """
    if FIELD.order() <= np.iinfo(np.int64).max:
        dtype = np.int64
    else:
        dtype = np.object_
    z = np.array([I(e) for e in poly.list()[::-1]], dtype=dtype).tolist()
    z = z if z != [] else [0]
    return z


def save_pickle(d, folder, name):
    print(f"  Saving {name}...")
    with open(os.path.join(folder, name), "wb") as f:
        pickle.dump(d, f)


def save_json(d, folder, name, indent=False):
    indent = 4 if indent else None
    with open(os.path.join(folder, name), "w") as f:
        json.dump(d, f, indent=indent)


def make_luts(field, sub_folder, seed, sparse=False):
    global FIELD, RING
    print(f"Making LUTs for {field}")

    ###############################################################################
    # Finite field arithmetic
    ###############################################################################
    folder = os.path.join(PATH, "fields", "data", sub_folder)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    FIELD = field
    RING = PolynomialRing(field, names="x")
    characteristic = int(field.characteristic())
    order = int(field.order())
    dtype = np.int64 if order <= np.iinfo(np.int64).max else object
    alpha = field.primitive_element()
    # assert field.gen() == field.multiplicative_generator()

    d = {
        "characteristic": int(field.characteristic()),
        "degree": int(field.degree()),
        "order": int(field.order()),
        "primitive_element": I(field.primitive_element()),
        "irreducible_poly": [int(c) for c in field.modulus().list()[::-1]],
    }
    save_json(d, folder, "properties.json", indent=True)

    ###############################################################################
    # Arithmetic
    ###############################################################################

    set_seed(seed + 1)
    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i, j] = I(F(XX[i, j]) + F(YY[i, j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "add.pkl")

    set_seed(seed + 2)
    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i, j] = I(F(XX[i, j]) - F(YY[i, j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "subtract.pkl")

    set_seed(seed + 3)
    X, Y, XX, YY, ZZ = io_2d(0, order, 0, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i, j] = I(F(XX[i, j]) * F(YY[i, j]))
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "multiply.pkl")

    set_seed(seed + 4)
    X, Y, XX, YY, ZZ = io_2d(0, order, -order - 2, order + 3, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i, j] = I(F(XX[i, j]) * YY[i, j])
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "scalar_multiply.pkl")

    set_seed(seed + 5)
    X, Y, XX, YY, ZZ = io_2d(0, order, 1, order, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i, j] = I(F(XX[i, j]) / F(YY[i, j]))
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
    X, Y, XX, YY, ZZ = io_2d(1, order, -order - 2, order + 3, sparse=sparse)
    for i in range(ZZ.shape[0]):
        for j in range(ZZ.shape[1]):
            ZZ[i, j] = I(F(XX[i, j]) ** YY[i, j])
    d = {"X": X, "Y": Y, "Z": ZZ}
    save_pickle(d, folder, "power.pkl")

    set_seed(seed + 9)
    X, Z = io_1d(1, order, sparse=sparse)
    for i in range(Z.shape[0]):
        try:
            Z[i] = I(field.fetch_int(X[i]).log(alpha))
        except:  # noqa: E722
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
        z = poly_to_list(p)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "characteristic_poly_element.pkl")

    set_seed(seed + 13)
    shapes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    X = []
    Z = []
    for i in range(len(shapes)):
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        p = x.charpoly()
        z = poly_to_list(p)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "characteristic_poly_matrix.pkl")

    set_seed(seed + 14)
    X, _ = io_1d(0, order, sparse=sparse)
    Z = []
    for i in range(len(X)):
        x = F(X[i])
        p = x.minpoly()
        z = poly_to_list(p)
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
    # Advanced arithmetic
    ###############################################################################

    set_seed(seed + 101)
    X = []
    Y = []
    Z = []
    for _ in range(3):
        x = randint_matrix(0, order, (10,))
        y = randint_matrix(0, order, (10,))
        X.append(x)
        Y.append(y)
        x = vector(FIELD, [F(xi) for xi in x])
        y = vector(FIELD, [F(yi) for yi in y])
        z = convolution(x, y)
        z = np.array([I(zi) for zi in z], dtype)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "convolve.pkl")

    ###############################################################################
    # Linear algebra
    ###############################################################################

    set_seed(seed + 201)
    X_shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4)]
    Y_shapes = [(2, 2), (3, 3), (2, 4), (3, 3), (4, 5)]
    X = []
    Y = []
    Z = []
    for i in range(len(shapes)):
        x = randint_matrix(0, order, X_shapes[i])
        y = randint_matrix(0, order, Y_shapes[i])
        X.append(x)
        Y.append(y)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        y = matrix(FIELD, [[F(e) for e in row] for row in y])
        z = x * y
        z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "matrix_multiply.pkl")

    set_seed(seed + 202)
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4)]
    X = []
    Z = []
    for i in range(len(shapes)):
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        z = x.rref()
        z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "row_reduce.pkl")

    set_seed(seed + 203)
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6)]
    X = []
    L = []
    U = []
    for i in range(len(shapes)):
        while True:
            # Ensure X has a PLU decomposition with P = I, which means it has an LU decomposition
            x = randint_matrix(0, order, shapes[i])
            x_orig = x.copy()
            x = matrix(FIELD, [[F(e) for e in row] for row in x])
            p, l, u = x.LU()
            if p == matrix.identity(FIELD, shapes[i][0]):
                break
        X.append(x_orig)
        l = np.array([[I(e) for e in row] for row in l], dtype)
        u = np.array([[I(e) for e in row] for row in u], dtype)
        L.append(l)
        U.append(u)
    d = {"X": X, "L": L, "U": U}
    save_pickle(d, folder, "lu_decompose.pkl")

    set_seed(seed + 204)
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 5), (6, 6)]
    X = []
    L = []
    U = []
    P = []
    for i in range(len(shapes)):
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        p, l, u = x.LU()
        p = np.array([[I(e) for e in row] for row in p], dtype)
        l = np.array([[I(e) for e in row] for row in l], dtype)
        u = np.array([[I(e) for e in row] for row in u], dtype)
        P.append(p)
        L.append(l)
        U.append(u)
    d = {"X": X, "P": P, "L": L, "U": U}
    save_pickle(d, folder, "plu_decompose.pkl")

    set_seed(seed + 205)
    shapes = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
    X = []
    Z = []
    for i in range(len(shapes)):
        while True:
            x = randint_matrix(0, order, shapes[i])
            x_orig = x.copy()
            x = matrix(FIELD, [[F(e) for e in row] for row in x])
            if x.rank() == shapes[i][0]:
                break
        X.append(x_orig)
        z = x.inverse()
        z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "matrix_inverse.pkl")

    set_seed(seed + 206)
    shapes = [
        (2, 2),
        (2, 2),
        (2, 2),
        (3, 3),
        (3, 3),
        (3, 3),
        (4, 4),
        (4, 4),
        (4, 4),
        (5, 5),
        (5, 5),
        (5, 5),
        (6, 6),
        (6, 6),
        (6, 6),
    ]
    X = []
    Z = []
    for i in range(len(shapes)):
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        z = I(x.determinant())
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "matrix_determinant.pkl")

    set_seed(seed + 207)
    shapes = [
        (2, 2),
        (2, 2),
        (2, 2),
        (3, 3),
        (3, 3),
        (3, 3),
        (4, 4),
        (4, 4),
        (4, 4),
        (5, 5),
        (5, 5),
        (5, 5),
        (6, 6),
        (6, 6),
        (6, 6),
    ]
    X = []
    Y = []
    Z = []
    for i in range(len(shapes)):
        while True:
            x = randint_matrix(0, order, shapes[i])
            x_orig = x.copy()
            x = matrix(FIELD, [[F(e) for e in row] for row in x])
            if x.rank() == shapes[i][0]:
                break
        X.append(x_orig)
        y = randint_matrix(0, order, shapes[i][1])  # 1-D vector
        Y.append(y)
        y = vector(FIELD, [F(e) for e in y])
        z = x.solve_right(y)
        z = np.array([I(e) for e in z], dtype)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "matrix_solve.pkl")

    set_seed(seed + 208)
    shapes = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (3, 3)]
    X = []
    Z = []
    for i in range(len(shapes)):
        deg = shapes[i][1]  # The degree of the vector space

        # Random matrix
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        z = x.row_space()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)

        # Reduce the row space by 1 by copying the 0th row to the jth row
        for j in range(1, shapes[i][0]):
            x = copy(x)
            x[j, :] = F(random.randint(0, order - 1)) * x[0, :]

            z = x.row_space()
            if z.dimension() == 0:
                z = randint_matrix(0, 1, (0, deg))
            else:
                z = z.basis_matrix()
                z = np.array([[I(e) for e in row] for row in z], dtype)
            X.append(np.array([[I(e) for e in row] for row in x], dtype))
            Z.append(z)

        # Zero matrix
        x = copy(x)
        x[:] = F(0)
        z = x.row_space()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        X.append(np.array([[I(e) for e in row] for row in x], dtype))
        Z.append(z)

    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "row_space.pkl")

    set_seed(seed + 209)
    shapes = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (3, 3)]
    X = []
    Z = []
    for i in range(len(shapes)):
        deg = shapes[i][0]  # The degree of the vector space

        # Random matrix
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        z = x.column_space()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)

        # Reduce the column space by 1 by copying the 0th column to the jth column
        for j in range(1, shapes[i][1]):
            x = copy(x)
            x[:, j] = F(random.randint(0, order - 1)) * x[:, 0]

            z = x.column_space()
            if z.dimension() == 0:
                z = randint_matrix(0, 1, (0, deg))
            else:
                z = z.basis_matrix()
                z = np.array([[I(e) for e in row] for row in z], dtype)
            X.append(np.array([[I(e) for e in row] for row in x], dtype))
            Z.append(z)

        # Zero matrix
        x = copy(x)
        x[:] = F(0)
        z = x.column_space()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        X.append(np.array([[I(e) for e in row] for row in x], dtype))
        Z.append(z)

    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "column_space.pkl")

    set_seed(seed + 210)
    shapes = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (3, 3)]
    X = []
    Z = []
    for i in range(len(shapes)):
        deg = shapes[i][0]  # The degree of the vector space

        # Random matrix
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        z = x.left_kernel()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)

        # Reduce the left null space by 1 by copying the 0th row to the jth row
        for j in range(1, shapes[i][0]):
            x = copy(x)
            x[j, :] = F(random.randint(0, order - 1)) * x[0, :]

            z = x.left_kernel()
            if z.dimension() == 0:
                z = randint_matrix(0, 1, (0, deg))
            else:
                z = z.basis_matrix()
                z = np.array([[I(e) for e in row] for row in z], dtype)
            X.append(np.array([[I(e) for e in row] for row in x], dtype))
            Z.append(z)

        # Zero matrix
        x = copy(x)
        x[:] = F(0)
        z = x.left_kernel()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        X.append(np.array([[I(e) for e in row] for row in x], dtype))
        Z.append(z)

    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "left_null_space.pkl")

    set_seed(seed + 211)
    shapes = [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (3, 3)]
    X = []
    Z = []
    for i in range(len(shapes)):
        deg = shapes[i][1]  # The degree of the vector space

        # Random matrix
        x = randint_matrix(0, order, shapes[i])
        X.append(x)
        x = matrix(FIELD, [[F(e) for e in row] for row in x])
        z = x.right_kernel()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)

        # Reduce the null space by 1 by copying the 0th column to the jth column
        for j in range(1, shapes[i][1]):
            x = copy(x)
            x[:, j] = F(random.randint(0, order - 1)) * x[:, 0]

            z = x.right_kernel()
            if z.dimension() == 0:
                z = randint_matrix(0, 1, (0, deg))
            else:
                z = z.basis_matrix()
                z = np.array([[I(e) for e in row] for row in z], dtype)
            X.append(np.array([[I(e) for e in row] for row in x], dtype))
            Z.append(z)

        # Zero matrix
        x = copy(x)
        x[:] = F(0)
        z = x.right_kernel()
        if z.dimension() == 0:
            z = randint_matrix(0, 1, (0, deg))
        else:
            z = z.basis_matrix()
            z = np.array([[I(e) for e in row] for row in z], dtype)
        X.append(np.array([[I(e) for e in row] for row in x], dtype))
        Z.append(z)

    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "null_space.pkl")

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
        x = list_to_poly(X[i])
        y = list_to_poly(Y[i])
        z = x + y
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "add.pkl")

    set_seed(seed + 102)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = list_to_poly(Y[i])
        z = x - y
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "subtract.pkl")

    set_seed(seed + 103)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = list_to_poly(Y[i])
        z = x * y
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "multiply.pkl")

    set_seed(seed + 104)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random.randint(1, 2 * characteristic) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = Y[i]
        z = x * y
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "scalar_multiply.pkl")

    set_seed(seed + 105)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    # Add some specific polynomial types
    X.append([0]), Y.append(random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS))  # 0 / y
    (
        X.append(random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS // 2)),
        Y.append(random_coeffs(0, order, MAX_COEFFS // 2, MAX_COEFFS)),
    )  # x / y with x.degree < y.degree
    X.append(random_coeffs(0, order, 2, MAX_COEFFS)), Y.append(random_coeffs(0, order, 1, 2))  # x / y with y.degree = 0
    Q = []
    R = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = list_to_poly(Y[i])
        q = x // y
        r = x % y
        q = poly_to_list(q)
        Q.append(q)
        r = poly_to_list(r)
        R.append(r)
    d = {"X": X, "Y": Y, "Q": Q, "R": R}
    save_pickle(d, folder, "divmod.pkl")

    set_seed(seed + 106)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(4)]
    X.append(random_coeffs(0, order, 1, 2))
    Y = [0, 1, 2, 3]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        ZZ = []
        for j in range(len(Y)):
            y = Y[j]
            z = x**y
            z = poly_to_list(z)
            ZZ.append(z)
        Z.append(ZZ)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "power.pkl")

    set_seed(seed + 107)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = arange(0, order, sparse=sparse)
    Z = np.array(np.zeros((len(X), len(Y))), dtype=dtype)
    for i in range(len(X)):
        for j in range(len(Y)):
            x = list_to_poly(X[i])
            y = F(Y[j])
            z = x(y)
            Z[i, j] = I(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "evaluate.pkl")

    set_seed(seed + 108)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [randint_matrix(0, order, (2, 2)) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = matrix(FIELD, [[F(e) for e in row] for row in Y[i]])
        z = x(y)
        z = np.array([[I(e) for e in row] for row in z], dtype)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "evaluate_matrix.pkl")

    set_seed(seed + 109)
    X = [random_coeffs(0, order, 1, 6) for i in range(20)]
    Y = [random_coeffs(0, order, 1, 6) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = list_to_poly(Y[i])
        z = x(y)
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "evaluate_poly.pkl")

    ###############################################################################
    # Polynomial arithmetic methods
    ###############################################################################

    set_seed(seed + 301)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        z = x.reverse()
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "reverse.pkl")

    set_seed(seed + 302)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    R = []
    M = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        roots = x.roots()
        RR, MM = [], []
        for root in roots:
            r = root[0]
            m = root[1]
            RR.append(I(r))
            MM.append(int(m))
        idxs = np.argsort(RR)  # Sort by ascending roots
        RR = (np.array(RR, dtype=dtype)[idxs]).tolist()
        MM = (np.array(MM, dtype=dtype)[idxs]).tolist()
        R.append(RR)
        M.append(MM)
    d = {"X": X, "R": R, "M": M}
    save_pickle(d, folder, "roots.pkl")

    set_seed(seed + 303)
    X = [random_coeffs(0, order, 2 * FIELD.degree(), 6 * FIELD.degree()) for i in range(20)]
    Y = [1] * 10 + [random.randint(2, FIELD.degree() + 1) for i in range(10)]
    Z = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        z = x.derivative(Y[i])
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "derivative.pkl")

    ###############################################################################
    # Polynomial arithmetic functions
    ###############################################################################

    set_seed(seed + 401)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Y = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    D = []
    S = []
    T = []
    for i in range(len(X)):
        x = list_to_poly(X[i])
        y = list_to_poly(Y[i])
        d, s, t = xgcd(x, y)
        d = poly_to_list(d)
        s = poly_to_list(s)
        t = poly_to_list(t)
        D.append(d)
        S.append(s)
        T.append(t)
    d = {"X": X, "Y": Y, "D": D, "S": S, "T": T}
    save_pickle(d, folder, "egcd.pkl")

    set_seed(seed + 402)
    X = []
    Z = []
    for i in range(20):
        XX = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(random.randint(2, 5))]
        X.append(XX)
        xx = [list_to_poly(XXi) for XXi in XX]
        z = lcm(xx)
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "lcm.pkl")

    set_seed(seed + 403)
    X = []
    Z = []
    for i in range(20):
        XX = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(random.randint(2, 5))]
        X.append(XX)
        xx = [list_to_poly(XXi) for XXi in XX]
        z = prod(xx)
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "prod.pkl")

    set_seed(seed + 404)
    X = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    E = [random.randint(2, 10) for i in range(20)]
    M = [random_coeffs(0, order, MIN_COEFFS, MAX_COEFFS) for i in range(20)]
    Z = []
    for i in range(20):
        x = list_to_poly(X[i])
        e = E[i]
        m = list_to_poly(M[i])
        z = (x**e) % m
        z = poly_to_list(z)
        Z.append(z)
    d = {"X": X, "E": E, "M": M, "Z": Z}
    save_pickle(d, folder, "modular_power.pkl")

    set_seed(seed + 405)
    X = [0] * 20  # The remainder
    Y = [0] * 20  # The modulus
    Z = [0] * 20  # The solution
    for i in range(20):
        n = random.randint(2, 4)  # The number of polynomials
        x, y = [], []
        for _ in range(n):
            d = random.randint(3, 5)
            x.append(random_coeffs(0, order, d, d + 1))
            y.append(random_coeffs(0, order, d + 1, d + 2))  # Ensure modulus degree is greater than remainder degree
        X[i] = x
        Y[i] = y
        try:
            x = [list_to_poly(xx) for xx in x]
            y = [list_to_poly(yy) for yy in y]
            z = crt(x, y)
            Z[i] = poly_to_list(z)
        except:  # noqa: E722
            Z[i] = None
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "crt.pkl")

    set_seed(seed + 406)
    X = [0] * 3  # The x values
    Y = [0] * 3  # The y values
    Z = [0] * 3  # The lagrange polynomial
    for i in range(3):
        N = min(10, order)
        if dtype is object:
            x = randint_matrix(0, order, (N,))
        else:
            x = np.array(random.sample(range(0, order), N), dtype=dtype)
        y = randint_matrix(0, order, (N,))
        points = [(F(x[i]), F(y[i])) for i in range(N)]
        z = RING.lagrange_polynomial(points)
        X[i] = x
        Y[i] = y
        Z[i] = poly_to_list(z)
    d = {"X": X, "Y": Y, "Z": Z}
    save_pickle(d, folder, "lagrange_poly.pkl")

    ###############################################################################
    # Special polynomials
    ###############################################################################

    set_seed(seed + 501)
    X = [random_coeffs(0, order, 1, 6) for _ in range(20)]
    Z = [False] * len(X)
    for i in range(len(X)):
        if random.choice(["one", "other"]) == "one":
            X[i][0] = 1
        x = list_to_poly(X[i])
        z = x.is_monic()
        Z[i] = bool(z)
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "is_monic.pkl")

    set_seed(seed + 502)
    IS = []
    IS_NOT = []
    if order <= 2**16:
        while len(IS) < 10:
            x = random_coeffs(0, order, 1, 6)
            f = list_to_poly(x)
            if f.is_irreducible():
                IS.append(x)
        while len(IS_NOT) < 10:
            x = random_coeffs(0, order, 1, 6)
            f = list_to_poly(x)
            if not f.is_irreducible():
                IS_NOT.append(x)
    d = {"IS": IS, "IS_NOT": IS_NOT}
    save_pickle(d, folder, "is_irreducible.pkl")

    set_seed(seed + 503)
    IS = []
    IS_NOT = []
    if order <= 2**16:
        while len(IS) < 10:
            x = random_coeffs(0, order, 1, 6)
            f = list_to_poly(x)
            # f = f / f.coefficients()[-1]  # Make monic
            # assert f.is_monic()
            if f.degree() == 1 and f.coefficients(sparse=False)[0] == 0:
                continue  # For some reason `is_primitive()` crashes on f(x) = a*x
            if not f.is_irreducible():
                continue  # Want to find an irreducible polynomial that is also primitive
            if f.is_primitive():
                IS.append(x)
        while len(IS_NOT) < 10:
            x = random_coeffs(0, order, 1, 6)
            f = list_to_poly(x)
            # f = f / f.coefficients()[-1]  # Make monic
            # assert f.is_monic()
            if f.degree() == 1 and f.coefficients(sparse=False)[0] == 0:
                continue  # For some reason `is_primitive()` crashes on f(x) = a*x
            if not f.is_irreducible():
                continue  # Want to find an irreducible polynomial that is not primitive
            if not f.is_primitive():
                IS_NOT.append(x)
    d = {"IS": IS, "IS_NOT": IS_NOT}
    save_pickle(d, folder, "is_primitive.pkl")

    set_seed(seed + 504)
    if order <= 2**16:
        X = [random_coeffs(0, order, 1, 6) for _ in range(20)]
        Z = [False] * len(X)
        for i in range(len(X)):
            x = list_to_poly(X[i])
            z = x.is_squarefree()
            Z[i] = bool(z)
    else:
        X = []
        Z = []
    d = {"X": X, "Z": Z}
    save_pickle(d, folder, "is_square_free.pkl")


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

    field = GF(2**8, "x", modulus=[1, 1, 0, 1, 1, 0, 0, 0, 1], repr="int")
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

    field = GF(7**3, "x", modulus=[6, 0, 6, 1], repr="int")
    sub_folder = "GF(7^3, 643, 244)"
    seed = 123456789 + 15000
    make_luts(field, sub_folder, seed)

    field = GF(109987**4, "x", modulus="primitive", repr="int")
    sub_folder = "GF(109987^4)"
    seed = 123456789 + 16000
    make_luts(field, sub_folder, seed, sparse=True)
