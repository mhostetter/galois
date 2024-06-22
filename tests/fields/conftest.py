"""
A pytest conftest module that provides pytest fixtures for galois/fields/ tests.
"""

import json
import os
import pickle
import random

import numpy as np
import pytest

import galois

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FIELDS = [
    # Binary field
    pytest.param("GF(2)"),
    # Binary extension fields
    pytest.param("GF(2^2)"),
    pytest.param("GF(2^3)"),
    pytest.param("GF(2^8)"),
    pytest.param("GF(2^32)"),
    pytest.param("GF(2^100)"),
    # Prime fields
    pytest.param("GF(5)"),
    pytest.param("GF(7)"),
    pytest.param("GF(31)"),
    pytest.param("GF(3191)"),
    pytest.param("GF(2147483647)"),
    pytest.param("GF(36893488147419103183)"),
    # Prime extension fields
    pytest.param("GF(7^3)"),
    pytest.param("GF(109987^4)"),
]

FIELDS_DIFF_MODES = [
    # Binary field
    pytest.param("GF(2)-jit-calculate"),
    pytest.param("GF(2)-python-calculate"),
    # Binary extension fields
    pytest.param("GF(2^2)-jit-lookup"),
    pytest.param("GF(2^2)-jit-calculate"),
    pytest.param("GF(2^3)-jit-lookup"),
    pytest.param("GF(2^3)-jit-calculate"),
    pytest.param("GF(2^8)-jit-lookup"),
    pytest.param("GF(2^8)-jit-calculate"),
    pytest.param("GF(2^8, 283, 19)-jit-lookup"),
    pytest.param("GF(2^8, 283, 19)-jit-calculate"),
    pytest.param("GF(2^32)-jit-calculate"),
    pytest.param("GF(2^100)-python-calculate"),
    # Prime fields
    pytest.param("GF(5)-jit-lookup"),
    pytest.param("GF(5)-jit-calculate"),
    pytest.param("GF(5)-python-calculate"),
    pytest.param("GF(7)-jit-lookup"),
    pytest.param("GF(7)-jit-calculate"),
    pytest.param("GF(7)-python-calculate"),
    pytest.param("GF(31)-jit-lookup"),
    pytest.param("GF(31)-jit-calculate"),
    pytest.param("GF(3191)-jit-lookup"),
    pytest.param("GF(3191)-jit-calculate"),
    pytest.param("GF(2147483647)-jit-calculate"),
    pytest.param("GF(36893488147419103183)-python-calculate"),
    # Prime extension fields
    pytest.param("GF(7^3)-jit-lookup"),
    pytest.param("GF(7^3)-jit-calculate"),
    pytest.param("GF(7^3, 643, 244)-jit-lookup"),
    pytest.param("GF(7^3, 643, 244)-jit-calculate"),
    pytest.param("GF(109987^4)-python-calculate"),
]


def construct_field(folder):
    if len(folder.split("-")) >= 2:
        folder, ufunc_mode = folder.split("-", maxsplit=1)
    else:
        ufunc_mode = "auto"

    if folder == "GF(2)":
        GF = galois.GF(2, compile=ufunc_mode)

    elif folder == "GF(5)":
        GF = galois.GF(5, compile=ufunc_mode)
    elif folder == "GF(7)":
        GF = galois.GF(7, compile=ufunc_mode)
    elif folder == "GF(31)":
        GF = galois.GF(31, compile=ufunc_mode)
    elif folder == "GF(3191)":
        GF = galois.GF(3191, compile=ufunc_mode)
    elif folder == "GF(2147483647)":
        GF = galois.GF(2147483647, compile=ufunc_mode)
    elif folder == "GF(36893488147419103183)":
        GF = galois.GF(36893488147419103183, compile=ufunc_mode)

    elif folder == "GF(2^2)":
        GF = galois.GF(2**2, compile=ufunc_mode)
    elif folder == "GF(2^3)":
        GF = galois.GF(2**3, compile=ufunc_mode)
    elif folder == "GF(2^8)":
        GF = galois.GF(2**8, compile=ufunc_mode)
    elif folder == "GF(2^8, 283, 19)":
        GF = galois.GF(2**8, irreducible_poly=283, primitive_element=19, compile=ufunc_mode)
    elif folder == "GF(2^32)":
        GF = galois.GF(2**32, compile=ufunc_mode)
    elif folder == "GF(2^100)":
        GF = galois.GF(2**100, compile=ufunc_mode)

    elif folder == "GF(7^3)":
        GF = galois.GF(7**3, compile=ufunc_mode)
    elif folder == "GF(7^3, 643, 244)":
        GF = galois.GF(7**3, irreducible_poly=643, primitive_element=244, compile=ufunc_mode)
    elif folder == "GF(109987^4)":
        GF = galois.GF(109987**4, compile=ufunc_mode)

    else:
        raise AssertionError(f"Test data folder {folder} not found")

    return GF, ufunc_mode, os.path.join(PATH, folder)


def read_json(field_folder, filename):
    GF, folder = field_folder
    with open(os.path.join(folder, filename), "rb") as f:
        d = json.load(f)
    return GF, d


def read_pickle(field_folder, filename):
    GF, folder = field_folder
    with open(os.path.join(folder, filename), "rb") as f:
        d = pickle.load(f)
    return GF, d


###############################################################################
# Fixtures for iterating over each finite field
###############################################################################


@pytest.fixture(scope="session", params=FIELDS)
def field(request):
    folder = request.param
    return construct_field(folder)[0]


@pytest.fixture(scope="session", params=FIELDS_DIFF_MODES)
def field_folder(request):
    folder = request.param
    field, _, folder = construct_field(folder)
    return field, folder


###############################################################################
# Fixtures for arithmetic over finite fields
###############################################################################


@pytest.fixture(scope="session")
def field_properties(field_folder):
    GF, d = read_json(field_folder, "properties.json")
    d["GF"] = GF
    d["characteristic"] = d["characteristic"]
    d["degree"] = d["degree"]
    d["order"] = d["order"]
    d["primitive_element"] = d["primitive_element"]
    d["irreducible_poly"] = galois.Poly(d["irreducible_poly"], field=galois.GF(d["characteristic"]))
    return d


@pytest.fixture(scope="session")
def field_add(field_folder):
    GF, d = read_pickle(field_folder, "add.pkl")
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_subtract(field_folder):
    GF, d = read_pickle(field_folder, "subtract.pkl")
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_multiply(field_folder):
    GF, d = read_pickle(field_folder, "multiply.pkl")
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_divide(field_folder):
    GF, d = read_pickle(field_folder, "divide.pkl")
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_additive_inverse(field_folder):
    GF, d = read_pickle(field_folder, "additive_inverse.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_multiplicative_inverse(field_folder):
    GF, d = read_pickle(field_folder, "multiplicative_inverse.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_scalar_multiply(field_folder):
    GF, d = read_pickle(field_folder, "scalar_multiply.pkl")
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = Y
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_power(field_folder):
    GF, d = read_pickle(field_folder, "power.pkl")
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = Y
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_log(field_folder):
    GF, d = read_pickle(field_folder, "log.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


###############################################################################
# Fixtures for advanced arithmetic over finite fields
###############################################################################


@pytest.fixture(scope="session")
def field_convolve(field_folder):
    GF, d = read_pickle(field_folder, "convolve.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Y"] = [GF(y) for y in d["Y"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


###############################################################################
# Fixtures for linear algebra over finite fields
###############################################################################


@pytest.fixture(scope="session")
def field_matrix_multiply(field_folder):
    GF, d = read_pickle(field_folder, "matrix_multiply.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Y"] = [GF(y) for y in d["Y"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_row_reduce(field_folder):
    GF, d = read_pickle(field_folder, "row_reduce.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_lu_decompose(field_folder):
    GF, d = read_pickle(field_folder, "lu_decompose.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["L"] = [GF(l) for l in d["L"]]
    d["U"] = [GF(u) for u in d["U"]]
    return d


@pytest.fixture(scope="session")
def field_plu_decompose(field_folder):
    GF, d = read_pickle(field_folder, "plu_decompose.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["P"] = [GF(p) for p in d["P"]]
    d["L"] = [GF(l) for l in d["L"]]
    d["U"] = [GF(u) for u in d["U"]]
    return d


@pytest.fixture(scope="session")
def field_matrix_inverse(field_folder):
    GF, d = read_pickle(field_folder, "matrix_inverse.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_matrix_determinant(field_folder):
    GF, d = read_pickle(field_folder, "matrix_determinant.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_matrix_solve(field_folder):
    GF, d = read_pickle(field_folder, "matrix_solve.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Y"] = [GF(y) for y in d["Y"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_row_space(field_folder):
    GF, d = read_pickle(field_folder, "row_space.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_column_space(field_folder):
    GF, d = read_pickle(field_folder, "column_space.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_left_null_space(field_folder):
    GF, d = read_pickle(field_folder, "left_null_space.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_null_space(field_folder):
    GF, d = read_pickle(field_folder, "null_space.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


###############################################################################
# Fixtures for arithmetic methods over finite fields
###############################################################################


@pytest.fixture(scope="session")
def field_additive_order(field_folder):
    GF, d = read_pickle(field_folder, "additive_order.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


@pytest.fixture(scope="session")
def field_multiplicative_order(field_folder):
    GF, d = read_pickle(field_folder, "multiplicative_order.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


@pytest.fixture(scope="session")
def field_characteristic_poly_element(field_folder):
    GF, d = read_pickle(field_folder, "characteristic_poly_element.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = [galois.Poly(p, field=GF.prime_subfield) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_characteristic_poly_matrix(field_folder):
    GF, d = read_pickle(field_folder, "characteristic_poly_matrix.pkl")
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_minimal_poly_element(field_folder):
    GF, d = read_pickle(field_folder, "minimal_poly_element.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = [galois.Poly(p, field=GF.prime_subfield) for p in d["Z"]]
    return d


# @pytest.fixture(scope="session")
# def field_minimal_poly_matrix(field_folder):
#     GF, d = read_pickle(field_folder, "minimal_poly_matrix.pkl")
#     d["GF"] = GF
#     d["X"] = [GF(x) for x in d["X"]]
#     d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
#     return d


@pytest.fixture(scope="session")
def field_trace(field_folder):
    GF, d = read_pickle(field_folder, "field_trace.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF.prime_subfield(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_norm(field_folder):
    GF, d = read_pickle(field_folder, "field_norm.pkl")
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF.prime_subfield(d["Z"])
    return d


###############################################################################
# Helper functions
###############################################################################

DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64, np.object_]


def array_equal(a, b):
    # Weird NumPy comparison bug, see https://github.com/mhostetter/galois/issues/37
    if a.dtype == np.object_:
        return np.array_equal(a, np.array(b, dtype=np.object_))
    return np.array_equal(a, b)


def randint(low, high, shape, dtype):
    if np.issubdtype(dtype, np.integer):
        array = np.random.default_rng().integers(low, high, shape, dtype=np.int64)
    else:
        # For dtype=object
        array = np.empty(shape, dtype=dtype)
        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for _ in iterator:
            array[iterator.multi_index] = random.randint(low, high - 1)
    if isinstance(shape, int) and shape == 1:
        return array.item()
    return array


def valid_dtype(GF):
    return random.choice(GF.dtypes)


def invalid_dtype(GF):
    return random.choice([dtype for dtype in DTYPES if dtype not in GF.dtypes])
