import json
import os
import pickle

import pytest
import numpy as np

import galois

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FIELDS = [
    pytest.param("GF(2)", marks=[pytest.mark.GF2]),

    pytest.param("GF(2^2)", marks=[pytest.mark.GF2m, pytest.mark.GF4]),
    pytest.param("GF(2^3)", marks=[pytest.mark.GF2m, pytest.mark.GF8]),
    pytest.param("GF(2^8)", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("GF(2^32)", marks=[pytest.mark.GF2m, pytest.mark.GF2_32]),
    pytest.param("GF(2^100)", marks=[pytest.mark.GF2m, pytest.mark.GF2_100]),

    pytest.param("GF(5)", marks=[pytest.mark.GFp, pytest.mark.GF5]),
    pytest.param("GF(7)", marks=[pytest.mark.GFp, pytest.mark.GF7]),
    pytest.param("GF(31)", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("GF(3191)", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("GF(2147483647)", marks=[pytest.mark.GFp, pytest.mark.GF2147483647]),
    pytest.param("GF(36893488147419103183)", marks=[pytest.mark.GFp, pytest.mark.GF36893488147419103183]),

    pytest.param("GF(7^3)", marks=[pytest.mark.GFpm, pytest.mark.GF7_3]),
    pytest.param("GF(109987^4)", marks=[pytest.mark.GFpm, pytest.mark.GF109987_4]),
]

FIELDS_DIFF_MODES = [
    pytest.param("GF(2)-jit-calculate", marks=[pytest.mark.GF2]),

    pytest.param("GF(2^2)-jit-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF4]),
    pytest.param("GF(2^2)-jit-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF4]),
    pytest.param("GF(2^3)-jit-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF8]),
    pytest.param("GF(2^3)-jit-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF8]),
    pytest.param("GF(2^8)-jit-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("GF(2^8)-jit-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("GF(2^8, 283, 19)-jit-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("GF(2^8, 283, 19)-jit-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("GF(2^32)-jit-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF2_32]),
    pytest.param("GF(2^100)-python-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF2_100]),

    pytest.param("GF(5)-jit-lookup", marks=[pytest.mark.GFp, pytest.mark.GF5]),
    pytest.param("GF(5)-jit-calculate", marks=[pytest.mark.GFp, pytest.mark.GF5]),
    pytest.param("GF(7)-jit-lookup", marks=[pytest.mark.GFp, pytest.mark.GF7]),
    pytest.param("GF(7)-jit-calculate", marks=[pytest.mark.GFp, pytest.mark.GF7]),
    pytest.param("GF(31)-jit-lookup", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("GF(31)-jit-calculate", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("GF(3191)-jit-lookup", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("GF(3191)-jit-calculate", marks=[pytest.mark.GFp, pytest.mark.GF3191]),
    pytest.param("GF(2147483647)-jit-calculate", marks=[pytest.mark.GFp, pytest.mark.GF2147483647]),
    pytest.param("GF(36893488147419103183)-python-calculate", marks=[pytest.mark.GFp, pytest.mark.GF36893488147419103183]),

    pytest.param("GF(7^3)-jit-lookup", marks=[pytest.mark.GFpm, pytest.mark.GF7_3]),
    pytest.param("GF(7^3)-jit-calculate", marks=[pytest.mark.GFpm, pytest.mark.GF7_3]),
    pytest.param("GF(7^3, 643, 244)-jit-lookup", marks=[pytest.mark.GFpm, pytest.mark.GF7_3]),
    pytest.param("GF(7^3, 643, 244)-jit-calculate", marks=[pytest.mark.GFpm, pytest.mark.GF7_3]),
    pytest.param("GF(109987^4)-python-calculate", marks=[pytest.mark.GFpm, pytest.mark.GF109987_4]),
]


def construct_field(folder):
    if len(folder.split("-")) >= 2:
        folder, ufunc_mode = folder.split("-", maxsplit=1)
    else:
        ufunc_mode = "auto"

    if folder == "GF(2)":
        GF = galois.GF2

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
    field, ufunc_mode, folder = construct_field(folder)
    return field, folder


###############################################################################
# Fixtures for arithmetic over finite fields
###############################################################################

@pytest.fixture(scope="session")
def field_properties(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "properties.json"), "rb") as f:
        d = json.load(f)
    d["GF"] = GF
    d["characteristic"] = d["characteristic"]
    d["degree"] = d["degree"]
    d["order"] = d["order"]
    d["primitive_element"] = d["primitive_element"]
    d["irreducible_poly"] = galois.Poly(d["irreducible_poly"], field=galois.GF(d["characteristic"]))
    return d


@pytest.fixture(scope="session")
def field_add(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "add.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_subtract(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "subtract.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_multiply(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "multiply.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_divide(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "divide.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = GF(Y)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_additive_inverse(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "additive_inverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_multiplicative_inverse(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "multiplicative_inverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_scalar_multiply(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "scalar_multiply.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = Y
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_power(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "power.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = Y
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def field_log(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "log.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


###############################################################################
# Fixtures for arithmetic methods over finite fields
###############################################################################

@pytest.fixture(scope="session")
def field_additive_order(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "additive_order.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


@pytest.fixture(scope="session")
def field_multiplicative_order(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "multiplicative_order.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


@pytest.fixture(scope="session")
def field_characteristic_poly_element(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "characteristic_poly_element.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = [galois.Poly(p, field=GF.prime_subfield) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def field_characteristic_poly_matrix(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "characteristic_poly_matrix.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [GF(x) for x in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d
