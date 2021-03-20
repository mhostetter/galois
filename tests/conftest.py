import json
import os
import pickle
import pytest
import numpy as np

import galois

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FIELDS = [
    pytest.param("gf2", marks=[pytest.mark.GF2]),

    pytest.param("gf5", marks=[pytest.mark.GFp, pytest.mark.GF5]),
    pytest.param("gf7", marks=[pytest.mark.GFp, pytest.mark.GF7]),
    pytest.param("gf31", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("gf3191", marks=[pytest.mark.GFp, pytest.mark.GF31]),

    pytest.param("gf2^2", marks=[pytest.mark.GF2m, pytest.mark.GF4]),
    pytest.param("gf2^3", marks=[pytest.mark.GF2m, pytest.mark.GF8]),
    pytest.param("gf2^8", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("gf2^32", marks=[pytest.mark.GF2m, pytest.mark.GF2_32]),
]

FIELDS_DIFF_MODES = [
    pytest.param("gf2", marks=[pytest.mark.GF2]),

    pytest.param("gf5-lookup", marks=[pytest.mark.GFp, pytest.mark.GF5]),
    pytest.param("gf7-lookup", marks=[pytest.mark.GFp, pytest.mark.GF7]),
    pytest.param("gf31-lookup", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("gf3191-lookup", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("gf5-calculate", marks=[pytest.mark.GFp, pytest.mark.GF5]),
    pytest.param("gf7-calculate", marks=[pytest.mark.GFp, pytest.mark.GF7]),
    pytest.param("gf31-calculate", marks=[pytest.mark.GFp, pytest.mark.GF31]),
    pytest.param("gf3191-calculate", marks=[pytest.mark.GFp, pytest.mark.GF31]),

    pytest.param("gf2^2-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF4]),
    pytest.param("gf2^3-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF8]),
    pytest.param("gf2^8-lookup", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("gf2^2-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF4]),
    pytest.param("gf2^3-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF8]),
    pytest.param("gf2^8-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF256]),
    pytest.param("gf2^32-calculate", marks=[pytest.mark.GF2m, pytest.mark.GF2_32]),
]

# DTYPES = [np.int64,]
DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]


# def pytest_addoption(parser):
#     parser.addoption(
#         "--one-dtype", action="store_true", default=False, help="Only loop arithmetic tests against one dtype, not all"
#     )


# def pytest_collection_modifyitems(config, items):
#     global DTYPES
#     if config.getoption("--one-dtype"):
#         DTYPES = [np.int64,]
#     else:
#         DTYPES = [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32, np.int64]

# print(DTYPES)


def construct_field(folder):
    if len(folder.split("-")) >= 2:
        folder, mode = folder.split("-")[0:2]
    else:
        mode = "auto"

    if folder == "gf2":
        GF = galois.GF2

    elif folder == "gf5":
        GF = galois.GF_factory(5, 1, mode=mode)
    elif folder == "gf7":
        GF = galois.GF_factory(7, 1, mode=mode)
    elif folder == "gf31":
        GF = galois.GF_factory(31, 1, mode=mode)
    elif folder == "gf3191":
        GF = galois.GF_factory(3191, 1, mode=mode)

    elif folder == "gf2^2":
        GF = galois.GF_factory(2, 2, mode=mode)
    elif folder == "gf2^3":
        GF = galois.GF_factory(2, 3, mode=mode)
    elif folder == "gf2^8":
        GF = galois.GF_factory(2, 8, mode=mode)
    elif folder == "gf2^32":
        GF = galois.GF_factory(2, 32, mode=mode)

    else:
        raise AssertionError(f"Test data folder {folder} not found")

    return GF, mode, os.path.join(PATH, folder)


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
    l = construct_field(folder)
    return l[0], l[2]


@pytest.fixture(scope="session", params=FIELDS_DIFF_MODES)
def field_classes(request):
    folder = request.param
    l = construct_field(folder)
    d = {
        "GF": l[0],
        "mode": l[1]
    }
    return d


@pytest.fixture(scope="session", params=DTYPES)
def dtype(request):
    return request.param


###############################################################################
# Fixtures for arithmetic over finite fields
###############################################################################

@pytest.fixture(scope="session")
def properties(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "properties.json"), "rb") as f:
        d = json.load(f)
    d["GF"] = GF
    d["prim_poly"] = d["prim_poly"]
    return d


@pytest.fixture(scope="session")
def add(field_folder):
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
def subtract(field_folder):
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
def multiply(field_folder):
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
def divide(field_folder):
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
def additive_inverse(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "additive_inverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def multiplicative_inverse(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "multiplicative_inverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d

@pytest.fixture(scope="session")
def multiple_add(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "multiple_add.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    X, Y = np.meshgrid(d["X"], d["Y"], indexing="ij")
    d["X"] = GF(X)
    d["Y"] = Y
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def power(field_folder):
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
def log(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "log.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = d["Z"]
    return d


###############################################################################
# Fixtures for polynomial arithmetic over finite fields
###############################################################################

def load_poly_luts(name, GF, folder):
    with open(os.path.join(folder, name), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_add(field_folder):
    GF, folder = field_folder
    return load_poly_luts("poly_add.pkl", GF, folder)


@pytest.fixture(scope="session")
def poly_subtract(field_folder):
    GF, folder = field_folder
    return load_poly_luts("poly_subtract.pkl", GF, folder)


@pytest.fixture(scope="session")
def poly_multiply(field_folder):
    GF, folder = field_folder
    return load_poly_luts("poly_multiply.pkl", GF, folder)


@pytest.fixture(scope="session")
def poly_divmod(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "poly_divmod.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Q"] = [galois.Poly(p, field=GF) for p in d["Q"]]
    d["R"] = [galois.Poly(p, field=GF) for p in d["R"]]
    return d


@pytest.fixture(scope="session")
def poly_power(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "poly_power.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = galois.Poly(d["X"], field=GF)
    d["Y"] = d["Y"]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_evaluate(field_folder):
    GF, folder = field_folder
    with open(os.path.join(folder, "poly_evaluate.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d
