"""
A pytest conftest module that provides pytest fixtures for galois/polys/ tests.
"""
import os
import pickle

import pytest
import numpy as np

import galois

from ..fields.conftest import field, field_folder


def convert_folder(folder):
    sub_folder = os.path.basename(folder)
    return os.path.join(folder, "..", "..", "..", "polys", "data", sub_folder)


###############################################################################
# Fixtures for polynomial arithmetic over finite fields
###############################################################################

@pytest.fixture(scope="session")
def poly_add(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "add.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_subtract(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "subtract.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_multiply(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "multiply.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_scalar_multiply(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "scalar_multiply.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = d["Y"]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_divmod(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "divmod.pkl"), "rb") as f:
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
    folder = convert_folder(folder)
    with open(os.path.join(folder, "power.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = d["Y"]
    d["Z"] = [[galois.Poly(p, field=GF) for p in l] for l in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_evaluate(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "evaluate.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def poly_evaluate_matrix(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "evaluate_matrix.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [GF(y) for y in d["Y"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


###############################################################################
# Fixtures for polynomial arithmetic methods
###############################################################################

@pytest.fixture(scope="session")
def poly_reverse(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "reverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_roots(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "roots.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["R"] = [GF(r) for r in d["R"]]
    d["M"] = [np.array(m) for m in d["M"]]
    return d


@pytest.fixture(scope="session")
def poly_derivative(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "derivative.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = d["Y"]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


###############################################################################
# Fixtures for polynomial arithmetic functions
###############################################################################

@pytest.fixture(scope="session")
def poly_egcd(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "egcd.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["D"] = [galois.Poly(p, field=GF) for p in d["D"]]
    d["S"] = [galois.Poly(p, field=GF) for p in d["S"]]
    d["T"] = [galois.Poly(p, field=GF) for p in d["T"]]
    return d


@pytest.fixture(scope="session")
def poly_lcm(field_folder):
    GF, folder = field_folder
    folder = convert_folder(folder)
    with open(os.path.join(folder, "lcm.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [[galois.Poly(p, field=GF) for p in X] for X in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d
