import os
import pickle

import pytest

import galois

from ..fields.conftest import field, field_folder


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
    with open(os.path.join(folder, "poly_add.pkl"), "rb") as f:
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
    with open(os.path.join(folder, "poly_subtract.pkl"), "rb") as f:
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
    with open(os.path.join(folder, "poly_multiply.pkl"), "rb") as f:
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
    with open(os.path.join(folder, "poly_scalar_multiply.pkl"), "rb") as f:
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
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = d["Y"]
    d["Z"] = [[galois.Poly(p, field=GF) for p in l] for l in d["Z"]]
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
