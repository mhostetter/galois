import json
import os
import pickle
import pytest
import numpy as np

import galois

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FIELDS = [
    pytest.param(
        (galois.GF2, os.path.join(PATH, "gf2")),
        marks=[pytest.mark.GF2]
    ),
    pytest.param(
        (galois.GF_factory(5, 1), os.path.join(PATH, "gf5")),
        marks=[pytest.mark.GFp, pytest.mark.GF5]
    ),
    pytest.param((
        galois.GF_factory(7, 1), os.path.join(PATH, "gf7")),
        marks=[pytest.mark.GFp, pytest.mark.GF7]
    ),
    pytest.param((
        galois.GF_factory(31, 1), os.path.join(PATH, "gf31")),
        marks=[pytest.mark.GFp, pytest.mark.GF31]
    ),
]


###############################################################################
# Fixtures for iterating over each finite field
###############################################################################

@pytest.fixture(scope="session", params=FIELDS)
def field(request):
    GF = request.param[0]
    return GF


@pytest.fixture(scope="session", params=FIELDS)
def properties(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "properties.json"), "rb") as f:
        d = json.load(f)
    d["GF"] = GF
    d["prim_poly"] = galois.Poly(d["prim_poly"])
    return d


###############################################################################
# Fixtures for arithmetic over finite fields
###############################################################################

@pytest.fixture(scope="session", params=FIELDS)
def add(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "add.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def subtract(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "subtract.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def multiply(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "multiply.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = np.array(d["Y"], dtype=int)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def divison(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "divison.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def additive_inverse(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "additive_inverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def multiplicative_inverse(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "multiplicative_inverse.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def power(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "power.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = d["Y"]
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def log(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "log.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
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


@pytest.fixture(scope="session", params=FIELDS)
def poly_add(request):
    return load_poly_luts("poly_add.pkl", request.param[0], request.param[1])


@pytest.fixture(scope="session", params=FIELDS)
def poly_subtract(request):
    return load_poly_luts("poly_subtract.pkl", request.param[0], request.param[1])


@pytest.fixture(scope="session", params=FIELDS)
def poly_multiply(request):
    return load_poly_luts("poly_multiply.pkl", request.param[0], request.param[1])


@pytest.fixture(scope="session", params=FIELDS)
def poly_divmod(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "poly_divmod.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Q"] = [galois.Poly(p, field=GF) for p in d["Q"]]
    d["R"] = [galois.Poly(p, field=GF) for p in d["R"]]
    return d


@pytest.fixture(scope="session", params=FIELDS)
def poly_power(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "poly_power.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = galois.Poly(d["X"], field=GF)
    d["Y"] = d["Y"]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session", params=FIELDS)
def poly_evaluate(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "poly_evaluate.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d
