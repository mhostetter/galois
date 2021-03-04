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
def sub(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "sub.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def mul(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "mul.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = np.array(d["Y"], dtype=int)
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def div(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "div.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def add_inv(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "add_inv.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def mul_inv(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "mul_inv.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = GF(d["X"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session", params=FIELDS)
def exp(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "exp.pkl"), "rb") as f:
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
def poly_sub(request):
    return load_poly_luts("poly_sub.pkl", request.param[0], request.param[1])


@pytest.fixture(scope="session", params=FIELDS)
def poly_mul(request):
    return load_poly_luts("poly_mul.pkl", request.param[0], request.param[1])


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
def poly_mod(request):
    return load_poly_luts("poly_mod.pkl", request.param[0], request.param[1])


@pytest.fixture(scope="session", params=FIELDS)
def poly_exp(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "poly_exp.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = galois.Poly(d["X"], field=GF)
    d["Y"] = d["Y"]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session", params=FIELDS)
def poly_eval(request):
    GF = request.param[0]
    folder = request.param[1]
    with open(os.path.join(folder, "poly_eval.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d
