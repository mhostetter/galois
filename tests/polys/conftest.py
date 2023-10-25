"""
A pytest conftest module that provides pytest fixtures for galois/polys/ tests.
"""
# ruff: noqa: F401, F811

import os
import pickle

import numpy as np
import pytest

import galois

from ..fields.conftest import field, field_folder

###############################################################################
# Helper functions
###############################################################################


def read_pickle(field_folder, filename):
    GF, folder = field_folder
    # Convert from folder in fields/data/ to polys/data/
    folder = os.path.join(folder, "..", "..", "..", "polys", "data", os.path.basename(folder))
    with open(os.path.join(folder, filename), "rb") as f:
        d = pickle.load(f)
    return GF, d


###############################################################################
# Fixtures for polynomial arithmetic over finite fields
###############################################################################


@pytest.fixture(scope="session")
def poly_add(field_folder):
    GF, d = read_pickle(field_folder, "add.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_subtract(field_folder):
    GF, d = read_pickle(field_folder, "subtract.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_multiply(field_folder):
    GF, d = read_pickle(field_folder, "multiply.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_scalar_multiply(field_folder):
    GF, d = read_pickle(field_folder, "scalar_multiply.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = d["Y"]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_divmod(field_folder):
    GF, d = read_pickle(field_folder, "divmod.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Q"] = [galois.Poly(p, field=GF) for p in d["Q"]]
    d["R"] = [galois.Poly(p, field=GF) for p in d["R"]]
    return d


@pytest.fixture(scope="session")
def poly_power(field_folder):
    GF, d = read_pickle(field_folder, "power.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = d["Y"]
    d["Z"] = [[galois.Poly(p, field=GF) for p in l] for l in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_evaluate(field_folder):
    GF, d = read_pickle(field_folder, "evaluate.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = GF(d["Y"])
    d["Z"] = GF(d["Z"])
    return d


@pytest.fixture(scope="session")
def poly_evaluate_matrix(field_folder):
    GF, d = read_pickle(field_folder, "evaluate_matrix.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [GF(y) for y in d["Y"]]
    d["Z"] = [GF(z) for z in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_evaluate_poly(field_folder):
    GF, d = read_pickle(field_folder, "evaluate_poly.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


###############################################################################
# Fixtures for polynomial arithmetic methods
###############################################################################


@pytest.fixture(scope="session")
def poly_reverse(field_folder):
    GF, d = read_pickle(field_folder, "reverse.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_roots(field_folder):
    GF, d = read_pickle(field_folder, "roots.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["R"] = [GF(r) for r in d["R"]]
    d["M"] = [np.array(m) for m in d["M"]]
    return d


@pytest.fixture(scope="session")
def poly_derivative(field_folder):
    GF, d = read_pickle(field_folder, "derivative.pkl")
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
    GF, d = read_pickle(field_folder, "egcd.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Y"] = [galois.Poly(p, field=GF) for p in d["Y"]]
    d["D"] = [galois.Poly(p, field=GF) for p in d["D"]]
    d["S"] = [galois.Poly(p, field=GF) for p in d["S"]]
    d["T"] = [galois.Poly(p, field=GF) for p in d["T"]]
    return d


@pytest.fixture(scope="session")
def poly_lcm(field_folder):
    GF, d = read_pickle(field_folder, "lcm.pkl")
    d["GF"] = GF
    d["X"] = [[galois.Poly(p, field=GF) for p in X] for X in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_prod(field_folder):
    GF, d = read_pickle(field_folder, "prod.pkl")
    d["GF"] = GF
    d["X"] = [[galois.Poly(p, field=GF) for p in X] for X in d["X"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_crt(field_folder):
    GF, d = read_pickle(field_folder, "crt.pkl")
    d["GF"] = GF
    d["X"] = [[galois.Poly(p, field=GF) for p in X] for X in d["X"]]
    d["Y"] = [[galois.Poly(p, field=GF) for p in Y] for Y in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) if p is not None else None for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_modular_power(field_folder):
    GF, d = read_pickle(field_folder, "modular_power.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["E"] = d["E"]
    d["M"] = [galois.Poly(p, field=GF) for p in d["M"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


@pytest.fixture(scope="session")
def poly_lagrange_poly(field_folder):
    GF, d = read_pickle(field_folder, "lagrange_poly.pkl")
    d["GF"] = GF
    d["X"] = [GF(X) for X in d["X"]]
    d["Y"] = [GF(Y) for Y in d["Y"]]
    d["Z"] = [galois.Poly(p, field=GF) for p in d["Z"]]
    return d


###############################################################################
# Fixtures for special polynomials
###############################################################################


@pytest.fixture(scope="session")
def poly_is_monic(field_folder):
    GF, d = read_pickle(field_folder, "is_monic.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Z"] = d["Z"]
    return d


@pytest.fixture(scope="session")
def poly_is_irreducible(field_folder):
    GF, d = read_pickle(field_folder, "is_irreducible.pkl")
    d["GF"] = GF
    d["IS"] = [galois.Poly(p, field=GF) for p in d["IS"]]
    d["IS_NOT"] = [galois.Poly(p, field=GF) for p in d["IS_NOT"]]
    return d


@pytest.fixture(scope="session")
def poly_is_primitive(field_folder):
    GF, d = read_pickle(field_folder, "is_primitive.pkl")
    d["GF"] = GF
    d["IS"] = [galois.Poly(p, field=GF) for p in d["IS"]]
    d["IS_NOT"] = [galois.Poly(p, field=GF) for p in d["IS_NOT"]]
    return d


@pytest.fixture(scope="session")
def poly_is_square_free(field_folder):
    GF, d = read_pickle(field_folder, "is_square_free.pkl")
    d["GF"] = GF
    d["X"] = [galois.Poly(p, field=GF) for p in d["X"]]
    d["Z"] = d["Z"]
    return d
