"""
A pytest conftest module that provides pytest fixtures for number theoretic functions.
"""
import os
import pickle

import pytest
import numpy as np

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


###############################################################################
# Helper functions
###############################################################################

def read_pickle(filename):
    with open(os.path.join(FOLDER, filename), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    return d


###############################################################################
# Fixtures for integer math functions
###############################################################################

@pytest.fixture(scope="session")
def egcd():
    return read_pickle("egcd.pkl")


@pytest.fixture(scope="session")
def lcm():
    return read_pickle("lcm.pkl")


@pytest.fixture(scope="session")
def prod():
    return read_pickle("prod.pkl")


@pytest.fixture(scope="session")
def power():
    return read_pickle("power.pkl")


@pytest.fixture(scope="session")
def crt():
    return read_pickle("crt.pkl")


@pytest.fixture(scope="session")
def isqrt():
    return read_pickle("isqrt.pkl")


@pytest.fixture(scope="session")
def iroot():
    return read_pickle("iroot.pkl")


@pytest.fixture(scope="session")
def ilog():
    return read_pickle("ilog.pkl")


###############################################################################
# Fixtures for Number Theory functions
###############################################################################

@pytest.fixture(scope="session")
def euler_phi():
    return read_pickle("euler_phi.pkl")


@pytest.fixture(scope="session")
def carmichael_lambda():
    return read_pickle("carmichael_lambda.pkl")


@pytest.fixture(scope="session")
def is_cyclic():
    return read_pickle("is_cyclic.pkl")
