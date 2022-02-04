"""
A pytest conftest module that provides pytest fixtures for number theoretic functions.
"""
import os
import pickle

import pytest
import numpy as np

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


###############################################################################
# Fixtures for integer math functions
###############################################################################

@pytest.fixture(scope="session")
def egcd():
    with open(os.path.join(FOLDER, "egcd.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    return d


@pytest.fixture(scope="session")
def lcm():
    with open(os.path.join(FOLDER, "lcm.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    return d


@pytest.fixture(scope="session")
def prod():
    with open(os.path.join(FOLDER, "prod.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    return d


@pytest.fixture(scope="session")
def power():
    with open(os.path.join(FOLDER, "power.pkl"), "rb") as f:
        print(f"Loading {f}...")
        d = pickle.load(f)
    return d
