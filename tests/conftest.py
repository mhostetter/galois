"""
A pytest conftest module that provides pytest fixtures for number theoretic functions.
"""

import os
import pickle

import numba
import numpy
import pytest

import galois

print("\nTested versions:")
print(f"  galois: {galois.__version__}")
print(f"  numpy: {numpy.__version__}")
print(f"  numba: {numba.__version__}")
print()

###############################################################################
# Helper functions
###############################################################################

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def read_pickle(filename):
    with open(os.path.join(FOLDER, filename), "rb") as f:
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


###############################################################################
# Fixtures for Number Theory functions
###############################################################################


@pytest.fixture(scope="session")
def primes():
    return read_pickle("primes.pkl")


@pytest.fixture(scope="session")
def kth_prime():
    return read_pickle("kth_prime.pkl")


@pytest.fixture(scope="session")
def prev_prime():
    return read_pickle("prev_prime.pkl")


@pytest.fixture(scope="session")
def next_prime():
    return read_pickle("next_prime.pkl")


@pytest.fixture(scope="session")
def is_prime():
    return read_pickle("is_prime.pkl")


@pytest.fixture(scope="session")
def is_prime_power():
    return read_pickle("is_prime_power.pkl")


@pytest.fixture(scope="session")
def is_perfect_power():
    return read_pickle("is_perfect_power.pkl")


@pytest.fixture(scope="session")
def is_square_free():
    return read_pickle("is_square_free.pkl")


@pytest.fixture(scope="session")
def is_smooth():
    return read_pickle("is_smooth.pkl")


@pytest.fixture(scope="session")
def is_powersmooth():
    return read_pickle("is_powersmooth.pkl")
