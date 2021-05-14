import pytest

import galois


ADDITIVE_GROUPS = [
    pytest.param("Z(17, +)", marks=[pytest.mark.Z17]),
    pytest.param("Z(10**20, +)", marks=[pytest.mark.Z10_20]),
]

MULTIPLICATIVE_GROUPS = [
    pytest.param("Z(17, *)", marks=[pytest.mark.Z17]),
    pytest.param("Z(10**20, *)", marks=[pytest.mark.Z10_20]),
]

GROUPS = ADDITIVE_GROUPS + MULTIPLICATIVE_GROUPS


def construct_group(folder):
    if folder == "Z(17, +)":
        G = galois.Group(17, "+")
    elif folder == "Z(10**20, +)":
        G = galois.Group(10**20, "+")

    elif folder == "Z(17, *)":
        G = galois.Group(17, "*")
    elif folder == "Z(10**20, *)":
        G = galois.Group(10**20, "*")

    else:
        raise AssertionError(f"Test data folder {folder} not found")

    return G


###############################################################################
# Fixtures for iterating over each finite group
###############################################################################

@pytest.fixture(scope="session", params=ADDITIVE_GROUPS)
def additive_group(request):
    folder = request.param
    return construct_group(folder)


@pytest.fixture(scope="session", params=MULTIPLICATIVE_GROUPS)
def multiplicative_group(request):
    folder = request.param
    return construct_group(folder)


@pytest.fixture(scope="session", params=GROUPS)
def group(request):
    folder = request.param
    return construct_group(folder)
