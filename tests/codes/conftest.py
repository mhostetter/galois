"""
A pytest conftest module that provides pytest fixtures for tests/codes/ tests.
"""

from __future__ import annotations

import pathlib
import pickle
import random

import numpy as np
import pytest

import galois


def get_filenames(path: pathlib.Path) -> list[str]:
    filenames = []
    for file in path.iterdir():
        if file.suffix != ".pkl":
            continue
        filenames.append(file.stem)
    return filenames


PATH = pathlib.Path(__file__).parent / "data"

BCH_PATH = PATH / "bch"
BCH_FILENAMES = get_filenames(BCH_PATH)

REED_SOLOMON_PATH = PATH / "reed_solomon"
REED_SOLOMON_FILENAMES = get_filenames(REED_SOLOMON_PATH)


def read_pickle(file):
    with open(file, "rb") as f:
        dict_ = pickle.load(f)
    return dict_


###############################################################################
# Fixtures for FEC codes
###############################################################################


@pytest.fixture(scope="session", params=BCH_FILENAMES)
def bch_codes(request):
    file = (BCH_PATH / request.param).with_suffix(".pkl")
    dict_ = read_pickle(file)

    n = dict_["n"]
    k = dict_["k"]
    d = dict_["d"]
    GF = galois.GF(dict_["q"])
    alpha = dict_["alpha"]
    c = dict_["c"]
    systematic = dict_["is_systematic"]

    code = galois.BCH(n, k, d=d, field=GF, alpha=alpha, c=c, systematic=systematic)
    dict_["code"] = code

    return dict_


@pytest.fixture(scope="session", params=REED_SOLOMON_FILENAMES)
def reed_solomon_codes(request):
    file = (REED_SOLOMON_PATH / request.param).with_suffix(".pkl")
    dict_ = read_pickle(file)

    n = dict_["n"]
    k = dict_["k"]
    d = dict_["d"]
    GF = galois.GF(dict_["q"])
    alpha = dict_["alpha"]
    c = dict_["c"]
    systematic = dict_["is_systematic"]

    code = galois.ReedSolomon(n, k, d=d, field=GF, alpha=alpha, c=c, systematic=systematic)
    dict_["code"] = code

    return dict_


###############################################################################
# Helper functions for unit tests
###############################################################################


def random_errors(GF, N, n, max_errors) -> tuple[np.ndarray, np.ndarray]:
    max_errors = min(n, max_errors)
    N_errors = np.random.default_rng().integers(0, max_errors + 1, N)
    N_errors[0] = max_errors  # Ensure the max number of errors is present at least once

    ERRORS = GF.Zeros((N, n))
    for i in range(N):
        ERRORS[i, random.sample(list(range(n)), N_errors[i])] = GF.Random(N_errors[i], low=1)

    return ERRORS, N_errors


def random_type(array):
    """
    Randomly vary the input type to encode()/decode() across various ArrayLike inputs.
    """
    x = random.randint(0, 2)
    if x == 0:
        # A FieldArray instance
        return array
    if x == 1:
        # A np.ndarray instance
        return array.view(np.ndarray)
    return array.tolist()


def verify_encode(
    code: galois._codes._linear.LinearCode,
    MESSAGES: np.ndarray,
    CODEWORDS: np.ndarray,
    is_systematic: bool,
    vector: bool,
):
    if vector:
        idx = np.random.randint(0, MESSAGES.shape[0])
        MESSAGES = MESSAGES[idx, :]
        CODEWORDS = CODEWORDS[idx, :]

    MESSAGES = random_type(MESSAGES)

    codewords = code.encode(MESSAGES)
    assert isinstance(codewords, code.field)
    assert np.array_equal(codewords, CODEWORDS)

    if is_systematic:
        parities = code.encode(MESSAGES, output="parity")
        assert isinstance(parities, code.field)
        assert np.array_equal(parities, CODEWORDS[..., code.k :])
    else:
        with pytest.raises(ValueError):
            code.encode(MESSAGES, output="parity")


def verify_encode_shortened(
    code: galois._codes._linear.LinearCode,
    MESSAGES: np.ndarray,
    CODEWORDS: np.ndarray,
    is_systematic: bool,
    vector: bool,
):
    if is_systematic:
        if vector:
            idx = np.random.randint(0, MESSAGES.shape[0])
            MESSAGES = MESSAGES[idx, :]
            CODEWORDS = CODEWORDS[idx, :]

        MESSAGES = random_type(MESSAGES)

        codewords = code.encode(MESSAGES)
        assert isinstance(codewords, code.field)
        assert np.array_equal(codewords, CODEWORDS)

        parities = code.encode(MESSAGES, output="parity")
        assert isinstance(parities, code.field)
        assert np.array_equal(parities, CODEWORDS[..., -(code.n - code.k) :])
    else:
        MESSAGES = [] if vector else [[]]

        with pytest.raises(ValueError):
            code.encode(MESSAGES)
        with pytest.raises(ValueError):
            code.encode(MESSAGES, output="parity")


def verify_decode(code: galois._codes._linear.LinearCode, N: int):
    GF = code.field

    MESSAGES = GF.Random((N, code.k))
    ERRORS, N_errors = random_errors(GF, N, code.n, code.t)
    if N == 1:
        MESSAGES = MESSAGES[0, :]
        ERRORS = ERRORS[0, :]
        N_errors = N_errors[0]

    CODEWORDS = code.encode(MESSAGES)
    RECEIVED_CODEWORDS = random_type(CODEWORDS + ERRORS)

    decoded_messages = code.decode(RECEIVED_CODEWORDS)
    assert type(decoded_messages) is GF
    assert np.array_equal(decoded_messages, MESSAGES)

    decoded_messages, N_corrected = code.decode(RECEIVED_CODEWORDS, errors=True)
    assert type(decoded_messages) is GF
    assert np.array_equal(decoded_messages, MESSAGES)
    assert np.array_equal(N_corrected, N_errors)

    decoded_codewords = code.decode(RECEIVED_CODEWORDS, output="codeword")
    assert type(decoded_codewords) is GF
    assert np.array_equal(decoded_codewords, CODEWORDS)


def verify_decode_shortened(code: galois._codes._linear.LinearCode, N: int, is_systematic: bool):
    if is_systematic:
        GF = code.field
        s = random.randint(0, code.k - 1)  # The number of shortened symbols
        MESSAGES = GF.Random((N, code.k - s))
        ERRORS, N_errors = random_errors(GF, N, code.n - s, code.t)
        if N == 1:
            MESSAGES = MESSAGES[0, :]
            ERRORS = ERRORS[0, :]
            N_errors = N_errors[0]

        CODEWORDS = code.encode(MESSAGES)
        RECEIVED_CODEWORDS = random_type(CODEWORDS + ERRORS)

        decoded_messages = code.decode(RECEIVED_CODEWORDS)
        assert type(decoded_messages) is GF
        assert np.array_equal(decoded_messages, MESSAGES)

        decoded_messages, N_corrected = code.decode(RECEIVED_CODEWORDS, errors=True)
        assert type(decoded_messages) is GF
        assert np.array_equal(decoded_messages, MESSAGES)
        assert np.array_equal(N_corrected, N_errors)
    else:
        RECEIVED_CODEWORDS = [] if N == 1 else [[]]

        with pytest.raises(ValueError):
            code.decode(RECEIVED_CODEWORDS)
        with pytest.raises(ValueError):
            code.decode(RECEIVED_CODEWORDS, errors=True)


# @pytest.mark.parametrize("size", CODES)
# def test_some_uncorrectable(self, size):
#     n, k = size[0], size[1]
#     N = 100
#     code = galois.BCH(n, k)
#     M = galois.GF2.Random((N, k))
#     C = code.encode(M)
#     E, N_errors = random_errors(galois.GF2, N, n, code.t + 1)
#     R = C + E

#     corr_idxs = np.where(N_errors <= code.t)[0]

#     RR = random_type(R)
#     DEC_M = code.decode(RR)
#     assert type(DEC_M) is galois.GF2
#     assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])

#     RR = random_type(R)
#     DEC_M, N_corr = code.decode(RR, errors=True)
#     assert type(DEC_M) is galois.GF2
#     assert np.array_equal(DEC_M[corr_idxs,:], M[corr_idxs,:])
#     assert np.array_equal(N_corr[corr_idxs], N_errors[corr_idxs])
