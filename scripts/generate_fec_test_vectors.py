"""
Script to generate unit test vectors for FEC codes.

Install SageMath:
* `sudo apt install sagemath`
"""

import os
import pickle
import shutil

import numpy as np
import sage
from sage.all import GF, ZZ, codes, matrix, vector

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "codes")


def to_int(field, element):
    """Convert from various finite field elements to an integer"""
    if isinstance(element, sage.rings.finite_rings.element_pari_ffelt.FiniteFieldElement_pari_ffelt):
        coeffs = element._vector_()
        characteristic = int(field.characteristic())
        return sum(int(c) * characteristic**i for i, c in enumerate(coeffs))
    try:
        return int(element)
    except TypeError:
        return element.integer_representation()


def to_field(field, integer):
    """Convert from an integer to various finite field elements"""
    if isinstance(field, sage.rings.finite_rings.finite_field_pari_ffelt.FiniteField_pari_ffelt):
        l = []
        characteristic = int(field.characteristic())
        degree = int(field.degree())
        for d in range(degree - 1, -1, -1):
            q = integer // characteristic**d
            l += [f"{q}*x^{d}"]
            integer -= q * characteristic**d
        return field(" + ".join(l))
    try:
        return field.fetch_int(int(integer))
    except:  # noqa: E722
        return field(integer)


def save_pickle(d, folder, name):
    print(f"  Saving {name}...")
    with open(os.path.join(folder, name), "wb") as f:
        pickle.dump(d, f)


def _convert_sage_generator_matrix(F, G, n, k, systematic=True):
    # Convert to a NumPy array
    G = matrix(G).numpy()
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i, j] = to_int(F, G[i, j])
    G = G.astype(np.int64)

    if systematic:
        # Flip the parity rows top-to-bottom and left-to-right
        G[:, k:] = np.flipud(G[:, k:])
        G[:, k:] = np.fliplr(G[:, k:])
    else:
        for i in range(k):
            idxs = np.nonzero(G[i, :])[0]
            j, k = idxs[0], idxs[-1]
            G[i, j : k + 1] = np.flip(G[i, j : k + 1])

    return G


def _convert_sage_parity_check_matrix(F, H, n, k):
    # Convert to a NumPy array
    H = matrix(H).numpy()
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i, j] = to_int(F, H[i, j])
    H = H.astype(np.int64)

    for i in range(n - k):
        idxs = np.nonzero(H[i, :])[0]
        j, k = idxs[0], idxs[-1]
        H[i, j : k + 1] = np.flip(H[i, j : k + 1])

    if H.size == 0:
        # The array shape should be (0, n) not (0, 0)
        H = H.reshape((0, n))

    return H


def _convert_to_sage_message(F, message, k):
    # Make the message a polynomial with x^0 as the first element
    message = message.tolist()
    for i in range(len(message)):
        message[i] = to_field(F, message[i])

    message = vector(F, message[0:k][::-1])

    return message


def _convert_sage_codeword(F, codeword, k, systematic=True):
    codeword = codeword.numpy()
    for i in range(codeword.shape[0]):
        codeword[i] = to_int(F, codeword[i])
    codeword = codeword.astype(np.int64)

    if systematic:
        codeword[0:k] = np.flip(codeword[0:k])
        codeword[k:] = np.flip(codeword[k:])
    else:
        codeword = np.flip(codeword)

    return codeword


def add_bch_lut(folder, q, m, n, d, alpha=None, c=1, systematic=True, rng=1):
    F = GF(q)
    EF = GF(q**m)
    if alpha is not None:
        alpha = to_field(EF, alpha)
    C = codes.BCHCode(F, n, d, primitive_root=alpha, offset=c)

    q = int(q)
    m = int(m)
    n = int(C.length())
    k = int(C.dimension())
    d = int(C.designed_distance())
    # d_min = int(C.minimum_distance())  # Takes FOREVER on some codes
    d_min = d
    alpha = to_int(EF, C.primitive_root())
    c = int(C.offset())
    is_systematic = bool(systematic)
    is_primitive = n == q**m - 1
    is_narrow_sense = c == 1
    generator_poly = str(C.generator_polynomial())
    parity_check_poly = str(C.check_polynomial())
    if systematic:
        G = _convert_sage_generator_matrix(F, C.generator_matrix("Systematic"), n, k, systematic=True)
    else:
        G = _convert_sage_generator_matrix(F, C.generator_matrix(), n, k, systematic=False)
    H = _convert_sage_parity_check_matrix(F, C.parity_check_matrix(), n, k)

    dict_ = {
        "q": q,
        "m": m,
        "n": n,
        "k": k,
        "d": d,
        "d_min": d_min,
        "alpha": alpha,
        "c": c,
        "is_systematic": is_systematic,
        "is_primitive": is_primitive,
        "is_narrow_sense": is_narrow_sense,
        "generator_poly": generator_poly,
        "parity_check_poly": parity_check_poly,
        "G": G,
        "H": H,
    }

    # Add encoding vectors
    N = 10  # Number of codewords
    messages = rng.integers(0, q, (N, k), dtype=np.int64)
    codewords = np.zeros((N, n), dtype=np.int64)
    for i in range(N):
        message_ = _convert_to_sage_message(F, messages[i, :], k)
        if systematic:
            codeword_ = C.encode(message_, "Systematic")
        else:
            codeword_ = C.encode(message_)
        codeword = _convert_sage_codeword(F, codeword_, k, systematic=systematic)
        codewords[i, :] = codeword
    dict_["encode"] = {
        "messages": messages,
        "codewords": codewords,
    }

    # Add shortened encoding vectors
    N = 10  # Number of codewords
    if d > 1 and systematic:
        s = rng.integers(0, k)  # The number of shortened symbols
        assert k - s > 0
        messages = rng.integers(0, q, (N, k), dtype=np.int64)
        messages[:, 0:s] = 0
        codewords = np.zeros((N, n), dtype=np.int64)
        for i in range(N):
            message_ = _convert_to_sage_message(F, messages[i, :], k)
            # print(C, list(range(0, s)))
            # CC = C.shortened(list(range(0, s)))  # The shortened code
            if systematic:
                codeword_ = C.encode(message_, "Systematic")
            else:
                codeword_ = C.encode(message_)
            codeword = _convert_sage_codeword(F, codeword_, k, systematic=systematic)
            codewords[i, :] = codeword
        dict_["encode_shortened"] = {
            "messages": messages[:, s:],
            "codewords": codewords[:, s:],
        }
    else:
        dict_["encode_shortened"] = {}

    if systematic:
        name = f"n{n}_k{k}_d{d}_alpha{alpha}_c{c}_sys.pkl"
    else:
        name = f"n{n}_k{k}_d{d}_alpha{alpha}_c{c}_nonsys.pkl"

    save_pickle(dict_, folder, name)


def make_bch_luts():
    folder = os.path.join(PATH, "data", "bch")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    # Binary primitive BCH codes
    rng = np.random.default_rng(1000)
    for d in range(1, 9):
        for systematic in [True, False]:
            add_bch_lut(folder, 2, 4, 2**4 - 1, d, systematic=systematic, rng=rng)
            add_bch_lut(folder, 2, 4, 2**4 - 1, d, alpha=9, systematic=systematic, rng=rng)
            add_bch_lut(folder, 2, 4, 2**4 - 1, d, c=3, systematic=systematic, rng=rng)

    # Primitive BCH codes over GF(3)
    rng = np.random.default_rng(2000)
    for d in range(1, 19):
        for systematic in [True, False]:
            add_bch_lut(folder, 3, 3, 3**3 - 1, d, systematic=systematic, rng=rng)
            add_bch_lut(folder, 3, 3, 3**3 - 1, d, alpha=17, systematic=systematic, rng=rng)
            add_bch_lut(folder, 3, 3, 3**3 - 1, d, c=3, systematic=systematic, rng=rng)

    # Non-primitive BCH codes over GF(3)
    rng = np.random.default_rng(3000)
    for d in range(1, 9):
        for systematic in [True, False]:
            add_bch_lut(folder, 3, 3, 13, d, alpha=9, systematic=systematic, rng=rng)
            add_bch_lut(folder, 3, 3, 13, d, alpha=6, systematic=systematic, rng=rng)
            add_bch_lut(folder, 3, 3, 13, d, alpha=9, c=3, systematic=systematic, rng=rng)


def add_reed_solomon_lut(folder, q, n, d, alpha=None, c=1, systematic=True, rng=1):
    k = n - (d - 1)
    F = GF(q, repr="int")
    if alpha is not None:
        alpha = to_field(F, alpha)
    else:
        assert c == 1
    alpha_c = alpha**c
    C = codes.ReedSolomonCode(F, ZZ(n), k, primitive_root=alpha_c)
    C_cyclic = codes.CyclicCode(code=C)

    q = int(q)
    n = int(C.length())
    k = int(C.dimension())
    d = int(C.minimum_distance())
    alpha = to_int(F, C.evaluation_points()[1])
    # c = int(C.offset())
    c = int(c)
    is_systematic = bool(systematic)
    is_primitive = n == q - 1
    is_narrow_sense = c == 1
    generator_poly = str(C_cyclic.generator_polynomial())
    if systematic:
        G = _convert_sage_generator_matrix(F, C_cyclic.generator_matrix("Systematic"), n, k, systematic=True)
    else:
        G = _convert_sage_generator_matrix(F, C_cyclic.generator_matrix(), n, k, systematic=False)
    H = _convert_sage_parity_check_matrix(F, C.parity_check_matrix(), n, k)

    dict_ = {
        "q": q,
        "n": n,
        "k": k,
        "d": d,
        "alpha": alpha,
        "c": c,
        "is_systematic": is_systematic,
        "is_primitive": is_primitive,
        "is_narrow_sense": is_narrow_sense,
        "generator_poly": generator_poly,
        "G": G,
        "H": H,
    }

    # Add encoding vectors
    N = 10  # Number of codewords
    messages = rng.integers(0, q, (N, k), dtype=np.int64)
    codewords = np.zeros((N, n), dtype=np.int64)
    for i in range(N):
        message_ = _convert_to_sage_message(F, messages[i, :], k)
        if systematic:
            codeword_ = C_cyclic.encode(message_, "Systematic")
        else:
            codeword_ = C_cyclic.encode(message_)
        codeword = _convert_sage_codeword(F, codeword_, k, systematic=systematic)
        codewords[i, :] = codeword
    dict_["encode"] = {
        "messages": messages,
        "codewords": codewords,
    }

    # Add shortened encoding vectors
    N = 10  # Number of codewords
    if d > 1 and systematic:
        s = rng.integers(0, k)  # The number of shortened symbols
        assert k - s > 0
        messages = rng.integers(0, q, (N, k), dtype=np.int64)
        messages[:, 0:s] = 0
        codewords = np.zeros((N, n), dtype=np.int64)
        for i in range(N):
            message_ = _convert_to_sage_message(F, messages[i, :], k)
            # print(C, list(range(0, s)))
            # CC = C.shortened(list(range(0, s)))  # The shortened code
            if systematic:
                codeword_ = C.encode(message_, "Systematic")
            else:
                codeword_ = C.encode(message_)
            codeword = _convert_sage_codeword(F, codeword_, k, systematic=systematic)
            codewords[i, :] = codeword
        dict_["encode_shortened"] = {
            "messages": messages[:, s:],
            "codewords": codewords[:, s:],
        }
    else:
        dict_["encode_shortened"] = {}

    if systematic:
        name = f"n{n}_k{k}_d{d}_alpha{alpha}_c{c}_sys.pkl"
    else:
        name = f"n{n}_k{k}_d{d}_alpha{alpha}_c{c}_nonsys.pkl"

    save_pickle(dict_, folder, name)


def make_reed_solomon_luts():
    folder = os.path.join(PATH, "data", "reed_solomon")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

    # Primitive RS codes over GF(2^4)
    rng = np.random.default_rng(1000)
    for d in range(2, 9):
        for systematic in [True, False]:
            add_reed_solomon_lut(folder, 2**4, 2**4 - 1, d, alpha=2, systematic=systematic, rng=rng)
            add_reed_solomon_lut(folder, 2**4, 2**4 - 1, d, alpha=9, systematic=systematic, rng=rng)
            # add_reed_solomon_lut(folder, 2**4, 2**4 - 1, d, alpha=9, c=3, systematic=systematic, rng=rng)

    # Non-primitive RS codes over GF(3^4)
    rng = np.random.default_rng(2000)
    for d in range(2, 9):
        for systematic in [True, False]:
            add_reed_solomon_lut(folder, 3**4, 16, d, alpha=31, systematic=systematic, rng=rng)
            add_reed_solomon_lut(folder, 3**4, 16, d, alpha=11, systematic=systematic, rng=rng)
            # add_reed_solomon_lut(folder, 3**4, 16, d, alpha=9, c=3, systematic=systematic, rng=rng)


if __name__ == "__main__":
    make_bch_luts()
    make_reed_solomon_luts()
