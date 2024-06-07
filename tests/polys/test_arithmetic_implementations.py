"""
A pytest module to test polynomial arithmetic using the "binary" and "sparse" implementations.

We don't need to verify the arithmetic is correct (that was already done in test_arithmetic.py). We
just need to make sure the arithmetic is the same for the different implementations.
"""

import random

import galois


def test_add(field):
    a = galois.Poly.Random(random.randint(0, 5), field=field)
    b = galois.Poly.Random(random.randint(0, 5), field=field)

    f_dense = a + b

    a._type = "sparse"
    b._type = "sparse"
    f_sparse = a + b
    assert f_sparse == f_dense

    if field is galois.GF2:
        a._type = "binary"
        b._type = "binary"
        f_binary = a + b
        assert f_binary == f_dense


def test_neg(field):
    a = galois.Poly.Random(random.randint(0, 5), field=field)

    f_dense = -a

    a._type = "sparse"
    f_sparse = -a
    assert f_sparse == f_dense

    if field is galois.GF2:
        a._type = "binary"
        f_binary = -a
        assert f_binary == f_dense


def test_subtract(field):
    a = galois.Poly.Random(random.randint(0, 5), field=field)
    b = galois.Poly.Random(random.randint(0, 5), field=field)

    f_dense = a - b

    a._type = "sparse"
    b._type = "sparse"
    f_sparse = a - b
    assert f_sparse == f_dense

    if field is galois.GF2:
        a._type = "binary"
        b._type = "binary"
        f_binary = a - b
        assert f_binary == f_dense


def test_multiply(field):
    for _ in range(100):
        a = galois.Poly.Random(random.randint(0, 5), field=field)
        b = galois.Poly.Random(random.randint(0, 5), field=field)

        f_dense = a * b

        a._type = "sparse"
        b._type = "sparse"
        f_sparse = a * b
        assert f_sparse == f_dense

        if field is galois.GF2:
            a._type = "binary"
            b._type = "binary"
            f_binary = a * b
            assert f_binary == f_dense


def test_divmod(field):
    a = galois.Poly.Random(random.randint(0, 5), field=field)
    b = galois.Poly.Random(random.randint(0, 5), field=field)

    q_dense, r_dense = divmod(a, b)

    a._type = "sparse"
    b._type = "sparse"
    q_sparse, r_sparse = divmod(a, b)
    assert q_sparse == q_dense
    assert r_sparse == r_dense

    if field is galois.GF2:
        a._type = "binary"
        b._type = "binary"
        q_binary, r_binary = divmod(a, b)
        assert q_binary == q_dense
        assert r_binary == r_dense


def test_floordiv(field):
    a = galois.Poly.Random(random.randint(0, 5), field=field)
    b = galois.Poly.Random(random.randint(0, 5), field=field)

    f_dense = a // b

    a._type = "sparse"
    b._type = "sparse"
    f_sparse = a // b
    assert f_sparse == f_dense

    if field is galois.GF2:
        a._type = "binary"
        b._type = "binary"
        f_binary = a // b
        assert f_binary == f_dense


def test_mod(field):
    a = galois.Poly.Random(random.randint(0, 5), field=field)
    b = galois.Poly.Random(random.randint(0, 5), field=field)

    f_dense = a % b

    a._type = "sparse"
    b._type = "sparse"
    f_sparse = a % b
    assert f_sparse == f_dense

    if field is galois.GF2:
        a._type = "binary"
        b._type = "binary"
        f_binary = a % b
        assert f_binary == f_dense
