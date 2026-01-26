"""
A pytest module to test methods of FieldArray subclasses.
"""

import numpy as np
import pytest

import galois

from .conftest import invalid_dtype, valid_dtype


def test_repr():
    GF = galois.GF(2**3)

    assert repr(GF(1)) == "GF(1, order=2^3)"
    assert repr(GF(0)) == "GF(0, order=2^3)"
    assert repr(GF(5)) == "GF(5, order=2^3)"
    assert repr(GF(2)) == "GF(2, order=2^3)"
    assert repr(GF([1, 0, 5, 2])) == "GF([1, 0, 5, 2], order=2^3)"

    GF.repr("poly")
    assert repr(GF(1)) == "GF(1, order=2^3)"
    assert repr(GF(0)) == "GF(0, order=2^3)"
    assert repr(GF(5)) == "GF(a^2 + 1, order=2^3)"
    assert repr(GF(2)) == "GF(a, order=2^3)"
    assert repr(GF([1, 0, 5, 2])) == "GF([      1,       0, a^2 + 1,       a], order=2^3)"

    GF.repr("power")
    assert repr(GF(1)) == "GF(1, order=2^3)"
    assert repr(GF(0)) == "GF(0, order=2^3)"
    assert repr(GF(5)) == "GF(g^6, order=2^3)"
    assert repr(GF(2)) == "GF(g, order=2^3)"
    assert repr(GF([1, 0, 5, 2])) == "GF([  1,   0, g^6,   g], order=2^3)"

    GF.repr()
    assert repr(GF(1)) == "GF(1, order=2^3)"
    assert repr(GF(0)) == "GF(0, order=2^3)"
    assert repr(GF(5)) == "GF(5, order=2^3)"
    assert repr(GF(2)) == "GF(2, order=2^3)"
    assert repr(GF([1, 0, 5, 2])) == "GF([1, 0, 5, 2], order=2^3)"


def test_str():
    GF = galois.GF(2**3)

    assert str(GF(1)) == "1"
    assert str(GF(0)) == "0"
    assert str(GF(5)) == "5"
    assert str(GF(2)) == "2"
    assert str(GF([1, 0, 5, 2])) == "[1 0 5 2]"

    GF.repr("poly")
    assert str(GF(1)) == "1"
    assert str(GF(0)) == "0"
    assert str(GF(5)) == "a^2 + 1"
    assert str(GF(2)) == "a"
    assert str(GF([1, 0, 5, 2])) == "[      1       0 a^2 + 1       a]"

    GF.repr("power")
    assert str(GF(1)) == "1"
    assert str(GF(0)) == "0"
    assert str(GF(5)) == "g^6"
    assert str(GF(2)) == "g"
    assert str(GF([1, 0, 5, 2])) == "[  1   0 g^6   g]"

    GF.repr()
    assert str(GF(1)) == "1"
    assert str(GF(0)) == "0"
    assert str(GF(5)) == "5"
    assert str(GF(2)) == "2"
    assert str(GF([1, 0, 5, 2])) == "[1 0 5 2]"


def test_repr_context_manager():
    GF = galois.GF(2**3)
    a = GF([1, 0, 5, 2])

    assert repr(a) == "GF([1, 0, 5, 2], order=2^3)"
    assert str(a) == "[1 0 5 2]"

    with GF.repr("poly"):
        assert repr(a) == "GF([      1,       0, a^2 + 1,       a], order=2^3)"
        assert str(a) == "[      1       0 a^2 + 1       a]"

    with GF.repr("power"):
        assert repr(a) == "GF([  1,   0, g^6,   g], order=2^3)"
        assert str(a) == "[  1   0 g^6   g]"

    assert repr(a) == "GF([1, 0, 5, 2], order=2^3)"
    assert str(a) == "[1 0 5 2]"


def test_repr_exceptions():
    GF = galois.GF(2**3)
    with pytest.raises(ValueError):
        GF.repr("invalid-display-type")


def test_arithmetic_table_exceptions():
    GF = galois.GF(2**3)
    with pytest.raises(ValueError):
        GF.arithmetic_table("invalid-arithmetic-type")


def test_arithmetic_table():
    GF = galois.GF(2**3)
    with GF.repr("int"):
        assert (
            GF.arithmetic_table("+")
            == "x + y | 0  1  2  3  4  5  6  7 \n------|------------------------\n    0 | 0  1  2  3  4  5  6  7 \n    1 | 1  0  3  2  5  4  7  6 \n    2 | 2  3  0  1  6  7  4  5 \n    3 | 3  2  1  0  7  6  5  4 \n    4 | 4  5  6  7  0  1  2  3 \n    5 | 5  4  7  6  1  0  3  2 \n    6 | 6  7  4  5  2  3  0  1 \n    7 | 7  6  5  4  3  2  1  0 "
        )
        assert (
            GF.arithmetic_table("-")
            == "x - y | 0  1  2  3  4  5  6  7 \n------|------------------------\n    0 | 0  1  2  3  4  5  6  7 \n    1 | 1  0  3  2  5  4  7  6 \n    2 | 2  3  0  1  6  7  4  5 \n    3 | 3  2  1  0  7  6  5  4 \n    4 | 4  5  6  7  0  1  2  3 \n    5 | 5  4  7  6  1  0  3  2 \n    6 | 6  7  4  5  2  3  0  1 \n    7 | 7  6  5  4  3  2  1  0 "
        )
        assert (
            GF.arithmetic_table("*")
            == "x * y | 0  1  2  3  4  5  6  7 \n------|------------------------\n    0 | 0  0  0  0  0  0  0  0 \n    1 | 0  1  2  3  4  5  6  7 \n    2 | 0  2  4  6  3  1  7  5 \n    3 | 0  3  6  5  7  4  1  2 \n    4 | 0  4  3  7  6  2  5  1 \n    5 | 0  5  1  4  2  7  3  6 \n    6 | 0  6  7  1  5  3  2  4 \n    7 | 0  7  5  2  1  6  4  3 "
        )
        assert (
            GF.arithmetic_table("/")
            == "x / y | 1  2  3  4  5  6  7 \n------|---------------------\n    0 | 0  0  0  0  0  0  0 \n    1 | 1  5  6  7  2  3  4 \n    2 | 2  1  7  5  4  6  3 \n    3 | 3  4  1  2  6  5  7 \n    4 | 4  2  5  1  3  7  6 \n    5 | 5  7  3  6  1  4  2 \n    6 | 6  3  2  4  7  1  5 \n    7 | 7  6  4  3  5  2  1 "
        )
    with GF.repr("poly"):
        assert (
            GF.arithmetic_table("+")
            == "      x + y |           0            1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n------------|--------------------------------------------------------------------------------------------------------\n          0 |           0            1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n          1 |           1            0        a + 1            a      a^2 + 1          a^2  a^2 + a + 1      a^2 + a \n          a |           a        a + 1            0            1      a^2 + a  a^2 + a + 1          a^2      a^2 + 1 \n      a + 1 |       a + 1            a            1            0  a^2 + a + 1      a^2 + a      a^2 + 1          a^2 \n        a^2 |         a^2      a^2 + 1      a^2 + a  a^2 + a + 1            0            1            a        a + 1 \n    a^2 + 1 |     a^2 + 1          a^2  a^2 + a + 1      a^2 + a            1            0        a + 1            a \n    a^2 + a |     a^2 + a  a^2 + a + 1          a^2      a^2 + 1            a        a + 1            0            1 \na^2 + a + 1 | a^2 + a + 1      a^2 + a      a^2 + 1          a^2        a + 1            a            1            0 "
        )
        assert (
            GF.arithmetic_table("-")
            == "      x - y |           0            1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n------------|--------------------------------------------------------------------------------------------------------\n          0 |           0            1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n          1 |           1            0        a + 1            a      a^2 + 1          a^2  a^2 + a + 1      a^2 + a \n          a |           a        a + 1            0            1      a^2 + a  a^2 + a + 1          a^2      a^2 + 1 \n      a + 1 |       a + 1            a            1            0  a^2 + a + 1      a^2 + a      a^2 + 1          a^2 \n        a^2 |         a^2      a^2 + 1      a^2 + a  a^2 + a + 1            0            1            a        a + 1 \n    a^2 + 1 |     a^2 + 1          a^2  a^2 + a + 1      a^2 + a            1            0        a + 1            a \n    a^2 + a |     a^2 + a  a^2 + a + 1          a^2      a^2 + 1            a        a + 1            0            1 \na^2 + a + 1 | a^2 + a + 1      a^2 + a      a^2 + 1          a^2        a + 1            a            1            0 "
        )
        assert (
            GF.arithmetic_table("*")
            == "      x * y |           0            1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n------------|--------------------------------------------------------------------------------------------------------\n          0 |           0            0            0            0            0            0            0            0 \n          1 |           0            1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n          a |           0            a          a^2      a^2 + a        a + 1            1  a^2 + a + 1      a^2 + 1 \n      a + 1 |           0        a + 1      a^2 + a      a^2 + 1  a^2 + a + 1          a^2            1            a \n        a^2 |           0          a^2        a + 1  a^2 + a + 1      a^2 + a            a      a^2 + 1            1 \n    a^2 + 1 |           0      a^2 + 1            1          a^2            a  a^2 + a + 1        a + 1      a^2 + a \n    a^2 + a |           0      a^2 + a  a^2 + a + 1            1      a^2 + 1        a + 1            a          a^2 \na^2 + a + 1 |           0  a^2 + a + 1      a^2 + 1            a            1      a^2 + a          a^2        a + 1 "
        )
        assert (
            GF.arithmetic_table("/")
            == "      x / y |           1            a        a + 1          a^2      a^2 + 1      a^2 + a  a^2 + a + 1 \n------------|-------------------------------------------------------------------------------------------\n          0 |           0            0            0            0            0            0            0 \n          1 |           1      a^2 + 1      a^2 + a  a^2 + a + 1            a        a + 1          a^2 \n          a |           a            1  a^2 + a + 1      a^2 + 1          a^2      a^2 + a        a + 1 \n      a + 1 |       a + 1          a^2            1            a      a^2 + a      a^2 + 1  a^2 + a + 1 \n        a^2 |         a^2            a      a^2 + 1            1        a + 1  a^2 + a + 1      a^2 + a \n    a^2 + 1 |     a^2 + 1  a^2 + a + 1        a + 1      a^2 + a            1          a^2            a \n    a^2 + a |     a^2 + a        a + 1            a          a^2  a^2 + a + 1            1      a^2 + 1 \na^2 + a + 1 | a^2 + a + 1      a^2 + a          a^2        a + 1      a^2 + 1            a            1 "
        )
    with GF.repr("power"):
        assert (
            GF.arithmetic_table("+")
            == "x + y |   0    1    g  g^2  g^3  g^4  g^5  g^6 \n------|----------------------------------------\n    0 |   0    1    g  g^2  g^3  g^4  g^5  g^6 \n    1 |   1    0  g^3  g^6    g  g^5  g^4  g^2 \n    g |   g  g^3    0  g^4    1  g^2  g^6  g^5 \n  g^2 | g^2  g^6  g^4    0  g^5    g  g^3    1 \n  g^3 | g^3    g    1  g^5    0  g^6  g^2  g^4 \n  g^4 | g^4  g^5  g^2    g  g^6    0    1  g^3 \n  g^5 | g^5  g^4  g^6  g^3  g^2    1    0    g \n  g^6 | g^6  g^2  g^5    1  g^4  g^3    g    0 "
        )
        assert (
            GF.arithmetic_table("-")
            == "x - y |   0    1    g  g^2  g^3  g^4  g^5  g^6 \n------|----------------------------------------\n    0 |   0    1    g  g^2  g^3  g^4  g^5  g^6 \n    1 |   1    0  g^3  g^6    g  g^5  g^4  g^2 \n    g |   g  g^3    0  g^4    1  g^2  g^6  g^5 \n  g^2 | g^2  g^6  g^4    0  g^5    g  g^3    1 \n  g^3 | g^3    g    1  g^5    0  g^6  g^2  g^4 \n  g^4 | g^4  g^5  g^2    g  g^6    0    1  g^3 \n  g^5 | g^5  g^4  g^6  g^3  g^2    1    0    g \n  g^6 | g^6  g^2  g^5    1  g^4  g^3    g    0 "
        )
        assert (
            GF.arithmetic_table("*")
            == "x * y |   0    1    g  g^2  g^3  g^4  g^5  g^6 \n------|----------------------------------------\n    0 |   0    0    0    0    0    0    0    0 \n    1 |   0    1    g  g^2  g^3  g^4  g^5  g^6 \n    g |   0    g  g^2  g^3  g^4  g^5  g^6    1 \n  g^2 |   0  g^2  g^3  g^4  g^5  g^6    1    g \n  g^3 |   0  g^3  g^4  g^5  g^6    1    g  g^2 \n  g^4 |   0  g^4  g^5  g^6    1    g  g^2  g^3 \n  g^5 |   0  g^5  g^6    1    g  g^2  g^3  g^4 \n  g^6 |   0  g^6    1    g  g^2  g^3  g^4  g^5 "
        )
        assert (
            GF.arithmetic_table("/")
            == "x / y |   1    g  g^2  g^3  g^4  g^5  g^6 \n------|-----------------------------------\n    0 |   0    0    0    0    0    0    0 \n    1 |   1  g^6  g^5  g^4  g^3  g^2    g \n    g |   g    1  g^6  g^5  g^4  g^3  g^2 \n  g^2 | g^2    g    1  g^6  g^5  g^4  g^3 \n  g^3 | g^3  g^2    g    1  g^6  g^5  g^4 \n  g^4 | g^4  g^3  g^2    g    1  g^6  g^5 \n  g^5 | g^5  g^4  g^3  g^2    g    1  g^6 \n  g^6 | g^6  g^5  g^4  g^3  g^2    g    1 "
        )


def test_repr_table_exceptions():
    GF = galois.GF(2**3)
    with pytest.raises(ValueError):
        GF.repr_table(sort="invalid-sort-type")


def test_repr_table():
    GF = galois.GF(2**3)
    assert (
        GF.repr_table()
        == " Power    Polynomial     Vector    Integer \n------- ------------- ----------- ---------\n   0          0        [0, 0, 0]      0     \n  a^0         1        [0, 0, 1]      1     \n  a^1         a        [0, 1, 0]      2     \n  a^2        a^2       [1, 0, 0]      4     \n  a^3       a + 1      [0, 1, 1]      3     \n  a^4      a^2 + a     [1, 1, 0]      6     \n  a^5    a^2 + a + 1   [1, 1, 1]      7     \n  a^6      a^2 + 1     [1, 0, 1]      5     "
    )
    assert (
        GF.repr_table(sort="int")
        == " Power    Polynomial     Vector    Integer \n------- ------------- ----------- ---------\n   0          0        [0, 0, 0]      0     \n  a^0         1        [0, 0, 1]      1     \n  a^1         a        [0, 1, 0]      2     \n  a^3       a + 1      [0, 1, 1]      3     \n  a^2        a^2       [1, 0, 0]      4     \n  a^6      a^2 + 1     [1, 0, 1]      5     \n  a^4      a^2 + a     [1, 1, 0]      6     \n  a^5    a^2 + a + 1   [1, 1, 1]      7     "
    )

    g = GF.primitive_elements[-1]
    assert (
        GF.repr_table(g)
        == "      Power         Polynomial     Vector    Integer \n----------------- ------------- ----------- ---------\n        0               0        [0, 0, 0]      0     \n (a^2 + a + 1)^0        1        [0, 0, 1]      1     \n (a^2 + a + 1)^1   a^2 + a + 1   [1, 1, 1]      7     \n (a^2 + a + 1)^2      a + 1      [0, 1, 1]      3     \n (a^2 + a + 1)^3        a        [0, 1, 0]      2     \n (a^2 + a + 1)^4     a^2 + 1     [1, 0, 1]      5     \n (a^2 + a + 1)^5     a^2 + a     [1, 1, 0]      6     \n (a^2 + a + 1)^6       a^2       [1, 0, 0]      4     "
    )
    assert (
        GF.repr_table(g, sort="int")
        == "      Power         Polynomial     Vector    Integer \n----------------- ------------- ----------- ---------\n        0               0        [0, 0, 0]      0     \n (a^2 + a + 1)^0        1        [0, 0, 1]      1     \n (a^2 + a + 1)^3        a        [0, 1, 0]      2     \n (a^2 + a + 1)^2      a + 1      [0, 1, 1]      3     \n (a^2 + a + 1)^6       a^2       [1, 0, 0]      4     \n (a^2 + a + 1)^4     a^2 + 1     [1, 0, 1]      5     \n (a^2 + a + 1)^5     a^2 + a     [1, 1, 0]      6     \n (a^2 + a + 1)^1   a^2 + a + 1   [1, 1, 1]      7     "
    )


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_vector_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Random(shape, dtype=dtype)

    v_dtype = valid_dtype(field.prime_subfield)
    v_shape = tuple(list(shape) + [field.degree])
    v = a.vector(dtype=v_dtype)

    assert np.all(v >= 0) and np.all(v < field.prime_subfield.order)
    assert type(v) is field.prime_subfield
    assert v.dtype == v_dtype
    assert v.shape == v_shape

    # Confirm the inverse operation reverts to the original array
    assert np.array_equal(field.Vector(v), a)


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_vector_invalid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Random(shape, dtype=dtype)

    v_dtype = invalid_dtype(field.prime_subfield)
    with pytest.raises(TypeError):
        a.vector(dtype=v_dtype)
