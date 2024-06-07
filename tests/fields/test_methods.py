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
    assert repr(GF(5)) == "GF(α^2 + 1, order=2^3)"
    assert repr(GF(2)) == "GF(α, order=2^3)"
    assert repr(GF([1, 0, 5, 2])) == "GF([      1,       0, α^2 + 1,       α], order=2^3)"

    GF.repr("power")
    assert repr(GF(1)) == "GF(1, order=2^3)"
    assert repr(GF(0)) == "GF(0, order=2^3)"
    assert repr(GF(5)) == "GF(α^6, order=2^3)"
    assert repr(GF(2)) == "GF(α, order=2^3)"
    assert repr(GF([1, 0, 5, 2])) == "GF([  1,   0, α^6,   α], order=2^3)"

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
    assert str(GF(5)) == "α^2 + 1"
    assert str(GF(2)) == "α"
    assert str(GF([1, 0, 5, 2])) == "[      1       0 α^2 + 1       α]"

    GF.repr("power")
    assert str(GF(1)) == "1"
    assert str(GF(0)) == "0"
    assert str(GF(5)) == "α^6"
    assert str(GF(2)) == "α"
    assert str(GF([1, 0, 5, 2])) == "[  1   0 α^6   α]"

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
        assert repr(a) == "GF([      1,       0, α^2 + 1,       α], order=2^3)"
        assert str(a) == "[      1       0 α^2 + 1       α]"

    with GF.repr("power"):
        assert repr(a) == "GF([  1,   0, α^6,   α], order=2^3)"
        assert str(a) == "[  1   0 α^6   α]"

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
            == "      x + y |           0            1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n------------|--------------------------------------------------------------------------------------------------------\n          0 |           0            1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n          1 |           1            0        α + 1            α      α^2 + 1          α^2  α^2 + α + 1      α^2 + α \n          α |           α        α + 1            0            1      α^2 + α  α^2 + α + 1          α^2      α^2 + 1 \n      α + 1 |       α + 1            α            1            0  α^2 + α + 1      α^2 + α      α^2 + 1          α^2 \n        α^2 |         α^2      α^2 + 1      α^2 + α  α^2 + α + 1            0            1            α        α + 1 \n    α^2 + 1 |     α^2 + 1          α^2  α^2 + α + 1      α^2 + α            1            0        α + 1            α \n    α^2 + α |     α^2 + α  α^2 + α + 1          α^2      α^2 + 1            α        α + 1            0            1 \nα^2 + α + 1 | α^2 + α + 1      α^2 + α      α^2 + 1          α^2        α + 1            α            1            0 "
        )
        assert (
            GF.arithmetic_table("-")
            == "      x - y |           0            1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n------------|--------------------------------------------------------------------------------------------------------\n          0 |           0            1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n          1 |           1            0        α + 1            α      α^2 + 1          α^2  α^2 + α + 1      α^2 + α \n          α |           α        α + 1            0            1      α^2 + α  α^2 + α + 1          α^2      α^2 + 1 \n      α + 1 |       α + 1            α            1            0  α^2 + α + 1      α^2 + α      α^2 + 1          α^2 \n        α^2 |         α^2      α^2 + 1      α^2 + α  α^2 + α + 1            0            1            α        α + 1 \n    α^2 + 1 |     α^2 + 1          α^2  α^2 + α + 1      α^2 + α            1            0        α + 1            α \n    α^2 + α |     α^2 + α  α^2 + α + 1          α^2      α^2 + 1            α        α + 1            0            1 \nα^2 + α + 1 | α^2 + α + 1      α^2 + α      α^2 + 1          α^2        α + 1            α            1            0 "
        )
        assert (
            GF.arithmetic_table("*")
            == "      x * y |           0            1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n------------|--------------------------------------------------------------------------------------------------------\n          0 |           0            0            0            0            0            0            0            0 \n          1 |           0            1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n          α |           0            α          α^2      α^2 + α        α + 1            1  α^2 + α + 1      α^2 + 1 \n      α + 1 |           0        α + 1      α^2 + α      α^2 + 1  α^2 + α + 1          α^2            1            α \n        α^2 |           0          α^2        α + 1  α^2 + α + 1      α^2 + α            α      α^2 + 1            1 \n    α^2 + 1 |           0      α^2 + 1            1          α^2            α  α^2 + α + 1        α + 1      α^2 + α \n    α^2 + α |           0      α^2 + α  α^2 + α + 1            1      α^2 + 1        α + 1            α          α^2 \nα^2 + α + 1 |           0  α^2 + α + 1      α^2 + 1            α            1      α^2 + α          α^2        α + 1 "
        )
        assert (
            GF.arithmetic_table("/")
            == "      x / y |           1            α        α + 1          α^2      α^2 + 1      α^2 + α  α^2 + α + 1 \n------------|-------------------------------------------------------------------------------------------\n          0 |           0            0            0            0            0            0            0 \n          1 |           1      α^2 + 1      α^2 + α  α^2 + α + 1            α        α + 1          α^2 \n          α |           α            1  α^2 + α + 1      α^2 + 1          α^2      α^2 + α        α + 1 \n      α + 1 |       α + 1          α^2            1            α      α^2 + α      α^2 + 1  α^2 + α + 1 \n        α^2 |         α^2            α      α^2 + 1            1        α + 1  α^2 + α + 1      α^2 + α \n    α^2 + 1 |     α^2 + 1  α^2 + α + 1        α + 1      α^2 + α            1          α^2            α \n    α^2 + α |     α^2 + α        α + 1            α          α^2  α^2 + α + 1            1      α^2 + 1 \nα^2 + α + 1 | α^2 + α + 1      α^2 + α          α^2        α + 1      α^2 + 1            α            1 "
        )
    with GF.repr("power"):
        assert (
            GF.arithmetic_table("+")
            == "x + y |   0    1    α  α^2  α^3  α^4  α^5  α^6 \n------|----------------------------------------\n    0 |   0    1    α  α^2  α^3  α^4  α^5  α^6 \n    1 |   1    0  α^3  α^6    α  α^5  α^4  α^2 \n    α |   α  α^3    0  α^4    1  α^2  α^6  α^5 \n  α^2 | α^2  α^6  α^4    0  α^5    α  α^3    1 \n  α^3 | α^3    α    1  α^5    0  α^6  α^2  α^4 \n  α^4 | α^4  α^5  α^2    α  α^6    0    1  α^3 \n  α^5 | α^5  α^4  α^6  α^3  α^2    1    0    α \n  α^6 | α^6  α^2  α^5    1  α^4  α^3    α    0 "
        )
        assert (
            GF.arithmetic_table("-")
            == "x - y |   0    1    α  α^2  α^3  α^4  α^5  α^6 \n------|----------------------------------------\n    0 |   0    1    α  α^2  α^3  α^4  α^5  α^6 \n    1 |   1    0  α^3  α^6    α  α^5  α^4  α^2 \n    α |   α  α^3    0  α^4    1  α^2  α^6  α^5 \n  α^2 | α^2  α^6  α^4    0  α^5    α  α^3    1 \n  α^3 | α^3    α    1  α^5    0  α^6  α^2  α^4 \n  α^4 | α^4  α^5  α^2    α  α^6    0    1  α^3 \n  α^5 | α^5  α^4  α^6  α^3  α^2    1    0    α \n  α^6 | α^6  α^2  α^5    1  α^4  α^3    α    0 "
        )
        assert (
            GF.arithmetic_table("*")
            == "x * y |   0    1    α  α^2  α^3  α^4  α^5  α^6 \n------|----------------------------------------\n    0 |   0    0    0    0    0    0    0    0 \n    1 |   0    1    α  α^2  α^3  α^4  α^5  α^6 \n    α |   0    α  α^2  α^3  α^4  α^5  α^6    1 \n  α^2 |   0  α^2  α^3  α^4  α^5  α^6    1    α \n  α^3 |   0  α^3  α^4  α^5  α^6    1    α  α^2 \n  α^4 |   0  α^4  α^5  α^6    1    α  α^2  α^3 \n  α^5 |   0  α^5  α^6    1    α  α^2  α^3  α^4 \n  α^6 |   0  α^6    1    α  α^2  α^3  α^4  α^5 "
        )
        assert (
            GF.arithmetic_table("/")
            == "x / y |   1    α  α^2  α^3  α^4  α^5  α^6 \n------|-----------------------------------\n    0 |   0    0    0    0    0    0    0 \n    1 |   1  α^6  α^5  α^4  α^3  α^2    α \n    α |   α    1  α^6  α^5  α^4  α^3  α^2 \n  α^2 | α^2    α    1  α^6  α^5  α^4  α^3 \n  α^3 | α^3  α^2    α    1  α^6  α^5  α^4 \n  α^4 | α^4  α^3  α^2    α    1  α^6  α^5 \n  α^5 | α^5  α^4  α^3  α^2    α    1  α^6 \n  α^6 | α^6  α^5  α^4  α^3  α^2    α    1 "
        )


def test_repr_table_exceptions():
    GF = galois.GF(2**3)
    with pytest.raises(ValueError):
        GF.repr_table(sort="invalid-sort-type")


def test_repr_table():
    GF = galois.GF(2**3)
    assert (
        GF.repr_table()
        == " Power    Polynomial     Vector    Integer \n------- ------------- ----------- ---------\n   0          0        [0, 0, 0]      0     \n  x^0         1        [0, 0, 1]      1     \n  x^1         x        [0, 1, 0]      2     \n  x^2        x^2       [1, 0, 0]      4     \n  x^3       x + 1      [0, 1, 1]      3     \n  x^4      x^2 + x     [1, 1, 0]      6     \n  x^5    x^2 + x + 1   [1, 1, 1]      7     \n  x^6      x^2 + 1     [1, 0, 1]      5     "
    )
    assert (
        GF.repr_table(sort="int")
        == " Power    Polynomial     Vector    Integer \n------- ------------- ----------- ---------\n   0          0        [0, 0, 0]      0     \n  x^0         1        [0, 0, 1]      1     \n  x^1         x        [0, 1, 0]      2     \n  x^3       x + 1      [0, 1, 1]      3     \n  x^2        x^2       [1, 0, 0]      4     \n  x^6      x^2 + 1     [1, 0, 1]      5     \n  x^4      x^2 + x     [1, 1, 0]      6     \n  x^5    x^2 + x + 1   [1, 1, 1]      7     "
    )

    alpha = GF.primitive_elements[-1]
    assert (
        GF.repr_table(alpha)
        == "      Power         Polynomial     Vector    Integer \n----------------- ------------- ----------- ---------\n        0               0        [0, 0, 0]      0     \n (x^2 + x + 1)^0        1        [0, 0, 1]      1     \n (x^2 + x + 1)^1   x^2 + x + 1   [1, 1, 1]      7     \n (x^2 + x + 1)^2      x + 1      [0, 1, 1]      3     \n (x^2 + x + 1)^3        x        [0, 1, 0]      2     \n (x^2 + x + 1)^4     x^2 + 1     [1, 0, 1]      5     \n (x^2 + x + 1)^5     x^2 + x     [1, 1, 0]      6     \n (x^2 + x + 1)^6       x^2       [1, 0, 0]      4     "
    )
    assert (
        GF.repr_table(alpha, sort="int")
        == "      Power         Polynomial     Vector    Integer \n----------------- ------------- ----------- ---------\n        0               0        [0, 0, 0]      0     \n (x^2 + x + 1)^0        1        [0, 0, 1]      1     \n (x^2 + x + 1)^3        x        [0, 1, 0]      2     \n (x^2 + x + 1)^2      x + 1      [0, 1, 1]      3     \n (x^2 + x + 1)^6       x^2       [1, 0, 0]      4     \n (x^2 + x + 1)^4     x^2 + 1     [1, 0, 1]      5     \n (x^2 + x + 1)^5     x^2 + x     [1, 1, 0]      6     \n (x^2 + x + 1)^1   x^2 + x + 1   [1, 1, 1]      7     "
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
