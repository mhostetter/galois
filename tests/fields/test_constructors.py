"""
A pytest module to test instantiation of new FieldArrays through alternate constructors.
"""

import numpy as np
import pytest

import galois

from .conftest import invalid_dtype, randint, valid_dtype

RANDOM_REPRODUCIBLE = {
    "GF(2)-42": galois.GF2([1, 0, 1, 0]),
    "GF(2)-1337": galois.GF2([1, 1, 1, 1]),
    "GF(2)-27182818284": galois.GF2([1, 1, 0, 1]),
    "GF(2^2)-42": galois.GF(2**2)([2, 0, 3, 0]),
    "GF(2^2)-1337": galois.GF(2**2)([2, 3, 3, 2]),
    "GF(2^2)-27182818284": galois.GF(2**2)([3, 3, 1, 3]),
    "GF(2^3)-42": galois.GF(2**3)([4, 1, 6, 0]),
    "GF(2^3)-1337": galois.GF(2**3)([4, 7, 6, 4]),
    "GF(2^3)-27182818284": galois.GF(2**3)([7, 6, 3, 7]),
    "GF(2^8)-42": galois.GF(2**8)([136, 38, 217, 22]),
    "GF(2^8)-1337": galois.GF(2**8)([129, 232, 219, 139]),
    "GF(2^8)-27182818284": galois.GF(2**8)([241, 199, 124, 251]),
    "GF(2^32)-42": galois.GF(2**32)([383329928, 3324115917, 2811363265, 1884968545]),
    "GF(2^32)-1337": galois.GF(2**32)([2346444929, 3771418944, 3122469425, 796836527]),
    "GF(2^32)-27182818284": galois.GF(2**32)([4219258865, 2369428822, 4215993831, 737067321]),
    "GF(2^100)-42": galois.GF(2**100)(
        [
            334597227169592314486858469975,
            160817104176174106146196623769,
            1073140825042556611459112948617,
            680511737545746878390210579424,
        ]
    ),
    "GF(2^100)-1337": galois.GF(2**100)(
        [
            857595242411244403701448342367,
            934069361206926877353993468533,
            975111178126741576966108234939,
            943790063887699295865451696011,
        ]
    ),
    "GF(2^100)-27182818284": galois.GF(2**100)(
        [
            331436691655515484372643008489,
            222081042415095477651329116333,
            738588762407569498067093382347,
            511609933311594721392744046328,
        ]
    ),
    "GF(5)-42": galois.GF(5)([2, 0, 4, 0]),
    "GF(5)-1337": galois.GF(5)([2, 4, 4, 2]),
    "GF(5)-27182818284": galois.GF(5)([4, 3, 2, 4]),
    "GF(7)-42": galois.GF(7)([3, 1, 5, 0]),
    "GF(7)-1337": galois.GF(7)([3, 6, 5, 3]),
    "GF(7)-27182818284": galois.GF(7)([6, 5, 3, 6]),
    "GF(31)-42": galois.GF(31)([16, 4, 26, 2]),
    "GF(31)-1337": galois.GF(31)([15, 28, 26, 16]),
    "GF(31)-27182818284": galois.GF(31)([29, 24, 30, 10]),
    "GF(3191)-42": galois.GF(3191)([480, 284, 3138, 2469]),
    "GF(3191)-1337": galois.GF(3191)([2898, 1743, 913, 326]),
    "GF(3191)-27182818284": galois.GF(3191)([2492, 3134, 1961, 1760]),
    "GF(2147483647)-42": galois.GF(2147483647)([191664963, 1662057957, 1405681631, 942484272]),
    "GF(2147483647)-1337": galois.GF(2147483647)([1173222463, 1885709471, 1561234711, 398418263]),
    "GF(2147483647)-27182818284": galois.GF(2147483647)([2109629431, 1184714410, 2107996914, 368533660]),
    "GF(36893488147419103183)-42": galois.GF(36893488147419103183)(
        [2053695854357871005, 5073395517033431291, 21020764468182013662, 30929646287488622503]
    ),
    "GF(36893488147419103183)-1337": galois.GF(36893488147419103183)(
        [35519858906168102147, 25195455184287478221, 13501076693576984818, 15207011548010871691]
    ),
    "GF(36893488147419103183)-27182818284": galois.GF(36893488147419103183)(
        [9039788033171530905, 30364060241532546935, 2417589984504447344, 14813783901409898527]
    ),
    "GF(7^3)-42": galois.GF(7**3)([51, 30, 337, 265]),
    "GF(7^3)-1337": galois.GF(7**3)([311, 187, 98, 301]),
    "GF(7^3)-27182818284": galois.GF(7**3)([267, 336, 210, 189]),
    "GF(109987^4)-42": galois.GF(109987**4)(
        [2053695854357871005, 23520139590742982907, 94807740763020220126, 104716622582326828967]
    ),
    "GF(109987^4)-1337": galois.GF(109987**4)(
        [90860091127296756995, 80535687405416133069, 31947820767286536434, 52100499695429974923]
    ),
    "GF(109987^4)-27182818284": galois.GF(109987**4)(
        [45933276180590634137, 104151036536370753399, 2417589984504447344, 14813783901409898527]
    ),
}


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_zeros(field, shape):
    a = field.Zeros(shape)
    assert np.all(a == 0)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_zeros_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Zeros(shape, dtype=dtype)
    assert np.all(a == 0)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_zeros_invalid_dtype(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        field.Zeros(shape, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_ones(field, shape):
    a = field.Ones(shape)
    assert np.all(a == 1)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_ones_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Ones(shape, dtype=dtype)
    assert np.all(a == 1)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_ones_invalid_dtype(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        field.Ones(shape, dtype=dtype)


def test_identity(field):
    size = 4
    a = field.Identity(size)
    for i in range(size):
        for j in range(size):
            assert a[i, j] == 1 if i == j else a[i, j] == 0
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == (size, size)


def test_identity_valid_dtype(field):
    dtype = valid_dtype(field)
    size = 4
    a = field.Identity(size, dtype=dtype)
    for i in range(size):
        for j in range(size):
            assert a[i, j] == 1 if i == j else a[i, j] == 0
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == (size, size)


def test_identity_invalid_dtype(field):
    dtype = invalid_dtype(field)
    size = 4
    with pytest.raises(TypeError):
        field.Identity(size, dtype=dtype)


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_random(field, shape):
    a = field.Random(shape, low=0, high=field.order)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("seed", [None, 42, np.int64(1337), np.ulonglong(27182818284), np.random.default_rng(123456)])
def test_random_valid_seed(field, seed):
    shape = (4, 4)
    low = randint(0, field.order, 1, field.dtypes[0])
    a = field.Random(shape, low=low, high=field.order, seed=seed)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == field.dtypes[0]
    assert a.shape == shape


@pytest.mark.parametrize("seed", ["hi", -1, 3.14, (1, 2, 3)])
def test_random_invalid_seed(field, seed):
    shape = (4, 4)
    with pytest.raises(ValueError):
        field.Random(shape, seed=seed)


def test_random_invalid_high(field):
    shape = (4,)
    with pytest.raises(ValueError):
        field.Random(shape, high=field.order + 1)


@pytest.mark.parametrize("seed", [42, 1337, 27182818284])
def test_random_reproducible(field, seed):
    shape = (4,)
    a = field.Random(shape, seed=seed)
    assert np.array_equal(a, RANDOM_REPRODUCIBLE[f"{field.name}-{seed}"])


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_random_valid_dtype(field, shape):
    dtype = valid_dtype(field)
    a = field.Random(shape, dtype=dtype)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_random_invalid_dtype(field, shape):
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        field.Random(shape, dtype=dtype)


def test_vandermonde_exceptions():
    GF = galois.GF(7)
    with pytest.raises(TypeError):
        GF.Vandermonde(2.0, 3, 3)
    with pytest.raises(TypeError):
        GF.Vandermonde(2, 3.0, 3)
    with pytest.raises(TypeError):
        GF.Vandermonde(2, 3, 3.0)
    with pytest.raises(ValueError):
        GF.Vandermonde(GF([2, 3]), 3, 3)
    with pytest.raises(ValueError):
        GF.Vandermonde(2, -3, 3)
    with pytest.raises(ValueError):
        GF.Vandermonde(2, 3, -3)


def test_vandermonde(field):
    m, n = 3, 4
    dtype = valid_dtype(field)
    V = field.Vandermonde(field.primitive_element, m, n, dtype=dtype)
    assert type(V) is field
    assert V.dtype == dtype
    assert V.shape == (m, n)

    assert np.all(V[0, :] == 1)
    assert np.all(V[:, 0] == 1)
    assert np.array_equal(V[1, :], field.primitive_element ** np.arange(0, n))


def test_vector_exceptions():
    GF = galois.GF(7**3)
    with pytest.raises(TypeError):
        GF.Vector([1, 2.0, 3])
    with pytest.raises(TypeError):
        GF.Vector([1, 2, 3], dtype=invalid_dtype(GF))
    with pytest.raises(ValueError):
        GF.Vector([1, 2, 3, 4])


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_vector_valid_dtype(field, shape):
    v_dtype = valid_dtype(field.prime_subfield)
    v_shape = list(shape) + [field.degree]
    v = field.prime_subfield.Random(v_shape, dtype=v_dtype)
    dtype = valid_dtype(field)
    a = field.Vector(v, dtype=dtype)
    assert np.all(a >= 0) and np.all(a < field.order)
    assert type(a) is field
    assert a.dtype == dtype
    assert a.shape == shape

    # Confirm the inverse operation reverts to the original array
    assert np.array_equal(a.vector(), v)


@pytest.mark.parametrize("shape", [(), (4,), (4, 4)])
def test_vector_invalid_dtype(field, shape):
    v_dtype = valid_dtype(field.prime_subfield)
    v_shape = list(shape) + [field.degree]
    v = field.prime_subfield.Random(v_shape, dtype=v_dtype)
    dtype = invalid_dtype(field)
    with pytest.raises(TypeError):
        field.Vector(v, dtype=dtype)
