"""
A pytest module to test polynomial arithmetic over Galois fields.
"""
import numpy as np

import galois

# pylint: disable=unidiomatic-typecheck


def test_add(poly_add):
    GF, X, Y, Z = poly_add["GF"], poly_add["X"], poly_add["Y"], poly_add["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        z = x + y
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_right_add(poly_add):
    GF = poly_add["GF"]
    a = GF.Random()
    b_poly = galois.Poly.Random(5, field=GF)
    assert a + b_poly == galois.Poly(a) + b_poly


def test_subtract(poly_subtract):
    GF, X, Y, Z = poly_subtract["GF"], poly_subtract["X"], poly_subtract["Y"], poly_subtract["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        z = x - y
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_right_subtract(poly_subtract):
    GF = poly_subtract["GF"]
    a = GF.Random()
    b_poly = galois.Poly.Random(5, field=GF)
    assert a - b_poly == galois.Poly(a) - b_poly


def test_multiply(poly_multiply):
    GF, X, Y, Z = poly_multiply["GF"], poly_multiply["X"], poly_multiply["Y"], poly_multiply["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        z = x * y
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_right_multiply(poly_multiply):
    GF = poly_multiply["GF"]
    a = GF.Random()
    b_poly = galois.Poly.Random(5, field=GF)
    assert a * b_poly == galois.Poly(a) * b_poly


def test_scalar_multiply(poly_scalar_multiply):
    GF, X, Y, Z = poly_scalar_multiply["GF"], poly_scalar_multiply["X"], poly_scalar_multiply["Y"], poly_scalar_multiply["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        z = x * y
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF

        z = y * x
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_divide(poly_divmod):
    GF, X, Y, Q = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["Q"]
    for x, y, q_truth in zip(X, Y, Q):
        q = x // y
        assert q == q_truth
        assert isinstance(q, galois.Poly)
        assert q.field is GF
        assert type(q.coeffs) is GF


def test_right_divide(poly_divmod):
    GF = poly_divmod["GF"]
    a = GF.Random()
    b_poly = galois.Poly.Random(5, field=GF)
    assert a // b_poly == galois.Poly(a) // b_poly


def test_mod(poly_divmod):
    # NOTE: Test modulo separately because there's a separate method to compute it without the quotient for space spacings
    GF, X, Y, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["R"]
    for x, y, r_truth in zip(X, Y, R):
        r = x % y
        assert r == r_truth
        assert isinstance(r, galois.Poly)
        assert r.field is GF
        assert type(r.coeffs) is GF


def test_right_mod(poly_divmod):
    GF = poly_divmod["GF"]
    a = GF.Random()
    b_poly = galois.Poly.Random(5, field=GF)
    assert a % b_poly == galois.Poly(a) % b_poly


def test_divmod(poly_divmod):
    GF, X, Y, Q, R = poly_divmod["GF"], poly_divmod["X"], poly_divmod["Y"], poly_divmod["Q"], poly_divmod["R"]
    for x, y, q_truth, r_truth in zip(X, Y, Q, R):
        q, r = divmod(x, y)

        assert q == q_truth
        assert isinstance(q, galois.Poly)
        assert q.field is GF
        assert type(q.coeffs) is GF

        assert r == r_truth
        assert isinstance(r, galois.Poly)
        assert r.field is GF
        assert type(r.coeffs) is GF


def test_right_divmod(poly_divmod):
    GF = poly_divmod["GF"]
    a = GF.Random()
    b_poly = galois.Poly.Random(5, field=GF)
    assert divmod(a, b_poly) == divmod(galois.Poly(a), b_poly)


def test_power(poly_power):
    GF, X, Y, Z = poly_power["GF"], poly_power["X"], poly_power["Y"], poly_power["Z"]
    for x, Zs in zip(X, Z):
        for y, z_truth in zip(Y, Zs):
            z = x**y
            assert z == z_truth
            assert isinstance(z, galois.Poly)
            assert z.field is GF
            assert type(z.coeffs) is GF


def test_modular_power(poly_modular_power):
    GF, X, E, M, Z = (
        poly_modular_power["GF"],
        poly_modular_power["X"],
        poly_modular_power["E"],
        poly_modular_power["M"],
        poly_modular_power["Z"],
    )
    for x, e, m, z_truth in zip(X, E, M, Z):
        z = pow(x, e, m)
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field is GF
        assert type(z.coeffs) is GF


def test_modular_power_large_exponent_jit():
    """
    Sage:
        F = GF(2^8, repr="int")
        R = PolynomialRing(F, names="x")
        f_coeffs = [255, 228, 34, 121, 243, 189, 6, 131, 102, 168, 82]
        g_coeffs = [193, 88, 107, 214, 72, 3]
        f = R([F.fetch_int(fi) for fi in f_coeffs[::-1]])
        g = R([F.fetch_int(gi) for gi in g_coeffs[::-1]])
        print(pow(f, 2**70 + 0, g))
        print(pow(f, 2**70 + 1234, g))
        print(pow(f, 2**70 + 7654, g))
        print(pow(f, 2**70 + 105030405, g))
    """
    GF = galois.GF(2**8)
    f = galois.Poly([255, 228, 34, 121, 243, 189, 6, 131, 102, 168, 82], field=GF)
    g = galois.Poly([193, 88, 107, 214, 72, 3], field=GF)

    assert pow(f, 2**70 + 0, g) == galois.Poly.Str("178*x^4 + 228*x^3 + 198*x^2 + 191*x + 211", field=GF)
    assert pow(f, 2**70 + 1234, g) == galois.Poly.Str("100*x^4 + 242*x^3 + 235*x^2 + 171*x + 43", field=GF)
    assert pow(f, 2**70 + 7654, g) == galois.Poly.Str("203*x^4 + 203*x^3 + 155*x^2 + 221*x + 151", field=GF)
    assert pow(f, 2**70 + 105030405, g) == galois.Poly.Str("180*x^4 + 206*x^3 + 223*x^2 + 126*x + 175", field=GF)


def test_modular_power_large_exponent_python():
    """
    Sage:
        F = GF(2^100, repr="int")
        R = PolynomialRing(F, names="x")
        f_coeffs = [255, 228, 34, 121, 243, 189, 6, 131, 102, 168, 82]
        g_coeffs = [193, 88, 107, 214, 72, 3]
        f = R([F.fetch_int(fi) for fi in f_coeffs[::-1]])
        g = R([F.fetch_int(gi) for gi in g_coeffs[::-1]])
        print([fi.integer_representation() for fi in pow(f, 2**70 + 0, g).list()[::-1]])
        print([fi.integer_representation() for fi in pow(f, 2**70 + 1234, g).list()[::-1]])
        print([fi.integer_representation() for fi in pow(f, 2**70 + 7654, g).list()[::-1]])
        print([fi.integer_representation() for fi in pow(f, 2**70 + 105030405, g).list()[::-1]])
    """
    GF = galois.GF(2**100)
    f = galois.Poly([255, 228, 34, 121, 243, 189, 6, 131, 102, 168, 82], field=GF)
    g = galois.Poly([193, 88, 107, 214, 72, 3], field=GF)

    assert pow(f, 2**70 + 0, g) == galois.Poly(
        [
            420013998870488935594333531316,
            467166943839280220379055289966,
            186006824455335245843600812277,
            96771878479768144633356244863,
            157326613576996636293122695271,
        ],
        field=GF,
    )
    assert pow(f, 2**70 + 1234, g) == galois.Poly(
        [
            22570526373096432759079317290,
            1022650052301719787915054353024,
            36488930895254982134146321994,
            232103113155652429788397015469,
            602929380923609768536867742066,
        ],
        field=GF,
    )
    assert pow(f, 2**70 + 7654, g) == galois.Poly(
        [
            1157532413047205128638237902356,
            731431734000747876200385228646,
            311313764490655270029408542359,
            81825181444198002714338087143,
            68173155813012544552855134791,
        ],
        field=GF,
    )
    assert pow(f, 2**70 + 105030405, g) == galois.Poly(
        [
            795758922378672681775973344546,
            1221486083569158504745962352000,
            474560121964431239726873721828,
            1008821918134362696532498449793,
            664177063731066580685161724661,
        ],
        field=GF,
    )


def test_evaluate_constant(poly_evaluate):
    GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
    for i in range(len(X)):  # pylint: disable=consider-using-enumerate
        j = np.random.default_rng().integers(0, Y.size)
        x = X[i]  # Polynomial
        y = Y[j]  # GF element
        z = x(y)  # GF element
        assert z == Z[i, j]
        assert type(z) is GF


def test_evaluate_vector(poly_evaluate):
    GF, X, Y, Z = poly_evaluate["GF"], poly_evaluate["X"], poly_evaluate["Y"], poly_evaluate["Z"]
    for i in range(len(X)):  # pylint: disable=consider-using-enumerate
        x = X[i]  # Polynomial
        y = Y  # GF array
        z = x(y)  # GF array
        assert np.array_equal(z, Z[i, :])
        assert type(z) is GF


def test_evaluate_matrix(poly_evaluate_matrix):
    GF, X, Y, Z = poly_evaluate_matrix["GF"], poly_evaluate_matrix["X"], poly_evaluate_matrix["Y"], poly_evaluate_matrix["Z"]
    for i in range(len(X)):  # pylint: disable=consider-using-enumerate
        x = X[i]  # Polynomial
        y = Y[i]  # GF square matrix
        z = x(y, elementwise=False)  # GF square matrix
        assert np.array_equal(z, Z[i])
        assert type(z) is GF


def test_evaluate_poly(poly_evaluate_poly):
    GF, X, Y, Z = poly_evaluate_poly["GF"], poly_evaluate_poly["X"], poly_evaluate_poly["Y"], poly_evaluate_poly["Z"]
    for x, y, z_truth in zip(X, Y, Z):
        z = x(y)
        assert z == z_truth
        assert isinstance(z, galois.Poly)
        assert z.field == GF
        assert type(z.coeffs) is GF
