"""
A pytest module to test the square roots in finite fields.

Sage:
    F = GF(7)
    x = []
    r = []
    for xi in range(0, F.order()):
        xi = F(xi)
        sqrt_x = sqrt(xi)
        if type(sqrt_x) == type(xi):
            x.append(xi)
            if -sqrt_x < sqrt_x:
                sqrt_x = -sqrt_x
            r.append(sqrt_x)

    print(x)
    print(r)

Sage:
    F = GF(3**3, repr="int")
    x = []
    r = []
    for xi in range(0, F.order()):
        xi = F.fetch_int(xi)
        try:
            sqrt_x = sqrt(xi)
            x.append(xi)
            if -sqrt_x < sqrt_x:
                sqrt_x = -sqrt_x
            r.append(sqrt_x)
        except:
            pass

    print(x)
    print(r)
"""

import numpy as np

import galois


def test_binary_field():
    GF = galois.GF(2)
    x = GF([0, 1])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1])
    assert isinstance(y, GF)


def test_prime_field_1():
    # p % 4 = 3, p % 8 != 5
    GF = galois.GF(7)
    x = GF([0, 1, 2, 4])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 3, 2])
    assert isinstance(y, GF)


def test_prime_field_2():
    # p % 4 != 3, p % 8 = 5
    GF = galois.GF(13)
    x = GF([0, 1, 3, 4, 9, 10, 12])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 4, 2, 3, 6, 5])
    assert isinstance(y, GF)


def test_prime_field_3():
    # p % 4 != 3, p % 8 != 5
    GF = galois.GF(17)
    x = GF([0, 1, 2, 4, 8, 9, 13, 15, 16])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 6, 2, 5, 3, 8, 7, 4])
    assert isinstance(y, GF)


def test_binary_extension_field_1():
    GF = galois.GF(2**3)
    x = GF([0, 1, 2, 3, 4, 5, 6, 7])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 6, 7, 2, 3, 4, 5])
    assert isinstance(y, GF)


def test_binary_extension_field_2():
    GF = galois.GF(2**4)
    x = GF([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 5, 4, 2, 3, 7, 6, 10, 11, 15, 14, 8, 9, 13, 12])
    assert isinstance(y, GF)


def test_binary_extension_field_3():
    GF = galois.GF(2**5)
    x = GF([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])  # fmt: skip
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 27, 26, 2, 3, 25, 24, 19, 18, 8, 9, 17, 16, 10, 11, 4, 5, 31, 30, 6, 7, 29, 28, 23, 22, 12, 13, 21, 20, 14, 15])  # fmt: skip
    assert isinstance(y, GF)


def test_prime_extension_field_1():
    # p % 4 = 3, p % 8 != 5
    GF = galois.GF(3**3)
    x = GF([0, 1, 6, 7, 8, 9, 11, 12, 13, 15, 16, 20, 22, 25])
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 17, 10, 14, 3, 13, 16, 5, 9, 4, 15, 12, 11])
    assert isinstance(y, GF)


def test_prime_extension_field_2():
    # p % 4 != 3, p % 8 = 5
    GF = galois.GF(5**3)
    x = GF([0, 1, 4, 6, 7, 10, 11, 13, 15, 17, 19, 23, 24, 25, 27, 28, 34, 36, 37, 39, 40, 41, 42, 44, 45, 46, 47, 49, 52, 53, 56, 60, 61, 62, 64, 66, 70, 71, 77, 78, 80, 84, 89, 90, 91, 93, 94, 99, 100, 102, 103, 105, 106, 108, 109, 110, 111, 113, 114, 116, 118, 119, 121])  # fmt: skip
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 2, 42, 46, 62, 29, 74, 31, 37, 53, 67, 59, 5, 49, 35, 8, 6, 58, 27, 66, 9, 61, 52, 34, 73, 40, 7, 57, 32, 72, 25, 43, 38, 65, 51, 47, 60, 64, 41, 69, 30, 28, 50, 45, 71, 56, 36, 10, 70, 68, 63, 14, 55, 39, 48, 26, 33, 13, 54, 44, 12, 11])  # fmt: skip
    assert isinstance(y, GF)


def test_prime_extension_field_3():
    # p % 4 != 3, p % 8 != 5
    GF = galois.GF(17**2)
    x = GF([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 20, 22, 24, 26, 28, 30, 31, 35, 38, 39, 40, 43, 44, 45, 48, 55, 56, 57, 59, 60, 61, 66, 67, 69, 70, 71, 73, 76, 78, 79, 80, 86, 87, 89, 93, 95, 96, 99, 100, 103, 105, 110, 112, 114, 115, 117, 118, 120, 123, 125, 128, 131, 132, 133, 134, 138, 139, 140, 141, 142, 143, 146, 152, 154, 160, 163, 164, 165, 166, 167, 168, 172, 173, 174, 175, 178, 181, 183, 186, 188, 189, 191, 192, 194, 196, 201, 203, 206, 207, 210, 211, 213, 217, 219, 220, 226, 227, 228, 230, 233, 235, 236, 237, 239, 240, 245, 246, 247, 249, 250, 251, 258, 261, 262, 263, 266, 267, 268, 271, 275, 276, 278, 280, 282, 284, 286, 287])  # fmt: skip
    y = np.sqrt(x)
    assert np.array_equal(y, [0, 1, 6, 116, 2, 58, 50, 83, 5, 3, 25, 149, 91, 8, 124, 7, 4, 81, 148, 61, 135, 109, 86, 46, 17, 42, 129, 79, 64, 98, 26, 102, 147, 38, 123, 93, 67, 112, 146, 18, 77, 75, 88, 105, 34, 145, 134, 53, 27, 19, 128, 47, 144, 115, 73, 100, 56, 43, 59, 95, 143, 71, 108, 122, 28, 90, 69, 20, 62, 118, 39, 142, 133, 127, 141, 84, 29, 35, 111, 85, 65, 97, 51, 121, 140, 82, 21, 48, 104, 114, 44, 139, 132, 92, 80, 30, 54, 78, 107, 126, 22, 40, 57, 87, 138, 99, 60, 31, 120, 36, 137, 117, 76, 74, 94, 110, 49, 136, 131, 63, 23, 32, 72, 45, 125, 89, 66, 103, 152, 41, 119, 70, 52, 101, 24, 113, 151, 68, 150, 55, 130, 106, 96, 37, 33])  # fmt: skip
    assert isinstance(y, GF)
