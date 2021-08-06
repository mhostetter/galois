"""
A pytest module to test modular exponentiation of polynomials over Galois fields.

Sage:
    def to_str(poly):
        c = poly.coefficients(sparse=False)[::-1]
        return f"galois.Poly({c}, field=GF{characteristic}_{degree})"

    N = 20
    for (characteristic, degree) in [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)]:
        print(f"POLY_POW_{characteristic}_{degree} = {{")
        R = GF(characteristic**degree, repr="int")["x"]
        for _ in range(N):
            base = R.random_element(randint(0, 10))
            exponent = randint(0, 1000)
            modulus = R.random_element(randint(0, 10))
            result = base**exponent % modulus
            print(f"    ({to_str(base)}, {exponent}, {to_str(modulus)}): {to_str(result)},")
        print("}\n")
"""
import pytest

import galois

GF2_1 = galois.GF(2**1)
GF2_8 = galois.GF(2**8)
GF3_1 = galois.GF(3**1)
GF3_5 = galois.GF(3**5)
GF5_1 = galois.GF(5**1)
GF5_4 = galois.GF(5**4)

POLY_POW_2_1 = {
    (galois.Poly([1, 0, 1, 1, 0, 0, 1, 0, 1], field=GF2_1), 677, galois.Poly([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0], field=GF2_1)): galois.Poly([1, 0, 0, 1, 1, 1, 1, 0, 1, 1], field=GF2_1),
    (galois.Poly([1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1], field=GF2_1), 731, galois.Poly([1], field=GF2_1)): galois.Poly([], field=GF2_1),
    (galois.Poly([1, 0], field=GF2_1), 91, galois.Poly([1, 0, 1, 0, 0, 0], field=GF2_1)): galois.Poly([1, 0, 0, 0], field=GF2_1),
    (galois.Poly([1, 0, 0], field=GF2_1), 298, galois.Poly([1, 1], field=GF2_1)): galois.Poly([1], field=GF2_1),
    (galois.Poly([1, 0], field=GF2_1), 420, galois.Poly([1, 0, 0], field=GF2_1)): galois.Poly([], field=GF2_1),
    (galois.Poly([1, 0, 0, 0, 1, 1, 0, 1], field=GF2_1), 548, galois.Poly([1, 1, 0, 0], field=GF2_1)): galois.Poly([1, 0, 1], field=GF2_1),
    (galois.Poly([1, 0, 1, 0, 1, 1, 1, 0], field=GF2_1), 89, galois.Poly([1, 1, 0, 1, 0, 0, 0, 0, 0], field=GF2_1)): galois.Poly([1, 0, 0, 0, 0, 0, 0, 0], field=GF2_1),
    (galois.Poly([1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], field=GF2_1), 144, galois.Poly([1, 1, 1, 1, 1, 0, 0], field=GF2_1)): galois.Poly([1], field=GF2_1),
    (galois.Poly([1, 1, 1, 0, 1, 1, 0, 1, 0], field=GF2_1), 548, galois.Poly([1, 0], field=GF2_1)): galois.Poly([], field=GF2_1),
    (galois.Poly([1, 0, 1, 0], field=GF2_1), 964, galois.Poly([1, 0], field=GF2_1)): galois.Poly([], field=GF2_1),
    (galois.Poly([1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1], field=GF2_1), 287, galois.Poly([1, 0, 0, 0], field=GF2_1)): galois.Poly([1, 1], field=GF2_1),
    (galois.Poly([1, 0, 1, 0], field=GF2_1), 211, galois.Poly([1, 1, 0, 0, 0, 1, 1, 1, 1], field=GF2_1)): galois.Poly([1, 0, 0, 0, 1], field=GF2_1),
    (galois.Poly([1, 0, 1, 1, 0, 1], field=GF2_1), 893, galois.Poly([1, 0, 1], field=GF2_1)): galois.Poly([], field=GF2_1),
    (galois.Poly([1, 0, 1, 0, 1, 0, 1], field=GF2_1), 676, galois.Poly([1, 1, 1, 0, 1], field=GF2_1)): galois.Poly([1, 1, 1, 1], field=GF2_1),
    (galois.Poly([1], field=GF2_1), 433, galois.Poly([1, 1], field=GF2_1)): galois.Poly([1], field=GF2_1),
    (galois.Poly([1, 1, 0, 0, 1, 0, 1, 1, 1], field=GF2_1), 70, galois.Poly([1, 1, 0, 0, 0, 0], field=GF2_1)): galois.Poly([1, 0, 1], field=GF2_1),
    (galois.Poly([1, 1, 0, 0, 0, 0, 0], field=GF2_1), 107, galois.Poly([1, 1, 1, 1, 0, 1, 0, 0, 1, 0], field=GF2_1)): galois.Poly([1, 0, 1, 0, 1, 1, 0, 0], field=GF2_1),
    (galois.Poly([1, 1, 1, 1, 1, 1, 1, 0, 0], field=GF2_1), 249, galois.Poly([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], field=GF2_1)): galois.Poly([1, 0, 1, 1, 0, 0, 0, 0, 0], field=GF2_1),
    (galois.Poly([1, 0, 1, 0, 1, 0, 1], field=GF2_1), 57, galois.Poly([1, 0, 1, 0, 0, 1, 0, 1], field=GF2_1)): galois.Poly([1, 1, 1, 1, 0, 0], field=GF2_1),
    (galois.Poly([1, 0, 1, 1, 1, 0, 1, 0, 0, 1], field=GF2_1), 27, galois.Poly([1, 1, 0, 1, 0, 1, 0, 1, 0], field=GF2_1)): galois.Poly([1, 0, 1, 1, 1, 0, 1], field=GF2_1),
}

POLY_POW_2_8 = {
    (galois.Poly([87, 64, 47, 124, 201, 50, 213, 178, 155, 64], field=GF2_8), 176, galois.Poly([45, 253, 56, 8, 42, 200, 177, 215, 20], field=GF2_8)): galois.Poly([21, 60, 228, 164, 150, 250, 217, 115], field=GF2_8),
    (galois.Poly([58, 230, 244, 226, 165], field=GF2_8), 160, galois.Poly([249, 249, 132, 117, 189], field=GF2_8)): galois.Poly([11, 239, 140, 116], field=GF2_8),
    (galois.Poly([164, 149, 51], field=GF2_8), 969, galois.Poly([26, 213, 162, 67, 95, 239, 165, 215, 100, 130, 43], field=GF2_8)): galois.Poly([222, 221, 18, 21, 21, 24, 97, 170, 14, 207], field=GF2_8),
    (galois.Poly([226, 233, 76, 252, 143, 215], field=GF2_8), 758, galois.Poly([39, 0, 98, 20, 237, 201, 70, 239, 79], field=GF2_8)): galois.Poly([42, 152, 190, 149, 75, 234, 23, 150], field=GF2_8),
    (galois.Poly([142], field=GF2_8), 769, galois.Poly([117, 174, 103, 222, 8, 101, 232, 72, 61, 112, 10], field=GF2_8)): galois.Poly([216], field=GF2_8),
    (galois.Poly([50, 65, 242, 77, 206, 135, 93, 13, 154, 75], field=GF2_8), 370, galois.Poly([172, 86, 142, 233, 229, 68, 45, 27], field=GF2_8)): galois.Poly([169, 173, 226, 135, 248, 110, 24], field=GF2_8),
    (galois.Poly([250, 237, 187, 250, 46, 208, 105, 179, 136], field=GF2_8), 904, galois.Poly([62, 105, 37, 172, 87, 14, 124, 60], field=GF2_8)): galois.Poly([191, 166, 81, 228, 216, 55, 4], field=GF2_8),
    (galois.Poly([190, 253, 137, 254, 14, 77, 169], field=GF2_8), 501, galois.Poly([1, 222, 1, 149, 209, 178, 162, 104, 0, 224], field=GF2_8)): galois.Poly([144, 5, 114, 47, 106, 122, 178, 57, 201], field=GF2_8),
    (galois.Poly([31, 16, 220, 58, 189, 94, 32, 45, 27, 246], field=GF2_8), 48, galois.Poly([246], field=GF2_8)): galois.Poly([], field=GF2_8),
    (galois.Poly([152, 22, 45, 212, 204, 16, 145, 46, 37, 92, 106], field=GF2_8), 147, galois.Poly([113, 43, 244, 27, 208, 161, 19, 70, 71], field=GF2_8)): galois.Poly([101, 189, 44, 52, 100, 177, 18, 139], field=GF2_8),
    (galois.Poly([163, 62, 183, 146, 173], field=GF2_8), 500, galois.Poly([168, 149, 45, 22, 194, 223, 231, 36, 9], field=GF2_8)): galois.Poly([12, 21, 164, 225, 54, 64, 43, 18], field=GF2_8),
    (galois.Poly([244, 1, 185, 208, 29, 118, 39, 78, 164, 204], field=GF2_8), 212, galois.Poly([151, 77, 21, 167], field=GF2_8)): galois.Poly([106, 14, 53], field=GF2_8),
    (galois.Poly([23, 128, 83, 87, 102, 106, 135, 15, 233, 101, 244], field=GF2_8), 441, galois.Poly([139, 254, 169, 128, 45, 129, 110], field=GF2_8)): galois.Poly([131, 2, 240, 244, 50, 47], field=GF2_8),
    (galois.Poly([113, 178, 55, 164], field=GF2_8), 437, galois.Poly([69, 164, 211, 234, 141, 232, 200, 47, 96, 86, 128], field=GF2_8)): galois.Poly([44, 141, 55, 8, 83, 223, 152, 168, 85, 201], field=GF2_8),
    (galois.Poly([20, 51, 74, 31, 129], field=GF2_8), 191, galois.Poly([175, 240, 43, 9, 151, 134, 111, 181, 239, 199], field=GF2_8)): galois.Poly([60, 58, 5, 226, 148, 168, 16, 227, 39], field=GF2_8),
    (galois.Poly([104, 66, 244, 26, 64, 213, 220, 124, 113], field=GF2_8), 273, galois.Poly([166, 17, 68, 88, 229, 218, 120], field=GF2_8)): galois.Poly([72, 189, 85, 202, 23, 37], field=GF2_8),
    (galois.Poly([56, 131, 159, 97, 74, 81], field=GF2_8), 430, galois.Poly([140, 201, 243, 101, 145], field=GF2_8)): galois.Poly([244, 131, 75, 249], field=GF2_8),
    (galois.Poly([117, 95, 192, 184, 213, 4, 5, 172, 85, 131], field=GF2_8), 905, galois.Poly([21, 224, 195, 185, 2, 63, 236, 111, 249, 5, 113], field=GF2_8)): galois.Poly([169, 113, 111, 102, 238, 35, 191, 157, 233, 185], field=GF2_8),
    (galois.Poly([72, 229, 218, 174, 70, 50, 97, 158, 146, 161, 142], field=GF2_8), 69, galois.Poly([251, 28, 167, 254], field=GF2_8)): galois.Poly([158, 88, 101], field=GF2_8),
    (galois.Poly([176, 119, 40, 25, 242, 166, 48], field=GF2_8), 797, galois.Poly([36, 137, 29, 212, 61, 213, 88, 22, 143, 78], field=GF2_8)): galois.Poly([26, 40, 147, 136, 90, 16, 55, 91, 39], field=GF2_8),
}

POLY_POW_3_1 = {
    (galois.Poly([1, 1, 2, 0, 0, 1, 0, 1, 2], field=GF3_1), 151, galois.Poly([1, 2, 0, 2, 0, 1, 1, 0], field=GF3_1)): galois.Poly([1, 0, 2, 2, 1, 2, 2], field=GF3_1),
    (galois.Poly([2, 1, 1, 1, 0, 2, 2], field=GF3_1), 714, galois.Poly([2, 1, 0, 0, 2, 2, 1], field=GF3_1)): galois.Poly([1, 1, 2, 1, 2, 0], field=GF3_1),
    (galois.Poly([1, 2, 1, 0, 2, 1, 1, 1], field=GF3_1), 690, galois.Poly([2, 0], field=GF3_1)): galois.Poly([1], field=GF3_1),
    (galois.Poly([2, 2], field=GF3_1), 540, galois.Poly([2, 0, 2, 2], field=GF3_1)): galois.Poly([2, 2, 0], field=GF3_1),
    (galois.Poly([2, 1, 1, 2, 0, 0, 0, 2, 0, 2, 1], field=GF3_1), 660, galois.Poly([1, 0, 2, 0, 0, 0, 0, 2, 1, 0, 1], field=GF3_1)): galois.Poly([2, 0, 1, 0, 0, 2, 0, 1, 2, 2], field=GF3_1),
    (galois.Poly([1, 2, 1, 0, 0, 2, 1, 2, 0, 1, 0], field=GF3_1), 822, galois.Poly([1], field=GF3_1)): galois.Poly([], field=GF3_1),
    (galois.Poly([1, 1, 1], field=GF3_1), 322, galois.Poly([2, 1, 2, 2, 1], field=GF3_1)): galois.Poly([2, 1, 2], field=GF3_1),
    (galois.Poly([1, 2, 1, 0, 1, 2, 2], field=GF3_1), 419, galois.Poly([2, 0, 1, 2, 2, 2, 1, 2, 0, 2], field=GF3_1)): galois.Poly([1, 1, 1, 2, 2, 1, 1, 2, 2], field=GF3_1),
    (galois.Poly([1, 2, 1, 1, 1, 0, 2], field=GF3_1), 193, galois.Poly([2, 2, 1, 2, 1, 2, 0, 0, 2, 2, 0], field=GF3_1)): galois.Poly([1, 0, 2, 1, 0, 0, 0, 2, 2], field=GF3_1),
    (galois.Poly([2, 2, 0, 1, 2, 2, 2, 0], field=GF3_1), 415, galois.Poly([1, 2], field=GF3_1)): galois.Poly([2], field=GF3_1),
    (galois.Poly([2, 0], field=GF3_1), 283, galois.Poly([2], field=GF3_1)): galois.Poly([], field=GF3_1),
    (galois.Poly([1, 2, 1, 1, 2, 2, 1, 1, 1], field=GF3_1), 677, galois.Poly([1, 0, 2, 0, 0, 1, 0, 2], field=GF3_1)): galois.Poly([2, 2, 0, 0, 1, 1, 0], field=GF3_1),
    (galois.Poly([1, 2, 2], field=GF3_1), 792, galois.Poly([1, 1], field=GF3_1)): galois.Poly([1], field=GF3_1),
    (galois.Poly([2, 0, 2, 0, 2], field=GF3_1), 966, galois.Poly([2, 2, 2], field=GF3_1)): galois.Poly([], field=GF3_1),
    (galois.Poly([1, 2, 2, 2, 1, 2, 2, 0, 2], field=GF3_1), 125, galois.Poly([2, 0, 2, 2, 2], field=GF3_1)): galois.Poly([2, 2, 2, 0], field=GF3_1),
    (galois.Poly([2, 1, 0, 1, 1, 0, 0, 0, 1], field=GF3_1), 639, galois.Poly([1, 1, 0, 2], field=GF3_1)): galois.Poly([1, 0, 2], field=GF3_1),
    (galois.Poly([2, 0, 0, 0, 0, 2], field=GF3_1), 7, galois.Poly([2, 1, 2, 0, 1], field=GF3_1)): galois.Poly([2, 1, 2, 2], field=GF3_1),
    (galois.Poly([2, 1, 2, 1, 1], field=GF3_1), 785, galois.Poly([2, 1, 1, 2, 2], field=GF3_1)): galois.Poly([1, 2, 1, 2], field=GF3_1),
    (galois.Poly([2, 0, 1, 2, 2, 0, 0, 0, 0, 2], field=GF3_1), 192, galois.Poly([2, 0, 2, 0, 1, 0, 2, 1, 1, 2, 2], field=GF3_1)): galois.Poly([2, 2, 1, 1, 0, 1, 1, 0, 0, 1], field=GF3_1),
    (galois.Poly([1, 2, 0, 1, 2, 0, 2, 2, 2, 0], field=GF3_1), 480, galois.Poly([2, 0, 0, 2, 1, 2, 2, 0, 2, 2, 2], field=GF3_1)): galois.Poly([1, 0, 2, 1, 2, 0, 1, 2, 1, 2], field=GF3_1),
}

POLY_POW_3_5 = {
    (galois.Poly([119, 216, 233, 150, 184, 91, 87], field=GF3_5), 517, galois.Poly([102, 49, 134, 180, 75, 126, 232, 120, 205, 23], field=GF3_5)): galois.Poly([98, 212, 70, 144, 174, 219, 69, 10, 125], field=GF3_5),
    (galois.Poly([97, 131, 124, 94, 30, 209, 42], field=GF3_5), 753, galois.Poly([216, 132, 234, 19, 174, 151, 37, 78, 226], field=GF3_5)): galois.Poly([89, 142, 34, 55, 239, 69, 206, 149], field=GF3_5),
    (galois.Poly([219, 167, 86, 213, 150, 165, 70, 95, 8, 51, 11], field=GF3_5), 755, galois.Poly([195, 159, 112, 234, 150, 227], field=GF3_5)): galois.Poly([225, 77, 21, 237, 146], field=GF3_5),
    (galois.Poly([171, 73], field=GF3_5), 113, galois.Poly([98, 192, 182], field=GF3_5)): galois.Poly([63, 6], field=GF3_5),
    (galois.Poly([45, 177, 127, 145], field=GF3_5), 288, galois.Poly([147, 182, 107, 209, 2, 44, 134, 202, 236, 86], field=GF3_5)): galois.Poly([10, 238, 217, 186, 112, 124, 190, 63, 146], field=GF3_5),
    (galois.Poly([113, 163, 69, 2, 216, 208, 35, 45, 94], field=GF3_5), 600, galois.Poly([153, 100, 6, 118, 107], field=GF3_5)): galois.Poly([23, 216, 26, 122], field=GF3_5),
    (galois.Poly([75, 183, 122], field=GF3_5), 986, galois.Poly([35], field=GF3_5)): galois.Poly([], field=GF3_5),
    (galois.Poly([35, 49, 51, 159, 229], field=GF3_5), 233, galois.Poly([183], field=GF3_5)): galois.Poly([], field=GF3_5),
    (galois.Poly([11, 214, 139, 12, 117, 239, 203, 94, 13], field=GF3_5), 353, galois.Poly([16, 178, 110, 47, 189, 38, 96, 140, 85, 187, 155], field=GF3_5)): galois.Poly([80, 2, 130, 101, 225, 140, 209, 225, 142, 194], field=GF3_5),
    (galois.Poly([35, 207], field=GF3_5), 44, galois.Poly([199, 35, 229, 44, 235, 103], field=GF3_5)): galois.Poly([67, 80, 185, 207, 63], field=GF3_5),
    (galois.Poly([11], field=GF3_5), 923, galois.Poly([52, 197, 17, 32, 198, 229, 119, 42], field=GF3_5)): galois.Poly([209], field=GF3_5),
    (galois.Poly([173, 180, 108, 57, 135, 7, 27, 136, 71, 33, 70], field=GF3_5), 47, galois.Poly([115, 241, 19, 73, 171, 140, 182, 214, 20, 124], field=GF3_5)): galois.Poly([64, 172, 81, 194, 231, 121, 77, 23, 0], field=GF3_5),
    (galois.Poly([48, 233, 112, 6, 10, 80, 135, 119, 219, 82], field=GF3_5), 985, galois.Poly([31, 193, 147, 63, 45, 232, 54, 156, 47, 27, 46], field=GF3_5)): galois.Poly([216, 200, 161, 170, 196, 67, 180, 140, 43, 3], field=GF3_5),
    (galois.Poly([175, 147, 75, 201, 215, 20, 120, 96, 18, 133], field=GF3_5), 536, galois.Poly([107, 109], field=GF3_5)): galois.Poly([81], field=GF3_5),
    (galois.Poly([27, 229, 51, 197], field=GF3_5), 20, galois.Poly([185, 236, 78, 95, 92, 89], field=GF3_5)): galois.Poly([211, 213, 17, 107, 234], field=GF3_5),
    (galois.Poly([94, 128, 67, 224], field=GF3_5), 638, galois.Poly([6, 40], field=GF3_5)): galois.Poly([157], field=GF3_5),
    (galois.Poly([112, 197, 123, 200, 115, 218, 114, 198, 145, 140, 77], field=GF3_5), 541, galois.Poly([127, 7, 45, 94, 55], field=GF3_5)): galois.Poly([89, 60, 1, 132], field=GF3_5),
    (galois.Poly([49, 85, 225, 156], field=GF3_5), 970, galois.Poly([135, 92, 35, 97, 129, 159, 29, 8, 168], field=GF3_5)): galois.Poly([50, 77, 161, 163, 174, 100, 89, 184], field=GF3_5),
    (galois.Poly([142, 130, 33, 194, 60, 110, 185, 202, 9], field=GF3_5), 17, galois.Poly([193, 30, 210, 220, 125, 51, 68, 168, 36], field=GF3_5)): galois.Poly([116, 6, 39, 180, 218, 116, 0, 78], field=GF3_5),
    (galois.Poly([225, 26, 143, 74, 140, 182, 231, 129, 0, 61], field=GF3_5), 97, galois.Poly([63, 107, 100, 143, 48, 234, 199, 100, 241, 126, 21], field=GF3_5)): galois.Poly([60, 67, 228, 142, 89, 208, 38, 187, 89, 196], field=GF3_5),
}

POLY_POW_5_1 = {
    (galois.Poly([4, 0, 1, 0, 1, 4, 1, 4, 3, 2], field=GF5_1), 27, galois.Poly([4, 1], field=GF5_1)): galois.Poly([], field=GF5_1),
    (galois.Poly([4, 2, 4, 0, 1, 4, 4, 0, 4, 4], field=GF5_1), 826, galois.Poly([4, 4, 2, 4, 1, 0, 1, 4, 0], field=GF5_1)): galois.Poly([3, 4, 3, 4, 1, 2, 1, 1], field=GF5_1),
    (galois.Poly([4, 0, 4, 0, 4, 2, 1], field=GF5_1), 657, galois.Poly([1, 1, 3, 3, 4], field=GF5_1)): galois.Poly([1, 0, 1, 3], field=GF5_1),
    (galois.Poly([4, 2, 4, 2, 2, 4, 0], field=GF5_1), 487, galois.Poly([4, 1, 1, 4, 4, 0, 0, 2, 1, 3], field=GF5_1)): galois.Poly([1, 1, 0, 2, 1, 1, 1, 3, 2], field=GF5_1),
    (galois.Poly([3, 3, 4, 4, 2, 2], field=GF5_1), 13, galois.Poly([3, 3, 2], field=GF5_1)): galois.Poly([1, 1], field=GF5_1),
    (galois.Poly([3, 4], field=GF5_1), 448, galois.Poly([4, 1, 4, 4, 1, 1, 1], field=GF5_1)): galois.Poly([1, 2, 3, 1, 1, 3], field=GF5_1),
    (galois.Poly([3, 0, 0, 2, 1, 3, 1], field=GF5_1), 54, galois.Poly([1, 0, 0, 4, 1, 2], field=GF5_1)): galois.Poly([1, 0, 4, 0, 4], field=GF5_1),
    (galois.Poly([2, 3, 1, 1, 4, 3, 2], field=GF5_1), 216, galois.Poly([3, 3], field=GF5_1)): galois.Poly([1], field=GF5_1),
    (galois.Poly([4], field=GF5_1), 771, galois.Poly([3, 1, 3, 1, 3, 4, 0, 1, 2, 4], field=GF5_1)): galois.Poly([4], field=GF5_1),
    (galois.Poly([2, 3, 1, 4, 3, 3, 2, 4, 1, 0, 1], field=GF5_1), 659, galois.Poly([3, 4, 0, 1, 0, 0, 0, 2], field=GF5_1)): galois.Poly([1, 2, 2, 4, 2, 1, 2], field=GF5_1),
    (galois.Poly([2, 4, 2, 2, 4, 0, 3, 4], field=GF5_1), 234, galois.Poly([2, 1, 4], field=GF5_1)): galois.Poly([1], field=GF5_1),
    (galois.Poly([1, 1, 0, 0, 3, 2, 2, 2, 1, 2], field=GF5_1), 842, galois.Poly([2, 4, 1, 3, 1, 1, 2, 0, 3, 3], field=GF5_1)): galois.Poly([4, 1, 0, 2, 4, 1, 3, 1, 0], field=GF5_1),
    (galois.Poly([3, 2], field=GF5_1), 227, galois.Poly([2, 1, 4, 2, 1, 3, 2, 0], field=GF5_1)): galois.Poly([1, 1, 4, 3, 3, 0, 3], field=GF5_1),
    (galois.Poly([2, 2, 1, 0, 1, 0, 1], field=GF5_1), 698, galois.Poly([3, 0, 3, 3, 3, 1, 4, 4, 3, 0], field=GF5_1)): galois.Poly([3, 3, 1, 2, 2, 0, 1, 3, 1], field=GF5_1),
    (galois.Poly([4, 0], field=GF5_1), 813, galois.Poly([4, 4, 1, 4, 3, 2, 1, 4, 1, 4], field=GF5_1)): galois.Poly([1, 1, 0, 1, 2, 1, 4, 0], field=GF5_1),
    (galois.Poly([2, 1, 4, 1, 4], field=GF5_1), 538, galois.Poly([2, 1, 3, 0, 1, 1, 1, 3, 2, 4, 4], field=GF5_1)): galois.Poly([3, 3, 1, 0, 2, 2, 2, 3, 1, 4], field=GF5_1),
    (galois.Poly([3, 4, 0, 3, 0, 0, 2, 2, 4], field=GF5_1), 965, galois.Poly([3, 1, 4, 1, 2, 4, 4, 4], field=GF5_1)): galois.Poly([1, 3, 2, 4, 2, 1, 4], field=GF5_1),
    (galois.Poly([2, 1, 4, 3, 2], field=GF5_1), 791, galois.Poly([3, 0, 1], field=GF5_1)): galois.Poly([4, 2], field=GF5_1),
    (galois.Poly([3, 0], field=GF5_1), 945, galois.Poly([4, 3, 3, 0, 0, 2, 1, 4, 4], field=GF5_1)): galois.Poly([4, 1, 3, 2, 2], field=GF5_1),
    (galois.Poly([2, 0, 0, 4, 4], field=GF5_1), 463, galois.Poly([2, 4], field=GF5_1)): galois.Poly([2], field=GF5_1),
}

POLY_POW_5_4 = {
    (galois.Poly([387, 255, 317, 534, 37], field=GF5_4), 94, galois.Poly([560, 212, 615, 271, 360], field=GF5_4)): galois.Poly([203, 330, 83, 119], field=GF5_4),
    (galois.Poly([177, 69, 30, 610, 516], field=GF5_4), 415, galois.Poly([519], field=GF5_4)): galois.Poly([], field=GF5_4),
    (galois.Poly([61], field=GF5_4), 763, galois.Poly([605, 251, 528, 491, 102, 279, 406, 132, 50, 556, 405], field=GF5_4)): galois.Poly([7], field=GF5_4),
    (galois.Poly([544, 610, 142, 487, 4], field=GF5_4), 862, galois.Poly([174, 508, 20, 395, 614, 164, 377, 437, 146, 46, 33], field=GF5_4)): galois.Poly([35, 30, 485, 294, 360, 619, 361, 538, 439, 590], field=GF5_4),
    (galois.Poly([388, 241, 203, 371, 42, 243, 39, 404, 520, 87, 83], field=GF5_4), 226, galois.Poly([180, 384, 169, 369, 491], field=GF5_4)): galois.Poly([211, 139, 598, 209], field=GF5_4),
    (galois.Poly([591, 4, 486, 578, 581], field=GF5_4), 879, galois.Poly([379], field=GF5_4)): galois.Poly([], field=GF5_4),
    (galois.Poly([327, 5, 489, 177, 15], field=GF5_4), 325, galois.Poly([563, 430], field=GF5_4)): galois.Poly([622], field=GF5_4),
    (galois.Poly([218, 87], field=GF5_4), 885, galois.Poly([416, 526, 249, 39, 181, 79, 351, 358, 490, 248], field=GF5_4)): galois.Poly([318, 231, 499, 97, 114, 173, 291, 260, 58], field=GF5_4),
    (galois.Poly([176, 202, 335, 425, 385, 218, 612, 411, 119, 621, 16], field=GF5_4), 13, galois.Poly([602, 570, 30, 425, 416, 180, 4], field=GF5_4)): galois.Poly([291, 154, 518, 381, 199, 16], field=GF5_4),
    (galois.Poly([510, 294, 172, 453, 72, 150], field=GF5_4), 458, galois.Poly([483, 437, 378, 118, 490, 422, 267, 132], field=GF5_4)): galois.Poly([183, 264, 310, 372, 262, 490, 452], field=GF5_4),
    (galois.Poly([159], field=GF5_4), 852, galois.Poly([320, 172, 431, 558, 571, 324, 86, 69, 477, 49, 369], field=GF5_4)): galois.Poly([1], field=GF5_4),
    (galois.Poly([124, 434, 186, 465, 526, 324, 216, 26, 268, 525], field=GF5_4), 918, galois.Poly([541, 552, 568, 275, 358, 509, 295, 547], field=GF5_4)): galois.Poly([600, 295, 254, 309, 585, 257, 576], field=GF5_4),
    (galois.Poly([123, 554, 133, 497, 72, 173, 560, 42, 239, 557, 337], field=GF5_4), 112, galois.Poly([475, 66, 411, 32, 546, 117, 125, 572, 415, 54], field=GF5_4)): galois.Poly([162, 153, 530, 149, 84, 136, 220, 136, 42], field=GF5_4),
    (galois.Poly([359, 220], field=GF5_4), 581, galois.Poly([589, 573, 287, 147, 583, 335, 399, 513], field=GF5_4)): galois.Poly([621, 185, 116, 381, 62, 63, 353], field=GF5_4),
    (galois.Poly([80, 298, 329, 19], field=GF5_4), 336, galois.Poly([404, 450, 522, 178, 576, 589, 5], field=GF5_4)): galois.Poly([48, 88, 457, 300, 550, 393], field=GF5_4),
    (galois.Poly([544, 608, 169, 336, 464, 488, 545, 383, 455], field=GF5_4), 638, galois.Poly([231, 611, 612, 459], field=GF5_4)): galois.Poly([395, 221, 339], field=GF5_4),
    (galois.Poly([32, 550, 440, 606, 192, 554, 564, 273, 459, 438], field=GF5_4), 152, galois.Poly([189, 128], field=GF5_4)): galois.Poly([368], field=GF5_4),
    (galois.Poly([69, 341, 99], field=GF5_4), 927, galois.Poly([493, 364, 598, 34, 224, 313, 370, 390, 100, 363, 481], field=GF5_4)): galois.Poly([118, 107, 8, 508, 596, 160, 428, 380, 449, 609], field=GF5_4),
    (galois.Poly([32, 153, 353, 412, 565, 134, 492], field=GF5_4), 348, galois.Poly([156], field=GF5_4)): galois.Poly([], field=GF5_4),
    (galois.Poly([67, 284, 29], field=GF5_4), 296, galois.Poly([114, 195, 267, 120], field=GF5_4)): galois.Poly([27, 482, 364], field=GF5_4),
}


def test_poly_pow_exceptions():
    GF = galois.GF(31)
    f = galois.Poly.Random(10, field=GF)
    g = galois.Poly.Random(7, field=GF)
    power = 20

    with pytest.raises(TypeError):
        galois.poly_pow(f.coeffs, power, g)
    with pytest.raises(TypeError):
        galois.poly_pow(f, float(power), g)
    with pytest.raises(TypeError):
        galois.poly_pow(f, power, g.coeffs)
    with pytest.raises(ValueError):
        galois.poly_pow(f, -power, g)


@pytest.mark.parametrize("characteristic,degree", [(2,1), (2,8), (3,1), (3,5), (5,1), (5,4)])
def test_poly_pow(characteristic, degree):
    LUT = eval(f"POLY_POW_{characteristic}_{degree}")
    for key in LUT:
        base, exponent, modulus = key
        assert galois.poly_pow(base, exponent, modulus) == LUT[key]
