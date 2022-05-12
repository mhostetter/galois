"""
A pytest module to test generating primitive elements.

Sage:
    p = 5
    for degree in range(2, 5):
        F.<x> = GF(p**degree, repr="int")
        print(f"PRIMITIVE_ELEMENTS_{p}_{degree} = [")
        for e in range(1, F.order()):
            e = F.fetch_int(e)
            if e.multiplicative_order() == F.order() - 1:
                c = e.polynomial().coefficients(sparse=False)[::-1]
                print(f"    galois.Poly({c}, field=GF{p}),")
        print("]\n")
"""
import random

import pytest

import galois

GF2 = galois.GF(2)
GF3 = galois.GF(3)
GF5 = galois.GF(5)

PRIMITIVE_ELEMENTS_2_2 = [
    galois.Poly([1, 0], field=GF2),
    galois.Poly([1, 1], field=GF2),
]

PRIMITIVE_ELEMENTS_2_3 = [
    galois.Poly([1, 0], field=GF2),
    galois.Poly([1, 1], field=GF2),
    galois.Poly([1, 0, 0], field=GF2),
    galois.Poly([1, 0, 1], field=GF2),
    galois.Poly([1, 1, 0], field=GF2),
    galois.Poly([1, 1, 1], field=GF2),
]

PRIMITIVE_ELEMENTS_2_4 = [
    galois.Poly([1, 0], field=GF2),
    galois.Poly([1, 1], field=GF2),
    galois.Poly([1, 0, 0], field=GF2),
    galois.Poly([1, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0], field=GF2),
]

PRIMITIVE_ELEMENTS_2_5 = [
    galois.Poly([1, 0], field=GF2),
    galois.Poly([1, 1], field=GF2),
    galois.Poly([1, 0, 0], field=GF2),
    galois.Poly([1, 0, 1], field=GF2),
    galois.Poly([1, 1, 0], field=GF2),
    galois.Poly([1, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 0], field=GF2),
    galois.Poly([1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0], field=GF2),
    galois.Poly([1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0], field=GF2),
    galois.Poly([1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0], field=GF2),
    galois.Poly([1, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 0, 0], field=GF2),
    galois.Poly([1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 0], field=GF2),
    galois.Poly([1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 0], field=GF2),
    galois.Poly([1, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 0], field=GF2),
    galois.Poly([1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0], field=GF2),
    galois.Poly([1, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0], field=GF2),
    galois.Poly([1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0], field=GF2),
    galois.Poly([1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0], field=GF2),
    galois.Poly([1, 1, 1, 1, 1], field=GF2),
]

PRIMITIVE_ELEMENTS_2_6 = [
    galois.Poly([1, 0], field=GF2),
    galois.Poly([1, 0, 0], field=GF2),
    galois.Poly([1, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 0], field=GF2),
    galois.Poly([1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 0, 0], field=GF2),
    galois.Poly([1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 0, 0], field=GF2),
    galois.Poly([1, 0, 0, 1, 1], field=GF2),
    galois.Poly([1, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 0], field=GF2),
    galois.Poly([1, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 1, 0], field=GF2),
    galois.Poly([1, 1, 1, 0, 0], field=GF2),
    galois.Poly([1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0], field=GF2),
    galois.Poly([1, 1, 1, 1, 1], field=GF2),
    galois.Poly([1, 0, 0, 0, 0, 0], field=GF2),
    galois.Poly([1, 0, 0, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 0, 1, 1, 0], field=GF2),
    galois.Poly([1, 0, 1, 0, 0, 0], field=GF2),
    galois.Poly([1, 0, 1, 0, 1, 0], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 0], field=GF2),
    galois.Poly([1, 0, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 0, 1, 1, 1, 0], field=GF2),
    galois.Poly([1, 1, 0, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 0, 0, 1, 0], field=GF2),
    galois.Poly([1, 1, 0, 1, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 0], field=GF2),
    galois.Poly([1, 1, 1, 0, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 0], field=GF2),
    galois.Poly([1, 1, 1, 0, 1, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 0], field=GF2),
    galois.Poly([1, 1, 1, 1, 0, 1], field=GF2),
    galois.Poly([1, 1, 1, 1, 1, 0], field=GF2),
]

PRIMITIVE_ELEMENTS_3_2 = [
    galois.Poly([1, 0], field=GF3),
    galois.Poly([1, 2], field=GF3),
    galois.Poly([2, 0], field=GF3),
    galois.Poly([2, 1], field=GF3),
]

PRIMITIVE_ELEMENTS_3_3 = [
    galois.Poly([1, 0], field=GF3),
    galois.Poly([1, 1], field=GF3),
    galois.Poly([1, 2], field=GF3),
    galois.Poly([1, 0, 1], field=GF3),
    galois.Poly([1, 1, 2], field=GF3),
    galois.Poly([1, 2, 2], field=GF3),
    galois.Poly([2, 0, 0], field=GF3),
    galois.Poly([2, 0, 1], field=GF3),
    galois.Poly([2, 1, 0], field=GF3),
    galois.Poly([2, 1, 2], field=GF3),
    galois.Poly([2, 2, 0], field=GF3),
    galois.Poly([2, 2, 2], field=GF3),
]

PRIMITIVE_ELEMENTS_3_4 = [
    galois.Poly([1, 0], field=GF3),
    galois.Poly([1, 2], field=GF3),
    galois.Poly([2, 0], field=GF3),
    galois.Poly([2, 1], field=GF3),
    galois.Poly([1, 1, 0], field=GF3),
    galois.Poly([1, 2, 2], field=GF3),
    galois.Poly([2, 1, 1], field=GF3),
    galois.Poly([2, 2, 0], field=GF3),
    galois.Poly([1, 0, 0, 0], field=GF3),
    galois.Poly([1, 0, 0, 2], field=GF3),
    galois.Poly([1, 0, 1, 0], field=GF3),
    galois.Poly([1, 0, 1, 2], field=GF3),
    galois.Poly([1, 0, 2, 2], field=GF3),
    galois.Poly([1, 1, 2, 0], field=GF3),
    galois.Poly([1, 1, 2, 2], field=GF3),
    galois.Poly([1, 2, 0, 0], field=GF3),
    galois.Poly([1, 2, 0, 1], field=GF3),
    galois.Poly([1, 2, 1, 0], field=GF3),
    galois.Poly([1, 2, 1, 1], field=GF3),
    galois.Poly([1, 2, 2, 2], field=GF3),
    galois.Poly([2, 0, 0, 0], field=GF3),
    galois.Poly([2, 0, 0, 1], field=GF3),
    galois.Poly([2, 0, 1, 1], field=GF3),
    galois.Poly([2, 0, 2, 0], field=GF3),
    galois.Poly([2, 0, 2, 1], field=GF3),
    galois.Poly([2, 1, 0, 0], field=GF3),
    galois.Poly([2, 1, 0, 2], field=GF3),
    galois.Poly([2, 1, 1, 1], field=GF3),
    galois.Poly([2, 1, 2, 0], field=GF3),
    galois.Poly([2, 1, 2, 2], field=GF3),
    galois.Poly([2, 2, 1, 0], field=GF3),
    galois.Poly([2, 2, 1, 1], field=GF3),
]

PRIMITIVE_ELEMENTS_5_2 = [
    galois.Poly([1, 0], field=GF5),
    galois.Poly([1, 4], field=GF5),
    galois.Poly([2, 0], field=GF5),
    galois.Poly([2, 3], field=GF5),
    galois.Poly([3, 0], field=GF5),
    galois.Poly([3, 2], field=GF5),
    galois.Poly([4, 0], field=GF5),
    galois.Poly([4, 1], field=GF5),
]

PRIMITIVE_ELEMENTS_5_3 = [
    galois.Poly([1, 0], field=GF5),
    galois.Poly([1, 3], field=GF5),
    galois.Poly([1, 4], field=GF5),
    galois.Poly([2, 2], field=GF5),
    galois.Poly([2, 4], field=GF5),
    galois.Poly([3, 1], field=GF5),
    galois.Poly([3, 3], field=GF5),
    galois.Poly([4, 0], field=GF5),
    galois.Poly([4, 1], field=GF5),
    galois.Poly([4, 2], field=GF5),
    galois.Poly([1, 0, 1], field=GF5),
    galois.Poly([1, 0, 4], field=GF5),
    galois.Poly([1, 1, 0], field=GF5),
    galois.Poly([1, 1, 1], field=GF5),
    galois.Poly([1, 1, 2], field=GF5),
    galois.Poly([1, 1, 3], field=GF5),
    galois.Poly([1, 2, 0], field=GF5),
    galois.Poly([1, 2, 3], field=GF5),
    galois.Poly([1, 3, 3], field=GF5),
    galois.Poly([1, 4, 3], field=GF5),
    galois.Poly([2, 0, 0], field=GF5),
    galois.Poly([2, 0, 1], field=GF5),
    galois.Poly([2, 0, 4], field=GF5),
    galois.Poly([2, 1, 0], field=GF5),
    galois.Poly([2, 1, 2], field=GF5),
    galois.Poly([2, 1, 3], field=GF5),
    galois.Poly([2, 1, 4], field=GF5),
    galois.Poly([2, 2, 3], field=GF5),
    galois.Poly([2, 3, 0], field=GF5),
    galois.Poly([2, 3, 2], field=GF5),
    galois.Poly([2, 3, 3], field=GF5),
    galois.Poly([2, 3, 4], field=GF5),
    galois.Poly([2, 4, 2], field=GF5),
    galois.Poly([2, 4, 3], field=GF5),
    galois.Poly([2, 4, 4], field=GF5),
    galois.Poly([3, 0, 0], field=GF5),
    galois.Poly([3, 0, 1], field=GF5),
    galois.Poly([3, 0, 4], field=GF5),
    galois.Poly([3, 1, 1], field=GF5),
    galois.Poly([3, 1, 2], field=GF5),
    galois.Poly([3, 1, 3], field=GF5),
    galois.Poly([3, 2, 0], field=GF5),
    galois.Poly([3, 2, 1], field=GF5),
    galois.Poly([3, 2, 2], field=GF5),
    galois.Poly([3, 2, 3], field=GF5),
    galois.Poly([3, 3, 2], field=GF5),
    galois.Poly([3, 4, 0], field=GF5),
    galois.Poly([3, 4, 1], field=GF5),
    galois.Poly([3, 4, 2], field=GF5),
    galois.Poly([3, 4, 3], field=GF5),
    galois.Poly([4, 0, 1], field=GF5),
    galois.Poly([4, 0, 4], field=GF5),
    galois.Poly([4, 1, 2], field=GF5),
    galois.Poly([4, 2, 2], field=GF5),
    galois.Poly([4, 3, 0], field=GF5),
    galois.Poly([4, 3, 2], field=GF5),
    galois.Poly([4, 4, 0], field=GF5),
    galois.Poly([4, 4, 2], field=GF5),
    galois.Poly([4, 4, 3], field=GF5),
    galois.Poly([4, 4, 4], field=GF5),
]

PRIMITIVE_ELEMENTS_5_4 = [
    galois.Poly([1, 0], field=GF5),
    galois.Poly([1, 1], field=GF5),
    galois.Poly([1, 3], field=GF5),
    galois.Poly([2, 0], field=GF5),
    galois.Poly([2, 1], field=GF5),
    galois.Poly([2, 2], field=GF5),
    galois.Poly([3, 0], field=GF5),
    galois.Poly([3, 3], field=GF5),
    galois.Poly([3, 4], field=GF5),
    galois.Poly([4, 0], field=GF5),
    galois.Poly([4, 2], field=GF5),
    galois.Poly([4, 4], field=GF5),
    galois.Poly([1, 0, 1], field=GF5),
    galois.Poly([1, 0, 4], field=GF5),
    galois.Poly([1, 1, 1], field=GF5),
    galois.Poly([1, 2, 0], field=GF5),
    galois.Poly([1, 2, 4], field=GF5),
    galois.Poly([1, 3, 3], field=GF5),
    galois.Poly([1, 3, 4], field=GF5),
    galois.Poly([1, 4, 0], field=GF5),
    galois.Poly([2, 0, 2], field=GF5),
    galois.Poly([2, 0, 3], field=GF5),
    galois.Poly([2, 1, 1], field=GF5),
    galois.Poly([2, 1, 3], field=GF5),
    galois.Poly([2, 2, 2], field=GF5),
    galois.Poly([2, 3, 0], field=GF5),
    galois.Poly([2, 4, 0], field=GF5),
    galois.Poly([2, 4, 3], field=GF5),
    galois.Poly([3, 0, 2], field=GF5),
    galois.Poly([3, 0, 3], field=GF5),
    galois.Poly([3, 1, 0], field=GF5),
    galois.Poly([3, 1, 2], field=GF5),
    galois.Poly([3, 2, 0], field=GF5),
    galois.Poly([3, 3, 3], field=GF5),
    galois.Poly([3, 4, 2], field=GF5),
    galois.Poly([3, 4, 4], field=GF5),
    galois.Poly([4, 0, 1], field=GF5),
    galois.Poly([4, 0, 4], field=GF5),
    galois.Poly([4, 1, 0], field=GF5),
    galois.Poly([4, 2, 1], field=GF5),
    galois.Poly([4, 2, 2], field=GF5),
    galois.Poly([4, 3, 0], field=GF5),
    galois.Poly([4, 3, 1], field=GF5),
    galois.Poly([4, 4, 4], field=GF5),
    galois.Poly([1, 0, 0, 3], field=GF5),
    galois.Poly([1, 0, 0, 4], field=GF5),
    galois.Poly([1, 0, 1, 1], field=GF5),
    galois.Poly([1, 0, 1, 2], field=GF5),
    galois.Poly([1, 0, 3, 0], field=GF5),
    galois.Poly([1, 0, 3, 4], field=GF5),
    galois.Poly([1, 0, 4, 1], field=GF5),
    galois.Poly([1, 0, 4, 4], field=GF5),
    galois.Poly([1, 1, 0, 0], field=GF5),
    galois.Poly([1, 1, 0, 1], field=GF5),
    galois.Poly([1, 1, 0, 3], field=GF5),
    galois.Poly([1, 1, 2, 0], field=GF5),
    galois.Poly([1, 1, 2, 1], field=GF5),
    galois.Poly([1, 1, 2, 2], field=GF5),
    galois.Poly([1, 1, 3, 0], field=GF5),
    galois.Poly([1, 1, 3, 1], field=GF5),
    galois.Poly([1, 1, 3, 3], field=GF5),
    galois.Poly([1, 2, 0, 2], field=GF5),
    galois.Poly([1, 2, 0, 4], field=GF5),
    galois.Poly([1, 2, 1, 0], field=GF5),
    galois.Poly([1, 2, 1, 1], field=GF5),
    galois.Poly([1, 2, 1, 3], field=GF5),
    galois.Poly([1, 2, 1, 4], field=GF5),
    galois.Poly([1, 2, 2, 3], field=GF5),
    galois.Poly([1, 2, 3, 0], field=GF5),
    galois.Poly([1, 2, 3, 3], field=GF5),
    galois.Poly([1, 3, 1, 0], field=GF5),
    galois.Poly([1, 3, 1, 2], field=GF5),
    galois.Poly([1, 3, 4, 3], field=GF5),
    galois.Poly([1, 4, 0, 3], field=GF5),
    galois.Poly([1, 4, 0, 4], field=GF5),
    galois.Poly([1, 4, 1, 0], field=GF5),
    galois.Poly([1, 4, 1, 4], field=GF5),
    galois.Poly([1, 4, 3, 0], field=GF5),
    galois.Poly([1, 4, 3, 1], field=GF5),
    galois.Poly([1, 4, 4, 1], field=GF5),
    galois.Poly([1, 4, 4, 4], field=GF5),
    galois.Poly([2, 0, 0, 1], field=GF5),
    galois.Poly([2, 0, 0, 3], field=GF5),
    galois.Poly([2, 0, 1, 0], field=GF5),
    galois.Poly([2, 0, 1, 3], field=GF5),
    galois.Poly([2, 0, 2, 2], field=GF5),
    galois.Poly([2, 0, 2, 4], field=GF5),
    galois.Poly([2, 0, 3, 2], field=GF5),
    galois.Poly([2, 0, 3, 3], field=GF5),
    galois.Poly([2, 1, 2, 0], field=GF5),
    galois.Poly([2, 1, 2, 4], field=GF5),
    galois.Poly([2, 1, 3, 1], field=GF5),
    galois.Poly([2, 2, 0, 0], field=GF5),
    galois.Poly([2, 2, 0, 1], field=GF5),
    galois.Poly([2, 2, 0, 2], field=GF5),
    galois.Poly([2, 2, 1, 0], field=GF5),
    galois.Poly([2, 2, 1, 1], field=GF5),
    galois.Poly([2, 2, 1, 2], field=GF5),
    galois.Poly([2, 2, 4, 0], field=GF5),
    galois.Poly([2, 2, 4, 2], field=GF5),
    galois.Poly([2, 2, 4, 4], field=GF5),
    galois.Poly([2, 3, 0, 1], field=GF5),
    galois.Poly([2, 3, 0, 3], field=GF5),
    galois.Poly([2, 3, 1, 0], field=GF5),
    galois.Poly([2, 3, 1, 2], field=GF5),
    galois.Poly([2, 3, 2, 0], field=GF5),
    galois.Poly([2, 3, 2, 3], field=GF5),
    galois.Poly([2, 3, 3, 2], field=GF5),
    galois.Poly([2, 3, 3, 3], field=GF5),
    galois.Poly([2, 4, 0, 3], field=GF5),
    galois.Poly([2, 4, 0, 4], field=GF5),
    galois.Poly([2, 4, 1, 0], field=GF5),
    galois.Poly([2, 4, 1, 1], field=GF5),
    galois.Poly([2, 4, 2, 0], field=GF5),
    galois.Poly([2, 4, 2, 1], field=GF5),
    galois.Poly([2, 4, 2, 2], field=GF5),
    galois.Poly([2, 4, 2, 3], field=GF5),
    galois.Poly([2, 4, 4, 1], field=GF5),
    galois.Poly([3, 0, 0, 2], field=GF5),
    galois.Poly([3, 0, 0, 4], field=GF5),
    galois.Poly([3, 0, 2, 2], field=GF5),
    galois.Poly([3, 0, 2, 3], field=GF5),
    galois.Poly([3, 0, 3, 1], field=GF5),
    galois.Poly([3, 0, 3, 3], field=GF5),
    galois.Poly([3, 0, 4, 0], field=GF5),
    galois.Poly([3, 0, 4, 2], field=GF5),
    galois.Poly([3, 1, 0, 1], field=GF5),
    galois.Poly([3, 1, 0, 2], field=GF5),
    galois.Poly([3, 1, 1, 4], field=GF5),
    galois.Poly([3, 1, 3, 0], field=GF5),
    galois.Poly([3, 1, 3, 2], field=GF5),
    galois.Poly([3, 1, 3, 3], field=GF5),
    galois.Poly([3, 1, 3, 4], field=GF5),
    galois.Poly([3, 1, 4, 0], field=GF5),
    galois.Poly([3, 1, 4, 4], field=GF5),
    galois.Poly([3, 2, 0, 2], field=GF5),
    galois.Poly([3, 2, 0, 4], field=GF5),
    galois.Poly([3, 2, 2, 2], field=GF5),
    galois.Poly([3, 2, 2, 3], field=GF5),
    galois.Poly([3, 2, 3, 0], field=GF5),
    galois.Poly([3, 2, 3, 2], field=GF5),
    galois.Poly([3, 2, 4, 0], field=GF5),
    galois.Poly([3, 2, 4, 3], field=GF5),
    galois.Poly([3, 3, 0, 0], field=GF5),
    galois.Poly([3, 3, 0, 3], field=GF5),
    galois.Poly([3, 3, 0, 4], field=GF5),
    galois.Poly([3, 3, 1, 0], field=GF5),
    galois.Poly([3, 3, 1, 1], field=GF5),
    galois.Poly([3, 3, 1, 3], field=GF5),
    galois.Poly([3, 3, 4, 0], field=GF5),
    galois.Poly([3, 3, 4, 3], field=GF5),
    galois.Poly([3, 3, 4, 4], field=GF5),
    galois.Poly([3, 4, 2, 4], field=GF5),
    galois.Poly([3, 4, 3, 0], field=GF5),
    galois.Poly([3, 4, 3, 1], field=GF5),
    galois.Poly([4, 0, 0, 1], field=GF5),
    galois.Poly([4, 0, 0, 2], field=GF5),
    galois.Poly([4, 0, 1, 1], field=GF5),
    galois.Poly([4, 0, 1, 4], field=GF5),
    galois.Poly([4, 0, 2, 0], field=GF5),
    galois.Poly([4, 0, 2, 1], field=GF5),
    galois.Poly([4, 0, 4, 3], field=GF5),
    galois.Poly([4, 0, 4, 4], field=GF5),
    galois.Poly([4, 1, 0, 1], field=GF5),
    galois.Poly([4, 1, 0, 2], field=GF5),
    galois.Poly([4, 1, 1, 1], field=GF5),
    galois.Poly([4, 1, 1, 4], field=GF5),
    galois.Poly([4, 1, 2, 0], field=GF5),
    galois.Poly([4, 1, 2, 4], field=GF5),
    galois.Poly([4, 1, 4, 0], field=GF5),
    galois.Poly([4, 1, 4, 1], field=GF5),
    galois.Poly([4, 2, 1, 2], field=GF5),
    galois.Poly([4, 2, 4, 0], field=GF5),
    galois.Poly([4, 2, 4, 3], field=GF5),
    galois.Poly([4, 3, 0, 1], field=GF5),
    galois.Poly([4, 3, 0, 3], field=GF5),
    galois.Poly([4, 3, 2, 0], field=GF5),
    galois.Poly([4, 3, 2, 2], field=GF5),
    galois.Poly([4, 3, 3, 2], field=GF5),
    galois.Poly([4, 3, 4, 0], field=GF5),
    galois.Poly([4, 3, 4, 1], field=GF5),
    galois.Poly([4, 3, 4, 2], field=GF5),
    galois.Poly([4, 3, 4, 4], field=GF5),
    galois.Poly([4, 4, 0, 0], field=GF5),
    galois.Poly([4, 4, 0, 2], field=GF5),
    galois.Poly([4, 4, 0, 4], field=GF5),
    galois.Poly([4, 4, 2, 0], field=GF5),
    galois.Poly([4, 4, 2, 2], field=GF5),
    galois.Poly([4, 4, 2, 4], field=GF5),
    galois.Poly([4, 4, 3, 0], field=GF5),
    galois.Poly([4, 4, 3, 3], field=GF5),
    galois.Poly([4, 4, 3, 4], field=GF5),
]


def test_primitive_element_exceptions():
    p = galois.conway_poly(2, 8)

    with pytest.raises(TypeError):
        galois.primitive_element(p.coeffs)
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Random(2)*galois.Poly.Random(2))
    with pytest.raises(ValueError):
        galois.primitive_element(p, method="invalid")
    with pytest.raises(ValueError):
        galois.primitive_element(galois.Poly.Str("x^4"))


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (3,2), (3,3), (3,4), (5,2), (5,3), (5,4)])
def test_primitive_element_min(characteristic, degree):
    LUT = eval(f"PRIMITIVE_ELEMENTS_{characteristic}_{degree}")
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p) == LUT[0]


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (3,2), (3,3), (3,4), (5,2), (5,3), (5,4)])
def test_primitive_element_max(characteristic, degree):
    LUT = eval(f"PRIMITIVE_ELEMENTS_{characteristic}_{degree}")
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p, method="max") == LUT[-1]


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (3,2), (3,3), (3,4), (5,2), (5,3), (5,4)])
def test_primitive_element_random(characteristic, degree):
    LUT = eval(f"PRIMITIVE_ELEMENTS_{characteristic}_{degree}")
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_element(p, method="max") in LUT


def test_primitive_elements_exceptions():
    p = galois.conway_poly(2, 8)

    with pytest.raises(TypeError):
        galois.primitive_elements(float(int(p)))
    with pytest.raises(ValueError):
        galois.primitive_elements(galois.Poly.Random(0))
    with pytest.raises(ValueError):
        galois.primitive_elements(galois.Poly.Random(2)*galois.Poly.Random(2))


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (3,2), (3,3), (3,4), (5,2), (5,3), (5,4)])
def test_primitive_elements(characteristic, degree):
    LUT = eval(f"PRIMITIVE_ELEMENTS_{characteristic}_{degree}")
    p = galois.GF(characteristic**degree).irreducible_poly
    assert galois.primitive_elements(p) == LUT


def test_is_primitive_element_exceptions():
    e = galois.Poly([1, 0, 1, 1])
    f = galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1])

    with pytest.raises(TypeError):
        galois.is_primitive_element(float(int(e)), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]))
    with pytest.raises(TypeError):
        galois.is_primitive_element(e, f.coeffs)
    with pytest.raises(ValueError):
        galois.is_primitive_element(galois.Poly([1, 0, 1, 1]), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1], field=galois.GF(3)))
    with pytest.raises(ValueError):
        galois.is_primitive_element(galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]), galois.Poly([1, 0, 0, 0, 1, 1, 1, 0, 1]))


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (3,2), (3,3), (3,4), (5,2), (5,3), (5,4)])
def test_is_primitive_element(characteristic, degree):
    LUT = eval(f"PRIMITIVE_ELEMENTS_{characteristic}_{degree}")
    p = galois.GF(characteristic**degree).irreducible_poly
    assert all(galois.is_primitive_element(e, p) for e in LUT)


@pytest.mark.parametrize("characteristic,degree", [(2,2), (2,3), (2,4), (2,5), (2,6), (3,2), (3,3), (3,4), (5,2), (5,3), (5,4)])
def test_is_not_primitive_element(characteristic, degree):
    LUT = eval(f"PRIMITIVE_ELEMENTS_{characteristic}_{degree}")
    p = galois.GF(characteristic**degree).irreducible_poly
    while True:
        e = galois.Poly.Int(random.randint(0, characteristic**degree - 1), field=galois.GF(characteristic))
        if e not in LUT:
            break
    assert not galois.is_primitive_element(e, p)
