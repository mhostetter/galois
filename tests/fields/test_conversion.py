"""
A pytest module to test FieldArray view casting.
"""
import numpy as np
import pytest

import galois

from .helper import DTYPES


class TestView:
    def test_valid_dtypes(self, field):
        for dtype in field.dtypes:
            v = np.array([0, 1, 0, 1], dtype=dtype)
            a = v.view(field)

    def test_small_integer_dtype(self, field):
        if np.int8 not in field.dtypes:
            v = np.array([0, 1, 0, 1], dtype=np.int8)
            with pytest.raises(TypeError):
                a = v.view(field)

    def test_non_valid_dtype(self, field):
        v = np.array([0, 1, 0, 1], dtype=float)
        with pytest.raises(TypeError):
            a = v.view(field)

    def test_valid_dtypes_out_of_range_values(self, field):
        for dtype in field.dtypes:
            if dtype != np.object_ and field.order > np.iinfo(dtype).max:
                # Skip for tests where order > dtype.max, like GF(2^8) for np.uint8 (order=256 > max=255). If we don't skip
                # the test then order will wrap and the ValueError won't be raised.
                continue
            v = np.array([0, 1, 0, field.order], dtype=dtype)
            with pytest.raises(ValueError):
                a = v.view(field)


class TestAsType:
    def test_valid_dtypes(self, field):
        a = field.Random(10)
        for dtype in field.dtypes:
            b = a.astype(dtype)

    def test_invalid_dtypes(self, field):
        a = field.Random(10)
        for dtype in [d for d in DTYPES if d not in field.dtypes]:
            with pytest.raises(TypeError):
                b = a.astype(dtype)
