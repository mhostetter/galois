from types import SimpleNamespace

import numpy as np
import pytest

from galois import _verify


def test_argument_names_fallbacks(monkeypatch):
    monkeypatch.setattr(_verify.inspect, "currentframe", lambda: None)
    assert _verify._argument_names() == ("<argument>",)

    class DummyFrame:
        pass

    monkeypatch.setattr(_verify.inspect, "currentframe", lambda: DummyFrame())
    monkeypatch.setattr(_verify.inspect, "getouterframes", lambda frame: [SimpleNamespace()])
    assert _verify._argument_names() == ("<argument>",)

    def getouterframes_with_frame(frame):
        return [SimpleNamespace(), SimpleNamespace(), SimpleNamespace(frame=SimpleNamespace())]

    monkeypatch.setattr(_verify.inspect, "getouterframes", getouterframes_with_frame)
    monkeypatch.setattr(_verify.inspect, "getframeinfo", lambda frame: SimpleNamespace(code_context=None))
    assert _verify._argument_names() == ("<argument>",)

    monkeypatch.setattr(_verify.inspect, "getframeinfo", lambda frame: SimpleNamespace(code_context=["no call here"]))
    assert _verify._argument_names() == ("<argument>",)


def test_verify_positional_args():
    assert _verify.verify_positional_args((1, 2), 2) == 2
    with pytest.raises(ValueError, match="max of 1 positional arguments"):
        _verify.verify_positional_args((1, 2), 1)


def test_verify_specified_and_not_specified():
    assert _verify.verify_specified(1) == 1
    with pytest.raises(ValueError, match="must be provided"):
        _verify.verify_specified(None)

    assert _verify.verify_not_specified(None) is None
    with pytest.raises(ValueError, match="must not be provided"):
        _verify.verify_not_specified(1)


def test_verify_only_one_and_at_least_one_specified():
    assert _verify.verify_only_one_specified(None, 1, None) == (None, 1, None)
    with pytest.raises(ValueError, match="Exactly one of the arguments"):
        _verify.verify_only_one_specified(None, 1, 2)

    assert _verify.verify_at_least_one_specified(None, 1) == (None, 1)
    with pytest.raises(ValueError, match="At least one of the arguments"):
        _verify.verify_at_least_one_specified(None, None)


def test_verify_isinstance():
    assert _verify.verify_isinstance(1, int) == 1
    assert _verify.verify_isinstance(None, int, optional=True) is None
    with pytest.raises(TypeError, match="must be an instance"):
        _verify.verify_isinstance("1", int)
    with pytest.raises(TypeError, match="must be an instance"):
        _verify.verify_isinstance(None, int, optional=False)


def test_verify_issubclass():
    assert _verify.verify_issubclass(int, int) is None
    assert _verify.verify_issubclass(None, int, optional=True) is None
    with pytest.raises(TypeError, match="must be a subclass"):
        _verify.verify_issubclass(1, int)


def test_verify_literal():
    assert _verify.verify_literal("a", ["a", "b"]) == "a"
    with pytest.raises(ValueError, match="must be one of"):
        _verify.verify_literal("c", ["a", "b"])


def test_verify_scalar_optional_and_array_rejection():
    assert _verify.verify_scalar(None, optional=True) is None
    with pytest.raises(TypeError, match="must be a scalar"):
        _verify.verify_scalar(np.array([1]))


def test_verify_scalar_type_checks():
    assert _verify.verify_scalar(1, int=True) == 1
    assert _verify.verify_scalar(1.5, float=True) == 1.5
    assert _verify.verify_scalar(1 + 2j, complex=True) == 1 + 2j

    with pytest.raises(TypeError, match="must be an int"):
        _verify.verify_scalar(1.5, int=True)
    with pytest.raises(TypeError):
        _verify.verify_scalar("1", float=True)
    with pytest.raises(TypeError):
        _verify.verify_scalar("1", complex=True)


def test_verify_scalar_numpy_integer_acceptance():
    assert _verify.verify_scalar(np.int64(2), int=True, accept_numpy=True) == np.int64(2)
    with pytest.raises(TypeError, match="must be an int"):
        _verify.verify_scalar(np.int64(2), int=True, accept_numpy=False)


def test_verify_scalar_real_imaginary():
    with pytest.raises(ValueError, match="must be real"):
        _verify.verify_scalar(np.complex64(1 + 1j), real=True)
    with pytest.raises(ValueError, match="must be complex"):
        _verify.verify_scalar(1, imaginary=True)

    assert _verify.verify_scalar(1 + 1j, imaginary=True) == 1 + 1j


def test_verify_scalar_sign_and_bounds():
    with pytest.raises(ValueError, match="must be negative"):
        _verify.verify_scalar(1, negative=True)
    with pytest.raises(ValueError, match="must be non-negative"):
        _verify.verify_scalar(-1, non_negative=True)
    with pytest.raises(ValueError, match="must be positive"):
        _verify.verify_scalar(0, positive=True)

    with pytest.raises(ValueError, match="must be at least"):
        _verify.verify_scalar(1, inclusive_min=2)
    with pytest.raises(ValueError, match="must be at most"):
        _verify.verify_scalar(3, inclusive_max=2)
    with pytest.raises(ValueError, match="must be greater than"):
        _verify.verify_scalar(2, exclusive_min=2)
    with pytest.raises(ValueError, match="must be less than"):
        _verify.verify_scalar(2, exclusive_max=2)

    assert _verify.verify_scalar(2, inclusive_min=2, inclusive_max=2) == 2


def test_verify_scalar_parity_and_power_of_two():
    with pytest.raises(TypeError, match="must be an int to be even"):
        _verify.verify_scalar(2.0, even=True)
    with pytest.raises(TypeError, match="must be an int to be odd"):
        _verify.verify_scalar(2.0, odd=True)

    with pytest.raises(ValueError, match="must be even"):
        _verify.verify_scalar(3, even=True)
    with pytest.raises(ValueError, match="must be odd"):
        _verify.verify_scalar(2, odd=True)

    with pytest.raises(TypeError, match="must be an int to be a power of two"):
        _verify.verify_scalar(2.0, power_of_two=True)
    with pytest.raises(ValueError, match="must be a power of two"):
        _verify.verify_scalar(0, power_of_two=True)
    with pytest.raises(ValueError, match="must be a power of two"):
        _verify.verify_scalar(3, power_of_two=True)

    assert _verify.verify_scalar(8, power_of_two=True) == 8


def test_verify_bool():
    assert _verify.verify_bool(True) is True
    assert _verify.verify_bool(np.bool_(True))
    assert _verify.verify_bool(np.array(True), convert_numpy=True) is True
    with pytest.raises(TypeError, match="must be a bool"):
        _verify.verify_bool(1)
    with pytest.raises(TypeError, match="must be a bool"):
        _verify.verify_bool(np.bool_(True), accept_numpy=False)


def test_verify_coprime_and_condition():
    _verify.verify_coprime(4, 9)
    with pytest.raises(ValueError, match="must be coprime"):
        _verify.verify_coprime(4, 6)

    _verify.verify_condition(True)
    with pytest.raises(ValueError, match="must satisfy the condition"):
        _verify.verify_condition(False)


def test_verify_arraylike_optional_and_dtype():
    assert _verify.verify_arraylike(None, optional=True) is None
    array = _verify.verify_arraylike([1, 2], dtype=int, int=True)
    assert array.dtype == np.dtype(int)
    with pytest.raises(TypeError, match="must be an int"):
        _verify.verify_arraylike([1.5], int=True)
    with pytest.raises(TypeError, match="must be an int or float"):
        _verify.verify_arraylike([1 + 1j], float=True)
    with pytest.raises(TypeError, match="must be an int or float or complex"):
        _verify.verify_arraylike(["1"], complex=True)


def test_verify_arraylike_real_imaginary_and_signs():
    with pytest.raises(ValueError, match="must be real"):
        _verify.verify_arraylike([1 + 1j], real=True)
    with pytest.raises(ValueError, match="must be complex"):
        _verify.verify_arraylike([1], imaginary=True)
    with pytest.raises(ValueError, match="must be negative"):
        _verify.verify_arraylike([1], negative=True)
    with pytest.raises(ValueError, match="must be non-negative"):
        _verify.verify_arraylike([-1], non_negative=True)
    with pytest.raises(ValueError, match="must be positive"):
        _verify.verify_arraylike([0], positive=True)


def test_verify_arraylike_bounds_and_shape_constraints():
    with pytest.raises(ValueError, match="must be at least"):
        _verify.verify_arraylike([1], inclusive_min=2)
    with pytest.raises(ValueError, match="must be at most"):
        _verify.verify_arraylike([3], inclusive_max=2)
    with pytest.raises(ValueError, match="must be greater than"):
        _verify.verify_arraylike([2], exclusive_min=2)
    with pytest.raises(ValueError, match="must be less than"):
        _verify.verify_arraylike([2], exclusive_max=2)

    scalar = _verify.verify_arraylike(1, atleast_1d=True)
    assert scalar.ndim == 1

    array2d = _verify.verify_arraylike([1, 2, 3], atleast_2d=True)
    assert array2d.ndim == 2

    array3d = _verify.verify_arraylike([1, 2, 3], atleast_3d=True)
    assert array3d.ndim == 3

    with pytest.raises(ValueError, match="must have 2 dimensions"):
        _verify.verify_arraylike([1, 2], ndim=2)
    with pytest.raises(ValueError, match="must have 3 elements"):
        _verify.verify_arraylike([1, 2], size=3)
    with pytest.raises(ValueError, match="must have one of"):
        _verify.verify_arraylike([1, 2], sizes=(1, 3))
    with pytest.raises(ValueError, match="size that is a multiple of 3"):
        _verify.verify_arraylike([1, 2], size_multiple=3)
    with pytest.raises(ValueError, match="must have shape"):
        _verify.verify_arraylike([1, 2], shape=(2, 1))


def test_verify_same_shape():
    _verify.verify_same_shape(np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="must have the same shape"):
        _verify.verify_same_shape(np.zeros((2, 2)), np.zeros((2, 3)))


def test_convert_to_scalar_and_output():
    assert _verify.convert_to_scalar(np.array(5)) == 5
    assert _verify.convert_to_scalar(np.int64(6)) == 6

    result = _verify.convert_output(np.array([[7]]), squeeze=True)
    assert result == 7
