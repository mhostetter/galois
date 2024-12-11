"""
A module that defines the GF(2) array class.
"""

from __future__ import annotations

import numpy
import numpy as np

from ..typing import ElementLike, ArrayLike, DTypeLike
from .._domains._lookup import (
    add_ufunc,
    divide_ufunc,
    log_ufunc,
    multiply_ufunc,
    negative_ufunc,
    power_ufunc,
    reciprocal_ufunc,
    sqrt_ufunc,
    subtract_ufunc,
)
from .._domains._ufunc import UFuncMixin, matmul_ufunc
from .._helper import export
from ._array import FieldArray


class reciprocal(reciprocal_ufunc):
    """
    A ufunc dispatcher for the multiplicative inverse in GF(2).
    """

    @staticmethod
    def calculate(a: int) -> int:  # pragma: no cover
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        return 1


class divide(divide_ufunc):
    """
    A ufunc dispatcher for division in GF(2).
    """

    @staticmethod
    def calculate(a: int, b: int) -> int:  # pragma: no cover
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        return a & b


class power(power_ufunc):
    """
    A ufunc dispatcher for exponentiation in GF(2).
    """

    @staticmethod
    def calculate(a: int, b: int) -> int:  # pragma: no cover
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        if b == 0:
            return 1
        return a


class log(log_ufunc):
    """
    A ufunc dispatcher for the logarithm in GF(2).
    """

    @staticmethod
    def calculate(a: int, b: int) -> int:  # pragma: no cover
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")
        return 0


class sqrt(sqrt_ufunc):
    """
    A ufunc dispatcher for the square root in GF(2).
    """

    def implementation(self, a: FieldArray) -> FieldArray:
        return a.copy()


class UFuncMixin_2_1(UFuncMixin):
    """
    A mixin class that provides explicit calculation arithmetic for GF(2).
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._add = add_ufunc(cls, override=np.bitwise_xor)
        cls._negative = negative_ufunc(cls, override=np.positive)
        cls._subtract = subtract_ufunc(cls, override=np.bitwise_xor)
        cls._multiply = multiply_ufunc(cls, override=np.bitwise_and)
        cls._reciprocal = reciprocal(cls)
        cls._divide = divide(cls)
        cls._power = power(cls)
        cls._log = log(cls)
        cls._sqrt = sqrt(cls)


class add_ufunc_bitpacked(add_ufunc):
    """
    Addition ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._unpacked_shape = inputs[0]._unpacked_shape
        return output


class subtract_ufunc_bitpacked(subtract_ufunc):
    """
    Subtraction ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._unpacked_shape = inputs[0]._unpacked_shape
        return output


class multiply_ufunc_bitpacked(multiply_ufunc):
    """
    Multiply ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._unpacked_shape = inputs[0]._unpacked_shape
        return output


class divide_ufunc_bitpacked(divide):
    """
    Divide ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._unpacked_shape = inputs[0]._unpacked_shape
        return output


class matmul_ufunc_bitpacked(matmul_ufunc):
    """
    Matmul ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        a, b = inputs

        assert isinstance(a, GF2BP) and isinstance(b, GF2BP)
        assert a.shape[-1] == b.shape[0]

        # a has rows packed, so repack b's columns
        b = np.packbits(
            np.unpackbits(b.view(np.ndarray), axis=-1, count=b._unpacked_shape[-1]),
            axis=0,
        ).view(GF2BP)

        inputs = (a, b)
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        unpacked_shape = output.shape
        output = self.field._view(np.packbits(output.view(np.ndarray), axis=-1))
        output._unpacked_shape = unpacked_shape

        return output


def not_implemented(*args, **kwargs):
    return NotImplemented


class UFuncMixin_2_1_BitPacked(UFuncMixin):
    """
    A mixin class that provides explicit calculation arithmetic for GF(2).
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._add = add_ufunc_bitpacked(cls, override=np.bitwise_xor)
        cls._negative = negative_ufunc(cls, override=np.positive)
        cls._subtract = subtract_ufunc_bitpacked(cls, override=np.bitwise_xor)
        cls._multiply = multiply_ufunc_bitpacked(cls, override=np.bitwise_and)
        cls._reciprocal = reciprocal(cls)
        cls._divide = divide_ufunc_bitpacked(cls)
        cls._power = power(cls)
        cls._log = log(cls)
        cls._sqrt = sqrt(cls)

    @classmethod
    def _assign_ufuncs(cls):
        super()._assign_ufuncs()

        # We have to set this here because ArrayMeta would override it.
        cls._matmul = matmul_ufunc_bitpacked(cls)


# NOTE: There is a "verbatim" block in the docstring because we were not able to monkey-patch GF2 like the
# other classes in docs/conf.py. So, technically, at doc-build-time issubclass(galois.GF2, galois.FieldArray) == False
# because galois.FieldArray is monkey-patched and GF2 is not. This all stems from an inability of Sphinx to
# document class properties... :(


@export
class GF2(
    FieldArray,
    UFuncMixin_2_1,
    characteristic=2,
    degree=1,
    order=2,
    irreducible_poly_int=3,
    is_primitive_poly=True,
    primitive_element=1,
):
    r"""
    A :obj:`~galois.FieldArray` subclass over $\mathrm{GF}(2)$.

    .. info::

        This class is a pre-generated :obj:`~galois.FieldArray` subclass generated with `galois.GF(2)` and is
        included in the API for convenience.

    Examples:
        This class is equivalent, and in fact identical, to the :obj:`~galois.FieldArray` subclass returned from the
        class factory :func:`~galois.GF`.

        .. ipython::

            In [2]: galois.GF2 is galois.GF(2)

            @verbatim
            In [3]: issubclass(galois.GF2, galois.FieldArray)
            Out[3]: True

            In [4]: print(galois.GF2.properties)

        Create a :obj:`~galois.FieldArray` instance using :obj:`~galois.GF2`'s constructor.

        .. ipython:: python

            x = galois.GF2([1, 0, 1, 1]); x
            isinstance(x, galois.GF2)

    Group:
        galois-fields
    """

    def astype(self, dtype, **kwargs):
        if dtype is GF2BP:
            return GF2BP(self)  # bits are packed in initialization

        return super().astype(dtype, **kwargs)


@export
class GF2BP(
    FieldArray,
    UFuncMixin_2_1_BitPacked,
    characteristic=2,
    degree=1,
    order=2,
    irreducible_poly_int=3,
    is_primitive_poly=True,
    primitive_element=1,
    bitpacked=True,
):
    r"""
    A :obj:`~galois.FieldArray` subclass over $\mathrm{GF}(2)$ with a bit-packed representation.

    .. info::

        This class is a pre-generated :obj:`~galois.FieldArray` subclass generated with `galois.GF(2)` and is
        included in the API for convenience.

    Examples:
        This class is equivalent, and in fact identical, to the :obj:`~galois.FieldArray` subclass returned from the
        class factory :func:`~galois.GF`.

        .. ipython::

            In [2]: galois.GF2 is galois.GF(2)

            @verbatim
            In [3]: issubclass(galois.GF2, galois.FieldArray)
            Out[3]: True

            In [4]: print(galois.GF2.properties)

        Create a :obj:`~galois.FieldArray` instance using :obj:`~galois.GF2`'s constructor.

        .. ipython:: python

            x = galois.GF2([1, 0, 1, 1]); x
            isinstance(x, galois.GF2)

    Group:
        galois-fields
    """

    def __init__(
        self, x: ElementLike | ArrayLike, dtype: DTypeLike | None = None, **kwargs
    ):
        if isinstance(x, np.ndarray):
            self.view(np.ndarray)[:] = np.packbits(x.view(np.ndarray), axis=-1)
            self._unpacked_shape = x.shape
            return

        raise RuntimeError("BLAH")
        # super().__init__(x, dtype, **kwargs)

    def astype(self, dtype, **kwargs):
        if dtype is GF2:
            return GF2(
                numpy.unpackbits(
                    self.view(np.ndarray),
                    axis=-1,
                    count=self._unpacked_shape[-1],
                )
            )

        return super().astype(dtype, **kwargs)


GF2._default_ufunc_mode = "jit-calculate"
GF2._ufunc_modes = ["jit-calculate", "python-calculate"]
GF2.compile("auto")

GF2BP._default_ufunc_mode = "jit-calculate"
GF2BP._ufunc_modes = ["jit-calculate", "python-calculate"]
GF2BP.compile("auto")
