"""
A module that defines the GF(2) array class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from typing_extensions import Literal, Self, Optional

from .._domains._linalg import row_reduce_jit
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
from .._domains._function import Function
from .._domains._array import Array
from .._helper import export, verify_isinstance
from ..typing import ArrayLike, DTypeLike, ElementLike
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

def packbits(a, axis=None, bitorder='big'):
    if isinstance(a, GF2BP):
        return a

    if not isinstance(a, GF2):
        raise TypeError("Bit-packing is only supported on instances of GF2.")

    axis = -1 if axis is None else axis
    axis_element_count = 1 if a.ndim == 0 else a.shape[axis]
    packed = GF2BP(np.packbits(a.view(np.ndarray), axis=axis, bitorder=bitorder), axis_element_count)
    return packed


def unpackbits(a, axis=None, count=None, bitorder='big'):
    if isinstance(a, GF2):
        return a

    if not isinstance(a, GF2BP):
        raise TypeError("Bit-unpacking is only supported on instances of GF2BP.")

    if axis is None:
        axis = -1

    return GF2(np.unpackbits(a.view(np.ndarray), axis=axis, count=a._axis_count if count is None else count, bitorder=bitorder))


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
        cls._packbits = packbits
        cls._unpackbits = unpackbits


class add_ufunc_bitpacked(add_ufunc):
    """
    Addition ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._axis_count = inputs[0]._axis_count
        return output


class subtract_ufunc_bitpacked(subtract_ufunc):
    """
    Subtraction ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._axis_count = max(i._axis_count for i in inputs)
        return output


class multiply_ufunc_bitpacked(multiply_ufunc):
    """
    Multiply ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._axis_count = max(i._axis_count for i in inputs)
        return output


class divide_ufunc_bitpacked(divide):
    """
    Divide ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        output = super().__call__(ufunc, method, inputs, kwargs, meta)
        output._axis_count = max(i._axis_count for i in inputs)
        return output


class matmul_ufunc_bitpacked(matmul_ufunc):
    """
    Matmul ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        a, b = inputs

        assert isinstance(a, GF2BP) and isinstance(b, GF2BP)

        # bit-packed matrices have columns packed by default, so unpack the second operand and repack to rows
        field = self.field
        row_axis_count = b.shape[0]
        b = field._view(
            np.packbits(
                np.unpackbits(b.view(np.ndarray), axis=-1, count=b._axis_count),
                axis=0,
            )
        )
        b._axis_count = row_axis_count

        # Make sure the inner dimensions match (e.g. (M, N) x (N, P) -> (M, P))
        assert a.shape[-1] == b.shape[0]
        if len(b.shape) == 1:
            final_shape = (a.shape[0],)
        else:
            final_shape = (a.shape[0], b.shape[-1])

        if len(b.shape) == 1:
            # matrix-vector multiplication
            output = np.bitwise_xor.reduce(np.unpackbits((a & b).view(np.ndarray), axis=-1), axis=-1)
        else:
            # matrix-matrix multiplication
            output = GF2.Zeros(final_shape)
            for i in range(b.shape[-1]):
                # TODO: Include alternate path for numpy < v2
                # output[:, i] = np.bitwise_xor.reduce(np.unpackbits((a & b[:, i]).view(np.ndarray), axis=-1), axis=-1)
                output[:, i] = np.bitwise_xor.reduce(np.bitwise_count((a & b[:, i]).view(np.ndarray)), axis=-1) % 2
        output = field._view(np.packbits(output.view(np.ndarray), axis=-1))
        output._axis_count = final_shape[-1]

        return output

class concatenate_bitpacked(Function):
    """Concatenates matrices together"""

    def __call__(self, arrays: list[Array], **kwargs):
        unpacked_arrays = []
        for array in arrays:
            verify_isinstance(array, self.field)
            unpacked_arrays.append(np.unpackbits(array))

        unpacked = np.concatenate(unpacked_arrays, **kwargs)
        return np.packbits(unpacked)


class inv_bitpacked(Function):
    """
    Computes the inverse of the square matrix.
    """

    def __call__(self, A: Array) -> Array:
        verify_isinstance(A, self.field)
        # if not (A.ndim == 2 and A.shape[0] == A.shape[1]):
        #     raise np.linalg.LinAlgError(f"Argument 'A' must be square, not {A.shape}.")

        n = A.shape[0]
        I = self.field.Identity(n, dtype=A.dtype)

        # Concatenate A and I to get the matrix AI = [A | I]
        AI = np.concatenate((A, I), axis=-1)

        # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
        AI_rre, _ = row_reduce_jit(self.field)(AI, ncols=n)

        # The rank is the number of non-zero rows of the row reduced echelon form
        rank = np.sum(~np.all(AI_rre[:, 0:n] == 0, axis=1))
        if not rank == n:
            raise np.linalg.LinAlgError(
                f"Argument 'A' is singular and not invertible because it does not have full rank of {n}, "
                f"but rank of {rank}."
            )

        A_inv = AI_rre[:, -n:]

        return A_inv


def not_implemented(*args, **kwargs):
    # TODO: Add a better error message about limited support
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
        cls._reciprocal = not_implemented # reciprocal(cls)
        cls._divide = divide_ufunc_bitpacked(cls)
        cls._power = not_implemented # power(cls)
        cls._log = not_implemented # log(cls)
        cls._sqrt = sqrt(cls)
        cls._packbits = packbits
        cls._unpackbits = unpackbits
        cls._concatenate = concatenate_bitpacked(cls)

    @classmethod
    def _assign_ufuncs(cls):
        super()._assign_ufuncs()

        # We have to set this here because ArrayMeta would override it.
        cls._matmul = matmul_ufunc_bitpacked(cls)
        cls._inv = inv_bitpacked(cls)


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

    def __new__(
        cls,
        x: ElementLike | ArrayLike,
        axis_element_count: Optional[int] = None,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0,
    ) -> Self:
        # axis_element_count is required, but by making it optional it allows us to catch uses of the class that are not
        # supported (e.g. Random)
        if isinstance(x, (tuple, list, np.ndarray, FieldArray)) and axis_element_count is not None:
            # NOTE: I'm not sure that we want to change the dtype specifically for the bit-packed version or how we verify
            # dtype = cls._get_dtype(dtype)
            # x = cls._verify_array_like_types_and_values(x)

            array = cls._view(np.array(x, dtype=dtype, copy=copy, order=order, ndmin=ndmin))
            array._axis_count = axis_element_count

            return array

        raise NotImplementedError(
            "GF2BP is a custom bit-packed GF2 class with limited functionality. "
            "If you were using an alternate constructor (e.g. Random), then use the GF2 class and convert it to the "
            "bit-packed version by using `np.packbits`."
        )

    def __init__(
        self,
        x: ElementLike | ArrayLike,
        axis_element_count: Optional[int] = None,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0,
    ):
        pass

    @classmethod
    def Identity(cls, size: int, dtype: DTypeLike | None = None) -> Self:
        r"""
        Creates an $n \times n$ identity matrix.

        Arguments:
            size: The size $n$ along one dimension of the identity matrix.
            dtype: The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest
                unsigned data type for this :obj:`~galois.Array` subclass (the first element in
                :obj:`~galois.Array.dtypes`).

        Returns:
            A 2-D identity matrix with shape `(size, size)`.
        """
        array = GF2.Identity(size, dtype=dtype)
        return np.packbits(array)

    def get_index_parameters(self, index):
        post_index = NotImplemented
        if isinstance(index, (Sequence, np.ndarray)):
            if len(index) == 2:
                row_index, col_index = index
                if isinstance(col_index, int):
                    post_index = (slice(None), col_index)
                    index = (row_index, col_index // 8)
                elif isinstance(col_index, slice):
                    post_index = (slice(None), col_index)
                    col_index = slice(col_index.start // 8, max(col_index.step // 8, 1), max(col_index.stop // 8, 1))
                    index = (row_index, col_index)
                elif isinstance(col_index, (Sequence, np.ndarray)):
                    if isinstance(row_index, np.ndarray):
                        post_index = np.array(range(len(row_index))).reshape(row_index.shape), col_index
                    else:
                        post_index = list(range(len(row_index))), col_index
                    col_index = tuple(s // 8 for s in col_index)
                    index = (row_index, col_index)
                elif col_index is None: # new axis
                    post_index = (slice(None), None)
                    index = (row_index,)
            elif ((isinstance(index, np.ndarray) and index.ndim == 1) or
                  (isinstance(index, list) and all(isinstance(x, int) for x in index))):
                post_index = index
                index = list(range((len(index) // 8) + 1))
            elif isinstance(index, tuple) and any(x is Ellipsis for x in index):
                post_index = index[1:]
                axis_adjustment = (slice(None),) if index[-1] is Ellipsis else (index[-1] // 8,)
                index = index[:-1] + axis_adjustment
            elif isinstance(index, tuple) and any(isinstance(x, slice) for x in index):
                post_index = index[1:]
                axis_adjustment = (slice(index.start // 8 if index.start is not None else index.start,
                                        max(index.step // 8, 1) if index.step is not None else index.step,
                                        max(index.stop // 8, 1) if index.stop is not None else index.stop)
                                   if isinstance(index[-1], slice) else (index[-1] // 8,))
                index = index[:-1] + axis_adjustment
        elif isinstance(index, slice):
            if self.ndim > 1:
                # Rows aren't packed, so we can index normally
                post_index = slice(None)
            if len(self.shape) == 1:
                # Array is 1-D, so we need to adjust
                post_index = index
                index = slice(index.start // 8 if index.start is not None else index.start,
                              max(index.step // 8, 1) if index.step is not None else index.step,
                              max(index.stop // 8, 1) if index.stop is not None else index.stop)
        elif isinstance(index, int):
            post_index = index
            index //= 8

        return index, post_index

    def get_unpacked_slice(self, index):
        # Numpy indexing is handled primarily in https://github.com/numpy/numpy/blob/maintenance/1.26.x/numpy/core/src/multiarray/mapping.c#L1435
        index, post_index = self.get_index_parameters(index)
        if post_index is NotImplemented:
            raise NotImplementedError(f"The following indexing scheme is not supported:\n{index}\n"
                                      "If you believe this scheme should be supported, "
                                      "please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\n"
                                      "If you'd like to perform this operation on the data, you should first call "
                                      "`array = array.view(np.ndarray)` and then call the function."
                                      )

        packed = self.view(np.ndarray)[index]
        if np.isscalar(packed):
            packed = GF2BP([packed], self._axis_count).view(np.ndarray)
        if packed.ndim == 1 and self.ndim > 1:
            packed = packed[:, None]
        unpacked = np.unpackbits(packed, axis=-1, count=self._axis_count)
        return GF2._view(unpacked[post_index])

    def __getitem__(self, item):
        return self.get_unpacked_slice(item)

    def set_unpacked_slice(self, index, value):
        assert not isinstance(value, GF2BP)

        packed_index, post_index = self.get_index_parameters(index)

        packed = self.view(np.ndarray)[packed_index]
        if np.isscalar(packed):
            packed = GF2BP([packed], self._axis_count).view(np.ndarray)
        if packed.ndim == 1 and self.ndim > 1:
            packed = packed[:, None]

        unpacked = np.unpackbits(packed, axis=-1, count=self._axis_count)
        unpacked[post_index] = value
        repacked = np.packbits(unpacked.view(np.ndarray), axis=-1)

        self.view(np.ndarray)[packed_index] = repacked[packed_index]

    def __setitem__(self, item, value):
        self.set_unpacked_slice(item, value)


GF2._default_ufunc_mode = "jit-calculate"
GF2._ufunc_modes = ["jit-calculate", "python-calculate"]
GF2.compile("auto")

# GF2BP._default_ufunc_mode = "jit-calculate"
# GF2BP._ufunc_modes = ["jit-calculate", "python-calculate"]
# GF2BP.compile("auto")
