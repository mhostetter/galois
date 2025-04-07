"""
A module that defines the GF(2) array class.
"""

from __future__ import annotations

import operator
from functools import reduce
from math import ceil, floor
from typing import Final, Optional, Sequence

import numpy as np
from typing_extensions import Literal, Self

from .._domains._array import Array
from .._domains._function import Function
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


def packbits(a, axis=None, bitorder="big"):
    if isinstance(a, GF2BP):
        return a

    if not isinstance(a, GF2):
        raise TypeError("Bit-packing is only supported on instances of GF2.")

    axis = GF2BP.DEFAULT_AXIS if axis is None else axis
    axis_element_count = 1 if a.ndim == 0 else a.shape[axis]
    packed = GF2BP(np.packbits(a.view(np.ndarray), axis=axis, bitorder=bitorder), axis, axis_element_count)
    return packed


def unpackbits(a, axis=None, count=None, bitorder="big"):
    if isinstance(a, GF2):
        return a

    if not isinstance(a, GF2BP):
        raise TypeError("Bit-unpacking is only supported on instances of GF2BP.")

    if axis is None:
        axis = a._axis

    if axis != a._axis:
        raise ValueError(f"You are requesting to unpack along a different axis ({axis} vs {a._axis})!")

    return GF2(
        np.unpackbits(a.view(np.ndarray), axis=axis, count=a._axis_count if count is None else count, bitorder=bitorder)
    )


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
        result_shape = np.broadcast_shapes(*(i.shape for i in inputs))
        if any(i.view(np.ndarray).shape != inputs[0].view(np.ndarray).shape for i in inputs):
            # We can't do simple bitwise addition when the shapes aren't the same due to broadcasting
            inputs = [np.unpackbits(i) for i in inputs]
            output = reduce(operator.add, inputs)  # We need this to use GF2's default addition
            output = np.packbits(output)
        else:
            output = super().__call__(ufunc, method, inputs, kwargs, meta)
            output._axis = inputs[0]._axis
        output._axis_count = result_shape[output._axis]
        return output


class subtract_ufunc_bitpacked(subtract_ufunc):
    """
    Subtraction ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        result_shape = np.broadcast_shapes(*(i.shape for i in inputs))
        if any(i.view(np.ndarray).shape != inputs[0].view(np.ndarray).shape for i in inputs):
            # We can't do simple bitwise subtraction when the shapes aren't the same due to broadcasting
            inputs = [np.unpackbits(i) for i in inputs]
            output = reduce(operator.sub, inputs)  # We need this to use GF2's default subtraction
            output = np.packbits(output)
        else:
            output = super().__call__(ufunc, method, inputs, kwargs, meta)
            output._axis = inputs[0]._axis
        output._axis_count = result_shape[GF2BP.DEFAULT_AXIS]
        return output


class multiply_ufunc_bitpacked(multiply_ufunc):
    """
    Multiply ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        is_outer_product = method == "outer"
        if is_outer_product and np.all(len(i.shape) == 1 for i in inputs):
            result_shape = reduce(operator.add, (i.shape for i in inputs))
        else:
            result_shape = np.broadcast_shapes(*(i.shape for i in inputs))

        output_axis = GF2BP.DEFAULT_AXIS
        if is_outer_product:
            assert len(inputs) == 2
            # Unpack the first argument and propagate the bitpacked second argument
            output_axis = inputs[-1]._axis
            inputs = [np.unpackbits(x).view(np.ndarray) if i == 0 else x.view(np.ndarray) for i, x in enumerate(inputs)]
            output = np.multiply.outer(*inputs)
        else:
            if any(i.view(np.ndarray).shape != inputs[0].view(np.ndarray).shape for i in inputs):
                # We can't do simple bitwise multiplication when the shapes aren't the same due to broadcasting
                inputs = [np.unpackbits(i) for i in inputs]
                output = reduce(operator.mul, inputs)  # We need this to use GF2's default multiply
                output = np.packbits(output)
            else:
                output = super().__call__(ufunc, method, inputs, kwargs, meta)
                output_axis = inputs[0]._axis

        output = self.field._view(output)
        output._axis = output_axis
        output._axis_count = result_shape[output_axis]
        return output


class divide_ufunc_bitpacked(divide):
    """
    Divide ufunc dispatcher w/ support for bit-packed fields.
    """

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        result_shape = np.broadcast_shapes(*(i.shape for i in inputs))
        if any(i.view(np.ndarray).shape != inputs[0].view(np.ndarray).shape for i in inputs):
            # We can't do simple bitwise division when the shapes aren't the same due to broadcasting
            inputs = [np.unpackbits(i) for i in inputs]
            output = reduce(operator.truediv, inputs)  # We need this to use GF2's default division
            output = np.packbits(output)
        else:
            output = super().__call__(ufunc, method, inputs, kwargs, meta)
            output._axis = inputs[0]._axis
        output._axis_count = result_shape[GF2BP.DEFAULT_AXIS]
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
                np.unpackbits(b.view(np.ndarray), axis=b._axis, count=b._axis_count),
                axis=0,
            )
        )
        b._axis = 0
        b._axis_count = row_axis_count

        # Make sure the inner dimensions match (e.g. (M, N) x (N, P) -> (M, P))
        a_packed_shape = a.view(np.ndarray).shape
        b_packed_shape = b.view(np.ndarray).shape
        if a_packed_shape[-1] != b_packed_shape[0]:
            # We can't do utilize packed m-v or m-m multiplication when the shapes aren't the same
            inputs = [np.unpackbits(i) for i in inputs]
            output = reduce(operator.matmul, inputs)  # We need this to use GF2's default matmul
            output = np.packbits(output, axis=a._axis)
        else:
            if len(b_packed_shape) == 1:
                final_shape = (a_packed_shape[0],)
            else:
                final_shape = (a_packed_shape[0], b_packed_shape[-1])

            if len(b.shape) == 1:
                # matrix-vector multiplication
                output = np.bitwise_xor.reduce(np.unpackbits((a & b).view(np.ndarray), axis=-1), axis=-1)
                output_axis = b._axis  # Output is a vector, so we use that axis (i.e. 0)
            else:
                # matrix-matrix multiplication
                output = GF2.Zeros(final_shape)
                for i in range(b_packed_shape[-1]):
                    # TODO: Include alternate path for numpy < v2
                    # output[:, i] = np.bitwise_xor.reduce(np.unpackbits((a & b[:, i]).view(np.ndarray), axis=-1), axis=-1)
                    output[:, i] = (
                        np.bitwise_xor.reduce(
                            np.bitwise_count((a & b.view(np.ndarray)[:, i]).view(np.ndarray)), axis=-1
                        )
                        % 2
                    )
                output_axis = a._axis

            output = field._view(np.packbits(output.view(np.ndarray), axis=output_axis))
            output._axis = output_axis
            output._axis_count = final_shape[output_axis]

        return output


class concatenate_bitpacked(Function):
    """Concatenates matrices together"""

    def __call__(self, arrays: list[Array], **kwargs):
        # TODO: Should we only unpack arrays that have column counts %8 != 0 and concatenate those first?
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
        cls._reciprocal = not_implemented  # reciprocal(cls)
        cls._divide = divide_ufunc_bitpacked(cls)
        cls._power = not_implemented  # power(cls)
        cls._log = not_implemented  # log(cls)
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

    DEFAULT_AXIS = -1  # The last axis
    BIT_WIDTH = 8

    def __new__(
        cls,
        x: ElementLike | ArrayLike,
        axis: int = DEFAULT_AXIS,
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
            array._axis = axis
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
        axis: int = DEFAULT_AXIS,
        axis_element_count: Optional[int] = None,
        dtype: DTypeLike | None = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0,
    ):
        pass

    def __array_finalize__(self, obj):
        """
        A NumPy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        super().__array_finalize__(obj)

        # Pass along attribute if they have been set already
        try:
            self._axis = obj._axis
            self._axis_count = obj._axis_count
        except AttributeError:
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

    @staticmethod
    def _normalize_indexing_to_tuple(index, shape, axis=0):
        """
        Normalize indexing into a tuple of positive-only slices, integers, and/or new axes.
        NOTE: Ellipsis indexing is converted to slice indexing.

        Args:
            index: The indexing expression (int, slice, list, etc.).
            shape: Tuple of integers representing the shape of the object being indexed.

        Returns:
            A tuple of positive integers, slices, and/or new axes.
        """
        if not isinstance(shape, tuple):
            raise TypeError("Shape must be a tuple of integers.")

        ndim = len(shape)

        if isinstance(index, int):
            if index < 0:
                index += shape[axis]
            return (index,)
        elif isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            step = step if step is not None else 1

            if step > 0:
                start = start if start is not None else 0
                stop = stop if stop is not None else shape[axis]

                # Adjust negative start/stop values
                if start < 0:
                    start += shape[axis]
                if stop < 0:
                    stop += shape[axis]
            else:
                start = start if start is not None else shape[axis] - 1
                stop = stop if stop is not None else -shape[axis] - 1

            return (slice(start, stop, step),)
        elif isinstance(index, list):
            for axis, i in enumerate(index):
                if i < 0:
                    index[axis] += shape[axis]
            return (index,)
        elif index is np.newaxis:
            return (index,)
        elif isinstance(index, tuple):
            normalized = []

            num_explicit_dims = sum(1 for i in index if i is not Ellipsis)
            for i in index:
                if i is Ellipsis:
                    span = ndim - num_explicit_dims
                    expanded_dims = [slice(None)] * span
                    for e_axis, e in enumerate(expanded_dims):
                        expanded_dims[e_axis] = GF2BP._normalize_indexing_to_tuple(e, shape, axis)[0]
                        axis += 1
                    normalized.extend(expanded_dims)
                else:
                    i_normalized = GF2BP._normalize_indexing_to_tuple(i, shape, axis)
                    normalized.extend(i_normalized)
                    if i_normalized is not (np.newaxis,):
                        axis += 1

            return tuple(normalized)
        elif isinstance(index, (Sequence, np.ndarray)):
            # Boolean mask or fancy indexing can remain as-is
            return (index,)
        else:
            raise TypeError(f"Unsupported indexing type: {type(index)}")

    def get_index_parameters(self, index):
        normalized_index = self._normalize_indexing_to_tuple(index, self.shape)

        assert isinstance(normalized_index, tuple)

        bit_width: Final[int] = self.BIT_WIDTH
        packed_shape = self.view(np.ndarray).shape
        packed_axis = len(packed_shape) - 1 if self._axis == -1 else self._axis
        packed_index = tuple()
        unpacked_index = tuple()
        shape = tuple()
        axes_in_index = len(normalized_index)
        axis = 0
        for i in normalized_index:
            is_unpacked_axis = axis != packed_axis
            if isinstance(i, int):
                if is_unpacked_axis and self.ndim > 1:
                    packed_index += (i,)

                    # If we have multidimensional indexing, then we will need to re-index the same way after reshaping
                    if axes_in_index > 1:
                        unpacked_index = (0,)
                        shape += (1,)
                else:
                    packed_index += (i // bit_width,)
                    unpacked_index += (i % bit_width,)
                    if axes_in_index > 1:
                        shape += (1,)
                    else:
                        shape += (packed_shape[axis],)
            elif isinstance(i, slice):
                if is_unpacked_axis:
                    packed_index += (i,)
                    # the packed index will already filter, so we just select everything after
                    unpacked_index += (slice(None),)
                else:
                    if i.step > 0:
                        packed_index += (
                            slice(
                                i.start // bit_width, max(int(ceil(i.stop / bit_width)), 1), max(i.step // bit_width, 1)
                            ),
                        )
                        unpacked_index += (slice(i.start % bit_width, i.start % bit_width + i.stop - i.start, i.step),)
                    else:
                        packed_index += (
                            slice(
                                i.start // bit_width,
                                max(int(floor(i.stop / bit_width)), -packed_shape[axis] - 1),
                                min(i.step // bit_width, -1),
                            ),
                        )
                        unpacked_index += (slice(i.start % bit_width, i.start % bit_width + i.stop - i.start, i.step),)

                packed_slice = packed_index[-1]
                abs_step = abs(packed_slice.step)
                slice_size = max(0, (packed_slice.stop - packed_slice.start + abs_step - 1) // abs_step)
                shape += (slice_size,)
            elif isinstance(i, (Sequence, np.ndarray)):
                if is_unpacked_axis:
                    shape += (len(i),)
                    packed_index += (i,)
                    unpacked_index += (slice(None),)
                else:
                    if isinstance(index, np.ndarray) and index.dtype == np.bool:
                        mask_packed = [False] * packed_shape[axis]
                        for j, _ in enumerate(i):
                            mask_packed[j // bit_width] |= True
                        packed_index = mask_packed
                        unpacked_index = i
                        shape += (max(sum(i) // bit_width, 1),)
                    else:
                        # adjust indexing for this packed axis
                        data = np.array([s // bit_width for s in i], dtype=self.dtype)
                        # remove duplicate entries, including nested arrays
                        if data.ndim > 1:
                            rows = []
                            for _, row_data in enumerate(data):
                                _, unique_indices = np.unique(row_data, return_index=True)
                                # maintain the original order
                                rows.append(row_data[np.sort(unique_indices)])
                            data = np.vstack(rows)
                        else:
                            _, unique_indices = np.unique(data, return_index=True)
                            # maintain the original order
                            data = data[np.sort(unique_indices)]

                        packed_index += (data,)
                        shape += (packed_shape[axis],)
                        if axes_in_index == 1:
                            unpacked_index += ([s % bit_width for s in i],)
            elif i is np.newaxis:
                unpacked_index += (i,)
                axis -= 1  # Don't count new axes
            else:
                raise NotImplementedError(
                    f"The following indexing scheme is not supported:\n{index}\n"
                    "If you believe this scheme should be supported, "
                    "please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\n"
                    "If you'd like to perform this operation on the data, you should first call "
                    "`array = array.view(np.ndarray)` and then call the function."
                )

            axis += 1

        # Catch any remaining indexing for the rest of the shape if not all axes were specified.
        if axis < len(packed_shape):
            shape += (-1,)

        return packed_index, unpacked_index, shape

    @property
    def shape(self):
        # A cast to np.ndarray is needed to get the packed shape
        packed_shape = list(self.view(np.ndarray).shape)
        packed_shape[self._axis] = self._axis_count
        unpacked_shape = tuple(packed_shape)
        return unpacked_shape

    def get_unpacked_value(self, index):
        # Numpy indexing is handled primarily in https://github.com/numpy/numpy/blob/maintenance/1.26.x/numpy/core/src/multiarray/mapping.c#L1435
        packed_index, unpacked_index, shape = self.get_index_parameters(index)

        packed = self.view(np.ndarray)[packed_index]

        if np.isscalar(packed):
            packed = GF2BP([packed], self._axis, self._axis_count).view(np.ndarray)

        if len(shape) > 0:
            packed = packed.reshape(shape)

        unpacked = np.unpackbits(packed, axis=self._axis, count=self._axis_count)
        value = unpacked[unpacked_index]
        if np.isscalar(value):
            return GF2(value, dtype=self.dtype)

        return GF2._view(value)

    def __getitem__(self, item):
        return self.get_unpacked_value(item)

    def set_unpacked_value(self, index, value):
        assert not isinstance(value, GF2BP)

        packed_index, unpacked_index, shape = self.get_index_parameters(index)

        packed = self.view(np.ndarray)[packed_index]
        original_packed_shape = packed.shape

        if np.isscalar(packed):
            packed = GF2BP([packed], self._axis, self._axis_count).view(np.ndarray)

        if len(shape) > 0:
            packed = packed.reshape(shape)

        unpacked = np.unpackbits(packed, axis=self._axis, count=self._axis_count)
        unpacked[unpacked_index] = value
        repacked = np.packbits(unpacked.view(np.ndarray), axis=self._axis)
        repacked = repacked.reshape(original_packed_shape)

        self.view(np.ndarray)[packed_index] = repacked

    def __setitem__(self, item, value):
        self.set_unpacked_value(item, value)


GF2._default_ufunc_mode = "jit-calculate"
GF2._ufunc_modes = ["jit-calculate", "python-calculate"]
GF2.compile("auto")

GF2BP._default_ufunc_mode = "jit-calculate"
GF2BP._ufunc_modes = ["jit-calculate", "python-calculate"]
GF2BP.compile("auto")
