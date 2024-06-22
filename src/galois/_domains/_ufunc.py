"""
A module that contains a NumPy ufunc dispatcher and an Array mixin class that override NumPy ufuncs. The ufunc
dispatcher classes have snake_case naming because they are act like functions.

The ufunc dispatchers (eg `add_ufunc`) and the UFuncMixin will be subclassed in `_lookup.py` to add the lookup
arithmetic and lookup table construction. Then in `_calculate.py` the ufunc dispatchers will be subclassed to add
unique explicit calculation algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Type

import numba
import numpy as np
from typing_extensions import Literal

from ._linalg import matmul_jit
from ._meta import ArrayMeta

if TYPE_CHECKING:
    from ._array import Array


class UFunc:
    """
    A ufunc dispatcher for Array objects. The dispatcher will invoke a JIT-compiled or pure-Python ufunc depending
    on the size of the Galois field or Galois ring.
    """

    type: Literal["unary", "binary"]

    _CACHE_CALCULATE = {}  # A cache of compiled ufuncs using explicit calculation
    _CACHE_LOOKUP = {}  # A cache of compiled ufuncs using lookup tables

    def __init__(self, field: Type[Array], override=None, always_calculate=False):
        self.field = field
        self.override = override  # A NumPy ufunc used instead of a custom one
        self.always_calculate = always_calculate  # Indicates to never use lookup tables for this ufunc

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        """
        Invokes the ufunc, either JIT-compiled or pure-Python, performing necessary verification and conversions.
        """
        raise NotImplementedError

    def set_calculate_globals(self):
        """
        Sets the global variables used in `calculate()` before JIT compiling it or before invoking it in pure Python.
        """
        return

    def set_lookup_globals(self):
        """
        Sets the global variables used in `lookup()` before JIT compiling it or before invoking it in pure Python.
        """
        return

    calculate: Callable
    """The explicit calculation implementation."""

    lookup: Callable
    """The lookup table implementation."""

    ###############################################################################
    # Various ufuncs based on implementation and compilation
    ###############################################################################

    @property
    def ufunc(self):
        """
        A ufunc based on the current state of `ufunc_mode`.
        """
        if self.field.ufunc_mode == "python-calculate":
            return self.python_calculate
        if self.field.ufunc_mode == "jit-lookup" and not self.always_calculate:
            return self.jit_lookup
        return self.jit_calculate

    @property
    def ufunc_call_only(self):
        """
        A ufunc (that only supports the `__call__()` method) based on the current state of `ufunc_mode`.

        This ufunc is the same as `ufunc`, however it fixes a bug in overridden pure-Python ufuncs
        (see https://github.com/mhostetter/galois/issues/358). This ufunc should be used in JIT function
        implementations (`Function.implementation`) to ensure bug #358 does not manifest.
        """
        if self.field.ufunc_mode == "python-calculate":
            # Specify `dtype=np.object_` for overridden ufuncs so Python int objects are returned, not np.intc (which
            # will eventually overflow and produce incorrect results).
            return self.python_calculate_call_only
        if self.field.ufunc_mode == "jit-lookup" and not self.always_calculate:
            return self.jit_lookup
        return self.jit_calculate

    @property
    def jit_calculate(self) -> numba.types.FunctionType:
        """
        A JIT-compiled ufunc implemented using explicit calculation.
        """
        if self.override:
            return self.override

        key_1 = (self.field.characteristic, self.field.degree, int(self.field.irreducible_poly))
        key_2 = str(self.__class__)
        self._CACHE_CALCULATE.setdefault(key_1, {})

        if key_2 not in self._CACHE_CALCULATE[key_1]:
            self.set_calculate_globals()  # Set the globals once before JIT compiling the function

            if self.type == "unary":
                ufunc = numba.vectorize(["int64(int64)"], nopython=True)(self.calculate)
            else:
                ufunc = numba.vectorize(["int64(int64, int64)"], nopython=True)(self.calculate)
            self._CACHE_CALCULATE[key_1][key_2] = ufunc

        return self._CACHE_CALCULATE[key_1][key_2]

    @property
    def jit_lookup(self) -> numba.types.FunctionType:
        """
        A JIT-compiled ufunc implemented using lookup tables.
        """
        if self.override:
            return self.override

        key_1 = (self.field.characteristic, self.field.degree, int(self.field.irreducible_poly))
        key_2 = (str(self.__class__), int(self.field.primitive_element))
        self._CACHE_LOOKUP.setdefault(key_1, {})

        if key_2 not in self._CACHE_LOOKUP[key_1]:
            # Ensure the lookup tables were created
            assert self.field._EXP.size > 0
            assert self.field._LOG.size > 0
            assert self.field._ZECH_LOG.size > 0
            self.set_lookup_globals()  # Set the globals once before JIT compiling the function

            if self.type == "unary":
                self._CACHE_LOOKUP[key_1][key_2] = numba.vectorize(["int64(int64)"], nopython=True)(self.lookup)
            else:
                self._CACHE_LOOKUP[key_1][key_2] = numba.vectorize(["int64(int64, int64)"], nopython=True)(self.lookup)

        return self._CACHE_LOOKUP[key_1][key_2]

    @property
    def python_calculate(self) -> Callable:
        """
        A pure-Python ufunc implemented using explicit calculation.
        """
        if self.override:
            return self.override

        self.set_calculate_globals()  # Set the globals each time before invoking the pure-Python function

        if self.type == "unary":
            return np.frompyfunc(self.calculate, 1, 1)

        return np.frompyfunc(self.calculate, 2, 1)

    @property
    def python_calculate_call_only(self) -> Callable:
        """
        A pure-Python ufunc (that only supports the `__call__()` method) implemented using explicit calculation.

        This ufunc is the same as `python_calculate_call_only`, however it fixes a bug in overridden pure-Python ufuncs
        (see https://github.com/mhostetter/galois/issues/358).
        """
        if self.override:
            # Specify `dtype=np.object_` for overridden ufuncs so Python int objects are returned, not np.intc (which
            # will eventually overflow and produce incorrect results).
            return lambda *args, **kwargs: self.override(*args, **kwargs, dtype=np.object_)

        return self.python_calculate

    ###############################################################################
    # Input/output verification
    ###############################################################################

    def _verify_unary_method_not_reduction(self, ufunc, method):
        if method in ["reduce", "accumulate", "reduceat", "outer"]:
            raise ValueError(
                f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. "
                "Reduction methods are only supported on binary functions."
            )

    def _verify_binary_method_not_reduction(self, ufunc, method):
        if method in ["reduce", "accumulate", "reduceat"]:
            raise ValueError(
                f"Ufunc method {method!r} is not supported on {ufunc.__name__!r} because it takes inputs "
                f"with type {self.field.name} array and integer array. Different types do not support reduction."
            )

    def _verify_method_only_call(self, ufunc, method):
        if not method == "__call__":
            raise ValueError(
                f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Only '__call__' is supported."
            )

    def _verify_operands_in_same_field(self, ufunc, inputs, meta):
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(
                f"Operation {ufunc.__name__!r} requires both operands to be {self.field.name} arrays, "
                f"not {[inputs[i] for i in meta['operands']]}."
            )

    def _verify_operands_in_field_or_int(self, ufunc, inputs, meta):
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if self.field.dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(
                            f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have "
                            f"integer dtype, not {inputs[i].dtype}."
                        )
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(
                            f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have "
                            f"integer dtype, not {inputs[i].dtype}."
                        )
            else:
                raise TypeError(
                    f"Operation {ufunc.__name__!r} requires operands that are not {self.field.name} arrays to be "
                    f"integers or an integer np.ndarray, not {type(inputs[i])}."
                )

    def _verify_operands_first_field_second_int(self, ufunc, inputs, meta):
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(
                f"Operation {ufunc.__name__!r} requires the first operand to be a {self.field.name} array, "
                f"not {meta['types'][meta['operands'][0]]}."
            )
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(
                f"Operation {ufunc.__name__!r} requires the second operand to be an integer array, "
                f"not {meta['types'][meta['operands'][1]]}."
            )

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return

        if isinstance(second, np.ndarray):
            if self.field.dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(
                        f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, "
                        f"not {second.dtype}."
                    )
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(
                        f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, "
                        f"not {second.dtype}."
                    )
        else:
            raise TypeError(
                f"Operation {ufunc.__name__!r} requires the second operand to be an integer or integer np.ndarray, "
                f"not {type(second)}."
            )

    ###############################################################################
    # Input/output conversion
    ###############################################################################

    def _convert_inputs_to_vector(self, inputs, kwargs):
        v_inputs = list(inputs)
        for i in range(len(inputs)):
            if issubclass(type(inputs[i]), self.field):
                v_inputs[i] = inputs[i].vector()

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if issubclass(type(output), self.field):
                    o = output.vector()
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _convert_output_from_vector(self, output, dtype):
        if output is None:
            return None

        return self.field.Vector(output, dtype=dtype)

    ###############################################################################
    # Input/output type viewing
    ###############################################################################

    def _view_inputs_as_ndarray(self, inputs, kwargs, dtype=None):
        # View all inputs that are FieldArrays as np.ndarray to avoid infinite recursion
        v_inputs = list(inputs)
        for i in range(len(inputs)):
            if issubclass(type(inputs[i]), self.field):
                v_inputs[i] = inputs[i].view(np.ndarray) if dtype is None else inputs[i].view(np.ndarray).astype(dtype)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if issubclass(type(output), self.field):
                    o = output.view(np.ndarray) if dtype is None else output.view(np.ndarray).astype(dtype)
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    def _view_output_as_field(self, output, field, dtype):
        if isinstance(type(output), field):
            return output
        if isinstance(output, np.ndarray):
            return field._view(output.astype(dtype))
        if output is None:
            return None
        return field(output, dtype=dtype)


###############################################################################
# Basic ufunc dispatchers, but they still need need lookup and calculate
# arithmetic implemented
###############################################################################


class add_ufunc(UFunc):
    """
    Default addition ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class negative_ufunc(UFunc):
    """
    Default additive inverse ufunc dispatcher.
    """

    type = "unary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class subtract_ufunc(UFunc):
    """
    Default subtraction ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class multiply_ufunc(UFunc):
    """
    Default multiplication ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        if len(meta["non_field_operands"]) > 0:
            # Scalar multiplication
            self._verify_operands_in_field_or_int(ufunc, inputs, meta)
            inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
            i = meta["non_field_operands"][0]  # Scalar multiplicand
            if meta["dtype"] == np.object_:
                # Need to explicitly cast to np.object_ in NumPy v2.0 or the integer will overflow
                inputs[i] = np.mod(inputs[i], self.field.characteristic, dtype=np.object_)
            else:
                inputs[i] = np.mod(inputs[i], self.field.characteristic)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class reciprocal_ufunc(UFunc):
    """
    Default multiplicative inverse ufunc dispatcher.
    """

    type = "unary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class divide_ufunc(UFunc):
    """
    Default division ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        if method == "__call__":
            # When dividing two arrays, instead multiply by the reciprocal. This is vastly
            # more efficient when the denominator is a scalar or smaller (broadcasted) array.
            inputs[1] = self.field._reciprocal.ufunc(inputs[1])
            output = getattr(self.field._multiply.ufunc, method)(*inputs, **kwargs)
        else:
            output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class divmod_ufunc(UFunc):
    """
    Default division with remainder ufunc dispatcher.

    NOTE: This does not need its own implementation. Instead, it invokes other ufuncs.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        q = getattr(np.divide, method)(*inputs, **kwargs)
        r = self.field.Zeros(q.shape, dtype=meta["dtype"])
        output = q, r
        return output


# TODO: Fix this atrocity
class remainder_ufunc(UFunc):
    """
    Default remainder ufunc dispatcher.

    NOTE: This does not need its own implementation. Instead, it invokes other ufuncs.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        # Perform dummy addition operation to get shape of output zeros
        x = getattr(np.add, method)(*inputs, **kwargs)
        output = self.field.Zeros(x.shape, dtype=meta["dtype"])
        return output


class power_ufunc(UFunc):
    """
    Default exponentiation ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_binary_method_not_reduction(ufunc, method)
        self._verify_operands_first_field_second_int(ufunc, inputs, meta)
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class square_ufunc(UFunc):
    """
    Default squaring ufunc dispatcher.

    NOTE: This does not need its own implementation. Instead, it invokes other ufuncs.
    """

    type = "unary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_unary_method_not_reduction(ufunc, method)
        inputs = list(inputs) + [2]
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.field._power.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class log_ufunc(UFunc):
    """
    Default logarithm ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_method_only_call(ufunc, method)
        inputs = list(inputs) + [int(self.field.primitive_element)]
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.ufunc, method)(*inputs, **kwargs)
        if output.dtype == np.object_:
            output = output.astype(int)
        return output


class sqrt_ufunc(UFunc):
    """
    Default square root ufunc dispatcher.
    """

    type = "unary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_method_only_call(ufunc, method)
        return self.implementation(*inputs)

    def implementation(self, a: Array) -> Array:
        """
        Computes the square root of an element in a Galois field or Galois ring.
        """
        raise NotImplementedError


class matmul_ufunc(UFunc):
    """
    Default matrix multiplication ufunc dispatcher.
    """

    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_method_only_call(ufunc, method)
        return matmul_jit(self.field)(*inputs, **kwargs)


###############################################################################
# Array mixin class
###############################################################################


class UFuncMixin(np.ndarray, metaclass=ArrayMeta):
    """
    An Array mixin class that overrides the invocation of NumPy ufuncs on Array objects.
    """

    _UNSUPPORTED_UFUNCS = [
        # Unary
        np.invert,
        np.log2,
        np.log10,
        np.exp,
        np.expm1,
        np.exp2,
        np.sin,
        np.cos,
        np.tan,
        np.sinh,
        np.cosh,
        np.tanh,
        np.arcsin,
        np.arccos,
        np.arctan,
        np.arcsinh,
        np.arccosh,
        np.arctanh,
        np.degrees,
        np.radians,
        np.deg2rad,
        np.rad2deg,
        np.floor,
        np.ceil,
        np.trunc,
        np.rint,
        # Binary
        np.hypot,
        np.arctan2,
        np.logaddexp,
        np.logaddexp2,
        np.fmod,
        np.modf,
        np.fmin,
        np.fmax,
    ]

    _UFUNCS_REQUIRING_VIEW = [
        np.bitwise_and,
        np.bitwise_or,
        np.bitwise_xor,
        np.left_shift,
        np.right_shift,
        np.positive,
    ]

    _OVERRIDDEN_UFUNCS = {
        np.add: "_add",
        np.negative: "_negative",
        np.subtract: "_subtract",
        np.multiply: "_multiply",
        np.reciprocal: "_reciprocal",
        np.floor_divide: "_divide",
        np.true_divide: "_divide",
        np.divmod: "_divmod",
        np.remainder: "_remainder",
        np.power: "_power",
        np.square: "_square",
        np.log: "_log",
        np.sqrt: "_sqrt",
        np.matmul: "_matmul",
    }

    # These need to be set in subclasses. Depending on the type of Galois field, different arithmetic must be used.
    _add: UFunc
    _negative: UFunc
    _subtract: UFunc
    _multiply: UFunc
    _reciprocal: UFunc
    _divide: UFunc
    _power: UFunc
    _log: UFunc
    _sqrt: UFunc

    # Special helper ufunc dispatchers that are used in other ufuncs
    _positive_power: UFunc

    # These ufuncs are implementation independent
    _divmod: UFunc
    _remainder: UFunc
    _square: UFunc
    _matmul: UFunc

    @classmethod
    def _assign_ufuncs(cls):
        cls._divmod = divmod_ufunc(cls)
        cls._remainder = remainder_ufunc(cls)
        cls._square = square_ufunc(cls)
        cls._matmul = matmul_ufunc(cls)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override the standard NumPy ufunc calls with the new finite field ufuncs.
        """
        field = type(self)

        meta = {}
        meta["types"] = [type(inputs[i]) for i in range(len(inputs))]
        meta["operands"] = list(range(len(inputs)))
        if method in ["at", "reduceat"]:
            # Remove the second argument for "at" ufuncs which is the indices list
            meta["operands"].pop(1)
        meta["field_operands"] = [i for i in meta["operands"] if isinstance(inputs[i], self.__class__)]
        meta["non_field_operands"] = [i for i in meta["operands"] if not isinstance(inputs[i], self.__class__)]
        self.field = self.__class__
        meta["dtype"] = self.dtype
        # meta["ufuncs"] = self._ufuncs

        if ufunc in field._OVERRIDDEN_UFUNCS:
            # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
            # to integers. We know this is safe because we already verified the inputs.
            if method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            # Need to set the intermediate dtype for reduction operations or an error will be thrown. We
            # use the largest valid dtype for this field.
            if method in ["reduce"]:
                kwargs["dtype"] = field.dtypes[-1]

            return getattr(field, field._OVERRIDDEN_UFUNCS[ufunc])(ufunc, method, inputs, kwargs, meta)

        if ufunc in field._UNSUPPORTED_UFUNCS:
            raise NotImplementedError(
                f"The NumPy ufunc {ufunc.__name__!r} is not supported on {field.name} arrays. "
                "If you believe this ufunc should be supported, "
                "please submit a GitHub issue at https://github.com/mhostetter/galois/issues."
            )

        # Process with our custom ufuncs
        if ufunc in [np.bitwise_and, np.bitwise_or, np.bitwise_xor] and method not in [
            "reduce",
            "accumulate",
            "at",
            "reduceat",
        ]:
            kwargs["casting"] = "unsafe"

        inputs, kwargs = UFunc(field)._view_inputs_as_ndarray(inputs, kwargs)
        output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

        if ufunc in field._UFUNCS_REQUIRING_VIEW and output is not None:
            output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

        return output

    def __pow__(self, other):
        # We call power here instead of `super().__pow__(other)` because when doing so `x ** GF(2)` will invoke
        # `np.square(x)` and not throw a TypeError. This way `np.power(x, GF(2))` is called which correctly checks
        # whether the second argument is an integer.
        return np.power(self, other)
