"""
A module that contains a NumPy ufunc dispatcher and an Array mixin class that override NumPy ufuncs. The ufunc
dispatcher classes have snake_case naming because they are act like functions.
"""
from __future__ import annotations

from typing import Type, Callable, TYPE_CHECKING
from typing_extensions import Literal

import numba
import numpy as np

from ._linalg import matmul_jit
from ._meta import ArrayMeta

if TYPE_CHECKING:
    from ._array import Array


class UFunc:
    """
    A ufunc dispatcher for Array objects. The dispatcher will invoke a JIT-compiled or pure-Python ufunc depending on the size
    of the Galois field or Galois ring.
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
        if self.field.ufunc_mode == "jit-lookup" and not self.always_calculate:
            return self.jit_lookup
        elif self.field.ufunc_mode == "python-calculate":
            return self.python_calculate
        else:
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
                self._CACHE_CALCULATE[key_1][key_2] = numba.vectorize(["int64(int64)"], nopython=True)(self.calculate)
            else:
                self._CACHE_CALCULATE[key_1][key_2] = numba.vectorize(["int64(int64, int64)"], nopython=True)(self.calculate)

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
        else:
            return np.frompyfunc(self.calculate, 2, 1)

    ###############################################################################
    # Input/output verification
    ###############################################################################

    def _verify_unary_method_not_reduction(self, ufunc, method):
        if method in ["reduce", "accumulate", "reduceat", "outer"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Reduction methods are only supported on binary functions.")

    def _verify_binary_method_not_reduction(self, ufunc, method):
        if method in ["reduce", "accumulate", "reduceat"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r} because it takes inputs with type {self.field.name} array and integer array. Different types do not support reduction.")

    def _verify_method_only_call(self, ufunc, method):
        if not method == "__call__":
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Only '__call__' is supported.")

    def _verify_operands_in_same_field(self, ufunc, inputs, meta):
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(f"Operation {ufunc.__name__!r} requires both operands to be {self.field.name} arrays, not {[inputs[i] for i in meta['operands']]}.")

    def _verify_operands_in_field_or_int(self, ufunc, inputs, meta):
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if self.field.dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
            else:
                raise TypeError(f"Operation {ufunc.__name__!r} requires operands that are not {self.field.name} arrays to be integers or an integer np.ndarray, not {type(inputs[i])}.")

    def _verify_operands_first_field_second_int(self, ufunc, inputs, meta):
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the first operand to be a {self.field.name} array, not {meta['types'][meta['operands'][0]]}.")
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer array, not {meta['types'][meta['operands'][1]]}.")

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return
        # elif type(second) is np.ndarray:
        #     if not np.issubdtype(second.dtype, np.integer):
        #         raise ValueError(f"Operation {ufunc.__name__!r} requires the second operand with type np.ndarray to have integer dtype, not {second.dtype}.")
        elif isinstance(second, np.ndarray):
            if self.field.dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
        else:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer or integer np.ndarray, not {type(second)}.")

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
        else:
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
        elif isinstance(output, np.ndarray):
            return field._view(output.astype(dtype))
        elif output is None:
            return None
        else:
            return field(output, dtype=dtype)


###############################################################################
# Default ufunc dispatchers that simply invoke other ufuncs
###############################################################################

class divmod_ufunc(UFunc):
    """
    Default division with remainder ufunc dispatcher.
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
    """
    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        # Perform dummy addition operation to get shape of output zeros
        x = getattr(np.add, method)(*inputs, **kwargs)
        output = self.field.Zeros(x.shape, dtype=meta["dtype"])
        return output


class square_ufunc(UFunc):
    """
    Default squaring ufunc dispatcher.
    """
    type = "unary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):
        self._verify_unary_method_not_reduction(ufunc, method)
        inputs = list(inputs) + [2]
        inputs, kwargs = self._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(self.field._power.ufunc, method)(*inputs, **kwargs)
        output = self._view_output_as_field(output, self.field, meta["dtype"])
        return output


class matmul_ufunc(UFunc):
    """
    Default matrix multiplication ufunc dispatcher.
    """
    type = "binary"

    def __call__(self, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
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
        np.log2, np.log10,
        np.exp, np.expm1, np.exp2,
        np.sin, np.cos, np.tan,
        np.sinh, np.cosh, np.tanh,
        np.arcsin, np.arccos, np.arctan,
        np.arcsinh, np.arccosh, np.arctanh,
        np.degrees, np.radians,
        np.deg2rad, np.rad2deg,
        np.floor, np.ceil, np.trunc, np.rint,
    # Binary
        np.hypot, np.arctan2,
        np.logaddexp, np.logaddexp2,
        np.fmod, np.modf,
        np.fmin, np.fmax,
    ]

    _UFUNCS_REQUIRING_VIEW = [
        np.bitwise_and, np.bitwise_or, np.bitwise_xor,
        np.left_shift, np.right_shift,
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

    # These ufuncs are implementation independent
    _divmod: UFunc
    _remainder: UFunc
    _square: UFunc
    _matmul: UFunc

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._divmod = divmod_ufunc(cls)
        cls._remainder = remainder_ufunc(cls)
        cls._square = square_ufunc(cls)
        cls._matmul = matmul_ufunc(cls)

    ###############################################################################
    # Lookup table construction
    ###############################################################################

    @classmethod
    def _build_lookup_tables(cls):
        """
        Construct EXP, LOG, and ZECH_LOG lookup tables to be used in the "lookup" arithmetic functions
        """
        # TODO: Make this faster by using JIT-compiled ufuncs and vector arithmetic, when possible
        primitive_element = int(cls._primitive_element)
        add = cls._add.python_calculate
        multiply = cls._multiply.python_calculate

        cls._EXP = np.zeros(2*cls.order, dtype=np.int64)
        cls._LOG = np.zeros(cls.order, dtype=np.int64)
        cls._ZECH_LOG = np.zeros(cls.order, dtype=np.int64)
        if cls.characteristic == 2:
            cls._ZECH_E = 0
        else:
            cls._ZECH_E = (cls.order - 1) // 2

        element = 1
        cls._EXP[0] = element
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, cls.order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element = multiply(element, primitive_element)
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < cls.order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, cls.order):
            one_plus_element = add(1, cls._EXP[i])
            cls._ZECH_LOG[i] = cls._LOG[one_plus_element]

        if not cls._EXP[cls.order - 1] == 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not cyclic with size {cls.order - 1}, which means the primitive element {cls._primitive_element} does not have multiplicative order {cls.order - 1} and therefore isn't a multiplicative generator for {cls.name}.")
        if not len(set(cls._EXP[0:cls.order - 1])) == cls.order - 1:
            raise RuntimeError(f"The anti-log lookup table for {cls.name} is not unique, which means the primitive element {cls._primitive_element} has order less than {cls.order - 1} and is not a multiplicative generator of {cls.name}.")
        if not len(set(cls._LOG[1:cls.order])) == cls.order - 1:
            raise RuntimeError(f"The log lookup table for {cls.name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[cls.order:2*cls.order] = cls._EXP[1:1 + cls.order]

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

        elif ufunc in field._UNSUPPORTED_UFUNCS:
            raise NotImplementedError(f"The NumPy ufunc {ufunc.__name__!r} is not supported on {field.name} arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            if ufunc in [np.bitwise_and, np.bitwise_or, np.bitwise_xor] and method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            inputs, kwargs = UFunc(field)._view_inputs_as_ndarray(inputs, kwargs)
            output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

            if ufunc in field._UFUNCS_REQUIRING_VIEW and output is not None:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

            return output

    def __pow__(self, other):
        # We call power here instead of `super().__pow__(other)` because when doing so `x ** GF(2)` will invoke `np.square(x)`
        # and not throw a TypeError. This way `np.power(x, GF(2))` is called which correctly checks whether the second argument
        # is an integer.
        return np.power(self, other)
