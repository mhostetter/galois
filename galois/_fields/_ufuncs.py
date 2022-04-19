"""
A module that contains a metaclass mixin that provides NumPy ufunc overriding for an ndarray subclass.
"""
import numba
from numba import int64
import numpy as np

from .._domains import Array

from ._meta import FieldMeta


class FieldCalculate(Array, metaclass=FieldMeta):
    """
    A mixin class that provides finite field arithmetic using explicit calculation. Specific implementations are
    overridded in GF2, GF2m, GFp, and GFpm.
    """
    # Function signatures for JIT-compiled explicit calculation arithmetic functions
    _UNARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64))
    _BINARY_CALCULATE_SIG = numba.types.FunctionType(int64(int64, int64, int64, int64, int64))

    _FUNC_CACHE_CALCULATE = {}
    _UFUNC_CACHE_CALCULATE = {}

    _UFUNC_TYPE = {
        "add": "binary",
        "negative": "unary",
        "subtract": "binary",
        "multiply": "binary",
        "reciprocal": "unary",
        "divide": "binary",
        "power": "binary",
        "log": "binary",
    }

    @classmethod
    def _set_globals(cls, name):
        """
        Sets the global variables in the GF*Meta metaclass mixins that are needed for linking during JIT compilation.
        """
        raise NotImplementedError

    @classmethod
    def _reset_globals(cls):
        """
        Resets the global variables in the GF*Meta metaclass mixins so when the pure-Python functions/ufuncs invoke these
        globals, they reference the correct pure-Python functions and not the JIT-compiled functions/ufuncs.
        """
        raise NotImplementedError

    @classmethod
    def _func_calculate(cls, name, reset=True):
        """
        Returns a JIT-compiled arithmetic function using explicit calculation. These functions are once-compiled and shared for all
        Galois fields. The only difference between Galois fields are the characteristic, degree, and irreducible polynomial that are
        passed in as inputs.
        """
        key = (name,)

        if key not in cls._FUNC_CACHE_CALCULATE:
            cls._set_globals(name)
            function = getattr(cls, f"_{name}_calculate")

            if cls._UFUNC_TYPE[name] == "unary":
                cls._FUNC_CACHE_CALCULATE[key] = numba.jit(["int64(int64, int64, int64, int64)"], nopython=True, cache=True)(function)
            else:
                cls._FUNC_CACHE_CALCULATE[key] = numba.jit(["int64(int64, int64, int64, int64, int64)"], nopython=True, cache=True)(function)

            if reset:
                cls._reset_globals()

        return cls._FUNC_CACHE_CALCULATE[key]

    @classmethod
    def _ufunc_calculate(cls, name):
        """
        Returns a JIT-compiled arithmetic ufunc using explicit calculation. These ufuncs are compiled for each Galois field since
        the characteristic, degree, and irreducible polynomial are compiled into the ufuncs as constants.
        """
        key = (name, cls._characteristic, cls._degree, int(cls._irreducible_poly))

        if key not in cls._UFUNC_CACHE_CALCULATE:
            cls._set_globals(name)
            function = getattr(cls, f"_{name}_calculate")

            # These variables must be locals and not class properties for Numba to compile them as literals
            characteristic = cls._characteristic
            degree = cls._degree
            irreducible_poly = int(cls._irreducible_poly)

            if cls._UFUNC_TYPE[name] == "unary":
                cls._UFUNC_CACHE_CALCULATE[key] = numba.vectorize(["int64(int64)"], nopython=True)(lambda a: function(a, characteristic, degree, irreducible_poly))
            else:
                cls._UFUNC_CACHE_CALCULATE[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(lambda a, b: function(a, b, characteristic, degree, irreducible_poly))

            cls._reset_globals()

        return cls._UFUNC_CACHE_CALCULATE[key]

    @classmethod
    def _func_python(cls, name):
        """
        Returns a pure-Python arithmetic function using explicit calculation. This lambda function wraps the arithmetic functions in
        GF2Meta, GF2mMeta, GFpMeta, and GFpmMeta by passing in the field's characteristic, degree, and irreducible polynomial.
        """
        return getattr(cls, f"_{name}_calculate")

    @classmethod
    def _ufunc_python(cls, name):
        """
        Returns a pure-Python arithmetic ufunc using explicit calculation.
        """
        function = getattr(cls, f"_{name}_calculate")

        # Pre-fetching these values into local variables allows Python to cache them as constants in the lambda function
        characteristic = cls._characteristic
        degree = cls._degree
        irreducible_poly = int(cls._irreducible_poly)

        if cls._UFUNC_TYPE[name] == "unary":
            return np.frompyfunc(lambda a: function(a, characteristic, degree, irreducible_poly), 1, 1)
        else:
            return np.frompyfunc(lambda a, b: function(a, b, characteristic, degree, irreducible_poly), 2, 1)

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError

    @staticmethod
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        raise NotImplementedError


class FieldLookup(FieldCalculate):
    """
    A mixin class that provides finite field arithmetic using lookup tables. These routines are the same
    for each finite field type. When building the lookup tables, the explicit calculation routines are used
    once.
    """
    # pylint: disable=abstract-method

    # Function signatures for JIT-compiled "lookup" arithmetic functions
    _UNARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64[:], int64[:], int64[:], int64))
    _BINARY_LOOKUP_SIG = numba.types.FunctionType(int64(int64, int64, int64[:], int64[:], int64[:], int64))

    _FUNC_CACHE_LOOKUP = {}
    _UFUNC_CACHE_LOOKUP = {}

    @classmethod
    def _build_lookup_tables(cls):
        """
        Construct EXP, LOG, and ZECH_LOG lookup tables to be used in the "lookup" arithmetic functions
        """
        primitive_element = int(cls._primitive_element)
        add = cls._ufunc_python("add")
        multiply = cls._ufunc_python("multiply")

        cls._EXP = np.zeros(2*cls._order, dtype=np.int64)
        cls._LOG = np.zeros(cls._order, dtype=np.int64)
        cls._ZECH_LOG = np.zeros(cls._order, dtype=np.int64)
        if cls._characteristic == 2:
            cls._ZECH_E = 0
        else:
            cls._ZECH_E = (cls._order - 1) // 2

        element = 1
        cls._EXP[0] = element
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, cls._order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element = multiply(element, primitive_element)
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < cls._order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, cls._order):
            one_plus_element = add(1, cls._EXP[i])
            cls._ZECH_LOG[i] = cls._LOG[one_plus_element]

        if not cls._EXP[cls._order - 1] == 1:
            raise RuntimeError(f"The anti-log lookup table for {cls._name} is not cyclic with size {cls._order - 1}, which means the primitive element {cls._primitive_element} does not have multiplicative order {cls._order - 1} and therefore isn't a multiplicative generator for {cls._name}.")
        if not len(set(cls._EXP[0:cls._order - 1])) == cls._order - 1:
            raise RuntimeError(f"The anti-log lookup table for {cls._name} is not unique, which means the primitive element {cls._primitive_element} has order less than {cls._order - 1} and is not a multiplicative generator of {cls._name}.")
        if not len(set(cls._LOG[1:cls._order])) == cls._order - 1:
            raise RuntimeError(f"The log lookup table for {cls._name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[cls._order:2*cls._order] = cls._EXP[1:1 + cls._order]

    @classmethod
    def _func_lookup(cls, name):  # pylint: disable=no-self-use
        """
        Returns an arithmetic function using lookup tables. These functions are once-compiled and shared for all Galois fields. The only difference
        between Galois fields are the lookup tables that are passed in as inputs.
        """
        key = (name,)

        if key not in cls._FUNC_CACHE_LOOKUP:
            function = getattr(cls, f"_{name}_lookup")
            if cls._UFUNC_TYPE[name] == "unary":
                cls._FUNC_CACHE_LOOKUP[key] = numba.jit("int64(int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)
            else:
                cls._FUNC_CACHE_LOOKUP[key] = numba.jit("int64(int64, int64, int64[:], int64[:], int64[:], int64)", nopython=True, cache=True)(function)

        return cls._FUNC_CACHE_LOOKUP[key]

    @classmethod
    def _ufunc_lookup(cls, name):
        """
        Returns an arithmetic ufunc using lookup tables. These ufuncs are compiled for each Galois field since the lookup tables are compiled
        into the ufuncs as constants.
        """
        key = (name, cls._characteristic, cls._degree, int(cls._irreducible_poly))

        if key not in cls._UFUNC_CACHE_LOOKUP:
            # These variables must be locals for Numba to compile them as literals
            EXP = cls._EXP
            LOG = cls._LOG
            ZECH_LOG = cls._ZECH_LOG
            ZECH_E = cls._ZECH_E

            function = getattr(cls, f"_{name}_lookup")
            if cls._UFUNC_TYPE[name] == "unary":
                cls._UFUNC_CACHE_LOOKUP[key] = numba.vectorize(["int64(int64)"], nopython=True)(lambda a: function(a, EXP, LOG, ZECH_LOG, ZECH_E))
            else:
                cls._UFUNC_CACHE_LOOKUP[key] = numba.vectorize(["int64(int64, int64)"], nopython=True)(lambda a, b: function(a, b, EXP, LOG, ZECH_LOG, ZECH_E))

        return cls._UFUNC_CACHE_LOOKUP[key]

    ###############################################################################
    # Arithmetic functions using lookup tables
    ###############################################################################

    # pylint: disable=unused-argument

    @staticmethod
    @numba.extending.register_jitable
    def _add_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a + b = α^m + α^n
              = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
              = α^m * α^ZECH_LOG(n - m)
              = α^(m + ZECH_LOG(n - m))
        """
        if a == 0:
            return b
        elif b == 0:
            return a

        m = LOG[a]
        n = LOG[b]

        if m > n:
            # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If m is
            # larger than n, switch a and b in the addition.
            m, n = n, m

        if n - m == ZECH_E:
            # zech_log(zech_e) = -Inf and α^(-Inf) = 0
            return 0
        else:
            return EXP[m + ZECH_LOG[n - m]]

    @staticmethod
    @numba.extending.register_jitable
    def _negative_lookup(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        -a = -α^m
           = -1 * α^m
           = α^e * α^m
           = α^(e + m)
        """
        if a == 0:
            return 0
        else:
            m = LOG[a]
            return EXP[ZECH_E + m]

    @staticmethod
    @numba.extending.register_jitable
    def _subtract_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a - b = α^m - α^n
              = α^m + (-α^n)
              = α^m + (-1 * α^n)
              = α^m + (α^e * α^n)
              = α^m + α^(e + n)
        """
        ORDER = LOG.size

        # Same as addition if n = log(b) + e
        m = LOG[a]
        n = LOG[b] + ZECH_E

        if b == 0:
            return a
        elif a == 0:
            return EXP[n]

        if m > n:
            # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If m is
            # larger than n, switch a and b in the addition.
            m, n = n, m

        z = n - m
        if z == ZECH_E:
            # zech_log(zech_e) = -Inf and α^(-Inf) = 0
            return 0
        if z >= ORDER - 1:
            # Reduce index of ZECH_LOG by the multiplicative order of the field `ORDER - 1`
            z -= ORDER - 1

        return EXP[m + ZECH_LOG[z]]

    @staticmethod
    @numba.extending.register_jitable
    def _multiply_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a * b = α^m * α^n
              = α^(m + n)
        """
        if a == 0 or b == 0:
            return 0
        else:
            m = LOG[a]
            n = LOG[b]
            return EXP[m + n]

    @staticmethod
    @numba.extending.register_jitable
    def _reciprocal_lookup(a, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        1 / a = 1 / α^m
              = α^(-m)
              = 1 * α^(-m)
              = α^(ORDER - 1) * α^(-m)
              = α^(ORDER - 1 - m)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        ORDER = LOG.size
        m = LOG[a]
        return EXP[(ORDER - 1) - m]

    @staticmethod
    @numba.extending.register_jitable
    def _divide_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a / b = α^m / α^n
              = α^(m - n)
              = 1 * α^(m - n)
              = α^(ORDER - 1) * α^(m - n)
              = α^(ORDER - 1 + m - n)
        """
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            ORDER = LOG.size
            m = LOG[a]
            n = LOG[b]
            return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative

    @staticmethod
    @numba.extending.register_jitable
    def _power_lookup(a, b, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b in Z

        a ** b = α^m ** b
               = α^(m * b)
               = α^(m * ((b // (ORDER - 1))*(ORDER - 1) + b % (ORDER - 1)))
               = α^(m * ((b // (ORDER - 1))*(ORDER - 1)) * α^(m * (b % (ORDER - 1)))
               = 1 * α^(m * (b % (ORDER - 1)))
               = α^(m * (b % (ORDER - 1)))
        """
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif a == 0:
            return 0
        else:
            ORDER = LOG.size
            m = LOG[a]
            return EXP[(m * b) % (ORDER - 1)]  # TODO: Do b % (ORDER - 1) first? b could be very large and overflow int64

    @staticmethod
    @numba.extending.register_jitable
    def _log_lookup(beta, alpha, EXP, LOG, ZECH_LOG, ZECH_E):  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        log(beta, α) = log(α^m, α)
                     = m
        """
        if beta == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        return LOG[beta]


class FieldUfunc(FieldLookup, FieldCalculate):
    """
    A mixin class that overrides NumPy ufuncs to perform finite field arithmetic, using either lookup tables or explicit
    calculation.
    """
    # pylint: disable=no-member

    _UNSUPPORTED_UFUNCS_UNARY = [
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
    ]

    _UNSUPPORTED_UFUNCS_BINARY = [
        np.hypot, np.arctan2,
        np.logaddexp, np.logaddexp2,
        np.fmod, np.modf,
        np.fmin, np.fmax,
    ]

    _UNSUPPORTED_UFUNCS = _UNSUPPORTED_UFUNCS_UNARY + _UNSUPPORTED_UFUNCS_BINARY

    _UFUNCS_REQUIRING_VIEW = [
        np.bitwise_and, np.bitwise_or, np.bitwise_xor,
        np.left_shift, np.right_shift,
        np.positive,
    ]

    _OVERRIDDEN_UFUNCS = {
        np.add: "_ufunc_routine_add",
        np.negative: "_ufunc_routine_negative",
        np.subtract: "_ufunc_routine_subtract",
        np.multiply: "_ufunc_routine_multiply",
        np.reciprocal: "_ufunc_routine_reciprocal",
        np.floor_divide: "_ufunc_routine_divide",
        np.true_divide: "_ufunc_routine_divide",
        np.divmod: "_ufunc_routine_divmod",
        np.remainder: "_ufunc_routine_remainder",
        np.power: "_ufunc_routine_power",
        np.square: "_ufunc_routine_square",
        np.log: "_ufunc_routine_log",
        np.sqrt: "_ufunc_routine_sqrt",
        np.matmul: "_ufunc_routine_matmul",
    }

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
        meta["field"] = self.__class__
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
                kwargs["dtype"] = field._dtypes[-1]

            return getattr(field, field._OVERRIDDEN_UFUNCS[ufunc])(ufunc, method, inputs, kwargs, meta)

        elif ufunc in field._UNSUPPORTED_UFUNCS:
            raise NotImplementedError(f"The NumPy ufunc {ufunc.__name__!r} is not supported on {field._name} arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            if ufunc in [np.bitwise_and, np.bitwise_or, np.bitwise_xor] and method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            inputs, kwargs = field._view_inputs_as_ndarray(inputs, kwargs)
            output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

            if ufunc in field._UFUNCS_REQUIRING_VIEW and output is not None:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

            return output

    @classmethod
    def _ufunc(cls, name):
        """
        Returns the ufunc for the specific type of arithmetic. The ufunc compilation is based on `ufunc_mode`.
        """
        if name not in cls._ufuncs:
            if cls._ufunc_mode == "jit-lookup":
                cls._ufuncs[name] = cls._ufunc_lookup(name)
            elif cls._ufunc_mode == "jit-calculate":
                cls._ufuncs[name] = cls._ufunc_calculate(name)
            else:
                cls._ufuncs[name] = cls._ufunc_python(name)
        return cls._ufuncs[name]

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a):
        raise NotImplementedError

    ###############################################################################
    # Input/output conversion functions
    ###############################################################################

    @classmethod
    def _verify_unary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat", "outer"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Reduction methods are only supported on binary functions.")

    @classmethod
    def _verify_binary_method_not_reduction(cls, ufunc, method):  # pylint: disable=no-self-use
        if method in ["reduce", "accumulate", "reduceat"]:
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r} because it takes inputs with type {cls._name} array and integer array. Different types do not support reduction.")

    @classmethod
    def _verify_method_only_call(cls, ufunc, method):  # pylint: disable=no-self-use
        if not method == "__call__":
            raise ValueError(f"Ufunc method {method!r} is not supported on {ufunc.__name__!r}. Only '__call__' is supported.")

    @classmethod
    def _verify_operands_in_same_field(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["non_field_operands"]) > 0:
            raise TypeError(f"Operation {ufunc.__name__!r} requires both operands to be {cls._name} arrays, not {[inputs[i] for i in meta['operands']]}.")

    @classmethod
    def _verify_operands_in_field_or_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        for i in meta["non_field_operands"]:
            if isinstance(inputs[i], (int, np.integer)):
                pass
            elif isinstance(inputs[i], np.ndarray):
                if meta["field"].dtypes == [np.object_]:
                    if not (inputs[i].dtype == np.object_ or np.issubdtype(inputs[i].dtype, np.integer)):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
                else:
                    if not np.issubdtype(inputs[i].dtype, np.integer):
                        raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {inputs[i].dtype}.")
            else:
                raise TypeError(f"Operation {ufunc.__name__!r} requires operands that are not {cls._name} arrays to be integers or an integer np.ndarray, not {type(inputs[i])}.")

    @classmethod
    def _verify_operands_first_field_second_int(cls, ufunc, inputs, meta):  # pylint: disable=no-self-use
        if len(meta["operands"]) == 1:
            return

        if not meta["operands"][0] == meta["field_operands"][0]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the first operand to be a {cls._name} array, not {meta['types'][meta['operands'][0]]}.")
        if len(meta["field_operands"]) > 1 and meta["operands"][1] == meta["field_operands"][1]:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer array, not {meta['types'][meta['operands'][1]]}.")

        second = inputs[meta["operands"][1]]
        if isinstance(second, (int, np.integer)):
            return
        # elif type(second) is np.ndarray:
        #     if not np.issubdtype(second.dtype, np.integer):
        #         raise ValueError(f"Operation {ufunc.__name__!r} requires the second operand with type np.ndarray to have integer dtype, not {second.dtype}.")
        elif isinstance(second, np.ndarray):
            if meta["field"].dtypes == [np.object_]:
                if not (second.dtype == np.object_ or np.issubdtype(second.dtype, np.integer)):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
            else:
                if not np.issubdtype(second.dtype, np.integer):
                    raise ValueError(f"Operation {ufunc.__name__!r} requires operands with type np.ndarray to have integer dtype, not {second.dtype}.")
        else:
            raise TypeError(f"Operation {ufunc.__name__!r} requires the second operand to be an integer or integer np.ndarray, not {type(second)}.")

    @classmethod
    def _view_inputs_as_ndarray(cls, inputs, kwargs, dtype=None):  # pylint: disable=no-self-use
        # View all inputs that are FieldArrays as np.ndarray to avoid infinite recursion
        v_inputs = list(inputs)
        for i in range(len(inputs)):
            if issubclass(type(inputs[i]), cls):
                v_inputs[i] = inputs[i].view(np.ndarray) if dtype is None else inputs[i].view(np.ndarray).astype(dtype)

        # View all output arrays as np.ndarray to avoid infinite recursion
        if "out" in kwargs:
            outputs = kwargs["out"]
            v_outputs = []
            for output in outputs:
                if issubclass(type(output), cls):
                    o = output.view(np.ndarray) if dtype is None else output.view(np.ndarray).astype(dtype)
                else:
                    o = output
                v_outputs.append(o)
            kwargs["out"] = tuple(v_outputs)

        return v_inputs, kwargs

    @classmethod
    def _view_output_as_field(cls, output, field, dtype):  # pylint: disable=no-self-use
        if isinstance(type(output), field):
            return output
        elif isinstance(output, np.ndarray):
            return field._view(output.astype(dtype))
        elif output is None:
            return None
        else:
            return field(output, dtype=dtype)

    ###############################################################################
    # Ufunc routines
    ###############################################################################

    @classmethod
    def _ufunc_routine_add(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("add"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_negative(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("negative"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_subtract(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("subtract"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_multiply(cls, ufunc, method, inputs, kwargs, meta):
        if len(meta["non_field_operands"]) > 0:
            # Scalar multiplication
            cls._verify_operands_in_field_or_int(ufunc, inputs, meta)
            inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
            inputs[meta["non_field_operands"][0]] = np.mod(inputs[meta["non_field_operands"][0]], cls._characteristic)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("multiply"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("reciprocal"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_divide(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("divide"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_divmod(cls, ufunc, method, inputs, kwargs, meta):
        q = cls._ufunc_routine_divide(ufunc, method, inputs, kwargs, meta)
        r = cls.Zeros(q.shape)
        output = q, r
        return output

    @classmethod
    def _ufunc_routine_remainder(cls, ufunc, method, inputs, kwargs, meta):
        # Perform dummy addition operation to get shape of output zeros
        x = cls._ufunc_routine_add(ufunc, method, inputs, kwargs, meta)
        output = cls.Zeros(x.shape)
        return output

    @classmethod
    def _ufunc_routine_power(cls, ufunc, method, inputs, kwargs, meta):
        cls._verify_binary_method_not_reduction(ufunc, method)
        cls._verify_operands_first_field_second_int(ufunc, inputs, meta)
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("power"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_unary_method_not_reduction(ufunc, method)
        inputs = list(inputs) + [2]
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("power"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    @classmethod
    def _ufunc_routine_log(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        inputs = list(inputs) + [int(cls._primitive_element)]
        inputs, kwargs = cls._view_inputs_as_ndarray(inputs, kwargs)
        output = getattr(cls._ufunc("log"), method)(*inputs, **kwargs)
        return output

    @classmethod
    def _ufunc_routine_sqrt(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        x = inputs[0]
        b = x.is_quadratic_residue()  # Boolean indicating if the inputs are quadratic residues
        if not np.all(b):
            raise ArithmeticError(f"Input array has elements that are quadratic non-residues (do not have a square root). Use `x.is_quadratic_residue()` to determine if elements have square roots in {cls._name}.\n{x[~b]}")
        return cls._sqrt(*inputs)

    @classmethod
    def _ufunc_routine_matmul(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        cls._verify_method_only_call(ufunc, method)
        return cls._matmul(*inputs, **kwargs)

    @classmethod
    def _matmul(cls, A, B, out=None, **kwargs):
        """
        Will be implemented in FieldFunctions.
        """
        raise NotImplementedError
