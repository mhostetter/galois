"""
A module that contains the main classes for Galois fields -- FieldArrayClass, FieldArray,
and Poly. They're all in one file because they have circular dependencies. The specific GF2
FieldArrayClass is also included.
"""
from __future__ import annotations

import contextlib
import inspect
import random
from typing import Tuple, List, Iterable, Optional, Union
from typing_extensions import Literal

import numpy as np

from .._array import ArrayClass, Array, ElementLike, ArrayLike, ShapeLike, DTypeLike
from .._modular import totatives
from .._overrides import set_module
from .._polys import Poly
from .._polys._conversions import integer_to_poly, str_to_integer, poly_to_str
from .._prime import divisors

from ._linalg import dot, row_reduce, lu_decompose, plu_decompose, row_space, column_space, left_null_space, null_space
from ._functions import FunctionMeta
from ._ufuncs import UfuncMeta

__all__ = ["FieldArrayClass", "FieldArray"]


###############################################################################
# NumPy ndarray subclass for Galois fields
###############################################################################

@set_module("galois")
class FieldArrayClass(ArrayClass, FunctionMeta, UfuncMeta):
    """
    Defines a metaclass for all :obj:`~galois.FieldArray` classes.

    Important
    ---------
    :obj:`~galois.FieldArrayClass` is a metaclass for :obj:`~galois.FieldArray` subclasses created with the class factory
    :func:`~galois.GF` and should not be instantiated directly. This metaclass gives :obj:`~galois.FieldArray` subclasses
    methods and attributes related to their Galois fields.

    This class is included in the API to allow the user to test if a class is a Galois field array class.

    .. ipython:: python

        GF = galois.GF(7)
        isinstance(GF, galois.FieldArrayClass)
    """
    # pylint: disable=no-value-for-parameter,unsupported-membership-test,abstract-method,too-many-public-methods

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._characteristic = kwargs.get("characteristic", 0)
        cls._degree = kwargs.get("degree", 0)
        cls._order = kwargs.get("order", 0)
        cls._ufunc_mode = None
        cls._dtypes = cls._determine_dtypes()

        if "irreducible_poly" in kwargs:
            cls._irreducible_poly = kwargs["irreducible_poly"]
            cls._irreducible_poly_int = int(cls._irreducible_poly)
        else:
            cls._irreducible_poly = None
            cls._irreducible_poly_int = 0
        cls._primitive_element = kwargs.get("primitive_element", None)

        cls._is_primitive_poly = kwargs.get("is_primitive_poly", None)
        cls._prime_subfield = None

        cls._display_mode = "int"

        if cls.degree == 1:
            cls._name = f"GF({cls._characteristic})"
            cls._order_str = f"order={cls.order}"
        else:
            cls._name = f"GF({cls._characteristic}^{cls._degree})"
            cls._order_str = f"order={cls.characteristic}^{cls.degree}"

        cls._element_fixed_width = None
        cls._element_fixed_width_counter = 0

        # By default, verify array elements are within the valid range when `.view()` casting
        cls._verify_on_view = True

    def __repr__(cls) -> str:
        """
        A terse string representation of the finite field class.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2); GF
            GF = galois.GF(2**8); GF
            GF = galois.GF(31); GF
            GF = galois.GF(7**5); GF
        """
        return super().__repr__()

    def __str__(cls) -> str:
        """
        A formatted string displaying relevant properties of the finite field.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2); print(GF)
            GF = galois.GF(2**8); print(GF)
            GF = galois.GF(31); print(GF)
            GF = galois.GF(7**5); print(GF)
        """
        if cls.prime_subfield is None:
            return repr(cls)

        with cls.prime_subfield.display("int"):
            irreducible_poly_str = str(cls.irreducible_poly)

        string = "Galois Field:"
        string += f"\n  name: {cls.name}"
        string += f"\n  characteristic: {cls.characteristic}"
        string += f"\n  degree: {cls.degree}"
        string += f"\n  order: {cls.order}"
        string += f"\n  irreducible_poly: {irreducible_poly_str}"
        string += f"\n  is_primitive_poly: {cls.is_primitive_poly}"
        string += f"\n  primitive_element: {poly_to_str(integer_to_poly(cls.primitive_element, cls.characteristic))}"

        return string

    ###############################################################################
    # Class methods
    ###############################################################################

    def compile(cls, mode: Literal["auto", "jit-lookup", "jit-calculate", "python-calculate"]):
        """
        Recompile the just-in-time compiled ufuncs for a new calculation mode.

        This function updates :obj:`ufunc_mode`.

        Parameters
        ----------
        mode
            The ufunc calculation mode.

            - `"auto"`: Selects `"jit-lookup"` for fields with order less than :math:`2^{20}`, `"jit-calculate"` for larger fields, and `"python-calculate"`
              for fields whose elements cannot be represented with :obj:`numpy.int64`.
            - `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
              In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
            - `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"` mode is designed for large
              fields that cannot or should not store lookup tables in RAM. Generally, the `"jit-calculate"` mode is slower than `"jit-lookup"`.
            - `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
              represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_` with Python :obj:`int` (which has arbitrary precision).
        """
        if not isinstance(mode, str):
            raise TypeError(f"Argument `mode` must be a string, not {type(mode)}.")
        if not mode in ["auto", "jit-lookup", "jit-calculate", "python-calculate"]:
            raise ValueError(f"Argument `mode` must be in ['auto', 'jit-lookup', 'jit-calculate', 'python-calculate'], not {mode!r}.")
        mode = cls.default_ufunc_mode if mode == "auto" else mode
        if mode not in cls.ufunc_modes:
            raise ValueError(f"Argument `mode` must be in {cls.ufunc_modes} for {cls.name}, not {mode!r}.")

        if mode == cls.ufunc_mode:
            # Don't need to rebuild these ufuncs
            return

        cls._ufunc_mode = mode
        cls._compile_ufuncs()

    def display(
        cls,
        mode: Literal["int", "poly", "power"] = "int"
    ) -> contextlib.AbstractContextManager:
        r"""
        Sets the display mode for all *Galois field arrays* from this field.

        The display mode can be set to either the integer representation, polynomial representation, or power
        representation. See :ref:`Field Element Representation` for a further discussion.

        This function updates :obj:`display_mode`.

        Warning
        -------
        For the power representation, :func:`numpy.log` is computed on each element. So for large fields without lookup
        tables, displaying arrays in the power representation may take longer than expected.

        Parameters
        ----------
        mode
            The field element representation.

            - `"int"`: Sets the display mode to the :ref:`integer representation <Integer representation>`.
            - `"poly"`: Sets the display mode to the :ref:`polynomial representation <Polynomial representation>`.
            - `"power"`: Sets the display mode to the :ref:`power representation <Power representation>`.

        Returns
        -------
        :
            A context manager for use in a `with` statement. If permanently setting the display mode, disregard the
            return value.

        Examples
        --------
        The default display mode is the integer representation.

        .. ipython:: python

            GF = galois.GF(3**2)
            x = GF.Elements(); x

        Permanently set the display mode by calling :func:`display`.

        .. tab-set::

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF.display("poly");
                    x

            .. tab-item:: Power

                .. ipython:: python

                    GF.display("power");
                    x
                    @suppress
                    GF.display()

        Temporarily modify the display mode by using :func:`display` as a context manager.

        .. tab-set::

            .. tab-item:: Polynomial

                .. ipython:: python

                    print(x)
                    with GF.display("poly"):
                        print(x)
                    # Outside the context manager, the display mode reverts to its previous value
                    print(x)

            .. tab-item:: Power

                .. ipython:: python

                    print(x)
                    with GF.display("power"):
                        print(x)
                    # Outside the context manager, the display mode reverts to its previous value
                    print(x)
                    @suppress
                    GF.display()
        """
        if not isinstance(mode, (type(None), str)):
            raise TypeError(f"Argument `mode` must be a string, not {type(mode)}.")
        if mode not in ["int", "poly", "power"]:
            raise ValueError(f"Argument `mode` must be in ['int', 'poly', 'power'], not {mode!r}.")

        prev_mode = cls._display_mode
        cls._display_mode = mode

        @set_module("galois")
        class context(contextlib.AbstractContextManager):
            """Simple display_mode context manager."""
            def __init__(self, mode):
                self.mode = mode

            def __enter__(self):
                # Don't need to do anything, we already set the new mode in the display() method
                pass

            def __exit__(self, exc_type, exc_value, traceback):
                cls._display_mode = self.mode

        return context(prev_mode)

    def repr_table(
        cls,
        element: Optional[ElementLike] = None,
        sort: Literal["power", "poly", "vector", "int"] = "power"
    ) -> str:
        r"""
        Generates a finite field element representation table comparing the power, polynomial, vector, and integer representations.

        Parameters
        ----------
        element
            The primitive element to use for the power representation. The default is `None` which uses the field's
            default primitive element, :obj:`FieldArrayClass.primitive_element`.
        sort
            The sorting method for the table. The default is `"power"`. Sorting by `"power"` will order the rows of the table by ascending
            powers of `element`. Sorting by any of the others will order the rows in lexicographically-increasing polynomial/vector
            order, which is equivalent to ascending order of the integer representation.

        Returns
        -------
        :
            A UTF-8 formatted table comparing the power, polynomial, vector, and integer representations of each
            field element.

        Examples
        --------
        Create a *Galois field array class* for :math:`\mathrm{GF}(2^4)`.

        .. ipython:: python

            GF = galois.GF(2**4)
            print(GF)

        .. tab-set::

            .. tab-item:: Default

                Generate a representation table for :math:`\mathrm{GF}(2^4)`. Since :math:`x^4 + x + 1` is a primitive polynomial,
                :math:`x` is a primitive element of the field. Notice, :math:`\textrm{ord}(x) = 15`.

                .. ipython:: python

                    print(GF.repr_table())

            .. tab-item:: Primitive element

                Generate a representation table for :math:`\mathrm{GF}(2^4)` using a different primitive element :math:`x^3 + x^2 + x`.
                Notice, :math:`\textrm{ord}(x^3 + x^2 + x) = 15`.

                .. ipython:: python

                    print(GF.repr_table("x^3 + x^2 + x"))

            .. tab-item:: Non-primitive element

                Generate a representation table for :math:`\mathrm{GF}(2^4)` using a non-primitive element :math:`x^3 + x^2`. Notice,
                :math:`\textrm{ord}(x^3 + x^2) = 5 \ne 15`.

                .. ipython:: python

                    print(GF.repr_table("x^3 + x^2"))
        """
        if sort not in ["power", "poly", "vector", "int"]:
            raise ValueError(f"Argument `sort` must be in ['power', 'poly', 'vector', 'int'], not {sort!r}.")
        if element is None:
            element = cls.primitive_element

        element = cls(element)
        degrees = np.arange(0, cls.order - 1)
        x = element**degrees
        if sort != "power":
            idxs = np.argsort(x)
            degrees, x = degrees[idxs], x[idxs]
        x = np.concatenate((np.atleast_1d(cls(0)), x))  # Add 0 = alpha**-Inf
        prim = poly_to_str(integer_to_poly(element, cls.characteristic))

        # Define print helper functions
        if len(prim) > 1:
            print_power = lambda power: "0" if power is None else f"({prim})^{power}"
        else:
            print_power = lambda power: "0" if power is None else f"{prim}^{power}"
        print_poly = lambda x: poly_to_str(integer_to_poly(x, cls.characteristic))
        print_vec = lambda x: str(integer_to_poly(x, cls.characteristic, degree=cls.degree-1))
        print_int = lambda x: str(int(x))

        # Determine column widths
        N_power = max([len(print_power(max(degrees))), len("Power")]) + 2
        N_poly = max([len(print_poly(e)) for e in x] + [len("Polynomial")]) + 2
        N_vec = max([len(print_vec(e)) for e in x] + [len("Vector")]) + 2
        N_int = max([len(print_int(e)) for e in x] + [len("Integer")]) + 2

        string = "+" + "-"*N_power + "+" + "-"*N_poly + "+" + "-"*N_vec + "+" + "-"*N_int + "+"
        string += "\n|" + "Power".center(N_power) + "|" + "Polynomial".center(N_poly) + "|" + "Vector".center(N_vec) + "|" + "Integer".center(N_int) + "|"
        string += "\n+" + "-"*N_power + "+" + "-"*N_poly + "+" + "-"*N_vec + "+" + "-"*N_int + "+"

        for i in range(x.size):
            d = None if i == 0 else degrees[i - 1]
            string += "\n|" + print_power(d).center(N_power) + "|" + poly_to_str(integer_to_poly(x[i], cls.characteristic)).center(N_poly) + "|" + str(integer_to_poly(x[i], cls.characteristic, degree=cls.degree-1)).center(N_vec) + "|" + cls._print_int(x[i]).center(N_int) + "|"

            if i < x.size - 1:
                string += "\n+" + "-"*N_power + "+" + "-"*N_poly + "+" + "-"*N_vec + "+" + "-"*N_int + "+"

        string += "\n+" + "-"*N_power + "+" + "-"*N_poly + "+"+ "-"*N_vec + "+" + "-"*N_int + "+"

        return string

    def arithmetic_table(
        cls,
        operation: Literal["+", "-", "*", "/"],
        x: Optional[FieldArray] = None,
        y: Optional[FieldArray] = None
    ) -> str:
        r"""
        Generates the specified arithmetic table for the finite field.

        Parameters
        ----------
        operation
            The arithmetic operation.
        x
            Optionally specify the :math:`x` values for the arithmetic table. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}`.
        y
            Optionally specify the :math:`y` values for the arithmetic table. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}` for addition, subtraction, and multiplication and
            :math:`\{1, \dots, p^m - 1\}` for division.

        Returns
        -------
        :
            A UTF-8 formatted arithmetic table.

        Examples
        --------
        Arithmetic tables can be displayed using the :ref:`integer representation <Integer representation>`.

        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    print(GF.arithmetic_table("+"))

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    print(GF.arithmetic_table("+"))

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    print(GF.arithmetic_table("+"))

        An arithmetic table may also be constructed from arbitrary :math:`x` and :math:`y`.

        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    x = GF([7, 2, 8]); x
                    y = GF([1, 4]); y
                    print(GF.arithmetic_table("+", x=x, y=y))

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    x = GF([7, 2, 8]); x
                    y = GF([1, 4]); y
                    print(GF.arithmetic_table("+", x=x, y=y))

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    x = GF([7, 2, 8]); x
                    y = GF([1, 4]); y
                    print(GF.arithmetic_table("+", x=x, y=y))
                    @suppress
                    GF.display()
        """
        if not operation in ["+", "-", "*", "/"]:
            raise ValueError(f"Argument `operation` must be in ['+', '-', '*', '/'], not {operation!r}.")

        if cls.display_mode == "power":
            # Order elements by powers of the primitive element
            x_default = np.concatenate((np.atleast_1d(cls(0)), cls.primitive_element**np.arange(0, cls.order - 1, dtype=cls.dtypes[-1])))
        else:
            x_default = cls.Elements()
        y_default = x_default if operation != "/" else x_default[1:]

        x = x_default if x is None else cls(x)
        y = y_default if y is None else cls(y)
        X, Y = np.meshgrid(x, y, indexing="ij")

        if operation == "+":
            Z = X + Y
        elif operation == "-":
            Z = X - Y
        elif operation == "*":
            Z = X * Y
        else:
            Z = X / Y

        if cls.display_mode == "int":
            print_element = cls._print_int
        elif cls.display_mode == "poly":
            print_element = cls._print_poly
        else:
            print_element = cls._print_power

        operation_str = f"x {operation} y"

        N = max([len(print_element(e)) for e in x]) + 2
        N_left = max(N, len(operation_str) + 2)

        string = "+" + "-"*N_left + "+" + ("-"*N + "+")*(y.size - 1) + "-"*N + "+"
        string += "\n|" + operation_str.rjust(N_left - 1) + " |"
        for j in range(y.size):
            string += print_element(y[j]).rjust(N - 1) + " "
            string += "|" if j < y.size - 1 else "|"
        string += "\n+" + "-"*N_left + "+" + ("-"*N + "+")*(y.size - 1) + "-"*N + "+"

        for i in range(x.size):
            string += "\n|" + print_element(x[i]).rjust(N_left - 1) + " |"
            for j in range(y.size):
                string += print_element(Z[i,j]).rjust(N - 1) + " "
                string += "|" if j < y.size - 1 else "|"

            if i < x.size - 1:
                string += "\n+" + "-"*N_left + "+" + ("-"*N + "+")*(y.size - 1) + "-"*N + "+"

        string += "\n+" + "-"*N_left + "+" + ("-"*N + "+")*(y.size - 1) + "-"*N + "+"

        return string

    def primitive_root_of_unity(cls, n: int) -> FieldArray:
        r"""
        Finds a primitive :math:`n`-th root of unity in the finite field.

        Parameters
        ----------
        n
            The root of unity.

        Returns
        -------
        :
            The primitive :math:`n`-th root of unity, a 0-D scalar array.

        Raises
        ------
        ValueError
            If no primitive :math:`n`-th roots of unity exist. This happens when :math:`n` is not a
            divisor of :math:`p^m - 1`.

        Notes
        -----
        A primitive :math:`n`-th root of unity :math:`\omega_n` is such that :math:`\omega_n^n = 1` and :math:`\omega_n^k \ne 1`
        for all :math:`1 \le k \lt n`.

        In :math:`\mathrm{GF}(p^m)`, a primitive :math:`n`-th root of unity exists when :math:`n` divides :math:`p^m - 1`.
        Then, the primitive root is :math:`\omega_n = \alpha^{(p^m - 1)/n}` where :math:`\alpha` is a primitive
        element of the field.

        Examples
        --------
        In :math:`\mathrm{GF}(31)`, primitive roots exist for all divisors of :math:`30`.

        .. ipython:: python

            GF = galois.GF(31)
            GF.primitive_root_of_unity(2)
            GF.primitive_root_of_unity(5)
            GF.primitive_root_of_unity(15)

        However, they do not exist for :math:`n` that do not divide :math:`30`.

        .. ipython:: python
            :okexcept:

            GF.primitive_root_of_unity(7)

        For :math:`\omega_5`, one can see that :math:`\omega_5^5 = 1` and :math:`\omega_5^k \ne 1` for :math:`1 \le k \lt 5`.

        .. ipython:: python

            root = GF.primitive_root_of_unity(5); root
            np.power.outer(root, np.arange(1, 5 + 1))
        """
        if not isinstance(n, (int, np.ndarray)):
            raise TypeError(f"Argument `n` must be an int, not {type(n)!r}.")
        if not 1 <= n < cls.order:
            raise ValueError(f"Argument `n` must be in [1, {cls.order}), not {n}.")
        if not (cls.order - 1) % n == 0:
            raise ValueError(f"There are no primitive {n}-th roots of unity in {cls.name}.")

        return cls.primitive_element ** ((cls.order - 1) // n)

    def primitive_roots_of_unity(cls, n: int) -> FieldArray:
        r"""
        Finds all primitive :math:`n`-th roots of unity in the finite field.

        Parameters
        ----------
        n
            The root of unity.

        Returns
        -------
        :
            All primitive :math:`n`-th roots of unity, a 1-D array. The roots are sorted in lexicographically-increasing
            order.

        Raises
        ------
        ValueError
            If no primitive :math:`n`-th roots of unity exist. This happens when :math:`n` is not a
            divisor of :math:`p^m - 1`.

        Notes
        -----
        A primitive :math:`n`-th root of unity :math:`\omega_n` is such that :math:`\omega_n^n = 1` and :math:`\omega_n^k \ne 1`
        for all :math:`1 \le k \lt n`.

        In :math:`\mathrm{GF}(p^m)`, a primitive :math:`n`-th root of unity exists when :math:`n` divides :math:`p^m - 1`.
        Then, the primitive root is :math:`\omega_n = \alpha^{(p^m - 1)/n}` where :math:`\alpha` is a primitive
        element of the field.

        Examples
        --------
        In :math:`\mathrm{GF}(31)`, primitive roots exist for all divisors of :math:`30`.

        .. ipython:: python

            GF = galois.GF(31)
            GF.primitive_roots_of_unity(2)
            GF.primitive_roots_of_unity(5)
            GF.primitive_roots_of_unity(15)

        However, they do not exist for :math:`n` that do not divide :math:`30`.

        .. ipython:: python
            :okexcept:

            GF.primitive_roots_of_unity(7)

        For :math:`\omega_5`, one can see that :math:`\omega_5^5 = 1` and :math:`\omega_5^k \ne 1` for :math:`1 \le k \lt 5`.

        .. ipython:: python

            root = GF.primitive_roots_of_unity(5); root
            np.power.outer(root, np.arange(1, 5 + 1))
        """
        if not isinstance(n, (int, np.ndarray)):
            raise TypeError(f"Argument `n` must be an int, not {type(n)!r}.")
        if not (cls.order - 1) % n == 0:
            raise ValueError(f"There are no primitive {n}-th roots of unity in {cls.name}.")

        roots = np.unique(cls.primitive_elements ** ((cls.order - 1) // n))
        roots = np.sort(roots)

        return roots

    ###############################################################################
    # Array display methods
    ###############################################################################

    def _formatter(cls, array):
        """
        Returns a NumPy printoptions "formatter" dictionary.
        """
        formatter = {}

        if cls.display_mode == "poly" and cls.is_extension_field:
            # The "poly" display mode for prime field's is the same as the integer representation
            formatter["int"] = cls._print_poly
            formatter["object"] = cls._print_poly
        elif cls.display_mode == "power":
            formatter["int"] = cls._print_power
            formatter["object"] = cls._print_power
        elif array.dtype == np.object_:
            formatter["object"] = cls._print_int

        return formatter

    def _print_int(cls, element):  # pylint: disable=no-self-use
        """
        Prints a single element in the integer representation. This is only needed for dtype=object arrays.
        """
        s = f"{int(element)}"

        if cls._element_fixed_width:
            s = s.rjust(cls._element_fixed_width)
        else:
            cls._element_fixed_width_counter = max(len(s), cls._element_fixed_width_counter)

        return s

    def _print_poly(cls, element):
        """
        Prints a single element in the polynomial representation.
        """
        poly = integer_to_poly(element, cls.characteristic)
        poly_var = "α" if cls.primitive_element == cls.characteristic else "x"
        s = poly_to_str(poly, poly_var=poly_var)

        if cls._element_fixed_width:
            s = s.rjust(cls._element_fixed_width)
        else:
            cls._element_fixed_width_counter = max(len(s), cls._element_fixed_width_counter)

        return s

    def _print_power(cls, element):
        """
        Prints a single element in the power representation.
        """
        if element in [0, 1]:
            s = f"{int(element)}"
        elif element == cls.primitive_element:
            s = "α"
        else:
            power = cls._ufunc("log")(element, cls.primitive_element)
            s = f"α^{power}"

        if cls._element_fixed_width:
            s = s.rjust(cls._element_fixed_width)
        else:
            cls._element_fixed_width_counter = max(len(s), cls._element_fixed_width_counter)

        return s

    ###############################################################################
    # Class attributes
    ###############################################################################

    @property
    def name(cls) -> str:
        """
        The finite field's name as a string `GF(p)` or `GF(p^m)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).name
            galois.GF(2**8).name
            galois.GF(31).name
            galois.GF(7**5).name
        """
        return cls._name

    @property
    def characteristic(cls) -> int:
        r"""
        The prime characteristic :math:`p` of the Galois field :math:`\mathrm{GF}(p^m)`. Adding
        :math:`p` copies of any element will always result in :math:`0`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).characteristic
            galois.GF(2**8).characteristic
            galois.GF(31).characteristic
            galois.GF(7**5).characteristic

        """
        return cls._characteristic

    @property
    def degree(cls) -> int:
        r"""
        The extension degree :math:`m` of the Galois field :math:`\mathrm{GF}(p^m)`. The degree is a positive integer.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).degree
            galois.GF(2**8).degree
            galois.GF(31).degree
            galois.GF(7**5).degree
        """
        return cls._degree

    @property
    def order(cls) -> int:
        r"""
        The order :math:`p^m` of the Galois field :math:`\mathrm{GF}(p^m)`. The order of the field is equal to the field's size.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).order
            galois.GF(2**8).order
            galois.GF(31).order
            galois.GF(7**5).order
        """
        return cls._order

    @property
    def irreducible_poly(cls) -> "Poly":
        r"""
        The irreducible polynomial :math:`f(x)` of the Galois field :math:`\mathrm{GF}(p^m)`. The irreducible
        polynomial is of degree :math:`m` over :math:`\mathrm{GF}(p)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).irreducible_poly
            galois.GF(2**8).irreducible_poly
            galois.GF(31).irreducible_poly
            galois.GF(7**5).irreducible_poly
        """
        return cls._irreducible_poly

    @property
    def is_primitive_poly(cls) -> bool:
        r"""
        Indicates whether the :obj:`FieldArrayClass.irreducible_poly` is a primitive polynomial. If so, :math:`x` is a primitive element
        of the finite field.

        The default irreducible polynomial is a Conway polynomial, see :func:`~galois.conway_poly`, which is a primitive
        polynomial. However, finite fields may be constructed from non-primitive, irreducible polynomials.

        Examples
        --------
        The default :math:`\mathrm{GF}(2^8)` field uses a primitive polynomial.

        .. ipython:: python

            GF = galois.GF(2**8)
            print(GF)
            GF.is_primitive_poly

        The :math:`\mathrm{GF}(2^8)` field from AES uses a non-primitive polynomial.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
            print(GF)
            GF.is_primitive_poly
        """
        return cls._is_primitive_poly

    @property
    def primitive_element(cls) -> "FieldArray":
        r"""
        A primitive element :math:`\alpha` of the Galois field :math:`\mathrm{GF}(p^m)`. A primitive element is a multiplicative
        generator of the field, such that :math:`\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m - 2}\}`.

        A primitive element is a root of the primitive polynomial :math:`f(x)`, such that :math:`f(\alpha) = 0` over
        :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).primitive_element
            galois.GF(2**8).primitive_element
            galois.GF(31).primitive_element
            galois.GF(7**5).primitive_element
        """
        # Ensure accesses of this property doesn't alter it
        return cls(cls._primitive_element)  # pylint: disable=no-value-for-parameter

    @property
    def primitive_elements(cls) -> "FieldArray":
        r"""
        All primitive elements :math:`\alpha` of the Galois field :math:`\mathrm{GF}(p^m)`. A primitive element is a multiplicative
        generator of the field, such that :math:`\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m - 2}\}`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).primitive_elements
            galois.GF(2**8).primitive_elements
            galois.GF(31).primitive_elements
            galois.GF(7**5).primitive_elements
        """
        if not hasattr(cls, "_primitive_elements"):
            n = cls.order - 1
            powers = np.array(totatives(n))
            cls._primitive_elements = np.sort(cls.primitive_element ** powers)
        return cls._primitive_elements.copy()

    @property
    def quadratic_residues(cls) -> "FieldArray":
        r"""
        All quadratic residues in the finite field.

        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic residue* if there exists a :math:`y` such that
        :math:`y^2 = x` in the field.

        In fields with characteristic 2, every element is a quadratic residue. In fields with characteristic greater than 2,
        exactly half of the nonzero elements are quadratic residues (and they have two unique square roots).

        See also :func:`FieldArray.is_quadratic_residue`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(11)
            x = GF.quadratic_residues; x
            r = np.sqrt(x); r
            r ** 2
            (-r) ** 2

        .. ipython:: python

            GF = galois.GF(2**4)
            x = GF.quadratic_residues; x
            r = np.sqrt(x); r
            r ** 2
            (-r) ** 2
        """
        x = cls.Elements()
        is_quadratic_residue = x.is_quadratic_residue()
        return x[is_quadratic_residue]

    @property
    def quadratic_non_residues(cls) -> "FieldArray":
        r"""
        All quadratic non-residues in the Galois field.

        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic non-residue* if there does not exist a :math:`y` such that
        :math:`y^2 = x` in the field.

        In fields with characteristic 2, no elements are quadratic non-residues. In fields with characteristic greater than 2,
        exactly half of the nonzero elements are quadratic non-residues.

        See also :func:`FieldArray.is_quadratic_residue`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(11)
            GF.quadratic_non_residues

        .. ipython:: python

            GF = galois.GF(2**4)
            GF.quadratic_non_residues
        """
        x = cls.Elements()
        is_quadratic_residue = x.is_quadratic_residue()
        return x[~is_quadratic_residue]

    @property
    def is_prime_field(cls) -> bool:
        """
        Indicates if the finite field is a prime field, not an extension field. This is true when the field's order is prime.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).is_prime_field
            galois.GF(2**8).is_prime_field
            galois.GF(31).is_prime_field
            galois.GF(7**5).is_prime_field
        """
        return cls._degree == 1

    @property
    def is_extension_field(cls) -> bool:
        """
        Indicates if the finite field is an extension field. This is true when the field's order is a prime power.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).is_extension_field
            galois.GF(2**8).is_extension_field
            galois.GF(31).is_extension_field
            galois.GF(7**5).is_extension_field
        """
        return cls._degree > 1

    @property
    def prime_subfield(cls) -> "FieldArrayClass":
        r"""
        The prime subfield :math:`\mathrm{GF}(p)` of the extension field :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).prime_subfield
            galois.GF(2**8).prime_subfield
            galois.GF(31).prime_subfield
            galois.GF(7**5).prime_subfield
        """
        return cls._prime_subfield

    @property
    def dtypes(cls) -> List[np.dtype]:
        r"""
        List of valid integer :obj:`numpy.dtype` values that are compatible with this finite field. Creating an array with an
        unsupported dtype will raise a `TypeError` exception.

        For finite fields whose elements cannot be represented with :obj:`numpy.int64`, the only valid data type is :obj:`numpy.object_`.

        Examples
        --------
        Some data types are too small for certain finite fields, such as :obj:`numpy.int16` for :math:`\mathrm{GF}(7^5)`.

        .. ipython:: python

            GF = galois.GF(31); GF.dtypes
            GF = galois.GF(7**5); GF.dtypes

        Large fields must use :obj:`numpy.object_` which uses Python :obj:`int` for its unlimited size.

        .. ipython:: python

            GF = galois.GF(2**100); GF.dtypes
            GF = galois.GF(36893488147419103183); GF.dtypes
        """
        return cls._dtypes

    @property
    def display_mode(cls) -> Literal["int", "poly", "power"]:
        r"""
        The current finite field element representation. This can be changed with :func:`display`.

        See :ref:`Field Element Representation` for a further discussion.

        Examples
        --------
        The default display mode is the integer representation.

        .. ipython:: python

            GF = galois.GF(3**2)
            x = GF.Elements(); x
            GF.display_mode

        Permanently modify the display mode by calling :func:`display`.

        .. ipython:: python

            GF.display("poly");
            x
            GF.display_mode
            @suppress
            GF.display()
        """
        return cls._display_mode

    @property
    def ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The current ufunc compilation mode. The ufuncs can be recompiled with :func:`compile`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).ufunc_mode
            galois.GF(2**8).ufunc_mode
            galois.GF(31).ufunc_mode
            galois.GF(7**5).ufunc_mode
        """
        return cls._ufunc_mode

    @property
    def ufunc_modes(cls) -> List[str]:
        """
        All supported ufunc compilation modes for this *Galois field array class*.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).ufunc_modes
            galois.GF(2**8).ufunc_modes
            galois.GF(31).ufunc_modes
            galois.GF(2**100).ufunc_modes
        """
        if cls.dtypes == [np.object_]:
            return ["python-calculate"]
        else:
            return ["jit-lookup", "jit-calculate"]

    @property
    def default_ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The default ufunc compilation mode for this *Galois field array class*.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).default_ufunc_mode
            galois.GF(2**8).default_ufunc_mode
            galois.GF(31).default_ufunc_mode
            galois.GF(2**100).default_ufunc_mode
        """
        if cls.dtypes == [np.object_]:
            return "python-calculate"
        elif cls.order <= 2**20:
            return "jit-lookup"
        else:
            return "jit-calculate"


class DirMeta(type):
    """
    A mixin metaclass that overrides __dir__() so that dir() and tab-completion in ipython of `FieldArray` classes
    (which are `FieldArrayClass` instances) include the methods and properties from the metaclass. Python does not
    natively include metaclass properties in dir().

    This is a separate class because it will be mixed in to `GF2Meta`, `GF2mMeta`, `GFpMeta`, and `GFpmMeta` separately. Otherwise, the
    sphinx documentation of `FieldArray` gets messed up.

    Since, `GF2` has this class mixed in, its docs are messed up. Because of that, we added a separate Sphinx template `class_only_init.rst`
    to suppress all the methods except __init__() so the docs are more presentable.
    """

    def __dir__(cls):
        if isinstance(cls, FieldArrayClass):
            meta_dir = dir(type(cls))
            classmethods = [attribute for attribute in super().__dir__() if attribute[0] != "_" and inspect.ismethod(getattr(cls, attribute))]
            return sorted(meta_dir + classmethods)
        else:
            return super().__dir__()


###############################################################################
# NumPy arrays over Galois fields
###############################################################################

@set_module("galois")
class FieldArray(Array, metaclass=FieldArrayClass):
    r"""
    A :obj:`~numpy.ndarray` subclass over :math:`\mathrm{GF}(p^m)`.

    Important
    ---------
        :obj:`~galois.FieldArray` is an abstract base class for all :ref:`Galois field array classes <Galois field array class>` and cannot
        be instantiated directly. Instead, :obj:`~galois.FieldArray` subclasses are created using the class factory :func:`~galois.GF`.

        This class is included in the API to allow the user to test if an array is a Galois field array subclass.

        .. ipython:: python

            GF = galois.GF(7)
            issubclass(GF, galois.FieldArray)
            x = GF([1, 2, 3]); x
            isinstance(x, galois.FieldArray)

    See :ref:`Galois Field Classes` for a detailed discussion of the relationship between :obj:`~galois.FieldArrayClass` and
    :obj:`~galois.FieldArray`.

    See :ref:`Array Creation` for a detailed discussion on creating arrays (with and without copying) from array-like
    objects, valid NumPy data types, and other :obj:`~galois.FieldArray` classmethods.

    Examples
    --------
    Create a :ref:`Galois field array class` using the class factory :func:`~galois.GF`.

    .. ipython:: python

        GF = galois.GF(3**5)
        print(GF)

    The *Galois field array class* `GF` is a subclass of :obj:`~galois.FieldArray`, with :obj:`~galois.FieldArrayClass` as its
    metaclass.

    .. ipython:: python

        isinstance(GF, galois.FieldArrayClass)
        issubclass(GF, galois.FieldArray)

    Create a :ref:`Galois field array` using `GF`'s constructor.

    .. ipython:: python

        x = GF([44, 236, 206, 138]); x

    The *Galois field array* `x` is an instance of the *Galois field array class* `GF`.

    .. ipython:: python

        isinstance(x, GF)
    """
    # pylint: disable=unsupported-membership-test,not-an-iterable,too-many-public-methods

    def __new__(
        cls,
        x: Union[ElementLike, ArrayLike],
        dtype: Optional[DTypeLike] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ) -> FieldArray:
        if cls is FieldArray:
            raise NotImplementedError("FieldArray is an abstract base class that cannot be directly instantiated. Instead, create a FieldArray subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` and instantiate an array using `x = GF(array_like)`.")
        return super().__new__(cls, x, dtype, copy, order, ndmin)

    def __init__(
        self,
        x: Union[ElementLike, ArrayLike],
        dtype: Optional[DTypeLike] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ):
        r"""
        Creates a :ref:`Galois field array` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        x
            A finite field scalar or array. See :ref:`Array Creation` for a detailed discussion about creating new arrays and array-like objects.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).
        copy
            The `copy` keyword argument from :func:`numpy.array`. The default is `True`.
        order
            The `order` keyword argument from :func:`numpy.array`. The default is `"K"`.
        ndmin
            The `ndmin` keyword argument from :func:`numpy.array`. The default is 0.
        """
        # pylint: disable=unused-argument,super-init-not-called
        # Adding __init__ and not doing anything is done to overwrite the superclass's __init__ docstring
        return

    ###############################################################################
    # Verification routines
    ###############################################################################

    @classmethod
    def _verify_array_like_types_and_values(cls, x: Union[ElementLike, ArrayLike]):
        """
        Verify the types of the array-like object. Also verify the values of the array are within the range [0, order).
        """
        if isinstance(x, (int, np.integer)):
            cls._verify_scalar_value(x)
        elif isinstance(x, cls):
            # This was a previously-created and vetted array -- there's no need to re-verify
            if x.ndim == 0:
                # Ensure that in "large" fields with dtype=object that FieldArray objects aren't assigned to the array. The arithmetic
                # functions are designed to operate on Python ints.
                x = int(x)
        elif isinstance(x, str):
            x = cls._convert_to_element(x)
            cls._verify_scalar_value(x)
        elif isinstance(x, (list, tuple)):
            x = cls._convert_iterable_to_elements(x)
            cls._verify_array_values(x)
        elif isinstance(x, np.ndarray):
            # If this a NumPy array, but not a FieldArray, verify the array
            if x.dtype == np.object_:
                x = cls._verify_element_types_and_convert(x, object_=True)
            elif not np.issubdtype(x.dtype, np.integer):
                raise TypeError(f"{cls.name} arrays must have integer dtypes, not {x.dtype}.")
            cls._verify_array_values(x)
        else:
            raise TypeError(f"{cls.name} arrays can be created with scalars of type int/str, lists/tuples, or ndarrays, not {type(x)}.")

        return x

    @classmethod
    def _verify_element_types_and_convert(cls, array: np.ndarray, object_=False) -> np.ndarray:
        """
        Iterate across each element and verify it's a valid type. Also, convert strings to integers along the way.
        """
        if array.size == 0:
            return array
        elif object_:
            return np.vectorize(cls._convert_to_element, otypes=[object])(array)
        else:
            return np.vectorize(cls._convert_to_element)(array)

    @classmethod
    def _verify_scalar_value(cls, scalar: np.ndarray):
        """
        Verify the single integer element is within the valid range [0, order).
        """
        if not 0 <= scalar < cls.order:
            raise ValueError(f"{cls.name} scalars must be in `0 <= x < {cls.order}`, not {scalar}.")

    @classmethod
    def _verify_array_values(cls, array: np.ndarray):
        """
        Verify all the elements of the integer array are within the valid range [0, order).
        """
        if np.any(array < 0) or np.any(array >= cls.order):
            idxs = np.logical_or(array < 0, array >= cls.order)
            values = array if array.ndim == 0 else array[idxs]
            raise ValueError(f"{cls.name} arrays must have elements in `0 <= x < {cls.order}`, not {values}.")

    ###############################################################################
    # Element conversion routines
    ###############################################################################

    @classmethod
    def _convert_to_element(cls, element) -> int:
        """
        Convert any element-like value to an integer.
        """
        if isinstance(element, (int, np.integer)):
            element = int(element)
        elif isinstance(element, str):
            element = str_to_integer(element, cls.prime_subfield)
        elif isinstance(element, FieldArray):
            element = int(element)
        else:
            raise TypeError(f"Valid element-like values are integers and string, not {type(element)}.")

        return element

    @classmethod
    def _convert_iterable_to_elements(cls, iterable: Iterable) -> np.ndarray:
        """
        Convert an iterable (recursive) to a NumPy integer array. Convert any strings to integers along the way.
        """
        if cls.dtypes == [np.object_]:
            array = np.array(iterable, dtype=object)
            array = cls._verify_element_types_and_convert(array, object_=True)
        else:
            # Try to convert to an integer array in one shot
            array = np.array(iterable)

            if not np.issubdtype(array.dtype, np.integer):
                # There are strings in the array, try again and manually convert each element
                array = np.array(iterable, dtype=object)
                array = cls._verify_element_types_and_convert(array)

        return array

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(cls, shape: ShapeLike, dtype: Optional[DTypeLike] = None) -> FieldArray:
        """
        Creates an array of all zeros.

        Parameters
        ----------
        shape
            A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M, N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            An array of zeros.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Zeros((2, 5))
        """
        dtype = cls._get_dtype(dtype)
        array = np.zeros(shape, dtype=dtype)
        return cls._view(array)

    @classmethod
    def Ones(cls, shape: ShapeLike, dtype: Optional[DTypeLike] = None) -> FieldArray:
        """
        Creates an array of all ones.

        Parameters
        ----------
        shape
            A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M, N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            An array of ones.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Ones((2, 5))
        """
        dtype = cls._get_dtype(dtype)
        array = np.ones(shape, dtype=dtype)
        return cls._view(array)

    @classmethod
    def Range(
        cls,
        start: ElementLike,
        stop: ElementLike,
        step: int = 1,
        dtype: Optional[DTypeLike] = None
    ) -> FieldArray:
        """
        Creates a 1-D array with a range of field elements.

        Parameters
        ----------
        start
            The starting finite field element (inclusive).
        stop
            The stopping finite field element (exclusive).
        step
            The increment between finite field elements. The default is 1.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            A 1-D array of a range of finite field elements.

        Examples
        --------
        For prime fields, the increment is simply a finite field element, since all elements are integers.

        .. ipython:: python

            GF = galois.GF(31)
            GF.Range(10, 20)
            GF.Range(10, 20, 2)

        For extension fields, the increment is the integer increment between finite field elements in their :ref:`integer representation <Integer representation>`.

        .. ipython:: python

            GF = galois.GF(3**3, display="poly")
            GF.Range(10, 20)
            GF.Range(10, 20, 2)
            @suppress
            GF.display()
        """
        # Coerce element-like values to integers in [0, p^m)
        if start != cls.order:
            start = int(cls(start))
        if stop != cls.order:
            stop = int(cls(stop))
        dtype = cls._get_dtype(dtype)

        if not 0 <= start <= cls.order:
            raise ValueError(f"Argument `start` must be within the field's order {cls.order}, not {start}.")
        if not 0 <= stop <= cls.order:
            raise ValueError(f"Argument `stop` must be within the field's order {cls.order}, not {stop}.")

        array = np.arange(start, stop, step=step, dtype=dtype)

        return cls._view(array)

    @classmethod
    def Random(
        cls,
        shape: ShapeLike = (),
        low: ElementLike = 0,
        high: Optional[ElementLike] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: Optional[DTypeLike] = None
    ) -> FieldArray:
        """
        Creates an array with random field elements.

        Parameters
        ----------
        shape
            A NumPy-compliant :obj:`~numpy.ndarray.shape` tuple. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M, N)`, represents a 2-D array with each element indicating the size in each dimension.
        low
            The smallest finite field element (inclusive). The default is 0.
        high
            The largest finite field element (exclusive). The default is `None` which represents the field's order :math:`p^m`.
        seed
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            An array of random finite field elements.

        Examples
        --------
        Generate a random matrix with an unpredictable seed.

        .. ipython:: python

            GF = galois.GF(31)
            GF.Random((2, 5))

        Generate a random array with a specified seed. This produces repeatable outputs.

        .. ipython:: python

            GF.Random(10, seed=123456789)
            GF.Random(10, seed=123456789)

        Generate a group of random arrays using a single global seed.

        .. ipython:: python

            rng = np.random.default_rng(123456789)
            GF.Random(10, seed=rng)
            GF.Random(10, seed=rng)
        """
        # Coerce element-like values to integers in [0, p^m)
        low = int(cls(low))
        if high is None:
            high = cls.order
        elif high != cls.order:
            high = int(cls(high))
        dtype = cls._get_dtype(dtype)

        if not 0 <= low < high <= cls.order:
            raise ValueError(f"Arguments must satisfy `0 <= low < high <= order`, not `0 <= {low} < {high} <= {cls.order}`.")

        if seed is not None:
            if not isinstance(seed, (int, np.integer, np.random.Generator)):
                raise ValueError("Seed must be an integer, a numpy.random.Generator or None.")
            if isinstance(seed, (int, np.integer)) and seed < 0:
                raise ValueError("Seed must be non-negative.")

        if dtype != np.object_:
            rng = np.random.default_rng(seed)
            array = rng.integers(low, high, shape, dtype=dtype)
        else:
            array = np.empty(shape, dtype=dtype)
            iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
            _seed = None
            if seed is not None:
                if isinstance(seed, np.integer):
                    # np.integers not supported by random and seeding based on hashing deprecated since Python 3.9
                    _seed = seed.item()
                elif isinstance(seed, np.random.Generator):
                    _seed = seed.bit_generator.state['state']['state']
                    seed.bit_generator.advance(1)
                else:  # int
                    _seed = seed
            random.seed(_seed)
            for _ in iterator:
                array[iterator.multi_index] = random.randint(low, high - 1)

        return cls._view(array)

    @classmethod
    def Elements(cls, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Creates a 1-D array of the finite field's elements :math:`\{0, \dots, p^m-1\}`.

        Parameters
        ----------
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            A 1-D array of all the finite field's elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Elements()

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            GF.Elements()
            @suppress
            GF.display()
        """
        return cls.Range(0, cls.order, step=1, dtype=dtype)

    @classmethod
    def Identity(cls, size: int, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Creates an :math:`n \times n` identity matrix.

        Parameters
        ----------
        size
            The size :math:`n` along one dimension of the identity matrix.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            A 2-D identity matrix with shape `(size, size)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Identity(4)
        """
        dtype = cls._get_dtype(dtype)
        array = np.identity(size, dtype=dtype)
        return cls._view(array)

    @classmethod
    def Vandermonde(cls, element: ElementLike, rows: int, cols: int, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Creates an :math:`m \times n` Vandermonde matrix of :math:`a \in \mathrm{GF}(q)`.

        Parameters
        ----------
        element
            An element :math:`a` of :math:`\mathrm{GF}(q)`.
        rows
            The number of rows :math:`m` in the Vandermonde matrix.
        cols
            The number of columns :math:`n` in the Vandermonde matrix.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            A :math:`m \times n` Vandermonde matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**3, display="power")
            a = GF.primitive_element; a
            V = GF.Vandermonde(a, 7, 7); V
            @suppress
            GF.display()
        """
        if not isinstance(element, (int, np.integer, cls)):
            raise TypeError(f"Argument `element` must be an integer or element of {cls.name}, not {type(element)}.")
        if not isinstance(rows, (int, np.integer)):
            raise TypeError(f"Argument `rows` must be an integer, not {type(rows)}.")
        if not isinstance(cols, (int, np.integer)):
            raise TypeError(f"Argument `cols` must be an integer, not {type(cols)}.")
        if not rows > 0:
            raise ValueError(f"Argument `rows` must be non-negative, not {rows}.")
        if not cols > 0:
            raise ValueError(f"Argument `cols` must be non-negative, not {cols}.")

        dtype = cls._get_dtype(dtype)
        element = cls(element, dtype=dtype)
        if not element.ndim == 0:
            raise ValueError(f"Argument `element` must be element scalar, not {element.ndim}-D.")

        v = element ** np.arange(0, rows)
        V = np.power.outer(v, np.arange(0, cols))

        return V

    @classmethod
    def Vector(cls, array: ArrayLike, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Creates an array over :math:`\mathrm{GF}(p^m)` from length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`vector` method.

        Parameters
        ----------
        array
            An array over :math:`\mathrm{GF}(p)` with last dimension :math:`m`. An array with shape `(n1, n2, m)` has output shape
            `(n1, n2)`. By convention, the vectors are ordered from highest degree to 0-th degree.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            An array over :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**3, display="poly")
            a = GF.Vector([[1, 0, 2], [0, 2, 1]]); a
            a.vector()
            @suppress
            GF.display()
        """
        dtype = cls._get_dtype(dtype)
        order = cls.prime_subfield.order
        degree = cls.degree

        x = cls.prime_subfield(array)  # Convert element-like objects into the prime subfield
        x = x.view(np.ndarray)  # Convert into an integer array
        if not x.shape[-1] == degree:
            raise ValueError(f"The last dimension of `array` must be the field extension dimension {cls.degree}, not {x.shape[-1]}.")

        degrees = np.arange(degree - 1, -1, -1, dtype=dtype)
        y = np.sum(x * order**degrees, axis=-1, dtype=dtype)

        if np.isscalar(y):
            y = cls(y, dtype=dtype)
        else:
            y = cls._view(y)

        return y

    ###############################################################################
    # Instance methods
    ###############################################################################

    def additive_order(self) -> Union[np.integer, np.ndarray]:
        r"""
        Computes the additive order of each element in :math:`x`.

        Returns
        -------
        :
            An integer array of the additive order of each element in :math:`x`. The return value is a single integer if the
            input array :math:`x` is a scalar.

        Notes
        -----
        The additive order :math:`a` of :math:`x` in :math:`\mathrm{GF}(p^m)` is the smallest integer :math:`a`
        such that :math:`x a = 0`. With the exception of :math:`0`, the additive order of every element is
        the finite field's characteristic.

        Examples
        --------
        Below is the additive order of each element of :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.Elements(); x
            order = x.additive_order(); order
            x * order
            @suppress
            GF.display()
        """
        x = self
        field = type(self)

        if x.ndim == 0:
            order = np.int64(1) if x == 0 else np.int64(field.characteristic)
        else:
            order = field.characteristic * np.ones(x.shape, dtype=np.int64)
            order[np.where(x == 0)] = 1

        return order

    def multiplicative_order(self) -> Union[np.integer, np.ndarray]:
        r"""
        Computes the multiplicative order :math:`\textrm{ord}(x)` of each element in :math:`x`.

        Returns
        -------
        :
            An integer array of the multiplicative order of each element in :math:`x`. The return value is a single integer if the
            input array :math:`x` is a scalar.

        Notes
        -----
        The multiplicative order :math:`\textrm{ord}(x) = a` of :math:`x` in :math:`\mathrm{GF}(p^m)` is the smallest power :math:`a`
        such that :math:`x^a = 1`. If :math:`a = p^m - 1`, :math:`a` is said to be a generator of the multiplicative group
        :math:`\mathrm{GF}(p^m)^\times`.

        The multiplicative order of :math:`0` is not defined and will raise an :obj:`ArithmeticError`.

        :func:`FieldArray.multiplicative_order` should not be confused with :obj:`FieldArrayClass.order`. The former is a method on a
        *Galois field array* that returns the multiplicative order of elements. The latter is a property of the field, namely
        the finite field's order or size.

        Examples
        --------
        Below is the multiplicative order of each non-zero element of :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            # The multiplicative order of 0 is not defined
            x = GF.Range(1, GF.order); x
            order = x.multiplicative_order(); order
            x ** order

        The elements with :math:`\textrm{ord}(x) = 8` are multiplicative generators of :math:`\mathrm{GF}(3^2)^\times`.

        .. ipython:: python

            GF.primitive_elements
            @suppress
            GF.display()
        """
        if not np.count_nonzero(self) == self.size:
            raise ArithmeticError("The multiplicative order of 0 is not defined.")

        x = self
        field = type(self)

        if field.ufunc_mode == "jit-lookup":
            # This algorithm is faster if np.log() has a lookup table
            # β = α^k
            # ord(α) = p^m - 1
            # ord(β) = (p^m - 1) / gcd(p^m - 1, k)
            k = np.log(x)  # x as an exponent of α
            order = (field.order - 1) // np.gcd(field.order - 1, k)
        else:
            d = np.array(divisors(field.order - 1))  # Divisors d such that d | p^m - 1
            y = np.power.outer(x, d)  # x^d -- the first divisor d for which x^d == 1 is the order of x
            idxs = np.argmin(y, axis=-1)  # First index of divisors, which is the order of x
            order = d[idxs]  # The order of each element of x

        return order

    def is_quadratic_residue(self) -> Union[np.bool_, np.ndarray]:
        r"""
        Determines if the elements of :math:`x` are quadratic residues in the finite field.

        Returns
        -------
        :
            A boolean array indicating if each element in :math:`x` is a quadratic residue. The return value is a single boolean if the
            input array :math:`x` is a scalar.

        Notes
        -----
        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic residue* if there exists a :math:`y` such that
        :math:`y^2 = x` in the field.

        In fields with characteristic 2, every element is a quadratic residue. In fields with characteristic greater than 2,
        exactly half of the nonzero elements are quadratic residues (and they have two unique square roots).

        References
        ----------
        * Section 3.5.1 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.

        Examples
        --------
        Since :math:`\mathrm{GF}(2^3)` has characteristic :math:`2`, every element has a square root.

        .. ipython:: python

            GF = galois.GF(2**3, display="poly")
            x = GF.Elements(); x
            x.is_quadratic_residue()
            @suppress
            GF.display()

        In :math:`\mathrm{GF}(11)`, the characteristic is greater than :math:`2` so only half of the elements have square
        roots.

        .. ipython:: python

            GF = galois.GF(11)
            x = GF.Elements(); x
            x.is_quadratic_residue()
        """
        x = self
        field = type(self)

        if field.characteristic == 2:
            # All elements are quadratic residues if the field's characteristic is 2
            return np.ones(x.shape, dtype=bool) if x.ndim > 0 else np.bool_(True)
        else:
            # Compute the Legendre symbol on each element
            return x ** ((field.order - 1)//2) != field.characteristic - 1

    def vector(self, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Converts an array over :math:`\mathrm{GF}(p^m)` to length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`Vector` constructor. For an array with shape `(n1, n2)`, the output shape
        is `(n1, n2, m)`. By convention, the vectors are ordered from highest degree to 0-th degree.

        Parameters
        ----------
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`~galois.FieldArrayClass.dtypes`).

        Returns
        -------
        :
            An array over :math:`\mathrm{GF}(p)` with last dimension :math:`m`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**3, display="poly")
            a = GF([11, 7]); a
            vec = a.vector(); vec
            GF.Vector(vec)
            @suppress
            GF.display()
        """
        field = type(self)
        subfield = field.prime_subfield
        order = subfield.order
        degree = field.degree

        x = np.array(self)  # The original array as an integer array
        shape = list(self.shape) + [degree,]  # The new shape
        y = subfield.Zeros(shape, dtype=dtype)

        if self.dtype == np.object_:
            # Need a separate "if" statement because divmod() does not work with dtype=object input and integer dtype outputs
            for i in range(degree - 1, -1, -1):
                q, r = x // order, x % order
                y[..., i] = r
                x = q
        else:
            for i in range(degree - 1, -1, -1):
                q, r = divmod(x, order)
                y[..., i] = r
                x = q

        return y

    def row_reduce(self, ncols: Optional[int] = None) -> FieldArray:
        r"""
        Performs Gaussian elimination on the matrix to achieve reduced row echelon form.

        Parameters
        ----------
        ncols
            The number of columns to perform Gaussian elimination over. The default is `None` which represents
            the number of columns of the matrix.

        Returns
        -------
        :
            The reduced row echelon form of the input matrix.

        Notes
        -----

        The elementary row operations in Gaussian elimination are: swap the position of any two rows, multiply any row by
        a non-zero scalar, and add any row to a scalar multiple of another row.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            A = GF([[16, 12, 1, 25], [1, 10, 27, 29], [1, 0, 3, 19]]); A
            A.row_reduce()
            np.linalg.matrix_rank(A)
        """
        A_rre, _ = row_reduce(self, ncols=ncols)
        return A_rre

    def lu_decompose(self) -> Tuple[FieldArray, FieldArray]:
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices.

        Returns
        -------
        L :
            The lower triangular matrix.
        U :
            The upper triangular matrix.

        Notes
        -----
        The LU decomposition of :math:`\mathbf{A}` is defined as :math:`\mathbf{A} = \mathbf{L} \mathbf{U}`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            # Not every square matrix has an LU decomposition
            A = GF([[22, 11, 25, 11], [30, 27, 10, 3], [21, 16, 29, 7]]); A
            L, U = A.lu_decompose()
            L
            U
            np.array_equal(A, L @ U)
        """
        L, U = lu_decompose(self)
        return L, U

    def plu_decompose(self) -> Tuple[FieldArray, FieldArray, FieldArray]:
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices using partial pivoting.

        Returns
        -------
        P :
            The column permutation matrix.
        L :
            The lower triangular matrix.
        U :
            The upper triangular matrix.

        Notes
        -----
        The PLU decomposition of :math:`\mathbf{A}` is defined as :math:`\mathbf{A} = \mathbf{P} \mathbf{L} \mathbf{U}`. This is equivalent to
        :math:`\mathbf{P}^T \mathbf{A} = \mathbf{L} \mathbf{U}`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            A = GF([[0, 29, 2, 9], [20, 24, 5, 1], [2, 24, 1, 7]]); A
            P, L, U = A.plu_decompose()
            P
            L
            U
            np.array_equal(A, P @ L @ U)
            np.array_equal(P.T @ A, L @ U)
        """
        P, L, U, _ = plu_decompose(self)
        return P, L, U

    def row_space(self) -> FieldArray:
        r"""
        Computes the row space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The row space basis matrix. The rows of the basis matrix are the basis vectors that span the row space.
            The number of rows of the basis matrix is the dimension of the row space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *row space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^n\}` defined by all linear combinations of the rows
        of :math:`\mathbf{A}`. The row space has at most dimension :math:`\textrm{min}(m, n)`.

        The row space has properties :math:`\mathcal{R}(\mathbf{A}) = \mathcal{C}(\mathbf{A}^T)` and
        :math:`\textrm{dim}(\mathcal{R}(\mathbf{A})) + \textrm{dim}(\mathcal{LN}(\mathbf{A})) = m`.

        Examples
        --------
        The :func:`row_space` method defines basis vectors (its rows) that span the row space of :math:`\mathbf{A}`.
        The dimension of the row space and left null space sum to :math:`m`.

        .. ipython:: python

            m, n = 5, 3
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            R = A.row_space(); R
            LN = A.left_null_space(); LN
            R.shape[0] + LN.shape[0] == m
        """
        return row_space(self)

    def column_space(self) -> FieldArray:
        r"""
        Computes the column space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The column space basis matrix. The rows of the basis matrix are the basis vectors that span the column space.
            The number of rows of the basis matrix is the dimension of the column space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *column space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^m\}` defined by all linear combinations of the columns
        of :math:`\mathbf{A}`. The column space has at most dimension :math:`\textrm{min}(m, n)`.

        The column space has properties :math:`\mathcal{C}(\mathbf{A}) = \mathcal{R}(\mathbf{A}^T)`  and
        :math:`\textrm{dim}(\mathcal{C}(\mathbf{A})) + \textrm{dim}(\mathcal{N}(\mathbf{A})) = n`.

        Examples
        --------
        The :func:`column_space` method defines basis vectors (its rows) that span the column space of :math:`\mathbf{A}`.
        The dimension of the column space and null space sum to :math:`n`.

        .. ipython:: python

            m, n = 3, 5
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            C = A.column_space(); C
            N = A.null_space(); N
            C.shape[0] + N.shape[0] == n
        """
        return column_space(self)

    def left_null_space(self) -> FieldArray:
        r"""
        Computes the left null space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The left null space basis matrix. The rows of the basis matrix are the basis vectors that span the left null space.
            The number of rows of the basis matrix is the dimension of the left null space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *left null space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^m\}` that annihilates the rows of :math:`\mathbf{A}`, i.e.
        :math:`\mathbf{x}\mathbf{A} = \mathbf{0}`.

        The left null space has properties :math:`\mathcal{LN}(\mathbf{A}) = \mathcal{N}(\mathbf{A}^T)` and
        :math:`\textrm{dim}(\mathcal{R}(\mathbf{A})) + \textrm{dim}(\mathcal{LN}(\mathbf{A})) = m`.

        Examples
        --------
        The :func:`left_null_space` method defines basis vectors (its rows) that span the left null space of :math:`\mathbf{A}`.
        The dimension of the row space and left null space sum to :math:`m`.

        .. ipython:: python

            m, n = 5, 3
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            R = A.row_space(); R
            LN = A.left_null_space(); LN
            R.shape[0] + LN.shape[0] == m

        The left null space is the set of vectors that sum the rows to 0.

        .. ipython:: python

            LN @ A
        """
        return left_null_space(self)

    def null_space(self) -> FieldArray:
        r"""
        Computes the null space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The null space basis matrix. The rows of the basis matrix are the basis vectors that span the null space.
            The number of rows of the basis matrix is the dimension of the null space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *null space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^n\}` that annihilates the columns of :math:`\mathbf{A}`, i.e.
        :math:`\mathbf{A}\mathbf{x} = \mathbf{0}`.

        The null space has properties :math:`\mathcal{N}(\mathbf{A}) = \mathcal{LN}(\mathbf{A}^T)` and
        :math:`\textrm{dim}(\mathcal{C}(\mathbf{A})) + \textrm{dim}(\mathcal{N}(\mathbf{A})) = n`.

        Examples
        --------
        The :func:`null_space` method defines basis vectors (its rows) that span the null space of :math:`\mathbf{A}`.
        The dimension of the column space and null space sum to :math:`n`.

        .. ipython:: python

            m, n = 3, 5
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            C = A.column_space(); C
            N = A.null_space(); N
            C.shape[0] + N.shape[0] == n

        The null space is the set of vectors that sum the columns to 0.

        .. ipython:: python

            A @ N.T
        """
        return null_space(self)

    def field_trace(self) -> FieldArray:
        r"""
        Computes the field trace :math:`\mathrm{Tr}_{L / K}(x)` of the elements of :math:`x`.

        Returns
        -------
        :
            The field trace of :math:`x` in the prime subfield :math:`\mathrm{GF}(p)`.

        Notes
        -----
        The `self` array :math:`x` is over the extension field :math:`L = \mathrm{GF}(p^m)`. The field trace of :math:`x` is
        over the subfield :math:`K = \mathrm{GF}(p)`. In other words, :math:`\mathrm{Tr}_{L / K}(x) : L \rightarrow K`.

        For finite fields, since :math:`L` is a Galois extension of :math:`K`, the field trace of :math:`x` is defined as a sum
        of the Galois conjugates of :math:`x`.

        .. math:: \mathrm{Tr}_{L / K}(x) = \sum_{i=0}^{m-1} x^{p^i}

        References
        ----------
        * https://en.wikipedia.org/wiki/Field_trace

        Examples
        --------
        The field trace of the elements of :math:`\mathrm{GF}(3^2)` is shown below.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.Elements(); x
            y = x.field_trace(); y
            @suppress
            GF.display()
        """
        field = type(self)
        x = self

        if field.is_prime_field:
            return x.copy()
        else:
            subfield = field.prime_subfield
            p = field.characteristic
            m = field.degree
            conjugates = np.power.outer(x, p**np.arange(0, m, dtype=field.dtypes[-1]))
            trace = np.add.reduce(conjugates, axis=-1)
            return subfield._view(trace)

    def field_norm(self) -> FieldArray:
        r"""
        Computes the field norm :math:`\mathrm{N}_{L / K}(x)` of the elements of :math:`x`.

        Returns
        -------
        :
            The field norm of :math:`x` in the prime subfield :math:`\mathrm{GF}(p)`.

        Notes
        -----
        The `self` array :math:`x` is over the extension field :math:`L = \mathrm{GF}(p^m)`. The field norm of :math:`x` is
        over the subfield :math:`K = \mathrm{GF}(p)`. In other words, :math:`\mathrm{N}_{L / K}(x) : L \rightarrow K`.

        For finite fields, since :math:`L` is a Galois extension of :math:`K`, the field norm of :math:`x` is defined as a product
        of the Galois conjugates of :math:`x`.

        .. math:: \mathrm{N}_{L / K}(x) = \prod_{i=0}^{m-1} x^{p^i} = x^{(p^m - 1) / (p - 1)}

        References
        ----------
        * https://en.wikipedia.org/wiki/Field_norm

        Examples
        --------
        The field norm of the elements of :math:`\mathrm{GF}(3^2)` is shown below.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.Elements(); x
            y = x.field_norm(); y
            @suppress
            GF.display()
        """
        field = type(self)
        x = self

        if field.is_prime_field:
            return x.copy()
        else:
            subfield = field.prime_subfield
            p = field.characteristic
            m = field.degree
            norm = x**((p**m - 1) // (p - 1))
            return subfield._view(norm)

    def characteristic_poly(self) -> Poly:
        r"""
        Computes the characteristic polynomial of a finite field element :math:`a` or a square matrix :math:`\mathbf{A}`.

        This function can be invoked on single finite field elements (scalar 0-D arrays) or square :math:`n \times n`
        matrices (2-D arrays).

        Returns
        -------
        :
            For scalar inputs, the degree-:math:`m` characteristic polynomial :math:`p_a(x)` of :math:`a` over :math:`\mathrm{GF}(p)`.
            For square :math:`n \times n` matrix inputs, the degree-:math:`n` characteristic polynomial :math:`p_A(x)` of
            :math:`\mathbf{A}` over :math:`\mathrm{GF}(p^m)`.

        Notes
        -----
        An element :math:`a` of :math:`\mathrm{GF}(p^m)` has characteristic polynomial :math:`p_a(x)` over :math:`\mathrm{GF}(p)`.
        The characteristic polynomial when evaluated in :math:`\mathrm{GF}(p^m)` annihilates :math:`a`, i.e. :math:`p_a(a) = 0`.
        In prime fields :math:`\mathrm{GF}(p)`, the characteristic polynomial of :math:`a` is simply :math:`p_a(x) = x - a`.

        An :math:`n \times n` matrix :math:`\mathbf{A}` has characteristic polynomial
        :math:`p_A(x) = \textrm{det}(x\mathbf{I} - \mathbf{A})` over :math:`\mathrm{GF}(p^m)`. The constant coefficient of the
        characteristic polynomial is :math:`\textrm{det}(-\mathbf{A})`. The :math:`x^{n-1}` coefficient of the characteristic
        polynomial is :math:`-\textrm{Tr}(\mathbf{A})`. The characteristic polynomial annihilates :math:`\mathbf{A}`, i.e.
        :math:`p_A(\mathbf{A}) = \mathbf{0}`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Characteristic_polynomial

        Examples
        --------
        The characteristic polynomial of the element :math:`a`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = GF.Random(); a
            poly = a.characteristic_poly(); poly
            # The characteristic polynomial annihilates a
            poly(a, field=GF)

        The characteristic polynomial of the square matrix :math:`\mathbf{A}`.

        .. ipython:: python

            GF = galois.GF(3**5)
            A = GF.Random((3,3)); A
            poly = A.characteristic_poly(); poly
            # The x^0 coefficient is det(-A)
            poly.coeffs[-1] == np.linalg.det(-A)
            # The x^n-1 coefficient is -Tr(A)
            poly.coeffs[1] == -np.trace(A)
            # The characteristic polynomial annihilates the matrix A
            poly(A, elementwise=False)
        """
        if self.ndim == 0:
            return self._characteristic_poly_element()
        elif self.ndim == 2:
            return self._characteristic_poly_matrix()
        else:
            raise ValueError(f"The array must be either 0-D to return the characteristic polynomial of a single element or 2-D to return the characteristic polynomial of a square matrix, not have shape {self.shape}.")

    def _characteristic_poly_element(self):
        field = type(self)
        a = self
        x = Poly.Identity(field)

        if field.is_prime_field:
            return x - a
        else:
            powers = a**(field.characteristic**np.arange(0, field.degree, dtype=field.dtypes[-1]))
            poly = Poly.Roots(powers, field=field)
            poly = Poly(poly.coeffs, field=field.prime_subfield)
            return poly

    def _characteristic_poly_matrix(self):
        if not self.shape[0] == self.shape[1]:
            raise ValueError(f"The 2-D array must be square to compute its characteristic polynomial, not have shape {self.shape}.")

        field = type(self)
        A = self

        # Compute P = xI - A
        P = np.zeros(self.shape, dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[0]):
                if i == j:
                    P[i,j] = Poly([1, -A[i,j]], field=field)
                else:
                    P[i,j] = Poly([-A[i,j]], field=field)

        # Compute det(P)
        return self._compute_poly_det(P)

    def _compute_poly_det(self, A):
        if A.shape == (2,2):
            return A[0,0]*A[1,1] - A[0,1]*A[1,0]

        field = type(self)
        n = A.shape[0]  # Size of the nxn matrix

        det = Poly.Zero(field)
        for i in range(n):
            idxs = np.delete(np.arange(0, n), i)
            if i % 2 == 0:
                det += A[0,i] * self._compute_poly_det(A[1:,idxs])
            else:
                det -= A[0,i] * self._compute_poly_det(A[1:,idxs])

        return det

    def minimal_poly(self) -> Poly:
        r"""
        Computes the minimal polynomial of a finite field element :math:`a`.

        This function can be invoked only on single finite field elements (scalar 0-D arrays).

        Returns
        -------
        :
            For scalar inputs, the minimal polynomial :math:`p_a(x)` of :math:`a` over :math:`\mathrm{GF}(p)`.

        Notes
        -----
        An element :math:`a` of :math:`\mathrm{GF}(p^m)` has minimal polynomial :math:`p_a(x)` over :math:`\mathrm{GF}(p)`.
        The minimal polynomial when evaluated in :math:`\mathrm{GF}(p^m)` annihilates :math:`a`, i.e. :math:`p_a(a) = 0`.
        The minimal polynomial always divides the characteristic polynomial. In prime fields :math:`\mathrm{GF}(p)`, the
        minimal polynomial of :math:`a` is simply :math:`p_a(x) = x - a`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Minimal_polynomial_(field_theory)
        * https://en.wikipedia.org/wiki/Minimal_polynomial_(linear_algebra)

        Examples
        --------
        The characteristic polynomial of the element :math:`a`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = GF.Random(); a
            poly = a.minimal_poly(); poly
            # The minimal polynomial annihilates a
            poly(a, field=GF)
            # The minimal polynomial always divides the characteristic polynomial
            a.characteristic_poly() // poly
        """
        if self.ndim == 0:
            return self._minimal_poly_element()
        # elif self.ndim == 2:
        #     return self._minimal_poly_matrix()
        else:
            raise ValueError(f"The array must be either 0-D to return the minimal polynomial of a single element or 2-D to return the minimal polynomial of a square matrix, not have shape {self.shape}.")

    def _minimal_poly_element(self):
        field = type(self)
        a = self
        x = Poly.Identity(field)

        if field.is_prime_field:
            return x - a
        else:
            conjugates = np.unique(a**(field.characteristic**np.arange(0, field.degree, dtype=field.dtypes[-1])))
            poly = Poly.Roots(conjugates, field=field)
            poly = Poly(poly.coeffs, field=field.prime_subfield)
            return poly

    ###############################################################################
    # Array creation functions that need redefined
    ###############################################################################

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
                kwargs["dtype"] = field.dtypes[-1]

            return getattr(field, field._OVERRIDDEN_UFUNCS[ufunc])(ufunc, method, inputs, kwargs, meta)

        elif ufunc in field._UNSUPPORTED_UFUNCS:
            raise NotImplementedError(f"The NumPy ufunc {ufunc.__name__!r} is not supported on {field.name} arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            if ufunc in [np.bitwise_and, np.bitwise_or, np.bitwise_xor] and method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            inputs, kwargs = field._view_inputs_as_ndarray(inputs, kwargs)
            output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

            if ufunc in field._UFUNCS_REQUIRING_VIEW and output is not None:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

            return output

    def __array_function__(self, func, types, args, kwargs):
        """
        Override the standard NumPy function calls with the new finite field functions.
        """
        field = type(self)

        if func in field._OVERRIDDEN_FUNCTIONS:
            output = getattr(field, field._OVERRIDDEN_FUNCTIONS[func])(*args, **kwargs)

        elif func in field._OVERRIDDEN_LINALG_FUNCTIONS:
            output = field._OVERRIDDEN_LINALG_FUNCTIONS[func](*args, **kwargs)

        elif func in field._UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(f"The NumPy function {func.__name__!r} is not supported on FieldArray. If you believe this function should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\nIf you'd like to perform this operation on the data, you should first call `array = array.view(np.ndarray)` and then call the function.")

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._verify_array_like_types_and_values(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in field._FUNCTIONS_REQUIRING_VIEW:
                output = field._view(output) if not np.isscalar(output) else field(output, dtype=self.dtype)

        return output

    ###############################################################################
    # Arithmetic functions that need redefiend
    ###############################################################################

    def __pow__(self, other):
        # We call power here instead of `super().__pow__(other)` because when doing so `x ** GF(2)` will invoke `np.square(x)`
        # and not throw a TypeError. This way `np.power(x, GF(2))` is called which correctly checks whether the second argument
        # is an integer.
        return np.power(self, other)

    ###############################################################################
    # Miscellaneous functions that need redefined
    ###############################################################################

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        """
        Before changing the array's data type, ensure it is a supported data type for this finite field.
        """
        if dtype not in type(self).dtypes:
            raise TypeError(f"{type(self).name} arrays can only be cast as integer dtypes in {type(self).dtypes}, not {dtype}.")
        return super().astype(dtype, **kwargs)

    def dot(self, b, out=None):
        """
        The `np.dot(a, b)` ufunc is also available as `a.dot(b)`. Need to override this method for consistent results.
        """
        return dot(self, b, out=out)

    ###############################################################################
    # Display methods
    ###############################################################################

    def __repr__(self) -> str:
        """
        Displays the array specifying the class and finite field order.

        This function prepends `GF(` and appends `, order=p^m)`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    x = GF([4, 2, 7, 5])
                    x

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    x = GF([4, 2, 7, 5])
                    x

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    x = GF([4, 2, 7, 5])
                    x
                    @suppress
                    GF.display()
        """
        return self._display("repr")

    def __str__(self) -> str:
        """
        Displays the array without specifying the class or finite field order.

        This function does not prepend `GF(` and or append `, order=p^m)`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    x = GF([4, 2, 7, 5])
                    print(x)

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    x = GF([4, 2, 7, 5])
                    print(x)

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    x = GF([4, 2, 7, 5])
                    print(x)
        """
        return self._display("str")

    def _display(self, mode: Literal["repr", "str"]) -> str:
        # View the array as an ndarray so that the scalar -> 0-D array conversion in __array_finalize__() for Galois field
        # arrays isn't continually invoked. This improves performance slightly.
        x = self.view(np.ndarray)
        field = type(self)

        separator = ", "
        prefix = "GF(" if mode == "repr" else ""
        order = field._order_str if mode == "repr" else ""
        suffix = ")" if mode == "repr" else ""
        formatter = field._formatter(self)

        field._element_fixed_width = None  # Do not print with fixed-width
        field._element_fixed_width_counter = 0  # Reset element width counter

        string = np.array2string(x, separator=separator, prefix=prefix, suffix=suffix, formatter=formatter)

        if formatter != {}:
            # We are using special print methods and must perform element alignment ourselves. We will print each element
            # a second time use the max width of any element observed on the first array2string() call.
            field._element_fixed_width = field._element_fixed_width_counter

            string = np.array2string(x, separator=separator, prefix=prefix, suffix=suffix, formatter=formatter)

        field._element_fixed_width = None
        field._element_fixed_width_counter = 0

        # Determine the width of the last line in the string
        if mode == "repr":
            idx = string.rfind("\n") + 1
            last_line_width = len(string[idx:] + ", " + order + suffix)

            if last_line_width <= np.get_printoptions()["linewidth"]:
                return prefix + string + ", " + order + suffix
            else:
                return prefix + string + ",\n" + " "*len(prefix) + order + suffix
        else:
            return prefix + string + suffix
