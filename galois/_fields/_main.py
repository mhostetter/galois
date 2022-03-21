"""
A module that contains the main classes for Galois fields -- FieldClass, FieldArray,
and Poly. They're all in one file because they have circular dependencies. The specific GF2
FieldClass is also included.
"""
import contextlib
import inspect
import math
import random
from typing import Tuple, List, Sequence, Iterable, Optional, Union, overload
from typing_extensions import Literal

import numba
import numpy as np

from .._overrides import set_module
from .._poly_conversion import integer_to_poly, poly_to_integer, str_to_integer, poly_to_str, sparse_poly_to_integer, sparse_poly_to_str, str_to_sparse_poly
from .._prime import divisors

from ._dtypes import DTYPES
from ._linalg import dot, row_reduce, lu_decompose, plu_decompose, row_space, column_space, left_null_space, null_space
from ._functions import FunctionMeta
from ._ufuncs import UfuncMeta

__all__ = ["FieldClass", "FieldArray", "GF2", "Poly"]


###############################################################################
# NumPy ndarray subclass for Galois fields
###############################################################################

@set_module("galois")
class FieldClass(FunctionMeta, UfuncMeta):
    """
    Defines a metaclass for all :obj:`galois.FieldArray` classes.

    Important
    ---------
    :obj:`galois.FieldClass` is a metaclass for :obj:`galois.FieldArray` subclasses created with the class factory
    :func:`galois.GF` and should not be instantiated directly. This metaclass gives :obj:`galois.FieldArray` subclasses
    methods and attributes related to their Galois fields.

    This class is included in the API to allow the user to test if a class is a Galois field array class.

    .. ipython:: python

        GF = galois.GF(7)
        isinstance(GF, galois.FieldClass)
    """
    # pylint: disable=no-value-for-parameter,unsupported-membership-test,abstract-method,too-many-public-methods

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._characteristic = kwargs.get("characteristic", 0)
        cls._degree = kwargs.get("degree", 0)
        cls._order = kwargs.get("order", 0)
        cls._order_str = None
        cls._ufunc_mode = None
        cls._ufunc_target = None
        cls._dtypes = cls._determine_dtypes()

        if "irreducible_poly" in kwargs:
            cls._irreducible_poly = kwargs["irreducible_poly"]
            cls._irreducible_poly_int = cls._irreducible_poly.integer
        else:
            cls._irreducible_poly = None
            cls._irreducible_poly_int = 0
        cls._primitive_element = kwargs.get("primitive_element", None)

        cls._is_primitive_poly = kwargs.get("is_primitive_poly", None)
        cls._prime_subfield = None

        cls._display_mode = "int"

        if cls.degree == 1:
            cls._order_str = f"order={cls.order}"
        else:
            cls._order_str = f"order={cls.characteristic}^{cls.degree}"

        cls._element_fixed_width = None
        cls._element_fixed_width_counter = 0

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

    def __repr__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

    ###############################################################################
    # Helper methods
    ###############################################################################

    def _determine_dtypes(cls):
        """
        Determine which NumPy integer data types are valid for this finite field. At a minimum, valid dtypes are ones that
        can hold x for x in [0, order).
        """
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

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

            * `"auto"`: Selects `"jit-lookup"` for fields with order less than :math:`2^{20}`, `"jit-calculate"` for larger fields, and `"python-calculate"`
              for fields whose elements cannot be represented with :obj:`numpy.int64`.
            * `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
              In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
            * `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The `"jit-calculate"` mode is designed for large
              fields that cannot or should not store lookup tables in RAM. Generally, the `"jit-calculate"` mode is slower than `"jit-lookup"`.
            * `"python-calculate"`: Uses pure-Python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
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

            * `"int"`: Sets the display mode to the :ref:`integer representation <Integer representation>`.
            * `"poly"`: Sets the display mode to the :ref:`polynomial representation <Polynomial representation>`.
            * `"power"`: Sets the display mode to the :ref:`power representation <Power representation>`.

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

        context = DisplayContext(cls)
        cls._display_mode = mode  # Set the new state

        return context

    def repr_table(
        cls,
        primitive_element: Optional[Union[int, str, np.ndarray, "FieldArray"]] = None,
        sort: Literal["power", "poly", "vector", "int"] = "power"
    ) -> str:
        r"""
        Generates a finite field element representation table comparing the power, polynomial, vector, and integer representations.

        Parameters
        ----------
        primitive_element
            The primitive element to use for the power representation. The default is `None` which uses the field's
            default primitive element, :obj:`FieldClass.primitive_element`. If an array, it must be a 0-D array.
        sort
            The sorting method for the table. The default is `"power"`. Sorting by `"power"` will order the rows of the table by ascending
            powers of the primitive element. Sorting by any of the others will order the rows in lexicographically-increasing polynomial/vector
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
        if primitive_element is None:
            primitive_element = cls.primitive_element

        primitive_element = cls(primitive_element)
        degrees = np.arange(0, cls.order - 1)
        x = primitive_element**degrees
        if sort != "power":
            idxs = np.argsort(x)
            degrees, x = degrees[idxs], x[idxs]
        x = np.concatenate((np.atleast_1d(cls(0)), x))  # Add 0 = alpha**-Inf
        prim = poly_to_str(integer_to_poly(primitive_element, cls.characteristic))

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
        x: Optional["FieldArray"] = None,
        y: Optional["FieldArray"] = None
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
        if cls._degree == 1:
            return f"GF({cls._characteristic})"
        else:
            return f"GF({cls._characteristic}^{cls._degree})"

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
        # Ensure accesses of this property don't alter it
        return cls._irreducible_poly.copy()

    @property
    def is_primitive_poly(cls) -> bool:
        r"""
        Indicates whether the :obj:`FieldClass.irreducible_poly` is a primitive polynomial. If so, :math:`x` is a primitive element
        of the finite field.

        The default irreducible polynomial is a Conway polynomial, see :func:`galois.conway_poly`, which is a primitive
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
        n = cls.order - 1
        totatives = [t for t in range(1, n + 1) if math.gcd(n, t) == 1]
        powers = np.array(totatives)
        return np.sort(cls.primitive_element ** powers)

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
    def prime_subfield(cls) -> "FieldClass":
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
    (which are `FieldClass` instances) include the methods and properties from the metaclass. Python does not
    natively include metaclass properties in dir().

    This is a separate class because it will be mixed in to `GF2Meta`, `GF2mMeta`, `GFpMeta`, and `GFpmMeta` separately. Otherwise, the
    sphinx documentation of `FieldArray` gets messed up.

    Since, `GF2` has this class mixed in, its docs are messed up. Because of that, we added a separate Sphinx template `class_only_init.rst`
    to suppress all the methods except __init__() so the docs are more presentable.
    """

    def __dir__(cls):
        if isinstance(cls, FieldClass):
            meta_dir = dir(type(cls))
            classmethods = [attribute for attribute in super().__dir__() if attribute[0] != "_" and inspect.ismethod(getattr(cls, attribute))]
            return sorted(meta_dir + classmethods)
        else:
            return super().__dir__()


class DisplayContext(contextlib.AbstractContextManager):
    """
    Simple context manager for the :obj:`FieldClass.display` method.
    """

    def __init__(self, cls):
        # Save the previous state
        self.cls = cls
        self.mode = cls.display_mode

    def __enter__(self):
        # Don't need to do anything, we already set the new mode in the display() method
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset mode and upon exiting the context
        self.cls._display_mode = self.mode


###############################################################################
# NumPy arrays over Galois fields
###############################################################################

@set_module("galois")
class FieldArray(np.ndarray, metaclass=FieldClass):
    r"""
    A :ref:`Galois field array` over :math:`\mathrm{GF}(p^m)`.

    Important
    ---------
        :obj:`galois.FieldArray` is an abstract base class for all :ref:`Galois field array classes <Galois field array class>` and cannot
        be instantiated directly. Instead, :obj:`galois.FieldArray` subclasses are created using the class factory :func:`galois.GF`.

        This class is included in the API to allow the user to test if an array is a Galois field array subclass.

        .. ipython:: python

            GF = galois.GF(7)
            issubclass(GF, galois.FieldArray)
            x = GF([1, 2, 3]); x
            isinstance(x, galois.FieldArray)

    See :ref:`Galois Field Classes` for a detailed discussion of the relationship between :obj:`galois.FieldClass` and
    :obj:`galois.FieldArray`.

    See :ref:`Array Creation` for a detailed discussion on creating arrays (with and without copying) from array-like
    objects, valid NumPy data types, and other :obj:`galois.FieldArray` classmethods.

    Examples
    --------
    Create a :ref:`Galois field array class` using the class factory :func:`galois.GF`.

    .. ipython:: python

        GF = galois.GF(3**5)
        print(GF)

    The *Galois field array class* `GF` is a subclass of :obj:`galois.FieldArray`, with :obj:`galois.FieldClass` as its
    metaclass.

    .. ipython:: python

        isinstance(GF, galois.FieldClass)
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
        array: Union[int, str, Iterable, np.ndarray, "FieldArray"],
        dtype: Optional[Union[np.dtype, int, object]] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ) -> "FieldArray":
        if cls is FieldArray:
            raise NotImplementedError("FieldArray is an abstract base class that cannot be directly instantiated. Instead, create a FieldArray subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` and instantiate an array using `x = GF(array_like)`.")
        return cls._array(array, dtype=dtype, copy=copy, order=order, ndmin=ndmin)

    def __init__(
        self,
        array: Union[int, str, Iterable, np.ndarray, "FieldArray"],
        dtype: Optional[Union[np.dtype, int, object]] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ):
        r"""
        Creates a :ref:`Galois field array` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        array
            The input array-like object to be converted to a *Galois field array*. See :ref:`Array Creation` for a detailed discussion
            about creating new arrays and array-like objects.

            * :obj:`int`: A single integer, which is the :ref:`integer representation <Integer representation>` of a finite field element,
              creates a 0-D array (scalar).
            * :obj:`str`: A single string, which is the :ref:`polynomial representation <Polynomial representation>` of a finite field element,
              creates a 0-D array (scalar).
            * :obj:`tuple`, :obj:`list`: A list or tuple (or nested lists/tuples) of integers or strings (which can be mixed and matched) creates
              an array of finite field elements from their integer or polynomial representations.
            * :obj:`numpy.ndarray`, :obj:`galois.FieldArray`: A NumPy array of integers creates a copy of the array over this specific field.

        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this class (the first element in :obj:`galois.FieldClass.dtypes`).
        copy
            The `copy` keyword argument from :func:`numpy.array`. The default is `True` which makes a copy of the input array.
        order
            The `order` keyword argument from :func:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
        ndmin
            The `ndmin` keyword argument from :func:`numpy.array`. The minimum number of dimensions of the output.
            The default is 0.
        """
        # pylint: disable=unused-argument,super-init-not-called
        # Adding __init__ and not doing anything is done to overwrite the superclass's __init__ docstring
        return

    @classmethod
    def _get_dtype(cls, dtype):
        if dtype is None:
            return cls.dtypes[0]

        # Convert "dtype" to a numpy dtype. This does platform specific conversion, if necessary.
        # For example, np.dtype(int) == np.int64 (on some systems).
        dtype = np.dtype(dtype)
        if dtype not in cls.dtypes:
            raise TypeError(f"{cls.name} arrays only support dtypes {[np.dtype(d).name for d in cls.dtypes]}, not {dtype.name!r}.")

        return dtype

    @classmethod
    def _array(cls, array_like, dtype=None, copy=True, order="K", ndmin=0):
        dtype = cls._get_dtype(dtype)
        array_like = cls._check_array_like_object(array_like)
        array = np.array(array_like, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
        return array.view(cls)

    @classmethod
    def _check_array_like_object(cls, array_like):
        if isinstance(array_like, cls):
            # If this was a previously-created and vetted array, there's no need to reverify
            return array_like

        if isinstance(array_like, str):
            # Convert the string to an integer and verify it's in range
            array_like = cls._check_string_value(array_like)
            cls._check_array_values(array_like)
        elif isinstance(array_like, (int, np.integer)):
            # Just check that the single int is in range
            cls._check_array_values(array_like)
        elif isinstance(array_like, (list, tuple)):
            # Recursively check the items in the iterable to ensure they're of the correct type
            # and that their values are in range
            array_like = cls._check_iterable_types_and_values(array_like)
        elif isinstance(array_like, np.ndarray):
            # If this a NumPy array, but not a FieldArray, verify the array
            if array_like.dtype == np.object_:
                array_like = cls._check_array_types_dtype_object(array_like)
            elif not np.issubdtype(array_like.dtype, np.integer):
                raise TypeError(f"{cls.name} arrays must have integer dtypes, not {array_like.dtype}.")
            cls._check_array_values(array_like)
        else:
            raise TypeError(f"{cls.name} arrays can be created with scalars of type int, not {type(array_like)}.")

        return array_like

    @classmethod
    def _check_iterable_types_and_values(cls, iterable):
        new_iterable = []
        for item in iterable:
            if isinstance(item, (list, tuple)):
                item = cls._check_iterable_types_and_values(item)
                new_iterable.append(item)
                continue

            if isinstance(item, str):
                item = cls._check_string_value(item)
            elif not isinstance(item, (int, np.integer, FieldArray)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with an iterable, each element must be an integer. Found type {type(item)}.")

            cls._check_array_values(item)
            # if not 0 <= item < cls.order:
            #     raise ValueError(f"{cls.name} arrays must have elements in 0 <= x < {cls.order}, not {item}.")

            # Ensure the type is int so dtype=object classes don't get all mixed up
            new_iterable.append(int(item))

        return new_iterable

    @classmethod
    def _check_array_types_dtype_object(cls, array):
        if array.size == 0:
            return array
        if array.ndim == 0:
            if not isinstance(array[()], (int, np.integer, FieldArray)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with `dtype=object`, each element must be an integer. Found type {type(array[()])}.")
            return int(array)

        iterator = np.nditer(array, flags=["multi_index", "refs_ok"])
        for _ in iterator:
            a = array[iterator.multi_index]
            if not isinstance(a, (int, np.integer, FieldArray)):
                raise TypeError(f"When {cls.name} arrays are created/assigned with a numpy array with `dtype=object`, each element must be an integer. Found type {type(a)}.")

            # Ensure the type is int so dtype=object classes don't get all mixed up
            array[iterator.multi_index] = int(a)

        return array

    @classmethod
    def _check_array_values(cls, array):
        if not isinstance(array, np.ndarray):
            # Convert single integer to array so next step doesn't fail
            array = np.array(array)

        # Check the value of the "field elements" and make sure they are valid
        if np.any(array < 0) or np.any(array >= cls.order):
            idxs = np.logical_or(array < 0, array >= cls.order)
            values = array if array.ndim == 0 else array[idxs]
            raise ValueError(f"{cls.name} arrays must have elements in `0 <= x < {cls.order}`, not {values}.")

    @classmethod
    def _check_string_value(cls, string):
        return str_to_integer(string, cls.prime_subfield)

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zeros(
        cls,
        shape: Union[int, Sequence[int]],
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates an array of all zeros.

        Parameters
        ----------
        shape
            A NumPy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M, N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        return array.view(cls)

    @classmethod
    def Ones(
        cls,
        shape: Union[int, Sequence[int]],
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates an array of all ones.

        Parameters
        ----------
        shape
            A NumPy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M, N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        return array.view(cls)

    @classmethod
    def Range(
        cls,
        start: int,
        stop: int,
        step: Optional[int] = 1,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates a 1-D array with a range of field elements.

        Parameters
        ----------
        start
            The starting finite field element (inclusive) in its :ref:`integer representation <Integer representation>`.
        stop
            The stopping finite field element (exclusive) in its :ref:`integer representation <Integer representation>`.
        step
            The increment between finite field element. The default is 1.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        if not stop <= cls.order:
            raise ValueError(f"The stopping value must be less than the field order of {cls.order}, not {stop}.")
        dtype = cls._get_dtype(dtype)
        array = np.arange(start, stop, step=step, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Random(
        cls,
        shape: Union[int, Sequence[int]] = (),
        low: Optional[int] = 0,
        high: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates an array with random field elements.

        Parameters
        ----------
        shape
            A NumPy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M, N)`, represents a 2-D array with each element indicating the size in each dimension.
        low
            The smallest finite field element (inclusive) in its :ref:`integer representation <Integer representation>`.
            The default is 0.
        high
            The largest finite field element (exclusive) in its :ref:`integer representation <Integer representation>`.
            The default is `None` which represents the field's order :math:`p^m`.
        seed
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        dtype = cls._get_dtype(dtype)
        high = cls.order if high is None else high
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

        return array.view(cls)

    @classmethod
    def Elements(
        cls,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        r"""
        Creates a 1-D array of the finite field's elements :math:`\{0, \dots, p^m-1\}`.

        Parameters
        ----------
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
    def Identity(
        cls,
        size: int,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        r"""
        Creates an :math:`n \times n` identity matrix.

        Parameters
        ----------
        size
            The size :math:`n` along one axis of the matrix. The resulting array has shape `(size, size)`.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        return array.view(cls)

    @classmethod
    def Vandermonde(
        cls,
        a: Union[int, "FieldArray"],
        m: int,
        n: int,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        r"""
        Creates an :math:`m \times n` Vandermonde matrix of :math:`a \in \mathrm{GF}(q)`.

        Parameters
        ----------
        a
            An element of :math:`\mathrm{GF}(q)`.
        m
            The number of rows in the Vandermonde matrix.
        n
            The number of columns in the Vandermonde matrix.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        if not isinstance(a, (int, np.integer, cls)):
            raise TypeError(f"Argument `a` must be an integer or element of {cls.name}, not {type(a)}.")
        if not isinstance(m, (int, np.integer)):
            raise TypeError(f"Argument `m` must be an integer, not {type(m)}.")
        if not isinstance(n, (int, np.integer)):
            raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
        if not m > 0:
            raise ValueError(f"Argument `m` must be non-negative, not {m}.")
        if not n > 0:
            raise ValueError(f"Argument `n` must be non-negative, not {n}.")

        dtype = cls._get_dtype(dtype)
        a = cls(a, dtype=dtype)
        if not a.ndim == 0:
            raise ValueError(f"Argument `a` must be a scalar, not {a.ndim}-D.")

        v = a ** np.arange(0, m)
        V = np.power.outer(v, np.arange(0, n))

        return V

    @classmethod
    def Vector(
        cls,
        array: Union[Iterable, np.ndarray, "FieldArray"],
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
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
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        order = cls.prime_subfield.order
        degree = cls.degree
        array = cls.prime_subfield(array).view(np.ndarray).astype(cls.dtypes[-1])  # Use the largest dtype so computation doesn't overflow
        if not array.shape[-1] == degree:
            raise ValueError(f"The last dimension of `array` must be the field extension dimension {cls.degree}, not {array.shape[-1]}.")
        degrees = np.arange(degree - 1, -1, -1, dtype=cls.dtypes[-1])
        array = np.sum(array * order**degrees, axis=-1)
        return cls(array, dtype=dtype)

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

        :func:`FieldArray.multiplicative_order` should not be confused with :obj:`FieldClass.order`. The former is a method on a
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

    def vector(
        self,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        r"""
        Converts an array over :math:`\mathrm{GF}(p^m)` to length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`Vector` constructor. For an array with shape `(n1, n2)`, the output shape
        is `(n1, n2, m)`. By convention, the vectors are ordered from highest degree to 0-th degree.

        Parameters
        ----------
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class (the first element in :obj:`galois.FieldClass.dtypes`).

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
        order = type(self).prime_subfield.order
        degree = type(self).degree
        array = self.view(np.ndarray)
        array = np.repeat(array, degree).reshape(*array.shape, degree)
        x = 0
        for i in range(degree):
            q = (array[...,i] - x) // order**(degree - 1 - i)
            array[...,i] = q
            x += q*order**(degree - 1 - i)
        return type(self).prime_subfield(array, dtype=dtype)  # pylint: disable=unexpected-keyword-arg

    def row_reduce(
        self,
        ncols: Optional[int] = None
    ) -> "FieldArray":
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

    def lu_decompose(self) -> Tuple["FieldArray", "FieldArray"]:
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

    def plu_decompose(self) -> Tuple["FieldArray", "FieldArray", "FieldArray"]:
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

    def row_space(self) -> "FieldArray":
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

    def column_space(self) -> "FieldArray":
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

    def left_null_space(self) -> "FieldArray":
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

    def null_space(self) -> "FieldArray":
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

    def field_trace(self) -> "FieldArray":
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
            return trace.view(subfield)

    def field_norm(self) -> "FieldArray":
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
            return norm.view(subfield)

    def characteristic_poly(self) -> "Poly":
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

    def minimal_poly(self) -> "Poly":
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
            a.characteristic_poly() / poly
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
    # NumPy getter/setter functions that need redefined
    ###############################################################################

    def __getitem__(self, key):
        """
        Ensure that slices that return a single value return a 0-D Galois field array and not a single integer. This
        ensures subsequent arithmetic with the finite field scalar works properly.
        """
        item = super().__getitem__(key)
        if np.isscalar(item):
            item = self.__class__(item, dtype=self.dtype)
        return item

    def __setitem__(self, key, value):
        """
        Before assigning new values to a Galois field array, ensure the values are valid finite field elements. That is,
        they are within [0, p^m).
        """
        value = self._check_array_like_object(value)
        super().__setitem__(key, value)

    ###############################################################################
    # Array creation functions that need redefined
    ###############################################################################

    def __array_finalize__(self, obj):
        """
        A NumPy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        if obj is not None and not isinstance(obj, FieldArray):
            # Only invoked on view casting
            if obj.dtype not in type(self).dtypes:
                raise TypeError(f"{type(self).name} can only have integer dtypes {type(self).dtypes}, not {obj.dtype}.")
            self._check_array_values(obj)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override the standard NumPy ufunc calls with the new finite field ufuncs.
        """
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

        if ufunc in type(self)._OVERRIDDEN_UFUNCS:
            # Set all ufuncs with "casting" keyword argument to "unsafe" so we can cast unsigned integers
            # to integers. We know this is safe because we already verified the inputs.
            if method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            # Need to set the intermediate dtype for reduction operations or an error will be thrown. We
            # use the largest valid dtype for this field.
            if method in ["reduce"]:
                kwargs["dtype"] = type(self).dtypes[-1]

            return getattr(type(self), type(self)._OVERRIDDEN_UFUNCS[ufunc])(ufunc, method, inputs, kwargs, meta)

        elif ufunc in type(self)._UNSUPPORTED_UFUNCS:
            raise NotImplementedError(f"The numpy ufunc {ufunc.__name__!r} is not supported on {type(self).name} arrays. If you believe this ufunc should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

        else:
            if ufunc in [np.bitwise_and, np.bitwise_or, np.bitwise_xor] and method not in ["reduce", "accumulate", "at", "reduceat"]:
                kwargs["casting"] = "unsafe"

            inputs, kwargs = type(self)._view_inputs_as_ndarray(inputs, kwargs)
            output = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)  # pylint: disable=no-member

            if ufunc in type(self)._UFUNCS_REQUIRING_VIEW and output is not None:
                output = output.view(type(self)) if not np.isscalar(output) else type(self)(output, dtype=self.dtype)

            return output

    def __array_function__(self, func, types, args, kwargs):
        """
        Override the standard NumPy function calls with the new finite field functions.
        """
        if func in type(self)._OVERRIDDEN_FUNCTIONS:
            output = getattr(type(self), type(self)._OVERRIDDEN_FUNCTIONS[func])(*args, **kwargs)

        elif func in type(self)._OVERRIDDEN_LINALG_FUNCTIONS:
            output = type(self)._OVERRIDDEN_LINALG_FUNCTIONS[func](*args, **kwargs)

        elif func in type(self)._UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(f"The numpy function {func.__name__!r} is not supported on Galois field arrays. If you believe this function should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\nIf you'd like to perform this operation on the data (but not necessarily a Galois field array), you should first call `array = array.view(np.ndarray)` and then call the function.")

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._check_array_like_object(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in type(self)._FUNCTIONS_REQUIRING_VIEW:
                output = output.view(type(self)) if not np.isscalar(output) else type(self)(output, dtype=self.dtype)

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


###############################################################################
# Special GF2 FieldArray subclass
###############################################################################

class GF2Meta(FieldClass, DirMeta):
    """
    A metaclass for the GF(2) class.
    """
    # pylint: disable=no-value-for-parameter

    # Need to have a unique cache of "calculate" functions for GF(2)
    _FUNC_CACHE_CALCULATE = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._prime_subfield = cls
        cls._is_primitive_poly = True

        cls.compile(kwargs["compile"])

    @property
    def ufunc_modes(cls):
        return ["jit-calculate"]

    @property
    def default_ufunc_mode(cls):
        return "jit-calculate"

    def _compile_ufuncs(cls):
        super()._compile_ufuncs()
        assert cls.ufunc_mode == "jit-calculate"

        cls._ufuncs["add"] = np.bitwise_xor
        cls._ufuncs["negative"] = np.positive
        cls._ufuncs["subtract"] = np.bitwise_xor
        cls._ufuncs["multiply"] = np.bitwise_and
        cls._ufuncs["reciprocal"] = np.positive
        cls._ufuncs["divide"] = np.bitwise_and

    ###############################################################################
    # Override ufunc routines to use native numpy bitwise ufuncs for GF(2)
    # arithmetic, which is faster than custom ufuncs
    ###############################################################################

    def _ufunc_routine_reciprocal(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, b in GF(2)
        b = 1 / a, a = 1 is the only valid element with a multiplicative inverse, which is 1
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        if np.count_nonzero(inputs[0]) != inputs[0].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufunc("reciprocal"), method)(*inputs, **kwargs)
        return output

    def _ufunc_routine_divide(cls, ufunc, method, inputs, kwargs, meta):
        """
        Need to re-implement this to manually throw ZeroDivisionError if necessary
        """
        cls._verify_operands_in_same_field(ufunc, inputs, meta)
        if np.count_nonzero(inputs[meta["operands"][-1]]) != inputs[meta["operands"][-1]].size:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")
        output = getattr(cls._ufunc("divide"), method)(*inputs, **kwargs)
        output = cls._view_output_as_field(output, meta["field"], meta["dtype"])
        return output

    def _ufunc_routine_square(cls, ufunc, method, inputs, kwargs, meta):  # pylint: disable=unused-argument
        """
        a, c in GF(2)
        c = a ** 2
          = a * a
          = a
        """
        cls._verify_unary_method_not_reduction(ufunc, method)
        return inputs[0]

    ###############################################################################
    # Arithmetic functions using explicit calculation
    #
    # NOTE: The ufunc inputs a and b are cast to integers at the beginning of each
    #       ufunc to prevent the non-JIT-compiled invocations (used in "large"
    #       fields with dtype=object) from performing infintely recursive
    #       arithmetic. Instead, the intended arithmetic inside the ufuncs is
    #       integer arithmetic.
    #       See https://github.com/mhostetter/galois/issues/253.
    ###############################################################################

    @staticmethod
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        a = int(a)
        b = int(b)

        return a ^ b

    @staticmethod
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.positive()` is faster.
        """
        a = int(a)

        return a

    @staticmethod
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        a = int(a)
        b = int(b)

        return a ^ b

    @staticmethod
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_and()` is faster.
        """
        a = int(a)
        b = int(b)

        return a & b

    @staticmethod
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        return 1

    @staticmethod
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        a = int(a)
        b = int(b)

        return a & b

    @staticmethod
    @numba.extending.register_jitable
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        a = int(a)
        b = int(b)

        if b == 0:
            return 1

        return a

    @staticmethod
    @numba.extending.register_jitable
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

        return 0

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a):
        return a.copy()


@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, compile="jit-calculate"):
    r"""
    A :ref:`Galois field array` over :math:`\mathrm{GF}(2)`.

    Important
    ---------
        This class is a pre-generated :obj:`galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
        for convenience.

        Only the constructor is documented on this page. See :obj:`galois.FieldArray` for all other classmethods and methods
        for :obj:`galois.GF2`.

    See :ref:`Galois Field Classes` for a detailed discussion of the relationship between :obj:`galois.FieldClass` and
    :obj:`galois.FieldArray`.

    See :ref:`Array Creation` for a detailed discussion on creating arrays (with and without copying) from array-like
    objects, valid NumPy data types, and other :obj:`galois.FieldArray` classmethods.

    Examples
    --------
    This class is equivalent, and in fact identical, to the subclass returned from the class factory :func:`galois.GF`.

    .. ipython:: python

        galois.GF2 is galois.GF(2)
        print(galois.GF2)

    The *Galois field array class* :obj:`galois.GF2` is a subclass of :obj:`galois.FieldArray`, with :obj:`galois.FieldClass` as its
    metaclass.

    .. ipython:: python

        isinstance(galois.GF2, galois.FieldClass)
        issubclass(galois.GF2, galois.FieldArray)

    Create a :ref:`Galois field array` using :obj:`galois.GF2`'s constructor.

    .. ipython:: python

        x = galois.GF2([1, 0, 1, 1]); x

    The *Galois field array* `x` is an instance of the *Galois field array class* :obj:`galois.GF2`.

    .. ipython:: python

        isinstance(x, galois.GF2)
    """


###############################################################################
# Polynomials over Galois fields
###############################################################################

# Values were obtained by running scripts/sparse_poly_performance_test.py
SPARSE_VS_BINARY_POLY_FACTOR = 0.00_05
SPARSE_VS_BINARY_POLY_MIN_COEFFS = int(1 / SPARSE_VS_BINARY_POLY_FACTOR)
SPARSE_VS_DENSE_POLY_FACTOR = 0.00_5
SPARSE_VS_DENSE_POLY_MIN_COEFFS = int(1 / SPARSE_VS_DENSE_POLY_FACTOR)


@set_module("galois")
class Poly:
    r"""
    A univariate polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Examples
    --------
    Create a polynomial over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        galois.Poly([1, 0, 1, 1])

    Create a polynomial over :math:`\mathrm{GF}(3^5)`.

    .. ipython:: python

        GF = galois.GF(3**5)
        galois.Poly([124, 0, 223, 0, 0, 15], field=GF)

    See :ref:`Polynomial Creation` and :ref:`Polynomial Arithmetic` for more examples.
    """
    # pylint: disable=too-many-public-methods

    # Increase my array priority so numpy will call my __radd__ instead of its own __add__
    __array_priority__ = 100

    def __new__(
        cls,
        coeffs: Union[Sequence[int], np.ndarray, FieldArray],
        field: Optional[FieldClass] = None,
        order: Literal["desc", "asc"] = "desc"
    ) -> "Poly":
        if not isinstance(coeffs, (list, tuple, np.ndarray, FieldArray)):
            raise TypeError(f"Argument `coeffs` must array-like, not {type(coeffs)}.")
        if not isinstance(field, (type(None), FieldClass)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")
        if not isinstance(order, str):
            raise TypeError(f"Argument `order` must be a str, not {type(order)}.")
        if isinstance(coeffs, (FieldArray, np.ndarray)) and not coeffs.ndim <= 1:
            raise ValueError(f"Argument `coeffs` can have dimension at most 1, not {coeffs.ndim}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order!r}.")

        if isinstance(coeffs, (FieldArray, np.ndarray)):
            coeffs = np.atleast_1d(coeffs)

        if order == "asc":
            coeffs = coeffs[::-1]  # Ensure it's in descending-degree order

        coeffs, field = cls._convert_coeffs(coeffs, field)

        if field is GF2:
            if len(coeffs) >= SPARSE_VS_BINARY_POLY_MIN_COEFFS and np.count_nonzero(coeffs) <= SPARSE_VS_BINARY_POLY_FACTOR*len(coeffs):
                degrees = np.arange(coeffs.size - 1, -1, -1)
                return SparsePoly(degrees, coeffs, field=field)
            else:
                integer = poly_to_integer(coeffs, 2)
                return BinaryPoly(integer)
        else:
            if len(coeffs) >= SPARSE_VS_DENSE_POLY_MIN_COEFFS and np.count_nonzero(coeffs) <= SPARSE_VS_DENSE_POLY_FACTOR*len(coeffs):
                degrees = np.arange(coeffs.size - 1, -1, -1)
                return SparsePoly(degrees, coeffs, field=field)
            else:
                return DensePoly(coeffs, field=field)

    def __init__(
        self,
        coeffs: Union[Sequence[int], np.ndarray, FieldArray],
        field: Optional[FieldClass] = None,
        order: Literal["desc", "asc"] = "desc"
    ):
        r"""
        Creates a polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

        The polynomial :math:`f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0` with degree :math:`d` has coefficients
        :math:`\{a_{d}, a_{d-1}, \dots, a_1, a_0\}` in :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        coeffs
            The polynomial coefficients :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}` with type :obj:`galois.FieldArray`. Alternatively,
            an iterable :obj:`tuple`, :obj:`list`, or :obj:`numpy.ndarray` may be provided and the Galois field domain is taken from
            the `field` keyword argument.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the coefficients are a :obj:`galois.FieldArray`, they won't be modified. If the coefficients
              are not explicitly in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using
              `galois.GF2(coeffs)`.
            * :obj:`galois.FieldClass`: The coefficients are explicitly converted to this Galois field `field(coeffs)`.

        order
            The interpretation of the coefficient degrees.

            * `"desc"` (default): The first element of `coeffs` is the highest degree coefficient, i.e. :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}`.
            * `"asc"`: The first element of `coeffs` is the lowest degree coefficient, i.e. :math:`\{a_0, a_1, \dots,  a_{d-1}, a_d\}`.
        """
        # pylint: disable=unused-argument,super-init-not-called
        return

    @classmethod
    def _convert_coeffs(cls, coeffs, field):
        if isinstance(coeffs, FieldArray) and field is None:
            # Use the field of the coefficients
            field = type(coeffs)
        else:
            # Convert coefficients to the specified field (or GF2 if unspecified), taking into
            # account negative coefficients
            field = GF2 if field is None else field
            coeffs = np.array(coeffs, dtype=field.dtypes[-1])
            idxs = coeffs < 0
            coeffs = field(np.abs(coeffs))
            coeffs[idxs] *= -1

        return coeffs, field

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zero(cls, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs the polynomial :math:`f(x) = 0` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x) = 0`.

        Examples
        --------
        Construct the zero polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Zero()

        Construct the zero polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Zero(GF)
        """
        return Poly([0], field=field)

    @classmethod
    def One(cls, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs the polynomial :math:`f(x) = 1` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x) = 1`.

        Examples
        --------
        Construct the one polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.One()

        Construct the one polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.One(GF)
        """
        return Poly([1], field=field)

    @classmethod
    def Identity(cls, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs the polynomial :math:`f(x) = x` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x) = x`.

        Examples
        --------
        Construct the identity polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Identity()

        Construct the identity polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Identity(GF)
        """
        return Poly([1, 0], field=field)

    @classmethod
    def Random(
        cls,
        degree: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
        field: Optional[FieldClass] = GF2
    ) -> "Poly":
        r"""
        Constructs a random polynomial over :math:`\mathrm{GF}(p^m)` with degree :math:`d`.

        Parameters
        ----------
        degree
            The degree of the polynomial.
        seed
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a random degree-:math:`5` polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Random(5)

        Construct a random degree-:math:`5` polynomial over :math:`\mathrm{GF}(3^5)` with a given seed. This produces repeatable results.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Random(5, seed=123456789, field=GF)
            galois.Poly.Random(5, seed=123456789, field=GF)

        Construct multiple polynomials with one global seed.

        .. ipython:: python

            rng = np.random.default_rng(123456789)
            galois.Poly.Random(5, seed=rng, field=GF)
            galois.Poly.Random(5, seed=rng, field=GF)
        """
        if not isinstance(degree, (int, np.integer)):
            raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
        if seed is not None:
            if not isinstance(seed, (int, np.integer, np.random.Generator)):
                raise ValueError("Seed must be an integer, a numpy.random.Generator or None.")
            if isinstance(seed, (int, np.integer)) and seed < 0:
                raise ValueError("Seed must be non-negative.")
        if not isinstance(field, FieldClass):
            raise TypeError(f"Argument `field` must be a Galois field class, not {type(field)}.")
        if not degree >= 0:
            raise ValueError(f"Argument `degree` must be non-negative, not {degree}.")

        rng = np.random.default_rng(seed)  # Make the seed a PRNG object so it can "step" its state if the below "if" statement is invoked
        coeffs = field.Random(degree + 1, seed=rng)
        if coeffs[0] == 0:
            coeffs[0] = field.Random(low=1, seed=rng)  # Ensure leading coefficient is non-zero

        return Poly(coeffs, field=field)

    @classmethod
    def Integer(cls, integer: int, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its integer representation.

        Parameters
        ----------
        integer
            The :ref:`integer representation <Integer representation>` of the polynomial :math:`f(x)`.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from its integer representation.

        .. ipython:: python

            galois.Poly.Integer(5)

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` from its integer representation.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Integer(186535908, field=GF)
            # The polynomial/integer equivalence
            13*GF.order**3 + 117
        """
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Argument `integer` be an integer, not {type(integer)}")
        if not isinstance(field, FieldClass):
            raise TypeError(f"Argument `field` must be a Galois field class, not {type(field)}.")
        if not integer >= 0:
            raise ValueError(f"Argument `integer` must be non-negative, not {integer}.")

        if field is GF2:
            # Explicitly create a binary poly
            return BinaryPoly(integer)
        else:
            coeffs = integer_to_poly(integer, field.order)
            return Poly(coeffs, field=field)

    @classmethod
    def Str(cls, string: str, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its string representation.

        :func:`galois.Poly.Str` and `str(poly)` are inverse operations.

        Parameters
        ----------
        string
            The string representation of the polynomial :math:`f(x)`.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Notes
        -----
        The string parsing rules include:

        * Either `^` or `**` may be used for indicating the polynomial degrees. For example, `"13x^3 + 117"` or `"13x**3 + 117"`.
        * Multiplication operators `*` may be used between coefficients and the polynomial indeterminate `x`, but are not required. For example,
          `"13x^3 + 117"` or `"13*x^3 + 117"`.
        * Polynomial coefficients of 1 may be specified or omitted. For example, `"x^3 + 117"` or `"1*x^3 + 117"`.
        * The polynomial indeterminate can be any single character, but must be consistent. For example, `"13x^3 + 117"` or `"13y^3 + 117"`.
        * Spaces are not required between terms. For example, `"13x^3 + 117"` or `"13x^3+117"`.
        * Any combination of the above rules is acceptable.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from its string representation.

        .. ipython:: python

            f = galois.Poly.Str("x^2 + 1"); f
            str(f)

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` from its string representation.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly.Str("13x^3 + 117", field=GF); f
            str(f)
        """
        if not isinstance(string, str):
            raise TypeError(f"Argument `string` be an string, not {type(string)}")

        return Poly.Degrees(*str_to_sparse_poly(string), field=field)

    @classmethod
    def Degrees(
        cls,
        degrees: Union[Sequence[int], np.ndarray],
        coeffs: Optional[Union[Sequence[int], np.ndarray, FieldArray]] = None,
        field: Optional[FieldClass] = None
    ) -> "Poly":
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its non-zero degrees.

        Parameters
        ----------
        degrees
            The polynomial degrees with non-zero coefficients.
        coeffs
            The corresponding non-zero polynomial coefficients with type :obj:`galois.FieldArray`. Alternatively, an iterable :obj:`tuple`,
            :obj:`list`, or :obj:`numpy.ndarray` may be provided and the Galois field domain is taken from the `field` keyword argument. The
            default is `None` which corresponds to all ones.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the coefficients are a :obj:`galois.FieldArray`, they won't be modified. If the coefficients are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(coeffs)`.
            * :obj:`galois.FieldClass`: The coefficients are explicitly converted to this Galois field `field(coeffs)`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            galois.Poly.Degrees([3, 1, 0])

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Degrees([3, 1, 0], coeffs=[214, 73, 185], field=GF)
        """
        if not isinstance(degrees, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `degrees` must array-like, not {type(degrees)}.")
        if not isinstance(coeffs, (type(None), list, tuple, np.ndarray, FieldArray)):
            raise TypeError(f"Argument `coeffs` must array-like, not {type(coeffs)}.")
        if not isinstance(field, (type(None), FieldClass)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {type(field)}.")

        degrees = np.array(degrees, dtype=np.int64)
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        coeffs, field = cls._convert_coeffs(coeffs, field)

        if not degrees.ndim <= 1:
            raise ValueError(f"Argument `degrees` can have dimension at most 1, not {degrees.ndim}.")
        if not degrees.size == np.unique(degrees).size:
            raise ValueError(f"Argument `degrees` must have unique entries, not {degrees}.")
        if not np.all(degrees >= 0):
            raise ValueError(f"Argument `degrees` must have non-negative values, not {degrees}.")
        if not coeffs.ndim <= 1:
            raise ValueError(f"Argument `coeffs` can have dimension at most 1, not {coeffs.ndim}.")
        if not degrees.size == coeffs.size:
            raise ValueError(f"Arguments `degrees` and `coeffs` must have the same length, not {degrees.size} and {coeffs.size}.")

        # No nonzero degrees means it's the zero polynomial
        if len(degrees) == 0:
            degrees, coeffs = np.array([0]), field([0])

        if field is GF2:
            if len(degrees) < SPARSE_VS_BINARY_POLY_FACTOR*max(degrees):
                # Explicitly create a sparse poly over GF(2)
                return SparsePoly(degrees, coeffs=coeffs, field=field)
            else:
                integer = sparse_poly_to_integer(degrees, coeffs, 2)
                return BinaryPoly(integer)
        else:
            if len(degrees) < SPARSE_VS_DENSE_POLY_FACTOR*max(degrees):
                # Explicitly create a sparse poly over GF(p^m)
                return SparsePoly(degrees, coeffs=coeffs, field=field)
            else:
                degree = max(degrees)  # The degree of the polynomial
                all_coeffs = type(coeffs).Zeros(degree + 1)
                all_coeffs[degree - degrees] = coeffs
                return DensePoly(all_coeffs)

    @classmethod
    def Roots(
        cls,
        roots: Union[Sequence[int], np.ndarray, FieldArray],
        multiplicities: Optional[Union[Sequence[int], np.ndarray]] = None,
        field: Optional[FieldClass] = None
    ) -> "Poly":
        r"""
        Constructs a monic polynomial over :math:`\mathrm{GF}(p^m)` from its roots.

        Parameters
        ----------
        roots
            The roots of the desired polynomial with type :obj:`galois.FieldArray`. Alternatively, an iterable :obj:`tuple`,
            :obj:`list`, or :obj:`numpy.ndarray` may be provided and the Galois field domain is taken from the `field` keyword argument.
        multiplicities
            The corresponding root multiplicities. The default is `None` which corresponds to all ones.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the roots are a :obj:`galois.FieldArray`, they won't be modified. If the roots are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(roots)`.
            * :obj:`galois.FieldClass`: The roots are explicitly converted to this Galois field `field(roots)`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Notes
        -----
        The polynomial :math:`f(x)` with :math:`k` roots :math:`\{r_1, r_2, \dots, r_k\}` with multiplicities
        :math:`\{m_1, m_2, \dots, m_k\}` is

        .. math::

            f(x) &= (x - r_1)^{m_1} (x - r_2)^{m_2} \dots (x - r_k)^{m_k} \\
                 &= a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0

        with degree :math:`d = \sum_{i=1}^{k} m_i`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from a list of its roots.

        .. ipython:: python

            roots = [0, 0, 1]
            f = galois.Poly.Roots(roots); f
            # Evaluate the polynomial at its roots
            f(roots)

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` from a list of its roots with specific multiplicities.

        .. ipython:: python

            GF = galois.GF(3**5)
            roots = [121, 198, 225]
            f = galois.Poly.Roots(roots, multiplicities=[1, 2, 1], field=GF); f
            # Evaluate the polynomial at its roots
            f(roots)
        """
        multiplicities = [1,]*len(roots) if multiplicities is None else multiplicities
        if not isinstance(roots, (tuple, list, np.ndarray, FieldArray)):
            raise TypeError(f"Argument `roots` must be array-like, not {type(roots)}.")
        if not isinstance(multiplicities, (tuple, list, np.ndarray)):
            raise TypeError(f"Argument `multiplicities` must be array-like, not {type(multiplicities)}.")
        if not isinstance(field, (type(None), FieldClass)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")

        roots, field = cls._convert_coeffs(roots, field)

        roots = field(roots).flatten()
        if not len(roots) == len(multiplicities):
            raise ValueError(f"Arguments `roots` and `multiplicities` must have the same length, not {len(roots)} and {len(multiplicities)}.")

        poly = Poly.One(field=field)
        x = Poly.Identity(field=field)
        for root, multiplicity in zip(roots, multiplicities):
            poly *= (x - root)**multiplicity

        return poly

    ###############################################################################
    # Methods
    ###############################################################################

    def coefficients(
        self,
        size: Optional[int] = None,
        order: Literal["desc", "asc"] = "desc"
    ) -> FieldArray:
        """
        Returns the polynomial coefficients in the order and size specified.

        Parameters
        ----------
        size
            The fixed size of the coefficient array. Zeros will be added for higher-order terms. This value must be
            at least `degree + 1` or a :obj:`ValueError` will be raised. The default is `None` which corresponds
            to `degree + 1`.

        order
            The order of the coefficient degrees, either descending (default) or ascending.

        Returns
        -------
        :
            An array of the polynomial coefficients with length `size`, either in descending order or ascending order.

        Notes
        -----
        This accessor is similar to the :obj:`coeffs` property, but it has more settings. By default, `Poly.coeffs == Poly.coefficients()`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            f.coeffs
            f.coefficients()
            # Return the coefficients in ascending order
            f.coefficients(order="asc")
            # Return the coefficients in ascending order with size 8
            f.coefficients(8, order="asc")
        """
        if not isinstance(size, (type(None), int, np.integer)):
            raise TypeError(f"Argument `size` must be an integer, not {type(size)}.")
        if not isinstance(order, str):
            raise TypeError(f"Argument `order` must be a str, not {type(order)}.")
        size = len(self) if size is None else size
        if not size >= len(self):
            raise ValueError(f"Argument `size` must be at least `degree + 1` which is {len(self)}, not {size}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order!r}.")

        coeffs = self.field.Zeros(size)
        coeffs[-len(self):] = self.coeffs
        if order == "asc":
            coeffs = np.flip(coeffs)

        return coeffs

    def copy(self) -> "Poly":
        """
        Deep copies the polynomial.

        Returns
        -------
        galois.Poly
            A copy of the original polynomial.
        """
        raise NotImplementedError

    def reverse(self) -> "Poly":
        r"""
        Returns the :math:`d`-th reversal :math:`x^d f(\frac{1}{x})` of the polynomial :math:`f(x)` with degree :math:`d`.

        Returns
        -------
        :
            The :math:`n`-th reversal :math:`x^n f(\frac{1}{x})`.

        Notes
        -----
        For a polynomial :math:`f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0` with degree :math:`d`, the :math:`d`-th
        reversal is equivalent to reversing the coefficients.

        .. math::
            \textrm{rev}_d f(x) = x^d f(x^{-1}) = a_0 x^d + a_{1} x^{d-1} + \dots + a_{d-1} x + a_d

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([5, 0, 3, 4], field=GF); f
            f.reverse()
        """
        return Poly(self.coeffs[::-1])

    @overload
    def roots(self, multiplicity: Literal[False] = False) -> FieldArray:
        ...
    @overload
    def roots(self, multiplicity: Literal[True] = True) -> Tuple[FieldArray, np.ndarray]:
        ...
    def roots(self, multiplicity=False):
        r"""
        Calculates the roots :math:`r` of the polynomial :math:`f(x)`, such that :math:`f(r) = 0`.

        Parameters
        ----------
        multiplicity : bool, optional
            Optionally return the multiplicity of each root. The default is `False` which only returns the unique
            roots.

        Returns
        -------
        galois.FieldArray
            Galois field array of roots of :math:`f(x)`. The roots are ordered in increasing order.
        numpy.ndarray
            The multiplicity of each root. This is only returned if `multiplicity=True`.

        Notes
        -----
        This implementation uses Chien's search to find the roots :math:`\{r_1, r_2, \dots, r_k\}` of the degree-:math:`d`
        polynomial

        .. math::
            f(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \dots + a_1x + a_0,

        where :math:`k \le d`. Then, :math:`f(x)` can be factored as

        .. math::
            f(x) = (x - r_1)^{m_1} (x - r_2)^{m_2} \dots (x - r_k)^{m_k},

        where :math:`m_i` is the multiplicity of root :math:`r_i` and :math:`d = \sum_{i=1}^{k} m_i`.

        The Galois field elements can be represented as :math:`\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m-2}\}`,
        where :math:`\alpha` is a primitive element of :math:`\mathrm{GF}(p^m)`.

        :math:`0` is a root of :math:`f(x)` if :math:`a_0 = 0`. :math:`1` is a root of :math:`f(x)` if :math:`\sum_{j=0}^{d} a_j = 0`. The
        remaining elements of :math:`\mathrm{GF}(p^m)` are powers of :math:`\alpha`. The following equations calculate :math:`f(\alpha^i)`,
        where :math:`\alpha^i` is a root of :math:`f(x)` if :math:`f(\alpha^i) = 0`.

        .. math::
            f(\alpha^i) &= a_{d}(\alpha^i)^{d} + a_{d-1}(\alpha^i)^{d-1} + \dots + a_1(\alpha^i) + a_0 \\
                        &\overset{\Delta}{=} \lambda_{i,d} + \lambda_{i,d-1} + \dots + \lambda_{i,1} + \lambda_{i,0} \\
                        &= \sum_{j=0}^{d} \lambda_{i,j}

        The next power of :math:`\alpha` can be easily calculated from the previous calculation.

        .. math::
            f(\alpha^{i+1}) &= a_{d}(\alpha^{i+1})^{d} + a_{d-1}(\alpha^{i+1})^{d-1} + \dots + a_1(\alpha^{i+1}) + a_0 \\
                            &= a_{d}(\alpha^i)^{d}\alpha^d + a_{d-1}(\alpha^i)^{d-1}\alpha^{d-1} + \dots + a_1(\alpha^i)\alpha + a_0 \\
                            &= \lambda_{i,d}\alpha^d + \lambda_{i,d-1}\alpha^{d-1} + \dots + \lambda_{i,1}\alpha + \lambda_{i,0} \\
                            &= \sum_{j=0}^{d} \lambda_{i,j}\alpha^j

        References
        ----------
        * https://en.wikipedia.org/wiki/Chien_search

        Examples
        --------
        Find the roots of a polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            f = galois.Poly.Roots([1, 0], multiplicities=[7, 3]); f
            f.roots()
            f.roots(multiplicity=True)

        Find the roots of a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly.Roots([18, 227, 153], multiplicities=[5, 7, 3], field=GF); f
            f.roots()
            f.roots(multiplicity=True)
        """
        if not isinstance(multiplicity, bool):
            raise TypeError(f"Argument `multiplicity` must be a bool, not {type(multiplicity)}.")

        roots = self.field._poly_roots(self.nonzero_degrees, self.nonzero_coeffs)

        if not multiplicity:
            return roots
        else:
            multiplicities = np.array([self._root_multiplicity(root) for root in roots])
            return roots, multiplicities

    def _root_multiplicity(self, root):
        poly = self.copy()
        multiplicity = 1

        while True:
            # If the root is also a root of the derivative, then its a multiple root.
            poly = poly.derivative()

            if poly == 0:
                # Cannot test whether p'(root) = 0 because p'(x) = 0. We've exhausted the non-zero derivatives. For
                # any Galois field, taking `characteristic` derivatives results in p'(x) = 0. For a root with multiplicity
                # greater than the field's characteristic, we need factor to the polynomial. Here we factor out (x - root)^m,
                # where m is the current multiplicity.
                poly = self.copy() // (Poly([1, -root], field=self.field)**multiplicity)

            if poly(root) == 0:
                multiplicity += 1
            else:
                break

        return multiplicity

    def derivative(self, k: int = 1) -> "Poly":
        r"""
        Computes the :math:`k`-th formal derivative :math:`\frac{d^k}{dx^k} f(x)` of the polynomial :math:`f(x)`.

        Parameters
        ----------
        k
            The number of derivatives to compute. 1 corresponds to :math:`p'(x)`, 2 corresponds to :math:`p''(x)`, etc.
            The default is 1.

        Returns
        -------
        :
            The :math:`k`-th formal derivative of the polynomial :math:`f(x)`.

        Notes
        -----
        For the polynomial

        .. math::
            f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0

        the first formal derivative is defined as

        .. math::
            f'(x) = (d) \cdot a_{d} x^{d-1} + (d-1) \cdot a_{d-1} x^{d-2} + \dots + (2) \cdot a_{2} x + a_1

        where :math:`\cdot` represents scalar multiplication (repeated addition), not finite field multiplication.
        The exponent that is "brought down" and multiplied by the coefficient is an integer, not a finite field element.
        For example, :math:`3 \cdot a = a + a + a`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Formal_derivative

        Examples
        --------
        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            f = galois.Poly.Random(7); f
            f.derivative()
            # p derivatives of a polynomial, where p is the field's characteristic, will always result in 0
            f.derivative(GF.characteristic)

        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(7)`.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly.Random(11, field=GF); f
            f.derivative()
            f.derivative(2)
            f.derivative(3)
            # p derivatives of a polynomial, where p is the field's characteristic, will always result in 0
            f.derivative(GF.characteristic)

        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly.Random(7, field=GF); f
            f.derivative()
            f.derivative(2)
            # p derivatives of a polynomial, where p is the field's characteristic, will always result in 0
            f.derivative(GF.characteristic)
        """
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
        if not k > 0:
            raise ValueError(f"Argument `k` must be a positive integer, not {k}.")

        if 0 in self.nonzero_degrees:
            # Cut off the 0th degree
            degrees = self.nonzero_degrees[:-1] - 1
            coeffs = self.nonzero_coeffs[:-1] * self.nonzero_degrees[:-1]  # Scalar multiplication
        else:
            degrees = self.nonzero_degrees - 1
            coeffs = self.nonzero_coeffs * self.nonzero_degrees  # Scalar multiplication

        p_prime = Poly.Degrees(degrees, coeffs, field=self.field)

        k -= 1
        if k > 0:
            return p_prime.derivative(k)
        else:
            return p_prime

    ###############################################################################
    # Overridden dunder methods
    ###############################################################################

    def __repr__(self) -> str:
        """
        A representation of the polynomial and the finite field it's over.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            f
        """
        return f"Poly({self}, {self.field.name})"

    def __str__(self) -> str:
        """
        The string representation of the polynomial, without specifying the finite field it's over.

        :func:`galois.Poly.Str` and `str(poly)` are inverse operations.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            str(f)
            print(f)
        """
        return sparse_poly_to_str(self.nonzero_degrees, self.nonzero_coeffs)

    def __hash__(self):
        t = tuple([self.field.order,] + self.nonzero_degrees.tolist() + self.nonzero_coeffs.tolist())
        return hash(t)

    def __call__(
        self,
        x: Union[int, Sequence[int], np.ndarray, FieldArray],
        field: Optional[FieldClass] = None,
        elementwise: bool = True
    ) -> FieldArray:
        r"""
        Evaluates the polynomial :math:`f(x)` at `x`.

        Parameters
        ----------
        x
            An array (or 0-D scalar) :math:`x` of finite field elements to evaluate the polynomial at.
        field
            The Galois field to evaluate the polynomial over. The default is `None` which represents
            the polynomial's current field, i.e. :obj:`field`.
        elementwise
            Indicates whether to evaluate :math:`x` elementwise. The default is `True`. If `False` (only valid
            for square matrices), the polynomial indeterminate :math:`x` is exponentiated using matrix powers
            (repeated matrix multiplication).

        Returns
        -------
        :
            The result of the polynomial evaluation :math:`f(x)`. The resulting array has the same shape as `x`.

        Examples
        --------
        Create a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly([37, 123, 0, 201], field=GF); f

        Evaluate the polynomial elementwise at :math:`x`.

        .. ipython:: python

            x = GF([185, 218, 84, 163]); x
            f(x)
            # The equivalent calculation
            GF(37)*x**3 + GF(123)*x**2 + GF(201)

        Evaluate the polynomial at the square matrix :math:`X`.

        .. ipython:: python

            X = GF([[185, 218], [84, 163]]); X
            f(X, elementwise=False)
            # The equivalent calculation
            GF(37)*np.linalg.matrix_power(X,3) + GF(123)*np.linalg.matrix_power(X,2) + GF(201)*GF.Identity(2)

        :meta public:
        """
        if not isinstance(field, (type(None), FieldClass)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {type(field)}.")

        field = self.field if field is None else field
        coeffs = field(self.coeffs)
        x = field(x)

        if elementwise:
            return field._poly_evaluate(coeffs, x)
        else:
            if not (x.ndim == 2 and x.shape[0] == x.shape[1]):
                raise ValueError(f"Argument `x` must be a square matrix when evaluating the polynomial not elementwise, not have shape {x.shape}.")
            return field._poly_evaluate_matrix(coeffs, x)

    def __len__(self) -> int:
        """
        Returns the length of the coefficient array `Poly.degree + 1`.

        Returns
        -------
        :
            The length of the coefficient array.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly([37, 123, 0, 201], field=GF); f
            f.coeffs
            len(f)
            f.degree + 1
        """
        return self.degree + 1

    def _check_inputs_are_polys(self, a, b):
        """
        Verify polynomial arithmetic operands are either galois.Poly or scalars in a finite field.
        """
        if not isinstance(a, (Poly, self.field)):
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {self.field.name}, not {type(a)}.")
        if not isinstance(b, (Poly, self.field)):
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {self.field.name}, not {type(b)}.")
        if (isinstance(a, Poly) and isinstance(b, Poly)) and not a.field is b.field:
            raise TypeError(f"Both polynomial operands must be over the same field, not {a.field.name} and {b.field.name}.")

    def _check_inputs_are_polys_or_ints(self, a, b):
        """
        Verify polynomial arithmetic operands are either galois.Poly, scalars in a finite field, or an integer (scalar multiplication).
        """
        if not isinstance(a, (Poly, self.field, int, np.integer)):
            raise TypeError(f"Both operands must be a galois.Poly, a single element of its field {self.field.name}, or an integer, not {type(a)}.")
        if not isinstance(b, (Poly, self.field, int, np.integer)):
            raise TypeError(f"Both operands must be a galois.Poly, a single element of its field {self.field.name}, or an integer, not {type(b)}.")
        if (isinstance(a, Poly) and isinstance(b, Poly)) and not a.field is b.field:
            raise TypeError(f"Both polynomial operands must be over the same field, not {a.field.name} and {b.field.name}.")

    def _convert_field_scalars_to_polys(self, a, b):
        """
        Convert finite field scalars to 0-degree polynomials in that field.
        """
        # Promote a single field element to a 0-degree polynomial
        if isinstance(a, self.field):
            if not a.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {a.size}.")
            a = Poly(np.atleast_1d(a))
        if isinstance(b, self.field):
            if not b.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {b.size}.")
            b = Poly(np.atleast_1d(b))

        return a, b

    @staticmethod
    def _determine_poly_class(a, b):
        """
        Determine the type of polynomial arithmetic to perform.
        """
        if isinstance(a, SparsePoly) or isinstance(b, SparsePoly):
            return SparsePoly
        elif isinstance(a, BinaryPoly) or isinstance(b, BinaryPoly):
            return BinaryPoly
        else:
            return DensePoly

    def __add__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        r"""
        Adds two polynomials in :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial).

        Returns
        -------
        :
            The polynomial :math:`a(x) + b(x)`.

        Examples
        --------
        Add two polynomials over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            a + b

        Add a polynomial and scalar over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            a + GF(75)
        """
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._add(a, b)

    def __radd__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._add(b, a)

    def __neg__(self):
        r"""
        Returns the additive inverse of the polynomial :math:`f(x)`.

        Returns
        -------
        :
            The polynomial :math:`-f(x)`.

        Examples
        --------
        Compute the additive inverse of a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            -a

        A polynomial added to its additive inverse is always zero.

        .. ipython:: python

            a + -a
        """
        raise NotImplementedError

    def __sub__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        r"""
        Subtracts two polynomials in :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial).

        Returns
        -------
        :
            The polynomial :math:`a(x) - b(x)`.

        Examples
        --------
        Subtract two polynomials over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            a - b

        Subtract a polynomial and scalar over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            a - GF(75)
        """
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._sub(a, b)

    def __rsub__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._sub(b, a)

    def __mul__(
        self,
        other: Union["Poly", FieldArray, int]
    ) -> "Poly":
        r"""
        Multiplies two polynomials in :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial). An integer
            :math:`b` may be passed, which performs scalar multiplication (repeated addition).

        Returns
        -------
        :
            The polynomial :math:`a(x)b(x)` or :math:`a(x) \cdot b`.

        Notes
        -----
        When the second argument is an integer, scalar multiplication is performed, which is equivalent to repeated addition.
        For example, :math:`b(x) \cdot 3 = b(x) + b(x) + b(x)`.

        Examples
        --------
        Multiply two polynomials over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            a * b

        Multiply a polynomial and finite field scalar over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            a * GF(4)

        Multiply a polynomial and integer scalar over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            a * 4
            a + a + a + a
        """
        self._check_inputs_are_polys_or_ints(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        if isinstance(a, (int, np.integer)):
            # Ensure the integer is in the second operand for scalar multiplication
            a, b = b, a
        cls = self._determine_poly_class(a, b)
        return cls._mul(a, b)

    def __rmul__(
        self,
        other: Union["Poly", FieldArray, int]
    ) -> "Poly":
        self._check_inputs_are_polys_or_ints(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        if isinstance(b, (int, np.integer)):
            # Ensure the integer is in the second operand for scalar multiplication
            b, a = a, b
        cls = self._determine_poly_class(a, b)
        return cls._mul(b, a)

    def __divmod__(
        self,
        other: Union["Poly", FieldArray]
    ) -> Tuple["Poly", "Poly"]:
        r"""
        Divides two polynomials over :math:`\mathrm{GF}(p^m)` and returns the quotient and remainder.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial).

        Returns
        -------
        :
            The quotient polynomial :math:`q(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.
        :
            The remainder polynomial :math:`r(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        Divide two polynomials over :math:`\mathrm{GF}(3^5)` and return the quotient and remainder.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            q, r = divmod(a, b)
            q
            r
            b*q + r == a

        Divide a polynomial and scalar over :math:`\mathrm{GF}(3^5)` and return the quotient and remainder.

        .. ipython:: python

            b = GF(75); b
            q, r = divmod(a, b)
            q
            r
            b*q + r == a
        """
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._divmod(a, b)

    def __rdivmod__(
        self,
        other: Union["Poly", FieldArray]
    ) -> Tuple["Poly", "Poly"]:
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._divmod(b, a)

    def __truediv__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        r"""
        Divides two polynomials over :math:`\mathrm{GF}(p^m)` and returns the quotient.

        True division and floor division are equivalent.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial).

        Returns
        -------
        :
            The quotient polynomial :math:`q(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        Divide two polynomials over :math:`\mathrm{GF}(3^5)` and return the quotient.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            a / b

        Divide a polynomial and scalar over :math:`\mathrm{GF}(3^5)` and return the quotient.

        .. ipython:: python

            b = GF(75); b
            a / b
        """
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._divmod(a, b)[0]

    def __rtruediv__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._divmod(b, a)[0]

    def __floordiv__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        r"""
        Divides two polynomials over :math:`\mathrm{GF}(p^m)` and returns the quotient.

        True division and floor division are equivalent.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial).

        Returns
        -------
        :
            The quotient polynomial :math:`q(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        Divide two polynomials over :math:`\mathrm{GF}(3^5)` and return the quotient.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            a // b

        Divide a polynomial and scalar over :math:`\mathrm{GF}(3^5)` and return the quotient.

        .. ipython:: python

            b = GF(75); b
            a // b
        """
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._divmod(a, b)[0]

    def __rfloordiv__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._divmod(b, a)[0]

    def __mod__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        r"""
        Divides two polynomials over :math:`\mathrm{GF}(p^m)` and returns the remainder.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial).

        Returns
        -------
        :
            The remainder polynomial :math:`r(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        Divide two polynomials over :math:`\mathrm{GF}(3^5)` and return the remainder.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            b = galois.Poly([47, 200, 75], field=GF); b
            a % b

        Divide a polynomial and scalar over :math:`\mathrm{GF}(3^5)` and return the remainder.

        .. ipython:: python

            b = GF(75); b
            a % b
        """
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._mod(a, b)

    def __rmod__(
        self,
        other: Union["Poly", FieldArray]
    ) -> "Poly":
        self._check_inputs_are_polys(self, other)
        a, b = self._convert_field_scalars_to_polys(self, other)
        cls = self._determine_poly_class(a, b)
        return cls._mod(b, a)

    def __pow__(
        self,
        other: int,
    ) -> "Poly":
        r"""
        Exponentiates the polynomial over :math:`\mathrm{GF}(p^m)` to an integer power.

        Parameters
        ----------
        other
            The non-negative integer exponent :math:`b`.

        Returns
        -------
        :
            The polynomial :math:`a(x)^b`.

        Examples
        --------
        Exponentiate a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = galois.Poly([163, 13, 0, 0, 0, 228], field=GF); a
            a ** 3
            a * a * a
        """
        if not isinstance(other, (int, np.integer)):
            raise TypeError(f"For polynomial exponentiation, the second argument must be an int, not {other}.")
        if not other >= 0:
            raise ValueError(f"Can only exponentiate polynomials to non-negative integers, not {other}.")
        a, power = self, other
        field = self.field

        # c(x) = a(x) ** power
        if power == 0:
            return Poly.One(field)

        c_square = a  # The "squaring" part
        c_mult = Poly.One(field)  # The "multiplicative" part

        while power > 1:
            if power % 2 == 0:
                c_square *= c_square
                power //= 2
            else:
                c_mult *= c_square
                power -= 1
        c = c_mult * c_square

        return c

    def __eq__(
        self,
        other: Union["Poly", FieldArray, int]
    ) -> bool:
        r"""
        Determines if two polynomials over :math:`\mathrm{GF}(p^m)` are equal.

        Parameters
        ----------
        other
            The polynomial :math:`b(x)` or a finite field scalar (equivalently a degree-:math:`0` polynomial). An integer
            may be passed and is interpreted as a degree-:math:`0` polynomial in the field :math:`a(x)` is over.

        Returns
        -------
        :
            `True` if the two polynomials have the same coefficients and are over the same finite field.

        Examples
        --------
        Compare two polynomials over the same field.

        .. ipython:: python

            a = galois.Poly([3, 0, 5], field=galois.GF(7)); a
            b = galois.Poly([3, 0, 5], field=galois.GF(7)); b
            a == b
            # They are still two distinct objects, however
            a is b

        Compare two polynomials with the same coefficients but over different fields.

        .. ipython:: python

            a = galois.Poly([3, 0, 5], field=galois.GF(7)); a
            b = galois.Poly([3, 0, 5], field=galois.GF(7**2)); b
            a == b

        Comparison with scalars is allowed for convenience.

        .. ipython:: python

            GF = galois.GF(7)
            a = galois.Poly([0], field=GF); a
            a == GF(0)
            a == 0
        """
        if isinstance(other, (int, np.integer)):
            # Compare poly to a integer scalar (assumed to be from the same field)
            return self.degree == 0 and np.array_equal(self.coeffs, [other])

        elif isinstance(other, FieldArray):
            # Compare poly to a finite field scalar (may or may not be from the same field)
            if not other.ndim == 0:
                raise ValueError(f"Can only compare galois.Poly to a 0-D galois.FieldArray scalar, not shape {other.shape}.")
            return self.field is type(other) and self.degree == 0 and np.array_equal(self.coeffs, np.atleast_1d(other))

        elif not isinstance(other, Poly):
            raise TypeError(f"Can only compare galois.Poly and galois.Poly / int / galois.FieldArray scalar objects, not {type(other)}.")

        else:
            # Compare two poly objects to each other
            return self.field is other.field and np.array_equal(self.nonzero_degrees, other.nonzero_degrees) and np.array_equal(self.nonzero_coeffs, other.nonzero_coeffs)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _add(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _sub(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _mul(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _divmod(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _mod(cls, a, b):
        raise NotImplementedError

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self) -> FieldClass:
        """
        The *Galois field array class* for the finite field the coefficients are over.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            a.field

        .. ipython:: python

            GF = galois.GF(2**8)
            b = galois.Poly.Random(5, field=GF); b
            b.field
        """
        raise NotImplementedError

    @property
    def degree(self) -> int:
        """
        The degree of the polynomial. The degree of a polynomial is the highest degree with a non-zero coefficient.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.degree
        """
        raise NotImplementedError

    @property
    def nonzero_degrees(self) -> np.ndarray:
        """
        An array of the polynomial degrees that have non-zero coefficients in descending order. The entries of
        :obj:`nonzero_degrees` are paired with :obj:`nonzero_coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.nonzero_degrees
        """
        raise NotImplementedError

    @property
    def nonzero_coeffs(self) -> FieldArray:
        """
        The non-zero coefficients of the polynomial in degree-descending order. The entries of :obj:`nonzero_degrees`
        are paired with :obj:`nonzero_coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.nonzero_coeffs
        """
        raise NotImplementedError

    @property
    def degrees(self) -> np.ndarray:
        """
        An array of the polynomial degrees in descending order. The entries of :obj:`degrees` are paired with :obj:`coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.degrees
        """
        raise NotImplementedError

    @property
    def coeffs(self) -> FieldArray:
        """
        The coefficients of the polynomial in degree-descending order. The entries of :obj:`degrees` are paired with :obj:`coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.coeffs
        """
        raise NotImplementedError

    @property
    def integer(self) -> int:
        r"""
        The integer representation of the polynomial. For the polynomial :math:`f(x) =  a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0`
        over the field :math:`\mathrm{GF}(p^m)`, the integer representation is :math:`i = a_d (p^m)^{d} + a_{d-1} (p^m)^{d-1} + \dots + a_1 (p^m) + a_0`
        using integer arithmetic, not finite field arithmetic.

        Said differently, if the polynomial coefficients :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}` are considered as the "digits" of a radix-:math:`p^m`
        value, the polynomial's integer representation is the decimal value (radix-:math:`10`).

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.integer
            p.integer == 3*GF.order**3 + 5*GF.order**1 + 2*GF.order**0
        """
        return sparse_poly_to_integer(self.nonzero_degrees, self.nonzero_coeffs, self.field.order)


class DensePoly(Poly):
    """
    Implementation of dense polynomials over Galois fields.
    """

    __slots__ = ["_coeffs"]

    def __new__(cls, coeffs, field=None):  # pylint: disable=signature-differs
        # Arguments aren't verified in Poly.__new__()
        obj = object.__new__(cls)
        obj._coeffs = coeffs

        if obj._coeffs.size > 1:
            # Remove leading zero coefficients
            idxs = np.nonzero(obj._coeffs)[0]
            if idxs.size > 0:
                obj._coeffs = obj._coeffs[idxs[0]:]
            else:
                obj._coeffs = obj._coeffs[-1]

        # Ensure the coefficient array isn't 0-dimensional
        obj._coeffs = np.atleast_1d(obj._coeffs)

        return obj

    def __init__(self, *args, **kwargs):  # pylint: disable=signature-differs
        # pylint: disable=unused-argument,super-init-not-called
        return

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        return DensePoly(self._coeffs.copy())

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __neg__(self):
        return DensePoly(-self._coeffs)

    @classmethod
    def _add(cls, a, b):
        field = a.field

        # c(x) = a(x) + b(x)
        c_coeffs = field.Zeros(max(a.coeffs.size, b.coeffs.size))
        c_coeffs[-a.coeffs.size:] = a.coeffs
        c_coeffs[-b.coeffs.size:] += b.coeffs

        return Poly(c_coeffs)

    @classmethod
    def _sub(cls, a, b):
        field = a.field

        # c(x) = a(x) + b(x)
        c_coeffs = field.Zeros(max(a.coeffs.size, b.coeffs.size))
        c_coeffs[-a.coeffs.size:] = a.coeffs
        c_coeffs[-b.coeffs.size:] -= b.coeffs

        return Poly(c_coeffs)

    @classmethod
    def _mul(cls, a, b):
        if isinstance(b, (int, np.integer)):
            # Scalar multiplication  (p * 3 = p + p + p)
            c_coeffs = a.coeffs * b
        else:
            # c(x) = a(x) * b(x)
            c_coeffs = np.convolve(a.coeffs, b.coeffs)

        return Poly(c_coeffs)

    @classmethod
    def _divmod(cls, a, b):
        field = a.field
        zero = Poly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            return Poly(a.coeffs // b.coeffs), zero

        elif a == 0:
            return zero, zero

        elif a.degree < b.degree:
            return zero, a.copy()

        else:
            q_coeffs, r_coeffs = field._poly_divmod(a.coeffs, b.coeffs)
            return Poly(q_coeffs), Poly(r_coeffs)

    @classmethod
    def _mod(cls, a, b):
        return cls._divmod(a, b)[1]

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        return type(self._coeffs)

    @property
    def degree(self):
        return self._coeffs.size - 1

    @property
    def nonzero_degrees(self):
        return self.degree - np.nonzero(self._coeffs)[0]

    @property
    def nonzero_coeffs(self):
        return self._coeffs[np.nonzero(self._coeffs)[0]]

    @property
    def degrees(self):
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        return self._coeffs.copy()


class BinaryPoly(Poly):
    """
    Implementation of polynomials over GF(2).
    """

    __slots__ = ["_integer", "_coeffs"]

    def __new__(cls, integer):  # pylint: disable=signature-differs
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Argument `integer` must be an integer, not {type(integer)}.")
        if not integer >= 0:
            raise ValueError(f"Argument `integer` must be non-negative, not {integer}.")

        obj = object.__new__(cls)
        obj._integer = integer
        obj._coeffs = None  # Only compute these if requested

        return obj

    def __init__(self, *args, **kwargs):  # pylint: disable=signature-differs
        # pylint: disable=unused-argument,super-init-not-called
        return

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        return BinaryPoly(self._integer)

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __neg__(self):
        return self.copy()

    @classmethod
    def _add(cls, a, b):
        return BinaryPoly(a.integer ^ b.integer)

    @classmethod
    def _sub(cls, a, b):
        return BinaryPoly(a.integer ^ b.integer)

    @classmethod
    def _mul(cls, a, b):
        if isinstance(b, (int, np.integer)):
            # Scalar multiplication  (p * 3 = p + p + p)
            return BinaryPoly(a.integer) if b % 2 == 1 else BinaryPoly(0)

        else:
            # Re-order operands such that a > b so the while loop has less loops
            a = a.integer
            b = b.integer
            if b > a:
                a, b = b, a

            c = 0
            while b > 0:
                if b & 0b1:
                    c ^= a  # Add a(x) to c(x)
                b >>= 1  # Divide b(x) by x
                a <<= 1  # Multiply a(x) by x

            return BinaryPoly(c)

    @classmethod
    def _divmod(cls, a, b):
        deg_a = a.degree
        deg_q = a.degree - b.degree
        deg_r = b.degree - 1
        a = a.integer
        b = b.integer

        q = 0
        mask = 1 << deg_a
        for i in range(deg_q, -1, -1):
            q <<= 1
            if a & mask:
                a ^= b << i
                q ^= 1  # Set the LSB then left shift
            assert a & mask == 0
            mask >>= 1

        # q = a >> deg_r
        mask = (1 << (deg_r + 1)) - 1  # The last deg_r + 1 bits of a
        r = a & mask

        return BinaryPoly(q), BinaryPoly(r)

    @classmethod
    def _mod(cls, a, b):
        return cls._divmod(a, b)[1]

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        return GF2

    @property
    def degree(self):
        if self._integer == 0:
            return 0
        else:
            return len(bin(self._integer)[2:]) - 1

    @property
    def nonzero_degrees(self):
        return self.degree - np.nonzero(self.coeffs)[0]

    @property
    def nonzero_coeffs(self):
        return self.coeffs[np.nonzero(self.coeffs)[0]]

    @property
    def degrees(self):
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        if self._coeffs is None:
            binstr = bin(self._integer)[2:]
            self._coeffs = GF2([int(b) for b in binstr])
        return self._coeffs.copy()

    @property
    def integer(self):
        return self._integer


class SparsePoly(Poly):
    """
    Implementation of sparse polynomials over Galois fields.
    """

    __slots__ = ["_degrees", "_coeffs"]

    def __new__(cls, degrees, coeffs=None, field=None):  # pylint: disable=signature-differs
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        if not isinstance(degrees, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `degrees` must be array-like, not {type(degrees)}.")
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `coeffs` must be array-like, not {type(coeffs)}.")
        if not len(degrees) == len(coeffs):
            raise ValueError(f"Arguments `degrees` and `coeffs` must have the same length, not {len(degrees)} and {len(coeffs)}.")
        if not all(degree >= 0 for degree in degrees):
            raise ValueError(f"Argument `degrees` must have non-negative values, not {degrees}.")

        obj = object.__new__(cls)

        if isinstance(coeffs, FieldArray) and field is None:
            obj._degrees = np.array(degrees)
            obj._coeffs = coeffs
        else:
            field = GF2 if field is None else field
            if isinstance(coeffs, np.ndarray):
                # Ensure coeffs is an iterable
                coeffs = coeffs.tolist()
            obj._degrees = np.array(degrees)
            obj._coeffs = field([-field(abs(c)) if c < 0 else field(c) for c in coeffs])

        # Sort the degrees and coefficients in descending order
        idxs = np.argsort(degrees)[::-1]
        obj._degrees = obj._degrees[idxs]
        obj._coeffs = obj._coeffs[idxs]

        # Remove zero coefficients
        idxs = np.nonzero(obj._coeffs)[0]
        obj._degrees = obj._degrees[idxs]
        obj._coeffs = obj._coeffs[idxs]

        return obj

    def __init__(self, *args, **kwargs):  # pylint: disable=signature-differs
        # pylint: disable=unused-argument,super-init-not-called
        return

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        return SparsePoly(self.degrees, self.coeffs)

    def reverse(self):
        return SparsePoly(self.degree - self.degrees, self.coeffs)

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __neg__(self):
        return SparsePoly(self._degrees, -self._coeffs)

    @classmethod
    def _add(cls, a, b):
        field = a.field

        # c(x) = a(x) + b(x)
        cc = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
        for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
            cc[b_degree] = cc.get(b_degree, field(0)) + b_coeff

        return Poly.Degrees(list(cc.keys()), list(cc.values()), field=field)

    @classmethod
    def _sub(cls, a, b):
        field = a.field

        # c(x) = a(x) - b(x)
        cc = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
        for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
            cc[b_degree] = cc.get(b_degree, field(0)) - b_coeff

        return Poly.Degrees(list(cc.keys()), list(cc.values()), field=field)

    @classmethod
    def _mul(cls, a, b):
        field = a.field

        if isinstance(b, (int, np.integer)):
            # Scalar multiplication  (p * 3 = p + p + p)
            return Poly.Degrees(a.nonzero_degrees, a.nonzero_coeffs * b)

        else:
            # c(x) = a(x) * b(x)
            cc = {}
            for a_degree, a_coeff in zip(a.nonzero_degrees, a.nonzero_coeffs):
                for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
                    cc[a_degree + b_degree] = cc.get(a_degree + b_degree, field(0)) + a_coeff*b_coeff

            return Poly.Degrees(list(cc.keys()), list(cc.values()), field=field)

    @classmethod
    def _divmod(cls, a, b):
        field = a.field
        zero = Poly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            q_degrees = a.nonzero_degrees
            q_coeffs = [a_coeff // b.coeffs[0] for a_coeff in a.nonzero_coeffs]
            return Poly.Degrees(q_degrees, q_coeffs, field=field), zero

        elif a == 0:
            return zero, zero

        elif a.degree < b.degree:
            return zero, a.copy()

        else:
            aa = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
            b_coeffs = b.coeffs

            q_degree = a.degree - b.degree
            r_degree = b.degree  # One larger than final remainder
            qq = {}
            r_coeffs = field.Zeros(r_degree + 1)

            # Preset remainder so we can rotate at the start of loop
            for i in range(0, b.degree):
                r_coeffs[1 + i] = aa.get(a.degree - i, 0)

            for i in range(0, q_degree + 1):
                r_coeffs = np.roll(r_coeffs, -1)
                r_coeffs[-1] = aa.get(a.degree - (i + b.degree), 0)

                if r_coeffs[0] > 0:
                    q = r_coeffs[0] // b_coeffs[0]
                    r_coeffs -= q*b_coeffs
                    qq[q_degree - i] = q

            return Poly.Degrees(list(qq.keys()), list(qq.values()), field=field), Poly(r_coeffs[1:])

    @classmethod
    def _mod(cls, a, b):
        field = a.field
        zero = Poly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            return zero

        elif a == 0:
            return zero

        elif a.degree < b.degree:
            return a.copy()

        else:
            aa = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
            b_coeffs = b.coeffs

            q_degree = a.degree - b.degree
            r_degree = b.degree  # One larger than final remainder
            r_coeffs = field.Zeros(r_degree + 1)

            # Preset remainder so we can rotate at the start of loop
            for i in range(0, b.degree):
                r_coeffs[1 + i] = aa.get(a.degree - i, 0)

            for i in range(0, q_degree + 1):
                r_coeffs = np.roll(r_coeffs, -1)
                r_coeffs[-1] = aa.get(a.degree - (i + b.degree), 0)

                if r_coeffs[0] > 0:
                    q = r_coeffs[0] // b_coeffs[0]
                    r_coeffs -= q*b_coeffs

            return Poly(r_coeffs[1:])

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        return type(self._coeffs)

    @property
    def degree(self):
        return 0 if self._degrees.size == 0 else int(np.max(self._degrees))

    @property
    def nonzero_degrees(self):
        return self._degrees.copy()

    @property
    def nonzero_coeffs(self):
        return self._coeffs.copy()

    @property
    def degrees(self):
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        # Assemble a full list of coefficients, including zeros
        coeffs = self.field.Zeros(self.degree + 1)
        if self.nonzero_degrees.size > 0:
            coeffs[self.degree - self.nonzero_degrees] = self.nonzero_coeffs
        return coeffs


# Define the GF(2) primitive polynomial here, not in _fields/_gf2.py, to avoid a circular dependency with `Poly`.
# The primitive polynomial is p(x) = x - alpha, where alpha = 1. Over GF(2), this is equivalent
# to p(x) = x + 1.
GF2._irreducible_poly = Poly([1, 1])  # pylint: disable=protected-access
