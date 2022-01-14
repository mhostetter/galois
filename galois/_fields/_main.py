"""
A module that contains the main classes for Galois fields -- FieldClass, FieldArray,
and Poly. They're all in one file because they have circular dependencies. The specific GF2
FieldClass is also included.
"""
import inspect
import math
import random
from typing import Tuple, List, Iterable, Optional, Union
from typing_extensions import Literal

import numba
import numpy as np

from .._overrides import set_module
from .._poly_conversion import integer_to_poly, poly_to_integer, str_to_integer, poly_to_str, sparse_poly_to_integer, sparse_poly_to_str, str_to_sparse_poly

from ._dtypes import DTYPES
from ._linalg import dot, row_reduce, lu_decompose, lup_decompose
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

    def __str__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

    def __repr__(cls):
        return str(cls)

    ###############################################################################
    # Helper methods
    ###############################################################################

    def _determine_dtypes(cls):
        """
        At a minimum, valid dtypes are ones that can hold x for x in [0, order).
        """
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    ###############################################################################
    # Class methods
    ###############################################################################

    def compile(cls, mode: str):
        """
        Recompile the just-in-time compiled numba ufuncs for a new calculation mode.

        This function updates :obj:`ufunc_mode`.

        Parameters
        ----------
        mode : str
            The ufunc calculation mode.

            * `"auto"`: Selects "jit-lookup" for fields with order less than :math:`2^{20}`, "jit-calculate" for larger fields, and "python-calculate"
              for fields whose elements cannot be represented with :obj:`numpy.int64`.
            * `"jit-lookup"`: JIT compiles arithmetic ufuncs to use Zech log, log, and anti-log lookup tables for efficient computation.
              In the few cases where explicit calculation is faster than table lookup, explicit calculation is used.
            * `"jit-calculate"`: JIT compiles arithmetic ufuncs to use explicit calculation. The "jit-calculate" mode is designed for large
              fields that cannot or should not store lookup tables in RAM. Generally, the "jit-calculate" mode is slower than "jit-lookup".
            * `"python-calculate"`: Uses pure-python ufuncs with explicit calculation. This is reserved for fields whose elements cannot be
              represented with :obj:`numpy.int64` and instead use :obj:`numpy.object_` with python :obj:`int` (which has arbitrary precision).
        """
        if not isinstance(mode, (type(None), str)):
            raise TypeError(f"Argument `mode` must be a string, not {type(mode)}.")
        # if not mode in ["auto", "jit-lookup", "jit-calculate", "python-calculate"]:
        #     raise ValueError(f"Argument `mode` must be in ['auto', 'jit-lookup', 'jit-calculate', 'python-calculate'], not {mode!r}.")
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
    ) -> "DisplayContext":
        r"""
        Sets the display mode for all Galois field arrays of this type.

        The display mode can be set to either the integer representation, polynomial representation, or power
        representation. This function updates :obj:`display_mode`.

        Warning
        -------
        For the power representation, :func:`np.log` is computed on each element. So for large fields without lookup
        tables, displaying arrays in the power representation may take longer than expected.

        Parameters
        ----------
        mode : str, optional
            The field element representation.

            * `"int"` (default): The element displayed as the integer representation of the polynomial. For example, :math:`2x^2 + x + 2` is an element of
              :math:`\mathrm{GF}(3^3)` and is equivalent to the integer :math:`23 = 2 \cdot 3^2 + 3 + 2`.
            * `"poly"`: The element as a polynomial over :math:`\mathrm{GF}(p)` of degree less than :math:`m`. For example, :math:`2x^2 + x + 2` is an element
              of :math:`\mathrm{GF}(3^3)`.
            * `"power"`: The element as a power of the primitive element, see :obj:`FieldClass.primitive_element`. For example, :math:`2x^2 + x + 2 = \alpha^5`
              in :math:`\mathrm{GF}(3^3)` with irreducible polynomial :math:`x^3 + 2x + 1` and primitive element :math:`\alpha = x`.

        Returns
        -------
        DisplayContext
            A context manager for use in a `with` statement. If permanently setting the display mode, disregard the
            return value.

        Examples
        --------
        Change the display mode by calling the :func:`display` method.

        .. ipython:: python

            GF = galois.GF(3**3)
            print(GF.properties)
            a = GF(23); a

            # Permanently set the display mode to the polynomial representation
            GF.display("poly"); a
            # Permanently set the display mode to the power representation
            GF.display("power"); a
            # Permanently reset the default display mode to the integer representation
            GF.display(); a

        The :func:`display` method can also be used as a context manager, as shown below.

        For the polynomial representation, when the primitive element is :math:`\alpha = x` in :math:`\mathrm{GF}(p)[x]` the polynomial
        indeterminate used is :math:`\alpha`.

        .. ipython:: python

            GF = galois.GF(2**8)
            print(GF.properties)
            a = GF.Random()
            print(GF.display_mode, a)
            with GF.display("poly"):
                print(GF.display_mode, a)
            with GF.display("power"):
                print(GF.display_mode, a)
            # The display mode is reset after exiting the context manager
            print(GF.display_mode, a)

        But when the primitive element is :math:`\alpha \ne x` in :math:`\mathrm{GF}(p)[x]`, the polynomial
        indeterminate used is :math:`x`.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly=galois.Poly.Degrees([8,4,3,1,0]))
            print(GF.properties)
            a = GF.Random()
            print(GF.display_mode, a)
            with GF.display("poly"):
                print(GF.display_mode, a)
            with GF.display("power"):
                print(GF.display_mode, a)
            # The display mode is reset after exiting the context manager
            print(GF.display_mode, a)
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
        Generates a field element representation table comparing the power, polynomial, vector, and integer representations.

        Parameters
        ----------
        primitive_element : int, str, np.ndarray, galois.FieldArray, optional
            The primitive element to use for the power representation. The default is `None` which uses the field's
            default primitive element, :obj:`primitive_element`. If an array, it must be a 0-D array.
        sort : str, optional
            The sorting method for the table, either `"power"` (default), `"poly"`, `"vector"`, or `"int"`. Sorting by "power" will order
            the rows of the table by ascending powers of the primitive element. Sorting by any of the others will order the rows in
            lexicographically-increasing polynomial/vector order, which is equivalent to ascending order of the integer representation.

        Returns
        -------
        str
            A UTF-8 formatted table comparing the power, polynomial, vector, and integer representations of each
            field element.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**4)
            print(GF.properties)

        Generate a representation table for :math:`\mathrm{GF}(2^4)`. Since :math:`x^4 + x + 1` is a primitive polynomial,
        :math:`x` is a primitive element of the field. Notice, :math:`\textrm{ord}(x) = 15`.

        .. ipython:: python

            print(GF.repr_table())

        Generate a representation table for :math:`\mathrm{GF}(2^4)` using a different primitive element :math:`x^3 + x^2 + x`.
        Notice, :math:`\textrm{ord}(x^3 + x^2 + x) = 15`.

        .. ipython:: python

            alpha = GF.primitive_elements[-1]
            print(GF.repr_table(alpha))

        Generate a representation table for :math:`\mathrm{GF}(2^4)` using a non-primitive element :math:`x^3 + x^2`. Notice,
        :math:`\textrm{ord}(x^3 + x^2) = 5 \ne 15`.

        .. ipython:: python

            beta = GF("x^3 + x^2")
            print(GF.repr_table(beta))
        """
        if sort not in ["power", "poly", "vector", "int"]:
            raise ValueError(f"Argument `sort` must be in ['power', 'poly', 'vector', 'int'], not {sort!r}.")
        if primitive_element is None:
            primitive_element = cls.primitive_element

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

        # Useful characters: https://www.utf8-chartable.de/unicode-utf8-table.pl?start=9472
        string = "╔" + "═"*N_power + "╤" + "═"*N_poly + "╤" + "═"*N_vec + "╤" + "═"*N_int + "╗"
        string += "\n║" + "Power".center(N_power) + "│" + "Polynomial".center(N_poly) + "│" + "Vector".center(N_vec) + "│" + "Integer".center(N_int) + "║"
        string += "\n║" + "═"*N_power + "╪" + "═"*N_poly + "╪" + "═"*N_vec + "╪" + "═"*N_int + "║"

        for i in range(x.size):
            d = None if i == 0 else degrees[i - 1]
            string += "\n║" + print_power(d).center(N_power) + "│" + poly_to_str(integer_to_poly(x[i], cls.characteristic)).center(N_poly) + "│" + str(integer_to_poly(x[i], cls.characteristic, degree=cls.degree-1)).center(N_vec) + "│" + cls._print_int(x[i]).center(N_int) + "║"

            if i < x.size - 1:
                string += "\n╟" + "─"*N_power + "┼" + "─"*N_poly + "┼" + "─"*N_vec + "┼" + "─"*N_int + "╢"

        string += "\n╚" + "═"*N_power + "╧" + "═"*N_poly + "╧"+ "═"*N_vec + "╧" + "═"*N_int + "╝"

        return string

    def arithmetic_table(
        cls,
        operation: Literal["+", "-", "*", "/"],
        x: Optional["FieldArray"] = None,
        y: Optional["FieldArray"] = None
    ) -> str:
        r"""
        Generates the specified arithmetic table for the Galois field.

        Parameters
        ----------
        operation : str
            The arithmetic operation, either `"+"`, `"-"`, `"*"`, or `"/"`.
        x : galois.FieldArray, optional
            Optionally specify the :math:`x` values for the arithmetic table. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}`.
        y : galois.FieldArray, optional
            Optionally specify the :math:`y` values for the arithmetic table. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}` for addition, subtraction, and multiplication and
            :math:`\{1, \dots, p^m - 1\}` for division.

        Returns
        -------
        str
            A UTF-8 formatted arithmetic table.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**2)
            print(GF.arithmetic_table("+"))

        .. ipython:: python

            GF.display("poly");
            print(GF.arithmetic_table("+"))

        .. ipython:: python

            GF.display("power");
            print(GF.arithmetic_table("+"))

        .. ipython:: python

            GF.display("poly");
            x = GF.Random(5); x
            y = GF.Random(3); y
            print(GF.arithmetic_table("+", x=x, y=y))
            GF.display();
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
            cls._set_print_power_vars(x)
            print_element = cls._print_power

        operation_str = f"x {operation} y"

        N = max([len(print_element(e)) for e in x]) + 2
        N_left = max(N, len(operation_str) + 2)

        # Useful characters: https://www.utf8-chartable.de/unicode-utf8-table.pl?start=9472
        string = "╔" + "═"*N_left + "╦" + ("═"*N + "╤")*(y.size - 1) + "═"*N + "╗"
        string += "\n║" + operation_str.rjust(N_left - 1) + " ║"
        for j in range(y.size):
            string += print_element(y[j]).center(N)
            string += "│" if j < y.size - 1 else "║"
        string += "\n╠" + "═"*N_left + "╬" + ("═"*N + "╪")*(y.size - 1) + "═"*N + "╣"

        for i in range(x.size):
            string += "\n║" + print_element(x[i]).rjust(N_left - 1) + " ║"
            for j in range(y.size):
                string += print_element(Z[i,j]).center(N)
                string += "│" if j < y.size - 1 else "║"

            if i < x.size - 1:
                string += "\n╟" + "─"*N_left + "╫" + ("─"*N + "┼")*(y.size - 1) + "─"*N + "╢"

        string += "\n╚" + "═"*N_left + "╩" + ("═"*N + "╧")*(y.size - 1) + "═"*N + "╝"

        return string

    ###############################################################################
    # Array display methods
    ###############################################################################

    def _formatter(cls, array):
        # pylint: disable=attribute-defined-outside-init
        formatter = {}
        if cls.display_mode == "poly":
            formatter["int"] = cls._print_poly
            formatter["object"] = cls._print_poly
        elif cls.display_mode == "power":
            cls._set_print_power_vars(array)
            formatter["int"] = cls._print_power
            formatter["object"] = cls._print_power
        elif array.dtype == np.object_:
            formatter["object"] = cls._print_int
        return formatter

    def _print_int(cls, element):  # pylint: disable=no-self-use
        return f"{int(element)}"

    def _print_poly(cls, element):
        poly = integer_to_poly(element, cls.characteristic)
        poly_var = "α" if cls.primitive_element == cls.characteristic else "x"
        return poly_to_str(poly, poly_var=poly_var)

    def _set_print_power_vars(cls, array):
        nonzero_idxs = np.nonzero(array)
        if array.ndim > 1:
            max_power = np.max(cls._ufunc("log")(array[nonzero_idxs], cls.primitive_element))
            if max_power > 1:
                cls._display_power_width = 2 + len(str(max_power))
            else:
                cls._display_power_width = 1
        else:
            cls._display_power_width = None

    def _print_power(cls, element):
        if element == 0:
            s = "0"
        else:
            power = cls._ufunc("log")(element, cls.primitive_element)
            if power > 1:
                s = f"α^{power}"
            elif power == 1:
                s = "α"
            else:
                s = "1"

        if cls._display_power_width:
            return s.rjust(cls._display_power_width)
        else:
            return s

    ###############################################################################
    # Class attributes
    ###############################################################################

    @property
    def name(cls) -> str:
        """
        str: The Galois field name.

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
        int: The prime characteristic :math:`p` of the Galois field :math:`\mathrm{GF}(p^m)`. Adding
        :math:`p` copies of any element will always result in :math:`0`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**8, display="poly")
            GF.characteristic
            a = GF.Random(low=1); a
            a * GF.characteristic
            @suppress
            GF.display();

        .. ipython:: python

            GF = galois.GF(31)
            GF.characteristic
            a = GF.Random(low=1); a
            a * GF.characteristic
        """
        return cls._characteristic

    @property
    def degree(cls) -> int:
        r"""
        int: The prime characteristic's degree :math:`m` of the Galois field :math:`\mathrm{GF}(p^m)`. The degree
        is a positive integer.

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
        int: The order :math:`p^m` of the Galois field :math:`\mathrm{GF}(p^m)`. The order of the field is also equal to
        the field's size.

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
        galois.Poly: The irreducible polynomial :math:`f(x)` of the Galois field :math:`\mathrm{GF}(p^m)`. The irreducible
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
        bool: Indicates whether the :obj:`irreducible_poly` is a primitive polynomial. If so, :math:`x` is a primitive element
        of the Galois field.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**8, display="poly")
            GF.irreducible_poly
            GF.primitive_element

            # The irreducible polynomial is a primitive polynomial if the primitive element is a root
            GF.irreducible_poly(GF.primitive_element, field=GF)
            GF.is_primitive_poly
            @suppress
            GF.display();

        Here is an example using the :math:`\mathrm{GF}(2^8)` field from AES, which does not use a primitive polynomial.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly=galois.Poly.Degrees([8,4,3,1,0]), display="poly")
            GF.irreducible_poly
            GF.primitive_element

            # The irreducible polynomial is a primitive polynomial if the primitive element is a root
            GF.irreducible_poly(GF.primitive_element, field=GF)
            GF.is_primitive_poly
            @suppress
            GF.display();
        """
        return cls._is_primitive_poly

    @property
    def primitive_element(cls) -> "FieldArray":
        r"""
        galois.FieldArray: A primitive element :math:`\alpha` of the Galois field :math:`\mathrm{GF}(p^m)`. A primitive element is a multiplicative
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
        galois.FieldArray: All primitive elements :math:`\alpha` of the Galois field :math:`\mathrm{GF}(p^m)`. A primitive element is a multiplicative
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
    def is_prime_field(cls) -> bool:
        """
        bool: Indicates if the field's order is prime.

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
        bool: Indicates if the field's order is a prime power.

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
        galois.FieldClass: The prime subfield :math:`\mathrm{GF}(p)` of the extension field :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            print(galois.GF(2).prime_subfield.properties)
            print(galois.GF(2**8).prime_subfield.properties)
            print(galois.GF(31).prime_subfield.properties)
            print(galois.GF(7**5).prime_subfield.properties)
        """
        return cls._prime_subfield

    @property
    def dtypes(cls) -> List[np.dtype]:
        """
        list: List of valid integer :obj:`numpy.dtype` values that are compatible with this Galois field. Creating an array with an
        unsupported dtype will throw a `TypeError` exception.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2); GF.dtypes
            GF = galois.GF(2**8); GF.dtypes
            GF = galois.GF(31); GF.dtypes
            GF = galois.GF(7**5); GF.dtypes

        For Galois fields that cannot be represented by :obj:`numpy.int64`, the only valid dtype is :obj:`numpy.object_`.

        .. ipython:: python

            GF = galois.GF(2**100); GF.dtypes
            GF = galois.GF(36893488147419103183); GF.dtypes
        """
        return cls._dtypes

    @property
    def display_mode(cls) -> str:
        r"""
        str: The representation of Galois field elements, either `"int"`, `"poly"`, or `"power"`. This can be
        changed with :func:`display`.

        Examples
        --------
        For the polynomial representation, when the primitive element is :math:`\alpha = x` in :math:`\mathrm{GF}(p)[x]` the polynomial
        indeterminate used is :math:`\alpha`.

        .. ipython:: python

            GF = galois.GF(2**8)
            print(GF.properties)
            a = GF.Random()
            print(GF.display_mode, a)
            with GF.display("poly"):
                print(GF.display_mode, a)
            with GF.display("power"):
                print(GF.display_mode, a)
            # The display mode is reset after exiting the context manager
            print(GF.display_mode, a)

        But when the primitive element is :math:`\alpha \ne x` in :math:`\mathrm{GF}(p)[x]`, the polynomial
        indeterminate used is :math:`x`.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly=galois.Poly.Degrees([8,4,3,1,0]))
            print(GF.properties)
            a = GF.Random()
            print(GF.display_mode, a)
            with GF.display("poly"):
                print(GF.display_mode, a)
            with GF.display("power"):
                print(GF.display_mode, a)
            # The display mode is reset after exiting the context manager
            print(GF.display_mode, a)

        The power representation displays elements as powers of :math:`\alpha` the primitive element, see
        :obj:`FieldClass.primitive_element`.

        .. ipython:: python

            with GF.display("power"):
                print(GF.display_mode, a)
            # The display mode is reset after exiting the context manager
            print(GF.display_mode, a)
        """
        return cls._display_mode

    @property
    def ufunc_mode(cls) -> str:
        """
        str: The mode for ufunc compilation, either `"jit-lookup"`, `"jit-calculate"`, or `"python-calculate"`.

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
        list: All supported ufunc modes for this Galois field array class.

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
    def default_ufunc_mode(cls) -> str:
        """
        str: The default ufunc arithmetic mode for this Galois field.

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

    @property
    def properties(cls) -> str:
        """
        str: A formatted string displaying relevant properties of the Galois field.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2); print(GF.properties)
            GF = galois.GF(2**8); print(GF.properties)
            GF = galois.GF(31); print(GF.properties)
            GF = galois.GF(7**5); print(GF.properties)
        """
        string = f"{cls.name}:"
        string += f"\n  characteristic: {cls.characteristic}"
        string += f"\n  degree: {cls.degree}"
        string += f"\n  order: {cls.order}"
        string += f"\n  irreducible_poly: {cls.irreducible_poly.string}"
        string += f"\n  is_primitive_poly: {cls.is_primitive_poly}"
        string += f"\n  primitive_element: {poly_to_str(integer_to_poly(cls.primitive_element, cls.characteristic))}"
        return string


class DirMeta(type):
    """
    A mixin metaclass that overrides __dir__ so that dir() and tab-completion in ipython of `FieldArray` classes
    (which are `FieldClass` instances) include the methods and properties from the metaclass. Python does not
    natively include metaclass properties in dir().

    This is a separate class because it will be mixed in to `GF2Meta`, `GF2mMeta`, `GFpMeta`, and `GFpmMeta` separately. Otherwise, the
    sphinx documentation of `FieldArray` gets messed up.

    Also, to not mess up the sphinx documentation of `GF2`, we had to create a custom sphinx template `class_gf2.rst` that
    manually includes all the classmethods and methods. This is because there is no way to redefine __dir__ for `GF2` and not have
    sphinx get confused when using autoclass.
    """

    def __dir__(cls):
        if isinstance(cls, FieldClass):
            meta_dir = dir(type(cls))
            classmethods = [attribute for attribute in super().__dir__() if attribute[0] != "_" and inspect.ismethod(getattr(cls, attribute))]
            return sorted(meta_dir + classmethods)
        else:
            return super().__dir__()


class DisplayContext:
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
    An array over :math:`\mathrm{GF}(p^m)`.

    Important
    ---------
    :obj:`galois.FieldArray` is an abstract base class for all Galois field array classes and cannot be instantiated
    directly. Instead, :obj:`galois.FieldArray` subclasses are created using the class factory :func:`galois.GF`.

    This class is included in the API to allow the user to test if an array is a Galois field array subclass.

    .. ipython:: python

        GF = galois.GF(7)
        issubclass(GF, galois.FieldArray)
        x = GF([1,2,3]); x
        isinstance(x, galois.FieldArray)

    Notes
    -----
    :obj:`galois.FieldArray` is an abstract base class and cannot be instantiated directly. Instead, the user creates a :obj:`galois.FieldArray`
    subclass for the field :math:`\mathrm{GF}(p^m)` by calling the class factory :func:`galois.GF`, e.g. `GF = galois.GF(p**m)`. In this case,
    `GF` is a subclass of :obj:`galois.FieldArray` and an instance of :obj:`galois.FieldClass`, a metaclass that defines special methods and attributes
    related to the Galois field.

    :obj:`galois.FieldArray`, and `GF`, is a subclass of :obj:`numpy.ndarray` and its constructor `x = GF(array_like)` has the same syntax as
    :func:`numpy.array`. The returned :obj:`galois.FieldArray` instance `x` is a :obj:`numpy.ndarray` that is acted upon like any other
    numpy array, except all arithmetic is performed in :math:`\mathrm{GF}(p^m)` not in :math:`\mathbb{Z}` or :math:`\mathbb{R}`.

    Examples
    --------
    Construct the Galois field class for :math:`\mathrm{GF}(2^8)` using the class factory :func:`galois.GF` and then display
    some relevant properties of the field. See :obj:`galois.FieldClass` for a complete list of Galois field array class
    methods and attributes.

    .. ipython:: python

        GF256 = galois.GF(2**8)
        GF256
        print(GF256.properties)

    Depending on the field's order, only certain numpy dtypes are supported. See :obj:`galois.FieldClass.dtypes` for more details.

    .. ipython:: python

        GF256.dtypes

    Galois field arrays can be created from existing numpy arrays.

    .. ipython:: python

        x = np.array([155, 232, 162, 159,  63,  29, 247, 141,  75, 189], dtype=int)

        # Explicit Galois field array creation -- a copy is performed
        GF256(x)

        # Or view an existing numpy array as a Galois field array -- no copy is performed
        x.view(GF256)

    Galois field arrays can also be created explicitly by converting an "array-like" object.

    .. ipython:: python

        # A scalar GF(2^8) element from its integer representation
        GF256(37)

        # A scalar GF(2^8) element from its polynomial representation
        GF256("x^5 + x^2 + 1")

        # A GF(2^8) array from a list of elements in their integer representation
        GF256([[142, 27], [92, 253]])

        # A GF(2^8) array from a list of elements in their integer and polynomial representations
        GF256([[142, "x^5 + x^2 + 1"], [92, 253]])

    There's also an alternate constructor :func:`Vector` (and accompanying :func:`vector` method) to convert an array of coefficients
    over :math:`\mathrm{GF}(p)` with last dimension :math:`m` into Galois field elements in :math:`\mathrm{GF}(p^m)`.

    .. ipython:: python

        # A scalar GF(2^8) element from its vector representation
        GF256.Vector([0, 0, 1, 0, 0, 1, 0, 1])

        # A GF(2^8) array from a list of elements in their vector representation
        GF256.Vector([[[1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0, 1, 1]], [[0, 1, 0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 1]]])

    Newly-created arrays will use the smallest unsigned dtype, unless otherwise specified.

    .. ipython:: python

        a = GF256([66, 166, 27, 182, 125]); a
        a.dtype
        b = GF256([66, 166, 27, 182, 125], dtype=np.int64); b
        b.dtype
    """
    # pylint: disable=unsupported-membership-test,not-an-iterable

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
        Creates an array over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        array : int, str, tuple, list, numpy.ndarray, galois.FieldArray
            The input array-like object to be converted to a Galois field array. See the examples section for demonstations of array creation
            using each input type. See see :func:`galois.FieldClass.display` and :obj:`galois.FieldClass.display_mode` for a description of the
            "integer" and "polynomial" representation of Galois field elements.

            * :obj:`int`: A single integer, which is the "integer representation" of a Galois field element, creates a 0-D array.
            * :obj:`str`: A single string, which is the "polynomial representation" of a Galois field element, creates a 0-D array.
            * :obj:`tuple`, :obj:`list`: A list or tuple (or nested lists/tuples) of ints or strings (which can be mix-and-matched) creates an array of
            Galois field elements from their integer or polynomial representations.
            * :obj:`numpy.ndarray`, :obj:`galois.FieldArray`: An array of ints creates a copy of the array over this specific field.

        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.
        copy : bool, optional
            The `copy` keyword argument from :func:`numpy.array`. The default is `True` which makes a copy of the input array.
        order : str, optional
            The `order` keyword argument from :func:`numpy.array`. Valid values are `"K"` (default), `"A"`, `"C"`, or `"F"`.
        ndmin : int, optional
            The `ndmin` keyword argument from :func:`numpy.array`. The minimum number of dimensions of the output.
            The default is 0.

        Returns
        -------
        galois.FieldArray
            An array over :math:`\mathrm{GF}(p^m)`.
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
        shape: Union[int, Tuple[()], Tuple[int]],
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates a Galois field array with all zeros.

        Parameters
        ----------
        shape : int, tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M,N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of zeros.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Zeros((2,5))
        """
        dtype = cls._get_dtype(dtype)
        array = np.zeros(shape, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Ones(
        cls,
        shape: Union[int, Tuple[()], Tuple[int]],
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates a Galois field array with all ones.

        Parameters
        ----------
        shape : int, tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M,N)`, represents a 2-D array with each element indicating the size in each dimension.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of ones.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Ones((2,5))
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
        Creates a 1-D Galois field array with a range of field elements.

        Parameters
        ----------
        start : int
            The starting Galois field value (inclusive) in its integer representation.
        stop : int
            The stopping Galois field value (exclusive) in its integer representation.
        step : int, optional
            The space between values. The default is 1.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A 1-D Galois field array of a range of field elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Range(10,20)
        """
        if not stop <= cls.order:
            raise ValueError(f"The stopping value must be less than the field order of {cls.order}, not {stop}.")
        dtype = cls._get_dtype(dtype)
        array = np.arange(start, stop, step=step, dtype=dtype)
        return array.view(cls)

    @classmethod
    def Random(
        cls,
        shape: Union[int, Tuple[()], Tuple[int]] = (),
        low: Optional[int] = 0,
        high: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        """
        Creates a Galois field array with random field elements.

        Parameters
        ----------
        shape : int, tuple
            A numpy-compliant `shape` tuple, see :obj:`numpy.ndarray.shape`. An empty tuple `()` represents a scalar.
            A single integer or 1-tuple, e.g. `N` or `(N,)`, represents the size of a 1-D array. A 2-tuple, e.g.
            `(M,N)`, represents a 2-D array with each element indicating the size in each dimension.
        low : int, optional
            The lowest value (inclusive) of a random field element in its integer representation. The default is 0.
        high : int, optional
            The highest value (exclusive) of a random field element in its integer representation. The default is `None`
            which represents the field's order :math:`p^m`.
        seed: int, numpy.random.Generator, optional
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed. If so,
            it is used directly when `dtype != np.object_`. Its state is used to seed `random.seed()`, otherwise.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of random field elements.

        Examples
        --------
        Generate a random matrix with an unpredictable seed.

        .. ipython:: python

            GF = galois.GF(31)
            GF.Random((2,5))

        Generate a random array with a specified seed. This produces repeatable outputs.

        .. ipython:: python

            GF.Random(10, seed=123456789)
            GF.Random(10, seed=123456789)

        Generate a group of random arrays with one global seed.

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
        Creates a 1-D Galois field array of the field's elements :math:`\{0, \dots, p^m-1\}`.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A 1-D Galois field array of all the field's elements.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**4)
            GF.Elements()

        As usual, Galois field elements can be displayed in either the "integer" (default), "polynomial", or "power" representation.
        This can be changed by calling :func:`galois.FieldClass.display`.

        .. ipython:: python

            # Permanently set the display mode to "poly"
            GF.display("poly");
            GF.Elements()
            # Temporarily set the display mode to "power"
            with GF.display("power"):
                print(GF.Elements())
            # Reset the display mode to "int"
            GF.display();
        """
        return cls.Range(0, cls.order, step=1, dtype=dtype)

    @classmethod
    def Identity(
        cls,
        size: int,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        r"""
        Creates an :math:`n \times n` Galois field identity matrix.

        Parameters
        ----------
        size : int
            The size :math:`n` along one axis of the matrix. The resulting array has shape `(size, size)`.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field identity matrix of shape `(size, size)`.

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
        Creates an :math:`m \times n` Vandermonde matrix of :math:`a \in \mathrm{GF}(p^m)`.

        Parameters
        ----------
        a : int, galois.FieldArray
            An element of :math:`\mathrm{GF}(p^m)`.
        m : int
            The number of rows in the Vandermonde matrix.
        n : int
            The number of columns in the Vandermonde matrix.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            The :math:`m \times n` Vandermonde matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**3)
            a = GF.primitive_element
            V = GF.Vandermonde(a, 7, 7)
            with GF.display("power"):
                print(V)
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
        Creates a Galois field array over :math:`\mathrm{GF}(p^m)` from length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`vector` method.

        Parameters
        ----------
        array : array_like
            The input array with field elements in :math:`\mathrm{GF}(p)` to be converted to a Galois field array in :math:`\mathrm{GF}(p^m)`.
            The last dimension of the input array must be :math:`m`. An input array with shape `(n1, n2, m)` has output shape `(n1, n2)`. By convention,
            the vectors are ordered from highest degree to 0-th degree.
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array over :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**6)
            vec = galois.GF2.Random((3,6)); vec
            a = GF.Vector(vec); a
            with GF.display("poly"):
                print(a)
            a.vector()
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

    def vector(
        self,
        dtype: Optional[Union[np.dtype, int, object]] = None
    ) -> "FieldArray":
        r"""
        Converts the Galois field array over :math:`\mathrm{GF}(p^m)` to length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`Vector` constructor. For an array with shape `(n1, n2)`, the output shape
        is `(n1, n2, m)`. By convention, the vectors are ordered from highest degree to 0-th degree.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            dtype for this class, i.e. the first element in :obj:`galois.FieldClass.dtypes`.

        Returns
        -------
        galois.FieldArray
            A Galois field array of length-:math:`m` vectors over :math:`\mathrm{GF}(p)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**6)
            a = GF.Random(3); a
            with GF.display("poly"):
                print(a)
            vec = a.vector(); vec
            GF.Vector(vec)
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

        **Row reduction operations**

        1. Swap the position of any two rows.
        2. Multiply a row by a non-zero scalar.
        3. Add one row to a scalar multiple of another row.

        Parameters
        ----------
        ncols : int, optional
            The number of columns to perform Gaussian elimination over. The default is `None` which represents
            the number of columns of the input array.

        Returns
        -------
        galois.FieldArray
            The reduced row echelon form of the input array.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        One column is a linear combination of another.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A[:,2] = A[:,1] * GF(17); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        One row is a linear combination of another.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A[3,:] = A[2,:] * GF(8); A
            A.row_reduce()
            np.linalg.matrix_rank(A)
        """
        return row_reduce(self, ncols=ncols)

    def lu_decompose(self) -> "FieldArray":
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices.

        Returns
        -------
        galois.FieldArray
            The lower triangular matrix.
        galois.FieldArray
            The upper triangular matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(5)

            # Not every square matrix has an LU decomposition
            A = GF([[2, 4, 4, 1], [3, 3, 1, 4], [4, 3, 4, 2], [4, 4, 3, 1]])
            L, U = A.lu_decompose()
            L
            U

            # A = L U
            np.array_equal(A, L @ U)
        """
        return lu_decompose(self)

    def lup_decompose(self) -> "FieldArray":
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices using partial pivoting.

        Returns
        -------
        galois.FieldArray
            The lower triangular matrix.
        galois.FieldArray
            The upper triangular matrix.
        galois.FieldArray
            The permutation matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(5)
            A = GF([[1, 3, 2, 0], [3, 4, 2, 3], [0, 2, 1, 4], [4, 3, 3, 1]])
            L, U, P = A.lup_decompose()
            L
            U
            P

            # P A = L U
            np.array_equal(P @ A, L @ U)
        """
        return lup_decompose(self)

    def field_trace(self) -> "FieldArray":
        r"""
        Computes the field trace :math:`\mathrm{Tr}_{L / K}(x)` of the elements of :math:`x`.

        Returns
        -------
        galois.FieldArray
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
        """
        if not type(self).is_extension_field:
            raise TypeError(f"The Galois field must be an extension field to compute the field trace, not {type(self)}.")
        field = type(self)
        subfield = field.prime_subfield
        p = field.characteristic
        m = field.degree
        conjugates = np.power.outer(self, p**np.arange(0, m, dtype=field.dtypes[-1]))
        trace = np.add.reduce(conjugates, axis=-1)
        return subfield(trace)

    def field_norm(self) -> "FieldArray":
        r"""
        Computes the field norm :math:`\mathrm{N}_{L / K}(x)` of the elements of :math:`x`.

        Returns
        -------
        galois.FieldArray
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
        """
        if not type(self).is_extension_field:
            raise TypeError(f"The Galois field must be an extension field to compute the field norm, not {type(self)}.")
        field = type(self)
        subfield = field.prime_subfield
        p = field.characteristic
        m = field.degree
        norm = self**((p**m - 1) // (p - 1))
        return subfield(norm)

    def characteristic_poly(self) -> "Poly":
        r"""
        Computes the characteristic polynomial of the square :math:`n \times n` matrix :math:`\mathbf{A}`.

        Returns
        -------
        Poly
            The degree-:math:`n` characteristic polynomial :math:`p_A(x)` of :math:`\mathbf{A}`.

        Notes
        -----
        An :math:`n \times n` matrix :math:`\mathbf{A}` has characteristic polynomial
        :math:`p_A(x) = \textrm{det}(x\mathbf{I} - \mathbf{A})`.

        Special properties of the characteristic polynomial are the constant coefficient is
        :math:`\textrm{det}(-\mathbf{A})` and the :math:`x^{n-1}` coefficient is :math:`-\textrm{Tr}(\mathbf{A})`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Characteristic_polynomial

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**5)
            A = GF.Random((3,3)); A
            poly = A.characteristic_poly(); poly

        .. ipython:: python

            # The x^0 coefficient is det(-A)
            poly.coeffs[-1] == np.linalg.det(-A)
            # The x^n-1 coefficient is -Tr(A)
            poly.coeffs[1] == -np.trace(A)
        """
        if not self.ndim == 2:
            raise ValueError(f"The array must be 2-D to compute its characteristic poly, not have shape {self.shape}.")
        if not self.shape[0] == self.shape[1]:
            raise ValueError(f"The array must be square to compute its characteristic poly, not have shape {self.shape}.")

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

    ###############################################################################
    # Special methods (redefined to add docstrings)
    ###############################################################################

    def __add__(self, other):  # pylint: disable=useless-super-delegation
        """
        Adds two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self + other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5); b
            a + b
        """
        return super().__add__(other)

    def __sub__(self, other):  # pylint: disable=useless-super-delegation
        """
        Subtracts two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self - other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5); b
            a - b
        """
        return super().__sub__(other)

    def __mul__(self, other):  # pylint: disable=useless-super-delegation
        """
        Multiplies two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field.

        Warning
        -------
        When both multiplicands are :obj:`galois.FieldArray`, that indicates a Galois field multiplication. When one
        multiplicand is an integer or integer :obj:`numpy.ndarray`, that indicates a scalar multiplication (repeated addition).
        Galois field multiplication and scalar multiplication are equivalent in prime fields, but not in extension fields.

        Parameters
        ----------
        other : numpy.ndarray, galois.FieldArray
            A :obj:`numpy.ndarray` of integers for scalar multiplication or a :obj:`galois.FieldArray` of Galois field elements
            for finite field multiplication.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self * other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5); b
            a * b

        When both multiplicands are Galois field elements, that indicates a Galois field multiplication.

        .. ipython:: python

            GF = galois.GF(2**4, display="poly")
            a = GF(7); a
            b = GF(2); b
            a * b
            @suppress
            GF.display();

        When one multiplicand is an integer, that indicates a scalar multiplication (repeated addition).

        .. ipython:: python

            a * 2
            a + a
        """
        return super().__mul__(other)

    def __truediv__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self / other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            a / b
        """
        return super().__truediv__(other)

    def __floordiv__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self // other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            a // b
        """
        return super().__floordiv__(other)  # pylint: disable=too-many-function-args

    def __divmod__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise and returns the quotient and remainder.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent. In Galois fields, the remainder
        is always zero.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self // other`.
        galois.FieldArray
            The Galois field array `self % other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            q, r = divmod(a, b)
            q, r
            b*q + r
        """
        return super().__divmod__(other)

    def __mod__(self, other):  # pylint: disable=useless-super-delegation
        """
        Divides two Galois field arrays element-wise and returns the remainder.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. Both arrays must be over
        the same Galois field. In Galois fields, true division and floor division are equivalent. In Galois fields, the remainder
        is always zero.

        Parameters
        ----------
        other : galois.FieldArray
            The other Galois field array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self % other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = GF.Random(5, low=1); b
            a % b
        """
        return super().__mod__(other)

    def __pow__(self, other):
        """
        Exponentiates a Galois field array element-wise.

        `Broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules apply. The first array must be a
        Galois field array and the second must be an integer or integer array.

        Parameters
        ----------
        other : int, numpy.ndarray
            The exponent(s) as an integer or integer array.

        Returns
        -------
        galois.FieldArray
            The Galois field array `self ** other`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            a = GF.Random((2,5)); a
            b = np.random.default_rng().integers(0, 10, 5); b
            a ** b
        """
        # NOTE: Calling power here instead of `super().__pow__(other)` because when doing so `x ** GF(2)` will invoke `np.square(x)` and not throw
        # an error. This way `np.power(x, GF(2))` is called which correctly checks whether the second argument is an integer.
        return np.power(self, other)

    ###############################################################################
    # Overridden numpy methods
    ###############################################################################

    def __array_finalize__(self, obj):
        """
        A numpy dunder method that is called after "new", "view", or "new from template". It is used here to ensure
        that view casting to a Galois field array has the appropriate dtype and that the values are in the field.
        """
        if obj is not None and not isinstance(obj, FieldArray):
            # Only invoked on view casting
            if obj.dtype not in type(self).dtypes:
                raise TypeError(f"{type(self).name} can only have integer dtypes {type(self).dtypes}, not {obj.dtype}.")
            self._check_array_values(obj)

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if np.isscalar(item):
            # Return scalar array elements as 0-dimensional Galois field arrays. This enables Galois field arithmetic
            # on scalars, which would otherwise be implemented using standard integer arithmetic.
            item = self.__class__(item, dtype=self.dtype)
        return item

    def __setitem__(self, key, value):
        # Verify the values to be written to the Galois field array are in the field
        value = self._check_array_like_object(value)
        super().__setitem__(key, value)

    def __array_function__(self, func, types, args, kwargs):
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
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

    def astype(self, dtype, **kwargs):  # pylint: disable=arguments-differ
        if dtype not in type(self).dtypes:
            raise TypeError(f"{type(self).name} arrays can only be cast as integer dtypes in {type(self).dtypes}, not {dtype}.")
        return super().astype(dtype, **kwargs)

    def dot(self, b, out=None):
        # `np.dot(a, b)` is also available as `a.dot(b)`. Need to override this here for proper results.
        return dot(self, b, out=out)

    ###############################################################################
    # Display methods
    ###############################################################################

    def __str__(self):
        return self.__repr__()
        # formatter = type(self)._formatter(self)

        # with np.printoptions(formatter=formatter):
        #     string = super().__str__()

        # return string

    def __repr__(self):
        formatter = type(self)._formatter(self)

        cls = type(self)
        class_name = cls.__name__
        with np.printoptions(formatter=formatter):
            cls.__name__ = "GF"  # Rename the class so very large fields don't create large indenting
            string = super().__repr__()
        cls.__name__ = class_name

        # Remove the dtype from the repr and add the Galois field order
        dtype_idx = string.find("dtype")
        if dtype_idx == -1:
            string = string[:-1] + f", {cls._order_str})"
        else:
            string = string[:dtype_idx] + f"{cls._order_str})"

        return string


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
    ###############################################################################

    @staticmethod
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.positive()` is faster.
        """
        return a

    @staticmethod
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_xor()` is faster.
        """
        return a ^ b

    @staticmethod
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Not actually used. `np.bitwise_and()` is faster.
        """
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

        return a & b

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        else:
            return a

    @staticmethod
    @numba.extending.register_jitable(inline="always")
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")
        if b != 1:
            raise ArithmeticError("In GF(2), 1 is the only multiplicative generator.")

        return 0


@set_module("galois")
class GF2(FieldArray, metaclass=GF2Meta, characteristic=2, degree=1, order=2, primitive_element=1, compile="jit-calculate"):
    r"""
    An array over :math:`\mathrm{GF}(2)`.

    This class is a pre-generated :obj:`galois.FieldArray` subclass generated with `galois.GF(2)` and is included in the API
    for convenience. See :obj:`galois.FieldArray` and :obj:`galois.FieldClass` for more complete documentation and examples.

    Examples
    --------
    This class is equivalent (and, in fact, identical) to the class returned from the Galois field class constructor.

    .. ipython:: python

        print(galois.GF2)
        GF2 = galois.GF(2); print(GF2)
        GF2 is galois.GF2

    The Galois field properties can be viewed by class attributes, see :obj:`galois.FieldClass`.

    .. ipython:: python

        # View a summary of the field's properties
        print(galois.GF2.properties)

        # Or access each attribute individually
        galois.GF2.irreducible_poly
        galois.GF2.is_prime_field

    The class's constructor mimics the call signature of :func:`numpy.array`.

    .. ipython:: python

        # Construct a Galois field array from an iterable
        galois.GF2([1,0,1,1,0,0,0,1])

        # Or an iterable of iterables
        galois.GF2([[1,0], [1,1]])

        # Or a single integer
        galois.GF2(1)
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
    Create a polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    The polynomial :math:`f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0` has coefficients :math:`\{a_{d}, a_{d-1}, \dots, a_1, a_0\}`
    in :math:`\mathrm{GF}(p^m)`.

    Parameters
    ----------
    coeffs : tuple, list, numpy.ndarray, galois.FieldArray
        The polynomial coefficients :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}` with type :obj:`galois.FieldArray`. Alternatively, an iterable :obj:`tuple`,
        :obj:`list`, or :obj:`numpy.ndarray` may be provided and the Galois field domain is taken from the `field` keyword argument.
    field : galois.FieldClass, optional
        The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

        * :obj:`None` (default): If the coefficients are a :obj:`galois.FieldArray`, they won't be modified. If the coefficients are not explicitly
          in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(coeffs)`.
        * :obj:`galois.FieldClass`: The coefficients are explicitly converted to this Galois field `field(coeffs)`.

    order : str, optional
        The interpretation of the coefficient degrees.

        * `"desc"` (default): The first element of `coeffs` is the highest degree coefficient, i.e. :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}`.
        * `"asc"`: The first element of `coeffs` is the lowest degree coefficient, i.e. :math:`\{a_0, a_1, \dots,  a_{d-1}, a_d\}`.

    Returns
    -------
    galois.Poly
        The polynomial :math:`f(x)`.

    Examples
    --------
    Create a polynomial over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        galois.Poly([1,0,1,1])
        galois.Poly.Degrees([3,1,0])

    Create a polynomial over :math:`\mathrm{GF}(2^8)`.

    .. ipython:: python

        GF = galois.GF(2**8)
        galois.Poly([124,0,223,0,0,15], field=GF)

        # Alternate way of constructing the same polynomial
        galois.Poly.Degrees([5,3,0], coeffs=[124,223,15], field=GF)

    Polynomial arithmetic using binary operators.

    .. ipython:: python

        a = galois.Poly([117,0,63,37], field=GF); a
        b = galois.Poly([224,0,21], field=GF); b

        a + b
        a - b

        # Compute the quotient of the polynomial division
        a / b

        # True division and floor division are equivalent
        a / b == a // b

        # Compute the remainder of the polynomial division
        a % b

        # Compute both the quotient and remainder in one pass
        divmod(a, b)
    """
    # pylint: disable=too-many-public-methods

    # Increase my array priority so numpy will call my __radd__ instead of its own __add__
    __array_priority__ = 100

    def __new__(
        cls,
        coeffs: Union[Tuple[int], List[int], np.ndarray, FieldArray],
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
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x) = 0`.

        Examples
        --------
        Construct the zero polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Zero()

        Construct the zero polynomial over :math:`\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Zero(field=GF)
        """
        return Poly([0], field=field)

    @classmethod
    def One(cls, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs the polynomial :math:`f(x) = 1` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x) = 1`.

        Examples
        --------
        Construct the one polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.One()

        Construct the one polynomial over :math:`\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.One(field=GF)
        """
        return Poly([1], field=field)

    @classmethod
    def Identity(cls, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs the polynomial :math:`f(x) = x` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x) = x`.

        Examples
        --------
        Construct the identity polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Identity()

        Construct the identity polynomial over :math:`\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Identity(field=GF)
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
        degree : int
            The degree of the polynomial.
        seed: int, numpy.random.Generator, optional
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed. If so,
            it is used directly when `dtype != np.object_`. Its state is used to seed `random.seed()`, otherwise.
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a random degree-:math:`5` polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Random(5)

        Construct a random degree-:math:`5` polynomial over :math:`\mathrm{GF}(2^8)` with a given seed. This produces repeatable results.

        .. ipython:: python

            GF = galois.GF(2**8)
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
        integer : int
            The integer representation of the polynomial :math:`f(x)`.
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Notes
        -----
        The integer value :math:`i` represents the polynomial :math:`f(x) = a_d x^{d} + a_{d-1} x^{d-1} + \dots + a_1 x + a_0`
        over the field :math:`\mathrm{GF}(p^m)` if :math:`i = a_{d}(p^m)^{d} + a_{d-1}(p^m)^{d-1} + \dots + a_1(p^m) + a_0` using integer arithmetic,
        not finite field arithmetic.

        Said differently, if the polynomial coefficients :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}` are considered as the "digits" of a radix-:math:`p^m`
        value, the polynomial's integer representation is the decimal value (radix-:math:`10`).

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from its integer representation.

        .. ipython:: python

            galois.Poly.Integer(5)

        Construct a polynomial over :math:`\mathrm{GF}(2^8)` from its integer representation.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Integer(13*256**3 + 117, field=GF)
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
    def String(cls, string: str, field: Optional[FieldClass] = GF2) -> "Poly":
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its string representation.

        Parameters
        ----------
        string : str
            The string representation of the polynomial :math:`f(x)`.
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
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

            galois.Poly.String("x^2 + 1")

        Construct a polynomial over :math:`\mathrm{GF}(2^8)` from its string representation.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.String("13x^3 + 117", field=GF)
        """
        if not isinstance(string, str):
            raise TypeError(f"Argument `string` be an string, not {type(string)}")

        return Poly.Degrees(*str_to_sparse_poly(string), field=field)


    @classmethod
    def Degrees(
        cls,
        degrees: Union[Tuple[int], List[int], np.ndarray],
        coeffs: Optional[Union[Tuple[int], List[int], np.ndarray, FieldArray]] = None,
        field: Optional[FieldClass] = None
    ) -> "Poly":
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its non-zero degrees.

        Parameters
        ----------
        degrees : tuple, list, numpy.ndarray
            The polynomial degrees with non-zero coefficients.
        coeffs : tuple, list, numpy.ndarray, galois.FieldArray, optional
            The corresponding non-zero polynomial coefficients with type :obj:`galois.FieldArray`. Alternatively, an iterable :obj:`tuple`,
            :obj:`list`, or :obj:`numpy.ndarray` may be provided and the Galois field domain is taken from the `field` keyword argument. The
            default is `None` which corresponds to all ones.
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the coefficients are a :obj:`galois.FieldArray`, they won't be modified. If the coefficients are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(coeffs)`.
            * :obj:`galois.FieldClass`: The coefficients are explicitly converted to this Galois field `field(coeffs)`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            galois.Poly.Degrees([3,1,0])

        Construct a polynomial over :math:`\mathrm{GF}(2^8)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Degrees([3,1,0], coeffs=[251,73,185], field=GF)
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
        roots: Union[Tuple[int], List[int], np.ndarray, FieldArray],
        multiplicities: Optional[Union[Tuple[int], List[int], np.ndarray]] = None,
        field: Optional[FieldClass] = None
    ) -> "Poly":
        r"""
        Constructs a monic polynomial over :math:`\mathrm{GF}(p^m)` from its roots.

        Parameters
        ----------
        roots : tuple, list, numpy.ndarray, galois.FieldArray
            The roots of the desired polynomial with type :obj:`galois.FieldArray`. Alternatively, an iterable :obj:`tuple`,
            :obj:`list`, or :obj:`numpy.ndarray` may be provided and the Galois field domain is taken from the `field` keyword argument.
        multiplicities : tuple, list, numpy.ndarray, optional
            The corresponding root multiplicities. The default is `None` which corresponds to all ones, i.e. `[1,]*len(roots)`.
        field : galois.FieldClass, optional
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the roots are a :obj:`galois.FieldArray`, they won't be modified. If the roots are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(roots)`.
            * :obj:`galois.FieldClass`: The roots are explicitly converted to this Galois field `field(roots)`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Notes
        -----
        The polynomial :math:`f(x)` with :math:`k` roots :math:`\{r_1, r_2, \dots, r_k\}` with multiplicities
        :math:`\{m_1, m_2, \dots, m_k\}` is

        .. math::

            f(x) &= (x - r_1)^{m_1} (x - r_2)^{m_2} \dots (x - r_k)^{m_k}

            f(x) &= a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0

        with degree :math:`d = \sum_{i=1}^{k} m_i`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from a list of its roots.

        .. ipython:: python

            roots = [0, 0, 1]
            p = galois.Poly.Roots(roots); p
            # Evaluate the polynomial at its roots
            p(roots)

        Construct a polynomial over :math:`\mathrm{GF}(2^8)` from a list of its roots with specific multiplicities.

        .. ipython:: python

            GF = galois.GF(2**8)
            roots = [121, 198, 225]
            multiplicities = [1, 2, 1]
            p = galois.Poly.Roots(roots, multiplicities=multiplicities, field=GF); p
            # Evaluate the polynomial at its roots
            p(roots)
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
        size : int, optional
            The fixed size of the coefficient array. Zeros will be added for higher-order terms. This value must be
            at least `degree + 1` or a :obj:`ValueError` will be raised. The default is `None` which corresponds
            to `degree + 1`.

        order : str, optional
            The interpretation of the coefficient degrees.

            * `"desc"` (default): The first element returned is the highest degree coefficient.
            * `"asc"`: The first element returned is the lowest degree coefficient.

        Returns
        -------
        galois.FieldArray
            An array of the polynomial coefficients with length `size`, either in ascending order or descending order.

        Notes
        -----
        This accessor is similar to :obj:`coeffs`, but it has more settings. By default, `Poly.coeffs == Poly.coefficients()`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.coeffs
            p.coefficients()
            # Return the coefficients in ascending order
            p.coefficients(order="asc")
            # Return the coefficients in ascending order with size 8
            p.coefficients(8, order="asc")
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
        galois.Poly
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

    def roots(self, multiplicity: bool = False) -> FieldArray:
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
        np.ndarray
            The multiplicity of each root, only returned if `multiplicity=True`.

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
            f(\alpha^i) &= a_{d}(\alpha^i)^{d} + a_{d-1}(\alpha^i)^{d-1} + \dots + a_1(\alpha^i) + a_0

            f(\alpha^i) &\overset{\Delta}{=} \lambda_{i,d} + \lambda_{i,d-1} + \dots + \lambda_{i,1} + \lambda_{i,0}

            f(\alpha^i) &= \sum_{j=0}^{d} \lambda_{i,j}

        The next power of :math:`\alpha` can be easily calculated from the previous calculation.

        .. math::
            f(\alpha^{i+1}) &= a_{d}(\alpha^{i+1})^{d} + a_{d-1}(\alpha^{i+1})^{d-1} + \dots + a_1(\alpha^{i+1}) + a_0

            f(\alpha^{i+1}) &= a_{d}(\alpha^i)^{d}\alpha^d + a_{d-1}(\alpha^i)^{d-1}\alpha^{d-1} + \dots + a_1(\alpha^i)\alpha + a_0

            f(\alpha^{i+1}) &= \lambda_{i,d}\alpha^d + \lambda_{i,d-1}\alpha^{d-1} + \dots + \lambda_{i,1}\alpha + \lambda_{i,0}

            f(\alpha^{i+1}) &= \sum_{j=0}^{d} \lambda_{i,j}\alpha^j

        References
        ----------
        * https://en.wikipedia.org/wiki/Chien_search

        Examples
        --------
        Find the roots of a polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            p = galois.Poly.Roots([0,]*7 + [1,]*13); p
            p.roots()
            p.roots(multiplicity=True)

        Find the roots of a polynomial over :math:`\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            p = galois.Poly.Roots([18,]*7 + [155,]*13 + [227,]*9, field=GF); p
            p.roots()
            p.roots(multiplicity=True)
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
        zero = Poly.Zero(self.field)
        poly = self.copy()
        multiplicity = 1

        while True:
            # If the root is also a root of the derivative, then its a multiple root.
            poly = poly.derivative()

            if poly == zero:
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
        k : int, optional
            The number of derivatives to compute. 1 corresponds to :math:`p'(x)`, 2 corresponds to :math:`p''(x)`, etc.
            The default is 1.

        Returns
        -------
        galois.Poly
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
        For example, :math:`3 \cdot a = a + a + a`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Formal_derivative

        Examples
        --------
        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            p = galois.Poly.Random(7); p
            p.derivative()

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(2)

        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(7)`.

        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly.Random(11, field=GF); p
            p.derivative()
            p.derivative(2)
            p.derivative(3)

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(7)

        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            p = galois.Poly.Random(7, field=GF); p
            p.derivative()

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(2)
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

    def __str__(self):
        return f"Poly({self.string}, {self.field.name})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        t = tuple([self.field.order,] + self.nonzero_degrees.tolist() + self.nonzero_coeffs.tolist())
        return hash(t)

    def __call__(self, x: FieldArray, field: Optional[FieldClass] = None) -> FieldArray:
        """
        Evaluate the polynomial.

        Parameters
        ----------
        x : galois.FieldArray
            An array (or 0-D scalar) of field elements to evaluate the polynomial over.
        field : galois.FieldClass, optional
            The Galois field to evaluate the polynomial over. The default is `None` which represents
            the polynomial's current field, i.e. :obj:`field`.

        Returns
        -------
        galois.FieldArray
            The result of the polynomial evaluation of the same shape as `x`.
        """
        if field is None:
            field = self.field
            coeffs = self.coeffs
        else:
            assert isinstance(field, FieldClass)
            coeffs = field(self.coeffs)
        if not isinstance(x, field):
            x = field(x)
        return field._poly_evaluate(coeffs, x)

    def _check_inputs_are_polys(self, a, b):
        if not isinstance(a, (Poly, self.field)):
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {b.field.name}, not {type(a)}.")
        if not isinstance(b, (Poly, self.field)):
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {a.field.name}, not {type(b)}.")

        # Promote a single field element to a 0-degree polynomial
        if not isinstance(a, Poly):
            if not a.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {a.size}.")
            a = Poly(np.atleast_1d(a))
        if not isinstance(b, Poly):
            if not b.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {b.size}.")
            b = Poly(np.atleast_1d(b))

        if not a.field is b.field:
            raise TypeError(f"Both polynomial operands must be over the same field, not {str(a.field)} and {str(b.field)}.")

        if isinstance(a, SparsePoly) or isinstance(b, SparsePoly):
            return SparsePoly, a, b
        elif isinstance(a, BinaryPoly) or isinstance(b, BinaryPoly):
            return BinaryPoly, a, b
        else:
            return DensePoly, a, b

    def __add__(self, other):
        """
        Adds two polynomials.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`c(x) = a(x) + b(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            a + b
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._add(a, b)

    def __radd__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._add(b, a)

    def __sub__(self, other):
        """
        Subtracts two polynomials.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`c(x) = a(x) - b(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            a - b
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._sub(a, b)

    def __rsub__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._sub(b, a)

    def __mul__(self, other):
        """
        Multiplies two polynomials.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`c(x) = a(x) b(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            a * b
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mul(a, b)

    def __rmul__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mul(b, a)

    def __divmod__(self, other):
        """
        Divides two polynomials and returns the quotient and remainder.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The quotient polynomial :math:`q(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.
        galois.Poly
            The remainder polynomial :math:`r(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            q, r = divmod(a, b)
            q, r
            b*q + r
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(a, b)

    def __rdivmod__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(b, a)

    def __truediv__(self, other):
        """
        Divides two polynomials and returns the quotient.

        True division and floor division are equivalent.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The quotient polynomial :math:`q(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            divmod(a, b)
            a / b
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(a, b)[0]

    def __rtruediv__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(b, a)[0]

    def __floordiv__(self, other):
        """
        Divides two polynomials and returns the quotient.

        True division and floor division are equivalent.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The quotient polynomial :math:`q(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            divmod(a, b)
            a // b
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(a, b)[0]

    def __rfloordiv__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(b, a)[0]

    def __mod__(self, other):
        """
        Divides two polynomials and returns the remainder.

        Parameters
        ----------
        other : galois.Poly
            The polynomial :math:`b(x)`.

        Returns
        -------
        galois.Poly
            The remainder polynomial :math:`r(x)` such that :math:`a(x) = b(x)q(x) + r(x)`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            b = galois.Poly.Random(3); b
            divmod(a, b)
            a % b
        """
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mod(a, b)

    def __rmod__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mod(b, a)

    def __pow__(self, other):
        """
        Exponentiates the polynomial to an integer power.

        Parameters
        ----------
        other : int
            The non-negative integer exponent.

        Returns
        -------
        galois.Poly
            The polynomial :math:`a(x)**b`.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            a**3
            a * a * a
        """
        if not isinstance(other, (int, np.integer)):
            raise TypeError(f"For polynomial exponentiation, the second argument must be an int, not {other}.")
        if not other >= 0:
            raise ValueError(f"Can only exponentiate polynomials to non-negative integers, not {other}.")
        field, a, power = self.field, self, other

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

    def __neg__(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Poly):
            if other == 0:
                addendum = " If you are trying to compare against 0, use `galois.Poly.Zero(GF)` or `galois.Poly([0], field=GF)`."
            elif other == 1:
                addendum = " If you are trying to compare against 1, use `galois.Poly.One(GF)` or `galois.Poly([1], field=GF)`."
            else:
                addendum = ""
            raise TypeError(f"Can't compare Poly and non-Poly objects, {other} is not a Poly object.{addendum}")

        return self.field is other.field and np.array_equal(self.nonzero_degrees, other.nonzero_degrees) and np.array_equal(self.nonzero_coeffs, other.nonzero_coeffs)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return self.degree + 1

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
        galois.FieldClass: The Galois field array class to which the coefficients belong.

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
        int: The degree of the polynomial, i.e. the highest degree with non-zero coefficient.

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
        numpy.ndarray: An array of the polynomial degrees that have non-zero coefficients, in degree-descending order. The entries of
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
        galois.FieldArray: The non-zero coefficients of the polynomial in degree-descending order. The entries of :obj:`nonzero_degrees`
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
        numpy.ndarray: An array of the polynomial degrees in degree-descending order. The entries of :obj:`degrees`
        are paired with :obj:`coeffs`.

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
        galois.FieldArray: The coefficients of the polynomial in degree-descending order. The entries of :obj:`degrees` are
        paired with :obj:`coeffs`.

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
        int: The integer representation of the polynomial. For the polynomial :math:`f(x) =  a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0`
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
            p.integer == 3*7**3 + 5*7**1 + 2*7**0
        """
        return sparse_poly_to_integer(self.nonzero_degrees, self.nonzero_coeffs, self.field.order)

    @property
    def string(self) -> str:
        """
        str: The string representation of the polynomial, without specifying the Galois field.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.string
        """
        return sparse_poly_to_str(self.nonzero_degrees, self.nonzero_coeffs)


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

        elif a == zero:
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

        elif a == zero:
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

        elif a == zero:
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
