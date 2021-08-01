import inspect

import numpy as np

from .._overrides import set_module

from ._meta_function import FunctionMeta
from ._meta_ufunc import UfuncMeta
from ._meta_properties import PropertiesMeta
from ._poly_conversion import integer_to_poly, poly_to_str

__all__ = ["FieldClass"]


@set_module("galois")
class FieldClass(UfuncMeta, FunctionMeta, PropertiesMeta):
    """
    Defines a metaclass for all :obj:`galois.FieldArray` classes.

    This metaclass gives :obj:`galois.FieldArray` classes returned from :func:`galois.GF` class methods and properties
    relating to its Galois field.
    """
    # pylint: disable=abstract-method,no-value-for-parameter,unsupported-membership-test

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        cls._characteristic = kwargs.get("characteristic", None)
        cls._degree = kwargs.get("degree", None)
        cls._order = kwargs.get("order", None)
        cls._order_str = None
        cls._ufunc_mode = None
        cls._ufunc_target = None
        super().__init__(name, bases, namespace, **kwargs)

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
            cls._order_str = "order={}".format(cls.order)
        else:
            cls._order_str = "order={}^{}".format(cls.characteristic, cls.degree)

    def __str__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

    def __repr__(cls):
        return str(cls)

    ###############################################################################
    # Class methods
    ###############################################################################

    def compile(cls, mode):
        """
        Recompile the just-in-time compiled numba ufuncs for a new calculation mode.

        Parameters
        ----------
        mode : str
            The method of field computation, either `"jit-lookup"`, `"jit-calculate"`, `"python-calculate"`. The "jit-lookup" mode will
            use Zech log, log, and anti-log lookup tables for speed. The "jit-calculate" mode will not store any lookup tables, but perform field
            arithmetic on the fly. The "jit-calculate" mode is designed for large fields that cannot store lookup tables in RAM.
            Generally, "jit-calculate" is slower than "jit-lookup". The "python-calculate" mode is reserved for extremely large fields. In
            this mode the ufuncs are not JIT-compiled, but are pur python functions operating on python ints. The list of valid
            modes for this field is in :obj:`galois.FieldClass.ufunc_modes`.
        """
        mode = cls.default_ufunc_mode if mode == "auto" else mode
        if mode not in cls.ufunc_modes:
            raise ValueError(f"Argument `mode` must be in {cls.ufunc_modes} for {cls.name}, not {mode}.")

        if mode == cls.ufunc_mode:
            # Don't need to rebuild these ufuncs
            return

        cls._ufunc_mode = mode
        cls._compile_ufuncs()

    def display(cls, mode="int"):
        r"""
        Sets the display mode for all Galois field arrays of this type.

        The display mode can be set to either the integer representation, polynomial representation, or power
        representation. This function updates :obj:`display_mode`.

        For the power representation, :func:`np.log` is computed on each element. So for large fields without lookup
        tables, this may take longer than desired.

        Parameters
        ----------
        mode : str, optional
            The field element display mode, either `"int"` (default), `"poly"`, or `"power"`.

        Examples
        --------
        Change the display mode by calling the :func:`display` method.

        .. ipython:: python

            GF = galois.GF(2**8)
            a = GF.Random(); a

            # Change the display mode going forward
            GF.display("poly"); a
            GF.display("power"); a

            # Reset to the default display mode
            GF.display(); a

        The :func:`display` method can also be used as a context manager, as shown below.

        For the polynomial representation, when the primitive element is :math:`x \in \mathrm{GF}(p)[x]` the polynomial
        indeterminate used is  `α`.

        .. ipython:: python

            GF = galois.GF(2**8)
            print(GF.properties)
            a = GF.Random(); a
            with GF.display("poly"):
                print(a)
            with GF.display("power"):
                print(a)

        But when the primitive element is not :math:`x \in \mathrm{GF}(p)[x]`, the polynomial
        indeterminate used is `x`.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly=galois.Poly.Degrees([8,4,3,1,0]))
            print(GF.properties)
            a = GF.Random(); a
            with GF.display("poly"):
                print(a)
            with GF.display("power"):
                print(a)
        """
        if mode not in ["int", "poly", "power"]:
            raise ValueError(f"Argument `mode` must be in ['int', 'poly', 'power'], not {mode}.")

        context = DisplayContext(cls)

        # Set the new state
        cls._display_mode = mode

        return context

    def repr_table(cls, primitive_element=None, sort="power"):
        """
        Generates a field element representation table comparing the power, polynomial, vector, and integer representations.

        Parameters
        ----------
        primitive_element : galois.FieldArray, optional
            The primitive element to use for the power representation. The default is `None` which uses the field's
            default primitive element, :obj:`primitive_element`.
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
            print(GF.repr_table())
            print(GF.repr_table(sort="int"))

        .. ipython:: python

            alpha = GF.primitive_elements[-1]
            print(GF.repr_table(alpha))
        """
        if sort not in ["power", "poly", "vector", "int"]:
            raise ValueError(f"Argument `sort` must be in ['power', 'poly', 'vector', 'int'], not {sort}.")
        if primitive_element is None:
            primitive_element = cls.primitive_element

        degrees = np.arange(0, cls.order - 1)
        x = primitive_element**degrees
        if sort != "power":
            idxs = np.argsort(x)
            degrees, x = degrees[idxs], x[idxs]
        x = np.concatenate((np.atleast_1d(cls(0)), x))  # Add 0 = alpha**-Inf
        prim = cls._print_poly(primitive_element)

        N_power = max(len("({})^{}".format(prim, str(cls.order - 1))), len("Power")) + 2
        N_poly = max([len(cls._print_poly(e)) for e in x] + [len("Polynomial")]) + 2
        N_vec = max([len(str(integer_to_poly(e, cls.characteristic, degree=cls.degree-1))) for e in x] + [len("Vector")]) + 2
        N_int = max([len(cls._print_int(e)) for e in x] + [len("Integer")]) + 2

        # Useful characters: https://www.utf8-chartable.de/unicode-utf8-table.pl?start=9472
        string = "╔" + "═"*N_power + "╤" + "═"*N_poly + "╤" + "═"*N_vec + "╤" + "═"*N_int + "╗"

        labels = "║" + "Power".center(N_power) + "│" + "Polynomial".center(N_poly) + "│" + "Vector".center(N_vec) + "│" + "Integer".center(N_int) + "║"
        string += "\n" + labels

        divider = "║" + "═"*N_power + "╪" + "═"*N_poly + "╪" + "═"*N_vec + "╪" + "═"*N_int + "║"
        string += "\n" + divider

        for i in range(x.size):
            if i  == 0:
                power = "0"
            else:
                power = "({})^{}".format(prim, degrees[i - 1]) if len(prim) > 1 else "{}^{}".format(prim, degrees[i - 1])
            line = "║" + power.center(N_power) + "│" + cls._print_poly(x[i]).center(N_poly) + "│" + str(integer_to_poly(x[i], cls.characteristic, degree=cls.degree-1)).center(N_vec) + "│" + cls._print_int(x[i]).center(N_int) + "║"
            string += "\n" + line

            if i < x.size - 1:
                divider = "╟" + "─"*N_power + "┼" + "─"*N_poly + "┼" + "─"*N_vec + "┼" + "─"*N_int + "╢"
                string += "\n" + divider

        bottom = "╚" + "═"*N_power + "╧" + "═"*N_poly + "╧"+ "═"*N_vec + "╧" + "═"*N_int + "╝"
        string += "\n" + bottom

        return string

    def arithmetic_table(cls, operation, x=None, y=None):
        r"""
        Generates the specified arithmetic table for the Galois field.

        Parameters
        ----------
        operation : str
            The arithmetic operation, either `"+"`, `"-"`, `"*"`, or `"/"`.
        x : galois.FieldArray, optional
            Optionally specify the :math:`x` values for the arithmetic operation. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}`.
        y : galois.FieldArray, optional
            Optionally specify the :math:`y` values for the arithmetic operation. The default is `None`
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
            raise ValueError(f"Argument `operation` must be in ['+', '-', '*', '/'], not {operation}.")

        if cls.display_mode == "power":
            # Order elements by powers of the primitive element
            x_default = np.concatenate((np.atleast_1d(cls(0)), cls.primitive_element**np.arange(0, cls.order - 1)))
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

        line = "║" + operation_str.rjust(N_left - 1) + " ║"
        for j in range(y.size):
            line += print_element(y[j]).center(N)
            line += "│" if j < y.size - 1 else "║"
        string += "\n" + line

        divider = "╠" + "═"*N_left + "╬" + ("═"*N + "╪")*(y.size - 1) + "═"*N + "╣"
        string += "\n" + divider

        for i in range(x.size):
            line = "║" + print_element(x[i]).rjust(N_left - 1) + " ║"
            for j in range(y.size):
                line += print_element(Z[i,j]).center(N)
                line += "│" if j < y.size - 1 else "║"
            string += "\n" + line

            if i < x.size - 1:
                divider = "╟" + "─"*N_left + "╫" + ("─"*N + "┼")*(y.size - 1) + "─"*N + "╢"
                string += "\n" + divider

        bottom = "╚" + "═"*N_left + "╩" + ("═"*N + "╧")*(y.size - 1) + "═"*N + "╝"
        string += "\n" + bottom

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
        return "{:d}".format(int(element))

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


class DirMeta(type):
    """
    A mixin metaclass that overrides __dir__ so that dir() and tab-completion in ipython of `FieldArray` classes
    (which are `FieldClass` instances) include the methods and properties from the metaclass. Python does not
    natively include metaclass properties in dir().

    This is a separate class because it will be mixed in to `GF2Meta`, `GF2mMeta`, `GFpMeta`, and `GFpmMeta` separately. Otherwise, the
    sphinx documentation of `FieldArray` is messed up.

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
