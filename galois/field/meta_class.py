import inspect

import numpy as np

from ..overrides import set_module

from .meta_function import FunctionMeta
from .meta_ufunc import UfuncMeta
from .meta_properties import PropertiesMeta
from .poly_conversion import integer_to_poly, poly_to_str

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

    def __dir__(cls):
        if isinstance(cls, FieldClass):
            meta_dir = dir(type(cls))
            classmethods = [attribute for attribute in super().__dir__() if attribute[0] != "_" and inspect.ismethod(getattr(cls, attribute))]
            return sorted(meta_dir + classmethods)
        else:
            return super().__dir__()

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
        """
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

        For the polynomial representation, when the primitive element is :math:`x \\in \\mathrm{GF}(p)[x]` the polynomial
        indeterminate used is  `α`.

        .. ipython:: python

            GF = galois.GF(2**8)
            print(GF.properties)
            a = GF.Random(); a
            with GF.display("poly"):
                print(a)
            with GF.display("power"):
                print(a)

        But when the primitive element is not :math:`x \\in \\mathrm{GF}(p)[x]`, the polynomial
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

    def repr_table(cls, primitive_element=None):
        """
        Generates an element representation table comparing the power, polynomial, vector, and integer representations.

        Parameters
        ----------
        primitive_element : galois.FieldArray, optional
            The primitive element to use for the power representation. The default is `None` which uses the field's
            default primitive element, :obj:`galois.FieldClass.primitive_element`.

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

        .. ipython:: python

            alpha = GF.primitive_elements[-1]
            print(GF.repr_table(alpha))
        """
        if primitive_element is None:
            primitive_element = cls.primitive_element

        degrees = np.arange(0, cls.order - 1)
        x = np.concatenate((np.atleast_1d(cls(0)), primitive_element**degrees))
        prim = cls._print_poly(primitive_element)

        N_power = max(len("({})^{}".format(prim, str(cls.order - 1))), len("Power")) + 2
        N_poly = max([len(cls._print_poly(e)) for e in x] + [len("Polynomial")]) + 2
        N_vec = max([len(str(integer_to_poly(e, cls.characteristic, degree=cls.degree-1))) for e in x] + [len("Vector")]) + 2
        N_int = max([len(cls._print_int(e)) for e in x] + [len("Integer")]) + 2

        string = "╔" + "═"*N_power + "╦" + "═"*N_poly + "╦" + "═"*N_vec + "╦" + "═"*N_int + "╗"

        labels = "║" + "Power".center(N_power) + "│" + "Polynomial".center(N_poly) + "│" + "Vector".center(N_vec) + "│" + "Integer".center(N_int) + "║"
        string += "\n" + labels

        divider = "║" + "═"*N_power + "╬" + "═"*N_poly + "╬" + "═"*N_vec + "╬" + "═"*N_int + "║"
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

        bottom = "╚" + "═"*N_power + "╩" + "═"*N_poly + "╩"+ "═"*N_vec + "╩" + "═"*N_int + "╝"
        string += "\n" + bottom

        return string

    def arithmetic_table(cls, operation, mode="int"):
        """
        Generates the specified arithmetic table for the Galois field.

        Parameters
        ----------
        operation : str
            Either `"+"`, `"-"`, `"*"`, or `"/"`.
        mode : str, optional
            The display mode to represent the field elements, either `"int"` (default), `"poly"`, or `"power"`.

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

            GF = galois.GF(3**2)
            print(GF.arithmetic_table("+", mode="poly"))
        """
        # pylint: disable=too-many-branches
        if not operation in ["+", "-", "*", "/"]:
            raise ValueError(f"Argument `operation` must be in ['+', '-', '*', '/'], not {operation}.")
        if mode not in ["int", "poly", "power"]:
            raise ValueError(f"Argument `mode` must be in ['int', 'poly', 'power'], not {mode}.")

        x = cls.Elements() if mode != "power" else np.concatenate((np.atleast_1d(cls(0)), cls.primitive_element**np.arange(0, cls.order - 1)))
        y = x if operation != "/" else x[1:]
        X, Y = np.meshgrid(x, y, indexing="ij")

        if operation == "+":
            Z = X + Y
        elif operation == "-":
            Z = X - Y
        elif operation == "*":
            Z = X * Y
        else:
            Z = X / Y

        if mode == "int":
            print_element = cls._print_int
        elif mode == "poly":
            print_element = cls._print_poly
        else:
            cls._set_print_power_vars(x)
            print_element = cls._print_power

        operation_str = f"x {operation} y"

        N = max([len(print_element(e)) for e in x]) + 2
        N_left = max(N, len(operation_str) + 2)

        string = "╔" + "═"*N_left + ("╦" + "═"*N)*y.size + "╗"

        line = "║" + operation_str.rjust(N_left - 1) + " ║"
        for j in range(y.size):
            line += print_element(y[j]).center(N)
            line += "│" if j < y.size - 1 else "║"
        string += "\n" + line

        divider = "╠" + "═"*N_left + ("╬" + "═"*N)*y.size + "╣"
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

        bottom = "╚" + "═"*N_left + ("╩" + "═"*N)*y.size + "╝"
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


class DisplayContext:
    """
    Simple context manager for the :obj:`FieldArrayMeta.display` method.
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
