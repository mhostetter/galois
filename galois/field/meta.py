import numpy as np

from ..modular import totatives
from ..overrides import set_module

from .meta_ufunc import FieldUfunc
from .meta_func import FieldFunc
from .poly_conversion import integer_to_poly, poly_to_str

__all__ = ["FieldMeta"]


@set_module("galois")
class FieldMeta(FieldUfunc, FieldFunc):
    """
    Defines a metaclass for all :obj:`galois.FieldArray` classes.

    This metaclass gives :obj:`galois.FieldArray` classes returned from :func:`galois.GF` class methods and properties
    relating to its Galois field.
    """
    # pylint: disable=no-value-for-parameter,too-many-public-methods

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
        return f"<class 'numpy.ndarray over {cls.name}'>"

    ###############################################################################
    # Methods
    ###############################################################################

    def compile(cls, mode, target="cpu"):
        """
        Recompile the just-in-time compiled numba ufuncs with a new calculation mode or target.

        Parameters
        ----------
        mode : str
            The method of field computation, either `"jit-lookup"`, `"jit-calculate"`, `"python-calculate"`. The "jit-lookup" mode will
            use Zech log, log, and anti-log lookup tables for speed. The "jit-calculate" mode will not store any lookup tables, but perform field
            arithmetic on the fly. The "jit-calculate" mode is designed for large fields that cannot store lookup tables in RAM.
            Generally, "jit-calculate" is slower than "jit-lookup". The "python-calculate" mode is reserved for extremely large fields. In
            this mode the ufuncs are not JIT-compiled, but are pur python functions operating on python ints. The list of valid
            modes for this field is in :obj:`galois.FieldMeta.ufunc_modes`.
        target : str, optional
            The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. The default
            is `"cpu"`. For extremely large fields the only supported target is `"cpu"` (which doesn't use numba it uses pure python to
            calculate the field arithmetic). The list of valid targets for this field is in :obj:`galois.FieldMeta.ufunc_targets`.
        """
        mode = cls.default_ufunc_mode if mode == "auto" else mode
        if mode not in cls.ufunc_modes:
            raise ValueError(f"Argument `mode` must be in {cls.ufunc_modes} for {cls.name}, not {mode}.")
        if target not in cls.ufunc_targets:
            raise ValueError(f"Argument `target` must be in {cls.ufunc_targets} for {cls.name}, not {target}.")

        if mode == cls.ufunc_mode and target == cls.ufunc_target:
            # Don't need to rebuild these ufuncs
            return

        cls._ufunc_mode = mode
        cls._ufunc_target = target
        cls._compile_ufuncs()

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
            max_power = np.max(np.log(array[nonzero_idxs]))
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
            power = np.log(element)
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
    # Class methods
    ###############################################################################

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
            default primitive element, :obj:`galois.FieldMeta.primitive_element`.

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
    # Class attributes
    ###############################################################################

    @property
    def name(cls):
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
    def characteristic(cls):
        """
        int: The prime characteristic :math:`p` of the Galois field :math:`\\mathrm{GF}(p^m)`. Adding
        :math:`p` copies of any element will always result in :math:`0`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**8)
            GF.characteristic
            a = GF.Random(); a
            a * GF.characteristic

        .. ipython:: python

            GF = galois.GF(31)
            GF.characteristic
            a = GF.Random(); a
            a * GF.characteristic
        """
        return cls._characteristic

    @property
    def degree(cls):
        """
        int: The prime characteristic's degree :math:`m` of the Galois field :math:`\\mathrm{GF}(p^m)`. The degree
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
    def order(cls):
        """
        int: The order :math:`p^m` of the Galois field :math:`\\mathrm{GF}(p^m)`. The order of the field is also equal to
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
    def irreducible_poly(cls):
        """
        galois.Poly: The irreducible polynomial :math:`f(x)` of the Galois field :math:`\\mathrm{GF}(p^m)`. The irreducible
        polynomial is of degree :math:`m` over :math:`\\mathrm{GF}(p)`.

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
        # return Poly(cls._irreducible_poly, field=cls.prime_subfield)

    @property
    def is_primitive_poly(cls):
        """
        bool: Indicates whether the :obj:`irreducible_poly` is a primitive polynomial.

        Examples
        --------

        .. ipython:: python

            GF = galois.GF(2**8)
            GF.irreducible_poly
            GF.primitive_element

            # The irreducible polynomial is a primitive polynomial is the primitive element is a root
            GF.irreducible_poly(GF.primitive_element, field=GF)
            GF.is_primitive_poly

        .. ipython:: python

            # Field used in AES
            GF = galois.GF(2**8, irreducible_poly=galois.Poly.Degrees([8,4,3,1,0]))
            GF.irreducible_poly
            GF.primitive_element

            # The irreducible polynomial is a primitive polynomial is the primitive element is a root
            GF.irreducible_poly(GF.primitive_element, field=GF)
            GF.is_primitive_poly
        """
        return cls._is_primitive_poly

    @property
    def primitive_element(cls):
        """
        int: A primitive element :math:`\\alpha` of the Galois field :math:`\\mathrm{GF}(p^m)`. A primitive element is a multiplicative
        generator of the field, such that :math:`\\mathrm{GF}(p^m) = \\{0, 1, \\alpha^1, \\alpha^2, \\dots, \\alpha^{p^m - 2}\\}`.

        A primitive element is a root of the primitive polynomial :math:`f(x)`, such that :math:`f(\\alpha) = 0` over
        :math:`\\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).primitive_element
            galois.GF(2**8).primitive_element
            galois.GF(31).primitive_element
            galois.GF(7**5).primitive_element
        """
        # Ensure accesses of this property doesn't alter it
        return cls(cls._primitive_element)

    @property
    def primitive_elements(cls):
        """
        int: All primitive elements :math:`\\alpha` of the Galois field :math:`\\mathrm{GF}(p^m)`. A primitive element is a multiplicative
        generator of the field, such that :math:`\\mathrm{GF}(p^m) = \\{0, 1, \\alpha^1, \\alpha^2, \\dots, \\alpha^{p^m - 2}\\}`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).primitive_elements
            galois.GF(2**8).primitive_elements
            galois.GF(31).primitive_elements
            galois.GF(7**5).primitive_elements
        """
        powers = np.array(totatives(cls.order - 1))
        return np.sort(cls.primitive_element ** powers)

    @property
    def is_prime_field(cls):
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
    def is_extension_field(cls):
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
    def prime_subfield(cls):
        """
        galois.FieldMeta: The prime subfield :math:`\\mathrm{GF}(p)` of the extension field :math:`\\mathrm{GF}(p^m)`.

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
    def dtypes(cls):
        """
        list: List of valid integer :obj:`numpy.dtype` objects that are compatible with this Galois field.

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
        raise NotImplementedError

    @property
    def display_mode(cls):
        """
        str: The representation of Galois field elements, either `"int"`, `"poly"`, or `"power"`. This can be
        changed with :func:`display`.

        Examples
        --------
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
        return cls._display_mode

    @property
    def ufunc_mode(cls):
        """
        str: The mode for ufunc compilation, either `"jit-lookup"`, `"jit-calculate"`, `"python-calculate"`.

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
    def ufunc_modes(cls):
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
    def default_ufunc_mode(cls):
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
    def ufunc_target(cls):
        """
        str: The numba target for the JIT-compiled ufuncs, either `"cpu"`, `"parallel"`, or `"cuda"`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).ufunc_target
            galois.GF(2**8).ufunc_target
            galois.GF(31).ufunc_target
            galois.GF(7**5).ufunc_target
        """
        return cls._ufunc_target

    @property
    def ufunc_targets(cls):
        """
        list: All supported ufunc targets for this Galois field array class.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).ufunc_targets
            galois.GF(2**8).ufunc_targets
            galois.GF(31).ufunc_targets
            galois.GF(2**100).ufunc_targets
        """
        if cls.dtypes == [np.object_]:
            return ["cpu"]
        else:
            return ["cpu", "parallel", "cuda"]

    @property
    def properties(cls):
        string = f"{cls.name}:"
        string += f"\n  characteristic: {cls.characteristic}"
        string += f"\n  degree: {cls.degree}"
        string += f"\n  order: {cls.order}"
        if cls.degree > 1:
            string += f"\n  irreducible_poly: {cls.irreducible_poly}"
            string += f"\n  is_primitive_poly: {cls.is_primitive_poly}"
            string += f"\n  primitive_element: {cls.primitive_element!r}"
        return string


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
