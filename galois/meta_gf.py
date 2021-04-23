import numpy as np

from .meta_mixin_target import TargetMixin
from .modular import totatives


class GFMeta(TargetMixin):
    """
    Defines a metaclass for all :obj:`galois.GFArray` classes.

    This metaclass gives :obj:`galois.GFArray` classes returned from :func:`galois.GF` class methods and properties
    relating to its Galois field.
    """
    # pylint: disable=no-value-for-parameter,comparison-with-callable,too-many-public-methods

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._characteristic = None
        cls._degree = None
        cls._order = None
        cls._irreducible_poly = None
        cls._is_primitive_poly = None
        cls._primitive_element = None
        cls._ground_field = None
        cls._ufunc_mode = None
        cls._ufunc_target = None
        cls._display_mode = "int"
        cls._display_poly_var = "α"

    def __str__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

    def __repr__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

    ###############################################################################
    # Class methods
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
            modes for this field is in :obj:`galois.GFMeta.ufunc_modes`.
        target : str, optional
            The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`. The default
            is `"cpu"`. For extremely large fields the only supported target is `"cpu"` (which doesn't use numba it uses pure python to
            calculate the field arithmetic). The list of valid targets for this field is in :obj:`galois.GFMeta.ufunc_targets`.
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

        if cls.ufunc_mode == "jit-lookup":
            cls._compile_jit_lookup(target)
        elif cls.ufunc_mode == "jit-calculate":
            cls._compile_jit_calculate(target)
        elif cls.ufunc_mode == "python-calculate":
            cls._compile_python_calculate()
        else:
            raise RuntimeError(f"Attribute `ufunc_mode` was not processed, {cls._ufunc_mode}. Please submit a GitHub issue at https://github.com/mhostetter/galois/issues.")

    def display(cls, mode="int", poly_var="α"):
        """
        Sets the display mode for all Galois field arrays of this type.

        The display mode can be set to either the integer representation or polynomial representation.
        This function updates :obj:`display_mode` and :obj:`display_poly_var`.

        Parameters
        ----------
        mode : str, optional
            The field element display mode, either `"int"` (default) or `"poly"`.
        poly_var : str, optional
            The polynomial representation's variable. The default is `"α"`.

        Examples
        --------
        Change the display mode by calling the :func:`display` method.

        .. ipython:: python

            GF = galois.GF(2**8)
            a = GF.Random(); a
            GF.display("poly"); a

            # Reset to the default display mode
            GF.display(); a

        The :func:`display` method can also be used as a context manager.

        .. ipython:: python

            # The original display mode
            a

            # The new display context
            with GF.display("poly"):
                print(a)

            # Returns to the default display mode
            a
        """
        if mode not in ["int", "poly"]:
            raise ValueError(f"Argument `mode` must be in ['int', 'poly'], not {mode}.")
        if not isinstance(poly_var, str):
            raise TypeError(f"Argument `poly_var` must be a string, not {type(poly_var)}.")

        context = DisplayContext(cls)

        # Set the new state
        cls._display_mode = mode
        cls._display_poly_var = poly_var

        return context

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
            # galois.GF(7**5).name
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
            # galois.GF(7**5).degree
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
            # galois.GF(7**5).order
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
            # galois.GF(7**5).irreducible_poly
        """
        # Ensure accesses of this property don't alter it
        return cls._irreducible_poly.copy()

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

        A primitive element is a root of the primitive polynomial :math:`\\f(x)`, such that :math:`f(\\alpha) = 0` over
        :math:`\\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).primitive_element
            galois.GF(2**8).primitive_element
            galois.GF(31).primitive_element
            # galois.GF(7**5).primitive_element
        """
        # Ensure accesses of this property don't alter it
        return np.copy(cls._primitive_element)

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
            # galois.GF(7**5).primitive_elements
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
            # galois.GF(7**5).is_prime_field
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
            # galois.GF(7**5).is_extension_field
        """
        return cls._degree > 1

    @property
    def ground_field(cls):
        """
        galois.GFMeta: The ground field :math:`\\mathrm{GF}(p)` of the extension field :math:`\\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            print(galois.GF(2).ground_field.properties)
            print(galois.GF(2**8).ground_field.properties)
            print(galois.GF(31).ground_field.properties)
            # print(galois.GF(7**5).ground_field.properties)
        """
        return cls._ground_field

    @property
    def dtypes(cls):
        """
        list: List of valid integer :obj:`numpy.dtype` objects that are compatible with this Galois field. Valid data
        types are signed and unsinged integers that can represent decimal values in :math:`[0, p^m)`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).dtypes
            galois.GF(2**8).dtypes
            galois.GF(31).dtypes
            # galois.GF(7**5).dtypes

        For field's with orders that cannot be represented by :obj:`numpy.int64`, the only valid dtype is :obj:`numpy.object_`.

        .. ipython:: python

            galois.GF(2**100).dtypes
            galois.GF(36893488147419103183).dtypes
        """
        raise NotImplementedError

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
            # galois.GF(7**5).ufunc_mode
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
            # galois.GF(7**5).ufunc_target
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
    def display_mode(cls):
        """
        str: The representation of Galois field elements, either `"int"` or `"poly"`. This can be
        changed with :func:`display`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**8)
            GF.display_mode
            a = GF.Random(); a
            with GF.display("poly"):
                print(GF.display_mode)
                print(a)
        """
        return cls._display_mode

    @property
    def display_poly_var(cls):
        """
        str: The polynomial indeterminate for the polynomial representation of the field elements. The default
        is `"α"`. This can be changed with :func:`display`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**8)
            GF.display_mode, GF.display_poly_var
            a = GF.Random(); a
            with GF.display("poly"):
                print(GF.display_mode, GF.display_poly_var)
                print(a)
        """
        return cls._display_poly_var

    @property
    def properties(cls):
        """
        str: A formmatted string displaying relevant properties of the Galois field.

        Examples
        --------
        .. ipython:: python

            print(galois.GF(2).properties)
            print(galois.GF(2**8).properties)
            print(galois.GF(31).properties)
            # print(galois.GF(7**5).properties)
        """
        string = f"{cls.name}:"
        string += f"\n  characteristic: {cls.characteristic}"
        string += f"\n  degree: {cls.degree}"
        string += f"\n  order: {cls.order}"
        string += f"\n  irreducible_poly: {cls.irreducible_poly}"
        string += f"\n  is_primitive_poly: {cls.is_primitive_poly}"
        string += f"\n  primitive_element: {cls.primitive_element!r}"
        return string


class DisplayContext:
    """
    Simple context manager for the :obj:`GFArrayMeta.display` method.
    """

    def __init__(self, cls):
        # Save the previous state
        self.cls = cls
        self.mode = cls.display_mode
        self.poly_var = cls.display_poly_var

    def __enter__(self):
        # Don't need to do anything, we already set the new mode and poly_var in the display() method
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset mode and poly_var upon exiting the context
        self.cls._display_mode = self.mode
        self.cls._display_poly_var = self.poly_var
