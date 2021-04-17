import numpy as np

from .meta_mixin_target import TargetMixin
from .modular import totatives


class GFMeta(TargetMixin):
    """
    Defines a metaclass for all :obj:`galois.GFArray` classes.

    This metaclass gives :obj:`galois.GFArray` classes returned from :func:`galois.GF` class methods and properties
    relating to its finite field.
    """
    # pylint: disable=no-value-for-parameter

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
        cls._dtypes = []
        cls._ufunc_mode = None
        cls._ufunc_target = None
        cls._display_mode = "int"
        cls._display_poly_var = "α"

    def __str__(cls):
        return f"<class 'numpy.ndarray' over {cls.name}>"

    def _valid_dtypes(cls):
        raise NotImplementedError

    ###############################################################################
    # Class methods
    ###############################################################################

    def target(cls, target, mode):
        """
        Retarget the just-in-time compiled numba ufuncs.

        Parameters
        ----------
        target : str
            The `target` keyword argument from :obj:`numba.vectorize`, either `"cpu"`, `"parallel"`, or `"cuda"`.
        mode : str
            The type of field computation, either `"lookup"` or `"calculate"`. The "lookup" mode will use Zech log, log,
            and anti-log lookup tables for speed. The "calculate" mode will not store any lookup tables, but perform field
            arithmetic on the fly. The "calculate" mode is designed for large fields that cannot store lookup tables in RAM.
            Generally, "calculate" will be slower than "lookup".
        """
        if target not in ["cpu", "parallel", "cuda"]:
            raise ValueError(f"Argument `target` must be in ['cpu', 'parallel', 'cuda'], not {target}.")
        if mode not in ["auto", "lookup", "calculate", "object"]:
            raise ValueError(f"Argument `mode` must be in ['auto', 'lookup', 'calculate', 'object'], not {mode}.")

        mode = cls._check_ufunc_mode(mode)
        new_mode = mode != cls._ufunc_mode
        new_target = target != cls._ufunc_target

        if not new_mode and not new_target:
            return

        cls._ufunc_target = target
        cls._ufunc_mode = mode

        if cls._ufunc_mode == "object":
            cls._target_python_calculate()
        elif cls._ufunc_mode == "lookup":
            cls._target_jit_lookup(target)
        elif cls._ufunc_mode == "calculate":
            cls._target_jit_calculate(target)
        else:
            raise AttributeError(f"Attribute `ufunc_mode` is invalid, {cls._ufunc_mode}.")

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
        galois.Poly: The irreducible polynomial :math:`\\pi(x)` of the Galois field :math:`\\mathrm{GF}(p^m)`. The irreducible
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

        A primitive element is a root of the primitive polynomial :math:`\\pi(x)`, such that :math:`\\pi(\\alpha) = 0`

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
        # Ensure accesses of this property don't alter it
        return list(cls._dtypes)

    @property
    def ufunc_mode(cls):
        """
        str: The mode for ufunc compilation, either `"lookup"`, `"calculate"`, `"object"`.

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
    def ufunc_target(cls):
        """
        str: The numba target for the JIT-compiled ufuncs, either `"cpu"`, `"parallel"`, `"cuda"`, or `None`.

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
        str: A formmatted string displaying all the Galois field's attributes.

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
        string += f"\n  dtypes: {[np.dtype(d).name for d in cls.dtypes]}"
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
