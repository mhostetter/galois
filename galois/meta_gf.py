import numpy as np

from .meta_mixin_ufunc import UfuncMixin
from .modular import totatives
from .overrides import set_module

__all__ = ["GFMeta"]


@set_module("galois")
class GFMeta(UfuncMixin):
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
        cls._characteristic = kwargs.get("characteristic", None)
        cls._degree = kwargs.get("degree", None)
        cls._order = kwargs.get("order", None)
        cls._irreducible_poly = kwargs.get("irreducible_poly", None)
        cls._primitive_element = kwargs.get("primitive_element", None)

        cls._is_primitive_poly = None
        cls._prime_subfield = None

        cls._ufunc_mode = None
        cls._ufunc_target = None
        cls._display_mode = "int"

    def __str__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

    def __repr__(cls):
        return f"<class 'numpy.ndarray over {cls.name}'>"

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
    def prime_subfield(cls):
        """
        galois.GFMeta: The prime subfield :math:`\\mathrm{GF}(p)` of the extension field :math:`\\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            print(galois.GF(2).prime_subfield.properties)
            print(galois.GF(2**8).prime_subfield.properties)
            print(galois.GF(31).prime_subfield.properties)
            # print(galois.GF(7**5).prime_subfield.properties)
        """
        return cls._prime_subfield

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

    def __enter__(self):
        # Don't need to do anything, we already set the new mode in the display() method
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset mode and upon exiting the context
        self.cls._display_mode = self.mode
