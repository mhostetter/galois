import numpy as np


class GFArrayMeta(type):
    """
    Defines a metaclass for all :obj:`GFArray` classes.

    This metaclass gives :obj:`GFArray` classes returned from :func:`galois.GF` class properties relating to its finite field
    attributes. This metaclass protects them from being written over when accessed. There are also classmethods that modify the class
    itself. For instance, :func:`GFArrayMeta.display` changes the way in which elements of this class are displayed
    by `str()` and `repr()`.
    """

    # These class attributes will be set in the subclasses of GFArray
    _characteristic = None
    _degree = None
    _order = None
    _prim_poly = None
    _alpha = None
    _dtypes = []
    _ufunc_mode = None
    _ufunc_target = None
    _display_mode = "int"
    _display_poly_var = "α"

    def __str__(cls):
        return f"<class 'numpy.ndarray' over {cls.name}>"

    ###############################################################################
    # Class methods
    ###############################################################################

    @classmethod
    def target(cls, target, mode, rebuild=False):  # pylint: disable=unused-argument
        """
        Retarget the just-in-time compiled numba ufuncs.
        """
        return

    @classmethod
    def display(cls, mode="int", poly_var="α"):
        """
        Sets the display mode for all arrays of this type to either the integer representation or
        polynomial representation.

        This function updates :obj:`galois.GFArrayMeta.display_mode` and :obj:`galois.GFArrayMeta.display_poly_var`.

        Parameters
        ----------
        mode : str, optional
            The field element display mode, either `"int"` (default) or `"poly"`.
        poly_var : str, optional
            The polynomial representation's variable. The default is `"x"`.

        Examples
        --------
        Change the display mode by calling the :obj:`galois.GFArrayMeta.display` method.

        .. ipython:: python

            GF = galois.GF(2**3)
            a = GF.Random(4); a
            GF.display("poly"); a
            GF.display("poly", "r"); a

            # Reset the print mode
            GF.display(); a

        The :obj:`galois.GFArrayMeta.display` method can also be used as a context manager.

        .. ipython:: python

            # The original display mode
            print(a)

            # The new display context
            with GF.display("poly"):
                print(a)

            # Returns to the original display mode
            print(a)
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

            galois.GF(2).characteristic
            galois.GF(2**8).characteristic
            galois.GF(31).characteristic
            # galois.GF(7**5).characteristic
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
    def prim_poly(cls):
        """
        galois.Poly: The primitive polynomial :math:`p(x)` of the Galois field :math:`\\mathrm{GF}(p^m)`. The primitive
        polynomial is of degree :math:`m` in :math:`\\mathrm{GF}(p)[x]`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).prim_poly
            galois.GF(2**8).prim_poly
            galois.GF(31).prim_poly
            # galois.GF(7**5).prim_poly
        """
        # Ensure accesses of this property don't alter it
        return cls._prim_poly.copy()

    @property
    def alpha(cls):
        """
        int: The primitive element :math:`\\alpha` of the Galois field :math:`\\mathrm{GF}(p^m)`. The primitive element is a root of the
        primitive polynomial :math:`p(x)`, such that :math:`p(\\alpha) = 0`. The primitive element is also a multiplicative
        generator of the field, such that :math:`\\mathrm{GF}(p^m) = \\{0, 1, \\alpha^1, \\alpha^2, \\dots, \\alpha^{p^m - 2}\\}`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).alpha
            galois.GF(2**8).alpha
            galois.GF(31).alpha
            # galois.GF(7**5).alpha
        """
        # Ensure accesses of this property don't alter it
        return np.copy(cls._alpha)

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
        str: The representation of Galois field elements, either `"int"` or `"poly"`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).display_mode
            galois.GF(2**8).display_mode
            galois.GF(31).display_mode
            # galois.GF(7**5).display_mode
        """
        return cls._display_mode

    @property
    def display_poly_var(cls):
        """
        str: The polynomial indeterminate for the polynomial representation. The default is `"x"`.

        Examples
        --------
        .. ipython:: python

            galois.GF(2).display_poly_var
            galois.GF(2**8).display_poly_var
            galois.GF(31).display_poly_var
            # galois.GF(7**5).display_poly_var
        """
        return cls._display_poly_var

    @property
    def properties(cls):
        """
        str: A formmatted string displaying all the Galois field's attributes in one dict.

        Examples
        --------
        .. ipython:: python

            print(galois.GF(2**8).properties)
        """
        string = f"{cls.name}:"
        string += f"\n  characteristic: {cls._characteristic}"
        string += f"\n  degree: {cls._degree}"
        string += f"\n  order: {cls._order}"
        string += f"\n  prim_poly: {cls._prim_poly}"
        string += f"\n  alpha: {cls._alpha!r}"
        string += f"\n  dtypes: {[np.dtype(d).name for d in cls._dtypes]}"
        string += f"\n  ufunc_mode: '{cls._ufunc_mode}'"
        string += f"\n  ufunc_target: '{cls._ufunc_target}'"
        string += f"\n  display_mode: '{cls._display_mode}'"
        string += f"\n  display_poly_var: '{cls._display_poly_var}'"
        return string


class DisplayContext:
    """
    Simple context manager for the :obj:`GFArrayMeta.display` method.
    """

    def __init__(self, cls):
        # Save the previous state
        self.cls = cls
        self.mode = cls._display_mode
        self.poly_var = cls._display_poly_var

    def __enter__(self):
        # Don't need to do anything, we already set the new mode and poly_var in the display() method
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Reset mode and poly_var upon exiting the context
        self.cls._display_mode = self.mode
        self.cls._display_poly_var = self.poly_var
