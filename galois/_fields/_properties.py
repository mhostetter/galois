"""
A module that contains a metaclass mixin that provides Galois field class properties.
"""
import math

import numpy as np

from .._poly_conversion import integer_to_poly, poly_to_str

from ._dtypes import DTYPES


class PropertiesMeta(type):
    """
    A mixin metaclass that contains Galois field properties.
    """

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
    def degree(cls):
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
    def order(cls):
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
    def irreducible_poly(cls):
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
        # return Poly(cls._irreducible_poly, field=cls.prime_subfield)

    @property
    def is_primitive_poly(cls):
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
    def primitive_element(cls):
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
    def primitive_elements(cls):
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
    def dtypes(cls):
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
        d = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1]
        if len(d) == 0:
            d = [np.object_]
        return d

    @property
    def display_mode(cls):
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
    def ufunc_mode(cls):
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
    def properties(cls):
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
