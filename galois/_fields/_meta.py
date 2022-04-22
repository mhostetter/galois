from typing import List, Type
from typing_extensions import Literal

import numpy as np

from .._domains._meta import ArrayMeta
from .._modular import totatives
from .._polys import Poly
from .._polys._conversions import integer_to_poly, poly_to_str


class FieldArrayMeta(ArrayMeta):
    """
    A metaclass that provides documented class properties for `FieldArray` subclasses.
    """
    # pylint: disable=no-value-for-parameter

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._irreducible_poly_int: int = kwargs.get("irreducible_poly_int", 0)
        cls._is_primitive_poly: bool = kwargs.get("is_primitive_poly", None)
        cls._primitive_element: int = kwargs.get("primitive_element", 0)

        if cls._degree == 1:
            cls._prime_subfield = cls
            cls._name = f"GF({cls._characteristic})"
            cls._order_str = f"order={cls._order}"
        else:
            cls._prime_subfield = kwargs["prime_subfield"]  # Must be provided
            cls._name = f"GF({cls._characteristic}^{cls._degree})"
            cls._order_str = f"order={cls._characteristic}^{cls._degree}"

        # Construct the irreducible polynomial from its integer representation
        cls._irreducible_poly = Poly.Int(cls._irreducible_poly_int, field=cls._prime_subfield)

        if "compile" in kwargs:
            cls.compile(kwargs["compile"])

    def __str__(cls) -> str:
        if cls._prime_subfield is None:
            return repr(cls)

        with cls._prime_subfield.display("int"):
            irreducible_poly_str = str(cls._irreducible_poly)

        string = "Galois Field:"
        string += f"\n  name: {cls._name}"
        string += f"\n  characteristic: {cls._characteristic}"
        string += f"\n  degree: {cls._degree}"
        string += f"\n  order: {cls._order}"
        string += f"\n  irreducible_poly: {irreducible_poly_str}"
        string += f"\n  is_primitive_poly: {cls._is_primitive_poly}"
        string += f"\n  primitive_element: {poly_to_str(integer_to_poly(int(cls._primitive_element), cls._characteristic))}"

        return string

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
        return cls._name

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
        return cls._irreducible_poly

    @property
    def is_primitive_poly(cls) -> bool:
        r"""
        Indicates whether the :obj:`~galois.FieldArray.irreducible_poly` is a primitive polynomial. If so, :math:`x` is a
        primitive element of the finite field.

        Examples
        --------
        The default :math:`\mathrm{GF}(2^8)` field uses a primitive polynomial.

        .. ipython:: python

            GF = galois.GF(2**8)
            GF.is_primitive_poly

        The :math:`\mathrm{GF}(2^8)` field from AES uses a non-primitive polynomial.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
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
        return cls(cls._primitive_element)

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
        if not hasattr(cls, "_primitive_elements"):
            n = cls.order - 1
            powers = np.array(totatives(n))
            cls._primitive_elements = np.sort(cls.primitive_element ** powers)
        return cls._primitive_elements.copy()

    @property
    def quadratic_residues(cls) -> "FieldArray":
        r"""
        All quadratic residues in the finite field.

        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic residue* if there exists a :math:`y` such that
        :math:`y^2 = x` in the field.

        In fields with characteristic 2, every element is a quadratic residue. In fields with characteristic greater than 2,
        exactly half of the nonzero elements are quadratic residues (and they have two unique square roots).

        See Also
        --------
        is_quadratic_residue

        Examples
        --------
        .. tab-set::

            .. tab-item:: Characteristic 2

                .. ipython:: python

                    GF = galois.GF(2**4)
                    x = GF.quadratic_residues; x
                    r = np.sqrt(x); r
                    np.array_equal(r ** 2, x)
                    np.array_equal((-r) ** 2, x)

            .. tab-item:: Characteristic > 2

                .. ipython:: python

                    GF = galois.GF(11)
                    x = GF.quadratic_residues; x
                    r = np.sqrt(x); r
                    np.array_equal(r ** 2, x)
                    np.array_equal((-r) ** 2, x)
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

        See Also
        --------
        is_quadratic_residue

        Examples
        --------
        .. tab-set::

            .. tab-item:: Characteristic 2

                .. ipython:: python

                    GF = galois.GF(2**4)
                    GF.quadratic_non_residues

            .. tab-item:: Characteristic > 2

                .. ipython:: python

                    GF = galois.GF(11)
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
    def prime_subfield(cls) -> Type["FieldArray"]:
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

        See :doc:`/basic-usage/element-representation` for a further discussion.

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
        The current ufunc compilation mode for this :obj:`~galois.FieldArray` subclass. The ufuncs may be recompiled
        with :func:`~galois.FieldArray.compile`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Small fields

                Fields with order less than :math:`2^{20}` are compiled, by default, using lookup tables for speed.

                .. ipython:: python

                    galois.GF(65537).ufunc_mode
                    galois.GF(2**16).ufunc_mode

            .. tab-item:: Medium fields

                Fields with order greater than :math:`2^{20}` are compiled, by default, using explicit calculation for
                memory savings. The field elements and arithmetic must still fit within :obj:`numpy.int64`.

                .. ipython:: python

                    galois.GF(2147483647).ufunc_mode
                    galois.GF(2**32).ufunc_mode

            .. tab-item:: Large fields

                Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` use pure-Python explicit calculation.

                .. ipython:: python

                    galois.GF(36893488147419103183).ufunc_mode
                    galois.GF(2**100).ufunc_mode
        """
        return cls._ufunc_mode

    @property
    def ufunc_modes(cls) -> List[str]:
        """
        All supported ufunc compilation modes for this :obj:`~galois.FieldArray` subclass.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Compiled fields

                Fields whose elements and arithmetic can fit within :obj:`numpy.int64` can be JIT compiled
                to use either lookup tables or explicit calculation.

                .. ipython:: python

                    galois.GF(65537).ufunc_modes
                    galois.GF(2**32).ufunc_modes

            .. tab-item:: Non-compiled fields

                Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` may only use pure-Python explicit
                calculation.

                .. ipython:: python

                    galois.GF(36893488147419103183).ufunc_modes
                    galois.GF(2**100).ufunc_modes
        """
        return cls._ufunc_modes

    @property
    def default_ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The default ufunc compilation mode for this :obj:`~galois.FieldArray` subclass. The ufuncs may be recompiled
        with :func:`~galois.FieldArray.compile`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Small fields

                Fields with order less than :math:`2^{20}` are compiled, by default, using lookup tables for speed.

                .. ipython:: python

                    galois.GF(65537).default_ufunc_mode
                    galois.GF(2**16).default_ufunc_mode

            .. tab-item:: Medium fields

                Fields with order greater than :math:`2^{20}` are compiled, by default, using explicit calculation for
                memory savings. The field elements and arithmetic must still fit within :obj:`numpy.int64`.

                .. ipython:: python

                    galois.GF(2147483647).default_ufunc_mode
                    galois.GF(2**32).default_ufunc_mode

            .. tab-item:: Large fields

                Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` use pure-Python explicit calculation.

                .. ipython:: python

                    galois.GF(36893488147419103183).default_ufunc_mode
                    galois.GF(2**100).default_ufunc_mode
        """
        return cls._default_ufunc_mode
