"""
A module that defines the metaclass for the abstract base class FieldArray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np
from typing_extensions import Literal

from .._domains._array import ArrayMeta
from .._modular import totatives
from .._polys import Poly
from .._polys._conversions import integer_to_poly, poly_to_str

# Obtain forward references
if TYPE_CHECKING:
    from ._array import FieldArray


class FieldArrayMeta(ArrayMeta):
    """
    A metaclass that provides documented class properties for `FieldArray` subclasses.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        cls._is_primitive_poly: bool = kwargs.get("is_primitive_poly", None)

        if cls._degree == 1:
            cls._is_prime_field = True
            cls._prime_subfield = cls
            cls._name = f"GF({cls._characteristic})"
            cls._order_str = f"order={cls._order}"
        else:
            cls._is_prime_field = False
            cls._prime_subfield = kwargs["prime_subfield"]  # Must be provided
            cls._name = f"GF({cls._characteristic}^{cls._degree})"
            cls._order_str = f"order={cls._characteristic}^{cls._degree}"

        # Construct the irreducible polynomial from its integer representation
        cls._irreducible_poly = Poly.Int(cls._irreducible_poly_int, field=cls._prime_subfield)

    ###############################################################################
    # Class properties
    ###############################################################################

    @property
    def properties(cls) -> str:
        """
        A formatted string of relevant properties of the Galois field.

        Examples:
            .. ipython:: python

                GF = galois.GF(7**5)
                print(GF.properties)

        Group:
            String representation

        Order:
            30
        """
        with cls.prime_subfield.repr("int"):
            irreducible_poly_str = str(cls._irreducible_poly)
        primitive_element_str = poly_to_str(integer_to_poly(int(cls.primitive_element), cls.characteristic))

        string = "Galois Field:"
        string += f"\n  name: {cls.name}"
        string += f"\n  characteristic: {cls.characteristic}"
        string += f"\n  degree: {cls.degree}"
        string += f"\n  order: {cls.order}"
        string += f"\n  irreducible_poly: {irreducible_poly_str}"
        string += f"\n  is_primitive_poly: {cls.is_primitive_poly}"
        string += f"\n  primitive_element: {primitive_element_str}"

        return string

    @property
    def name(cls) -> str:
        """
        The finite field's name as a string `GF(p)` or `GF(p^m)`.

        Examples:
            .. ipython:: python

                galois.GF(2).name
                galois.GF(2**8).name
                galois.GF(31).name
                galois.GF(7**5).name
        """
        return super().name

    @property
    def characteristic(cls) -> int:
        r"""
        The prime characteristic $p$ of the Galois field $\mathrm{GF}(p^m)$.

        Notes:
            Adding $p$ copies of any element will always result in 0.

        Examples:
            .. ipython:: python

                galois.GF(2).characteristic
                galois.GF(2**8).characteristic
                galois.GF(31).characteristic
                galois.GF(7**5).characteristic
        """
        return super().characteristic

    @property
    def degree(cls) -> int:
        r"""
        The extension degree $m$ of the Galois field $\mathrm{GF}(p^m)$.

        Notes:
            The degree is a positive integer. For prime fields, the degree is 1.

        Examples:
            .. ipython:: python

                galois.GF(2).degree
                galois.GF(2**8).degree
                galois.GF(31).degree
                galois.GF(7**5).degree
        """
        return super().degree

    @property
    def order(cls) -> int:
        r"""
        The order $p^m$ of the Galois field $\mathrm{GF}(p^m)$.

        Notes:
            The order of the field is equal to the field's size.

        Examples:
            .. ipython:: python

                galois.GF(2).order
                galois.GF(2**8).order
                galois.GF(31).order
                galois.GF(7**5).order
        """
        return super().order

    @property
    def irreducible_poly(cls) -> Poly:
        r"""
        The irreducible polynomial $f(x)$ of the Galois field $\mathrm{GF}(p^m)$.

        Notes:
            The irreducible polynomial is of degree $m$ over $\mathrm{GF}(p)$.

        Examples:
            .. ipython:: python

                galois.GF(2).irreducible_poly
                galois.GF(2**8).irreducible_poly
                galois.GF(31).irreducible_poly
                galois.GF(7**5).irreducible_poly
        """
        return super().irreducible_poly

    @property
    def is_primitive_poly(cls) -> bool:
        r"""
        Indicates whether the :obj:`~galois.FieldArray.irreducible_poly` is a primitive polynomial.

        Notes:
            If the irreducible polynomial is primitive, then $x$ is a primitive element of the finite field.

        Examples:
            The default $\mathrm{GF}(2^8)$ field uses a primitive polynomial.

            .. ipython:: python

                GF = galois.GF(2**8)
                print(GF.properties)
                GF.is_primitive_poly

            The $\mathrm{GF}(2^8)$ field from AES uses a non-primitive polynomial.

            .. ipython:: python

                GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
                print(GF.properties)
                GF.is_primitive_poly
        """
        return cls._is_primitive_poly

    @property
    def elements(cls) -> FieldArray:
        r"""
        All of the finite field's elements $\{0, \dots, p^m-1\}$.

        Examples:
            All elements of the prime field $\mathrm{GF}(31)$ in increasing order.

            .. ipython-with-reprs:: int,power

                GF = galois.GF(31)
                GF.elements

            All elements of the extension field $\mathrm{GF}(5^2)$ in lexicographical order.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(5**2)
                GF.elements

        Group:
            Elements

        Order:
            22
        """
        return super().elements

    @property
    def units(cls) -> FieldArray:
        r"""
        All of the finite field's units $\{1, \dots, p^m-1\}$.

        Notes:
            A unit is an element with a multiplicative inverse.

        Examples:
            All units of the prime field $\mathrm{GF}(31)$ in increasing order.

            .. ipython-with-reprs:: int,power

                GF = galois.GF(31)
                GF.units

            All units of the extension field $\mathrm{GF}(5^2)$ in lexicographical order.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(5**2)
                GF.units

        Group:
            Elements

        Order:
            22
        """
        return super().units

    @property
    def primitive_element(cls) -> FieldArray:
        r"""
        A primitive element $\alpha$ of the Galois field $\mathrm{GF}(p^m)$.

        Notes:
            A primitive element is a multiplicative generator of the field, such that
            $\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m - 2}\}$. A primitive element is a
            root of the primitive polynomial $f(x)$, such that $f(\alpha) = 0$ over
            $\mathrm{GF}(p^m)$.

        Examples:
            The smallest primitive element of the prime field $\mathrm{GF}(31)$.

            .. ipython-with-reprs:: int,power

                GF = galois.GF(31)
                GF.primitive_element

            The smallest primitive element of the extension field $\mathrm{GF}(5^2)$.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(5**2)
                GF.primitive_element

        Group:
            Elements

        Order:
            22
        """
        return super().primitive_element

    @property
    def primitive_elements(cls) -> FieldArray:
        r"""
        All primitive elements $\alpha$ of the Galois field $\mathrm{GF}(p^m)$.

        Notes:
            A primitive element is a multiplicative generator of the field, such that
            $\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m - 2}\}$. A primitive element is a
            root of the primitive polynomial $f(x)$, such that $f(\alpha) = 0$ over
            $\mathrm{GF}(p^m)$.

        Examples:
            All primitive elements of the prime field $\mathrm{GF}(31)$ in increasing order.

            .. ipython-with-reprs:: int,power

                GF = galois.GF(31)
                GF.primitive_elements

            All primitive elements of the extension field $\mathrm{GF}(5^2)$ in lexicographical order.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(5**2)
                GF.primitive_elements

        Group:
            Elements

        Order:
            22
        """
        if not hasattr(cls, "_primitive_elements"):
            n = cls.order - 1
            powers = np.array(totatives(n))
            cls._primitive_elements = np.sort(cls.primitive_element**powers)
        return cls._primitive_elements.copy()

    @property
    def squares(cls) -> FieldArray:
        r"""
        All squares in the finite field.

        Notes:
            An element $x$ in $\mathrm{GF}(p^m)$ is a *square* if there exists a $y$ such that
            $y^2 = x$ in the field.

        See Also:
            is_square

        Examples:
            In fields with characteristic 2, every element is a square (with two identical square roots).

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(2**3)
                x = GF.squares; x
                y1 = np.sqrt(x); y1
                y2 = -y1; y2
                np.array_equal(y1 ** 2, x)
                np.array_equal(y2 ** 2, x)

            In fields with characteristic greater than 2, exactly half of the nonzero elements are squares
            (with two unique square roots).

            .. ipython-with-reprs:: int,power

                GF = galois.GF(11)
                x = GF.squares; x
                y1 = np.sqrt(x); y1
                y2 = -y1; y2
                np.array_equal(y1 ** 2, x)
                np.array_equal(y2 ** 2, x)

        Group:
            Elements

        Order:
            22
        """
        x = cls.elements
        is_square = x.is_square()
        return x[is_square]

    @property
    def non_squares(cls) -> FieldArray:
        r"""
        All non-squares in the Galois field.

        Notes:
            An element $x$ in $\mathrm{GF}(p^m)$ is a *non-square* if there does not exist a $y$
            such that $y^2 = x$ in the field.

        See Also:
            is_square

        Examples:
            In fields with characteristic 2, no elements are non-squares.

            .. ipython-with-reprs:: int,poly,power

                GF = galois.GF(2**3)
                GF.non_squares

            In fields with characteristic greater than 2, exactly half of the nonzero elements are non-squares.

            .. ipython-with-reprs:: int,power

                GF = galois.GF(11)
                GF.non_squares

        Group:
            Elements

        Order:
            22
        """
        x = cls.elements
        is_square = x.is_square()
        return x[~is_square]

    @property
    def is_prime_field(cls) -> bool:
        """
        Indicates if the finite field is a prime field, having prime order.

        Examples:
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
        Indicates if the finite field is an extension field, having prime power order.

        Examples:
            .. ipython:: python

                galois.GF(2).is_extension_field
                galois.GF(2**8).is_extension_field
                galois.GF(31).is_extension_field
                galois.GF(7**5).is_extension_field
        """
        return cls._degree > 1

    @property
    def prime_subfield(cls) -> Type[FieldArray]:
        r"""
        The prime subfield $\mathrm{GF}(p)$ of the extension field $\mathrm{GF}(p^m)$.

        Notes:
            For the prime field $\mathrm{GF}(p)$, the prime subfield is itself.

        Examples:
            .. ipython:: python

                galois.GF(2).prime_subfield
                galois.GF(2**8).prime_subfield
                galois.GF(31).prime_subfield
                galois.GF(7**5).prime_subfield
        """
        return cls._prime_subfield

    @property
    def dtypes(cls) -> list[np.dtype]:
        r"""
        List of valid integer :obj:`numpy.dtype` values that are compatible with this finite field.

        Notes:
            Creating an array with an unsupported dtype will raise a `TypeError` exception.

            For finite fields whose elements cannot be represented with :obj:`numpy.int64`, the only valid data type
            is :obj:`numpy.object_`.

        Examples:
            For small finite fields, all integer data types are acceptable, with the exception of :obj:`numpy.uint64`.
            This is because all arithmetic is done using :obj:`numpy.int64`.

            .. ipython:: python

                GF = galois.GF(31); GF.dtypes

            Some data types are too small for certain finite fields, such as :obj:`numpy.int16` for
            $\mathrm{GF}(7^5)$.

            .. ipython:: python

                GF = galois.GF(7**5); GF.dtypes

            Large fields must use :obj:`numpy.object_` which uses Python :obj:`int` for its unlimited size.

            .. ipython:: python

                GF = galois.GF(2**100); GF.dtypes
                GF = galois.GF(36893488147419103183); GF.dtypes

        Group:
            Arithmetic compilation

        Order:
            32
        """
        return super().dtypes

    @property
    def element_repr(cls) -> Literal["int", "poly", "power"]:
        r"""
        The current finite field element representation.

        Notes:
            This can be changed with :func:`~galois.FieldArray.repr`. See :doc:`/basic-usage/element-representation`
            for a further discussion.

        Examples:
            The default element representation is the integer representation.

            .. ipython:: python

                GF = galois.GF(3**2)
                x = GF.elements; x
                GF.element_repr

            Permanently modify the element representation by calling :func:`~galois.FieldArray.repr`.

            .. ipython:: python

                GF.repr("poly");
                x
                GF.element_repr
                @suppress
                GF.repr()

        Group:
            Element representation

        Order:
            31
        """
        return super().element_repr

    @property
    def ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The current ufunc compilation mode for this :obj:`~galois.FieldArray` subclass.

        Notes:
            The ufuncs may be recompiled with :func:`~galois.FieldArray.compile`.

        Examples:
            Fields with order less than $2^{20}$ are compiled, by default, using lookup tables for speed.

            .. ipython:: python

                galois.GF(65537).ufunc_mode
                galois.GF(2**16).ufunc_mode

            Fields with order greater than $2^{20}$ are compiled, by default, using explicit calculation for
            memory savings. The field elements and arithmetic must still fit within :obj:`numpy.int64`.

            .. ipython:: python

                galois.GF(2147483647).ufunc_mode
                galois.GF(2**32).ufunc_mode

            Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` use pure-Python explicit
            calculation.

            .. ipython:: python

                galois.GF(36893488147419103183).ufunc_mode
                galois.GF(2**100).ufunc_mode

        Group:
            Arithmetic compilation

        Order:
            32
        """
        return super().ufunc_mode

    @property
    def ufunc_modes(cls) -> list[str]:
        """
        All supported ufunc compilation modes for this :obj:`~galois.FieldArray` subclass.

        Notes:
            The ufuncs may be recompiled with :func:`~galois.FieldArray.compile`.

        Examples:
            Fields whose elements and arithmetic can fit within :obj:`numpy.int64` can be JIT compiled
            to use either lookup tables or explicit calculation.

            .. ipython:: python

                galois.GF(65537).ufunc_modes
                galois.GF(2**32).ufunc_modes

            Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` may only use pure-Python explicit
            calculation.

            .. ipython:: python

                galois.GF(36893488147419103183).ufunc_modes
                galois.GF(2**100).ufunc_modes

        Group:
            Arithmetic compilation

        Order:
            32
        """
        return super().ufunc_modes

    @property
    def default_ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The default ufunc compilation mode for this :obj:`~galois.FieldArray` subclass.

        Notes:
            The ufuncs may be recompiled with :func:`~galois.FieldArray.compile`.

        Examples:
            Fields with order less than $2^{20}$ are compiled, by default, using lookup tables for speed.

            .. ipython:: python

                galois.GF(65537).default_ufunc_mode
                galois.GF(2**16).default_ufunc_mode

            Fields with order greater than $2^{20}$ are compiled, by default, using explicit calculation for
            memory savings. The field elements and arithmetic must still fit within :obj:`numpy.int64`.

            .. ipython:: python

                galois.GF(2147483647).default_ufunc_mode
                galois.GF(2**32).default_ufunc_mode

            Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` use pure-Python explicit
            calculation.

            .. ipython:: python

                galois.GF(36893488147419103183).default_ufunc_mode
                galois.GF(2**100).default_ufunc_mode

        Group:
            Arithmetic compilation

        Order:
            32
        """
        return super().default_ufunc_mode
