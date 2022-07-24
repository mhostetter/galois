"""
A module that defines the abstract base class FieldArray.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Union, Type
from typing_extensions import Literal

import numpy as np

from .._domains import Array, _linalg
from .._domains._array import ArrayMeta
from .._modular import totatives
from .._overrides import set_module, extend_docstring, SPHINX_BUILD
from .._polys import Poly
from .._polys._conversions import integer_to_poly, str_to_integer, poly_to_str
from .._prime import divisors
from ..typing import ElementLike, IterableLike, ArrayLike, ShapeLike, DTypeLike

__all__ = ["FieldArray"]


class FieldArrayMeta(ArrayMeta):
    """
    A metaclass that provides documented class properties for `FieldArray` subclasses.
    """
    # pylint: disable=no-value-for-parameter,too-many-public-methods

    def __new__(cls, name, bases, namespace, **kwargs):  # pylint: disable=unused-argument
        return super().__new__(cls, name, bases, namespace)

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

        :group: String representation
        :order: 21

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7**5)
            print(GF.properties)
        """
        with cls.prime_subfield.display("int"):
            irreducible_poly_str = str(cls._irreducible_poly)

        string = "Galois Field:"
        string += f"\n  name: {cls.name}"
        string += f"\n  characteristic: {cls.characteristic}"
        string += f"\n  degree: {cls.degree}"
        string += f"\n  order: {cls.order}"
        string += f"\n  irreducible_poly: {irreducible_poly_str}"
        string += f"\n  is_primitive_poly: {cls.is_primitive_poly}"
        string += f"\n  primitive_element: {poly_to_str(integer_to_poly(int(cls.primitive_element), cls.characteristic))}"

        return string

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
        return super().name

    @property
    def characteristic(cls) -> int:
        r"""
        The prime characteristic :math:`p` of the Galois field :math:`\mathrm{GF}(p^m)`. Adding
        :math:`p` copies of any element will always result in 0.

        Examples
        --------
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
        The extension degree :math:`m` of the Galois field :math:`\mathrm{GF}(p^m)`. The degree is a positive integer.

        Examples
        --------
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
        The order :math:`p^m` of the Galois field :math:`\mathrm{GF}(p^m)`. The order of the field is equal to the field's size.

        Examples
        --------
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
        return super().irreducible_poly

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
            print(GF.properties)
            GF.is_primitive_poly

        The :math:`\mathrm{GF}(2^8)` field from AES uses a non-primitive polynomial.

        .. ipython:: python

            GF = galois.GF(2**8, irreducible_poly="x^8 + x^4 + x^3 + x + 1")
            print(GF.properties)
            GF.is_primitive_poly
        """
        return cls._is_primitive_poly

    @property
    def elements(cls) -> FieldArray:
        r"""
        All of the finite field's elements :math:`\{0, \dots, p^m-1\}`.

        Examples
        --------
        All elements of the prime field :math:`\mathrm{GF}(31)` in increasing order.

        .. ipython:: python

            GF = galois.GF(31)
            GF.elements

        All elements of the extension field :math:`\mathrm{GF}(5^2)` in lexicographically-increasing order.

        .. ipython:: python

            GF = galois.GF(5**2, display="poly")
            GF.elements
            @suppress
            GF.display()
        """
        return super().elements

    @property
    def units(cls) -> FieldArray:
        r"""
        All of the finite field's units :math:`\{1, \dots, p^m-1\}`. A unit is an element with a multiplicative inverse.

        Examples
        --------
        All units of the prime field :math:`\mathrm{GF}(31)` in increasing order.

        .. ipython:: python

            GF = galois.GF(31)
            GF.units

        All units of the extension field :math:`\mathrm{GF}(5^2)` in lexicographically-increasing order.

        .. ipython:: python

            GF = galois.GF(5**2, display="poly")
            GF.units
            @suppress
            GF.display()
        """
        return super().units

    @property
    def primitive_element(cls) -> FieldArray:
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
        return super().primitive_element

    @property
    def primitive_elements(cls) -> FieldArray:
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
    def quadratic_residues(cls) -> FieldArray:
        r"""
        All quadratic residues in the finite field.

        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic residue* if there exists a :math:`y` such that
        :math:`y^2 = x` in the field.

        See Also
        --------
        is_quadratic_residue

        Examples
        --------
        In fields with characteristic 2, every element is a quadratic residue.

        .. ipython:: python

            GF = galois.GF(2**4)
            x = GF.quadratic_residues; x
            r = np.sqrt(x); r
            np.array_equal(r ** 2, x)
            np.array_equal((-r) ** 2, x)

        In fields with characteristic greater than 2,exactly half of the nonzero elements are quadratic residues
        (and they have two unique square roots).

        .. ipython:: python

            GF = galois.GF(11)
            x = GF.quadratic_residues; x
            r = np.sqrt(x); r
            np.array_equal(r ** 2, x)
            np.array_equal((-r) ** 2, x)
        """
        x = cls.elements
        is_quadratic_residue = x.is_quadratic_residue()
        return x[is_quadratic_residue]

    @property
    def quadratic_non_residues(cls) -> FieldArray:
        r"""
        All quadratic non-residues in the Galois field.

        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic non-residue* if there does not exist a :math:`y` such that
        :math:`y^2 = x` in the field.

        See Also
        --------
        is_quadratic_residue

        Examples
        --------
        In fields with characteristic 2, no elements are quadratic non-residues.

        .. ipython:: python

            GF = galois.GF(2**4)
            GF.quadratic_non_residues

        In fields with characteristic greater than 2, exactly half of the nonzero elements are quadratic non-residues.

        .. ipython:: python

            GF = galois.GF(11)
            GF.quadratic_non_residues
        """
        x = cls.elements
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
    def prime_subfield(cls) -> Type[FieldArray]:
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
        For small finite fields, all integer data types are acceptable, with the exception of :obj:`numpy.uint64`. This is
        because all arithmetic is done using :obj:`numpy.int64`.

        .. ipython:: python

            GF = galois.GF(31); GF.dtypes

        Some data types are too small for certain finite fields, such as :obj:`numpy.int16` for :math:`\mathrm{GF}(7^5)`.

        .. ipython:: python

            GF = galois.GF(7**5); GF.dtypes

        Large fields must use :obj:`numpy.object_` which uses Python :obj:`int` for its unlimited size.

        .. ipython:: python

            GF = galois.GF(2**100); GF.dtypes
            GF = galois.GF(36893488147419103183); GF.dtypes
        """
        return super().dtypes

    @property
    def display_mode(cls) -> Literal["int", "poly", "power"]:
        r"""
        The current finite field element representation. This can be changed with :func:`~galois.FieldArray.display`.

        See :doc:`/basic-usage/element-representation` for a further discussion.

        Examples
        --------
        The default display mode is the integer representation.

        .. ipython:: python

            GF = galois.GF(3**2)
            x = GF.elements; x
            GF.display_mode

        Permanently modify the display mode by calling :func:`~galois.FieldArray.display`.

        .. ipython:: python

            GF.display("poly");
            x
            GF.display_mode
            @suppress
            GF.display()
        """
        return super().display_mode

    @property
    def ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The current ufunc compilation mode for this :obj:`~galois.FieldArray` subclass. The ufuncs may be recompiled
        with :func:`~galois.FieldArray.compile`.

        Examples
        --------
        Fields with order less than :math:`2^{20}` are compiled, by default, using lookup tables for speed.

        .. ipython:: python

            galois.GF(65537).ufunc_mode
            galois.GF(2**16).ufunc_mode

        Fields with order greater than :math:`2^{20}` are compiled, by default, using explicit calculation for
        memory savings. The field elements and arithmetic must still fit within :obj:`numpy.int64`.

        .. ipython:: python

            galois.GF(2147483647).ufunc_mode
            galois.GF(2**32).ufunc_mode

        Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` use pure-Python explicit calculation.

        .. ipython:: python

            galois.GF(36893488147419103183).ufunc_mode
            galois.GF(2**100).ufunc_mode
        """
        return super().ufunc_mode

    @property
    def ufunc_modes(cls) -> List[str]:
        """
        All supported ufunc compilation modes for this :obj:`~galois.FieldArray` subclass.

        Examples
        --------
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
        """
        return super().ufunc_modes

    @property
    def default_ufunc_mode(cls) -> Literal["jit-lookup", "jit-calculate", "python-calculate"]:
        """
        The default ufunc compilation mode for this :obj:`~galois.FieldArray` subclass. The ufuncs may be recompiled
        with :func:`~galois.FieldArray.compile`.

        Examples
        --------
        Fields with order less than :math:`2^{20}` are compiled, by default, using lookup tables for speed.

        .. ipython:: python

            galois.GF(65537).default_ufunc_mode
            galois.GF(2**16).default_ufunc_mode

        Fields with order greater than :math:`2^{20}` are compiled, by default, using explicit calculation for
        memory savings. The field elements and arithmetic must still fit within :obj:`numpy.int64`.

        .. ipython:: python

            galois.GF(2147483647).default_ufunc_mode
            galois.GF(2**32).default_ufunc_mode

        Fields whose elements and arithmetic cannot fit within :obj:`numpy.int64` use pure-Python explicit calculation.

        .. ipython:: python

            galois.GF(36893488147419103183).default_ufunc_mode
            galois.GF(2**100).default_ufunc_mode
        """
        return super().default_ufunc_mode


@set_module("galois")
class FieldArray(Array, metaclass=FieldArrayMeta):
    r"""
    An abstract :obj:`~numpy.ndarray` subclass over :math:`\mathrm{GF}(p^m)`.

    Important
    ---------
    :obj:`~galois.FieldArray` is an abstract base class and cannot be instantiated directly. Instead, :obj:`~galois.FieldArray`
    subclasses are created using the class factory :func:`~galois.GF`.

    Examples
    --------
    Create a :obj:`~galois.FieldArray` subclass over :math:`\mathrm{GF}(3^5)` using the class factory :func:`~galois.GF`.

    .. ipython:: python

        GF = galois.GF(3**5)
        issubclass(GF, galois.FieldArray)
        print(GF.properties)

    Create a :obj:`~galois.FieldArray` instance using `GF`'s constructor.

    .. ipython:: python

        x = GF([44, 236, 206, 138]); x
        isinstance(x, GF)

    :group: galois-fields
    """
    # pylint: disable=no-value-for-parameter,abstract-method,too-many-public-methods

    def __new__(
        cls,
        x: Union[ElementLike, ArrayLike],
        dtype: Optional[DTypeLike] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ) -> FieldArray:
        if not SPHINX_BUILD and cls is FieldArray:
            raise NotImplementedError("FieldArray is an abstract base class that cannot be directly instantiated. Instead, create a FieldArray subclass for GF(p^m) arithmetic using `GF = galois.GF(p**m)` and instantiate an array using `x = GF(array_like)`.")
        return super().__new__(cls, x, dtype, copy, order, ndmin)

    def __init__(
        self,
        x: Union[ElementLike, ArrayLike],
        dtype: Optional[DTypeLike] = None,
        copy: bool = True,
        order: Literal["K", "A", "C", "F"] = "K",
        ndmin: int = 0
    ):
        r"""
        Creates an array over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        x
            A finite field scalar or array.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.FieldArray` subclass (the first element in :obj:`~galois.FieldArray.dtypes`).
        copy
            The `copy` keyword argument from :func:`numpy.array`. The default is `True`.
        order
            The `order` keyword argument from :func:`numpy.array`. The default is `"K"`.
        ndmin
            The `ndmin` keyword argument from :func:`numpy.array`. The default is 0.
        """
        # pylint: disable=unused-argument,super-init-not-called
        # Adding __init__ and not doing anything is done to overwrite the superclass's __init__ docstring
        return

    ###############################################################################
    # Verification routines
    ###############################################################################

    @classmethod
    def _verify_array_like_types_and_values(cls, x: Union[ElementLike, ArrayLike]) -> Union[ElementLike, ArrayLike]:
        if isinstance(x, (int, np.integer)):
            cls._verify_scalar_value(x)
        elif isinstance(x, cls):
            # This was a previously-created and vetted array -- there's no need to re-verify
            if x.ndim == 0:
                # Ensure that in "large" fields with dtype=object that FieldArray objects aren't assigned to the array. The arithmetic
                # functions are designed to operate on Python ints.
                x = int(x)
        elif isinstance(x, str):
            x = cls._convert_to_element(x)
            cls._verify_scalar_value(x)
        elif isinstance(x, (list, tuple)):
            x = cls._convert_iterable_to_elements(x)
            cls._verify_array_values(x)
        elif isinstance(x, np.ndarray):
            # If this a NumPy array, but not a FieldArray, verify the array
            if x.dtype == np.object_:
                x = cls._verify_element_types_and_convert(x, object_=True)
            elif not np.issubdtype(x.dtype, np.integer):
                raise TypeError(f"{cls.name} arrays must have integer dtypes, not {x.dtype}.")
            cls._verify_array_values(x)
        else:
            raise TypeError(f"{cls.name} arrays can be created with scalars of type int/str, lists/tuples, or ndarrays, not {type(x)}.")

        return x

    @classmethod
    def _verify_element_types_and_convert(cls, array: np.ndarray, object_=False) -> np.ndarray:
        if array.size == 0:
            return array
        elif object_:
            return np.vectorize(cls._convert_to_element, otypes=[object])(array)
        else:
            return np.vectorize(cls._convert_to_element)(array)

    @classmethod
    def _verify_scalar_value(cls, scalar: int):
        if not 0 <= scalar < cls.order:
            raise ValueError(f"{cls.name} scalars must be in `0 <= x < {cls.order}`, not {scalar}.")

    @classmethod
    def _verify_array_values(cls, array: np.ndarray):
        if np.any(array < 0) or np.any(array >= cls.order):
            idxs = np.logical_or(array < 0, array >= cls.order)
            values = array if array.ndim == 0 else array[idxs]
            raise ValueError(f"{cls.name} arrays must have elements in `0 <= x < {cls.order}`, not {values}.")

    ###############################################################################
    # Element conversion routines
    ###############################################################################

    @classmethod
    def _convert_to_element(cls, element: ElementLike) -> int:
        if isinstance(element, (int, np.integer)):
            element = int(element)
        elif isinstance(element, str):
            element = str_to_integer(element, cls.prime_subfield)
        elif isinstance(element, FieldArray):
            element = int(element)
        else:
            raise TypeError(f"Valid element-like values are integers and string, not {type(element)}.")

        return element

    @classmethod
    def _convert_iterable_to_elements(cls, iterable: IterableLike) -> np.ndarray:
        if cls.dtypes == [np.object_]:
            array = np.array(iterable, dtype=object)
            array = cls._verify_element_types_and_convert(array, object_=True)
        else:
            # Try to convert to an integer array in one shot
            array = np.array(iterable)

            if not np.issubdtype(array.dtype, np.integer):
                # There are strings in the array, try again and manually convert each element
                array = np.array(iterable, dtype=object)
                array = cls._verify_element_types_and_convert(array)

        return array

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    @extend_docstring(Array.Zeros, {"Array": "FieldArray"})
    def Zeros(cls, shape: ShapeLike, dtype: Optional[DTypeLike] = None) -> FieldArray:
        """
        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Zeros((2, 5))
        """
        return super().Zeros(shape, dtype=dtype)

    @classmethod
    @extend_docstring(Array.Ones, {"Array": "FieldArray"})
    def Ones(cls, shape: ShapeLike, dtype: Optional[DTypeLike] = None) -> FieldArray:
        """
        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Ones((2, 5))
        """
        return super().Ones(shape, dtype=dtype)

    @classmethod
    @extend_docstring(Array.Range, {"Array": "FieldArray"})
    def Range(
        cls,
        start: ElementLike,
        stop: ElementLike,
        step: int = 1,
        dtype: Optional[DTypeLike] = None
    ) -> FieldArray:
        """
        Examples
        --------
        For prime fields, the increment is simply a finite field element, since all elements are integers.

        .. ipython:: python

            GF = galois.GF(31)
            GF.Range(10, 20)
            GF.Range(10, 20, 2)

        For extension fields, the increment is the integer increment between finite field elements in their :ref:`integer representation <int-repr>`.

        .. ipython:: python

            GF = galois.GF(3**3, display="poly")
            GF.Range(10, 20)
            GF.Range(10, 20, 2)
            @suppress
            GF.display()
        """
        return super().Range(start, stop, step=step, dtype=dtype)

    @classmethod
    @extend_docstring(Array.Random, {"Array": "FieldArray"})
    def Random(
        cls,
        shape: ShapeLike = (),
        low: ElementLike = 0,
        high: Optional[ElementLike] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: Optional[DTypeLike] = None
    ) -> FieldArray:
        """
        Examples
        --------
        Generate a random matrix with an unpredictable seed.

        .. ipython:: python

            GF = galois.GF(31)
            GF.Random((2, 5))

        Generate a random array with a specified seed. This produces repeatable outputs.

        .. ipython:: python

            GF.Random(10, seed=123456789)
            GF.Random(10, seed=123456789)

        Generate a group of random arrays using a single global seed.

        .. ipython:: python

            rng = np.random.default_rng(123456789)
            GF.Random(10, seed=rng)
            GF.Random(10, seed=rng)
        """
        return super().Random(shape=shape, low=low, high=high, seed=seed, dtype=dtype)

    @classmethod
    @extend_docstring(Array.Identity, {"Array": "FieldArray"})
    def Identity(cls, size: int, dtype: Optional[DTypeLike] = None) -> FieldArray:
        """
        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            GF.Identity(4)
        """
        return super().Identity(size, dtype=dtype)

    @classmethod
    def Vandermonde(cls, element: ElementLike, rows: int, cols: int, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Creates an :math:`m \times n` Vandermonde matrix of :math:`a \in \mathrm{GF}(q)`.

        Parameters
        ----------
        element
            An element :math:`a` of :math:`\mathrm{GF}(q)`.
        rows
            The number of rows :math:`m` in the Vandermonde matrix.
        cols
            The number of columns :math:`n` in the Vandermonde matrix.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.FieldArray` subclass (the first element in :obj:`~galois.FieldArray.dtypes`).

        Returns
        -------
        :
            A :math:`m \times n` Vandermonde matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**3, display="power")
            a = GF.primitive_element; a
            V = GF.Vandermonde(a, 7, 7); V
            @suppress
            GF.display()
        """
        if not isinstance(element, (int, np.integer, cls)):
            raise TypeError(f"Argument `element` must be an integer or element of {cls.name}, not {type(element)}.")
        if not isinstance(rows, (int, np.integer)):
            raise TypeError(f"Argument `rows` must be an integer, not {type(rows)}.")
        if not isinstance(cols, (int, np.integer)):
            raise TypeError(f"Argument `cols` must be an integer, not {type(cols)}.")
        if not rows > 0:
            raise ValueError(f"Argument `rows` must be non-negative, not {rows}.")
        if not cols > 0:
            raise ValueError(f"Argument `cols` must be non-negative, not {cols}.")

        dtype = cls._get_dtype(dtype)
        element = cls(element, dtype=dtype)
        if not element.ndim == 0:
            raise ValueError(f"Argument `element` must be element scalar, not {element.ndim}-D.")

        v = element ** np.arange(0, rows)
        V = np.power.outer(v, np.arange(0, cols))

        return V

    @classmethod
    def Vector(cls, array: ArrayLike, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Creates an array over :math:`\mathrm{GF}(p^m)` from length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`vector` method.

        Parameters
        ----------
        array
            An array over :math:`\mathrm{GF}(p)` with last dimension :math:`m`. An array with shape `(n1, n2, m)` has output shape
            `(n1, n2)`. By convention, the vectors are ordered from degree :math:`m-1` to degree 0.
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.FieldArray` subclass (the first element in :obj:`~galois.FieldArray.dtypes`).

        Returns
        -------
        :
            An array over :math:`\mathrm{GF}(p^m)`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**3, display="poly")
            a = GF.Vector([[1, 0, 2], [0, 2, 1]]); a
            a.vector()
            @suppress
            GF.display()
        """
        dtype = cls._get_dtype(dtype)
        order = cls.prime_subfield.order
        degree = cls.degree

        x = cls.prime_subfield(array)  # Convert element-like objects into the prime subfield
        x = x.view(np.ndarray)  # Convert into an integer array
        if not x.shape[-1] == degree:
            raise ValueError(f"The last dimension of `array` must be the field extension dimension {cls.degree}, not {x.shape[-1]}.")

        degrees = np.arange(degree - 1, -1, -1, dtype=dtype)
        y = np.sum(x * order**degrees, axis=-1, dtype=dtype)

        if np.isscalar(y):
            y = cls(y, dtype=dtype)
        else:
            y = cls._view(y)

        return y

    ###############################################################################
    # Class methods
    ###############################################################################

    @classmethod
    def repr_table(cls, element: Optional[ElementLike] = None, sort: Literal["power", "poly", "vector", "int"] = "power") -> str:
        r"""
        Generates a finite field element representation table comparing the power, polynomial, vector, and integer representations.

        Parameters
        ----------
        element
            An element to use as the exponent base in the power representation. The default is `None` which corresponds to
            :obj:`~galois.FieldArray.primitive_element`.
        sort
            The sorting method for the table. The default is `"power"`. Sorting by `"power"` will order the rows of the table by ascending
            powers of `element`. Sorting by any of the others will order the rows in lexicographically-increasing polynomial/vector
            order, which is equivalent to ascending order of the integer representation.

        Returns
        -------
        :
            A string representation of the table comparing the power, polynomial, vector, and integer representations of each
            field element.

        Examples
        --------
        Create a :obj:`~galois.FieldArray` subclass for :math:`\mathrm{GF}(3^3)`.

        .. ipython:: python

            GF = galois.GF(3**3)
            print(GF.properties)

        Generate a representation table for :math:`\mathrm{GF}(3^3)`. Since :math:`x^3 + 2x + 1` is a primitive polynomial,
        :math:`x` is a primitive element of the field. Notice, :math:`\textrm{ord}(x) = 26`.

        .. ipython:: python

            print(GF.repr_table())
            GF("x").multiplicative_order()

        Generate a representation table for :math:`\mathrm{GF}(3^3)` using a different primitive element :math:`2x^2 + 2x + 2`.
        Notice, :math:`\textrm{ord}(2x^2 + 2x + 2) = 26`.

        .. ipython:: python

            print(GF.repr_table("2x^2 + 2x + 2"))
            GF("2x^2 + 2x + 2").multiplicative_order()

        Generate a representation table for :math:`\mathrm{GF}(3^3)` using a non-primitive element :math:`x^2`. Notice,
        :math:`\textrm{ord}(x^2) = 13 \ne 26`.

        .. ipython:: python

            print(GF.repr_table("x^2"))
            GF("x^2").multiplicative_order()
        """
        if sort not in ["power", "poly", "vector", "int"]:
            raise ValueError(f"Argument `sort` must be in ['power', 'poly', 'vector', 'int'], not {sort!r}.")
        if element is None:
            element = cls.primitive_element

        element = cls(element)
        degrees = np.arange(0, cls.order - 1)
        x = element**degrees
        if sort != "power":
            idxs = np.argsort(x)
            degrees, x = degrees[idxs], x[idxs]
        x = np.concatenate((np.atleast_1d(cls(0)), x))  # Add 0 = alpha**-Inf
        prim = poly_to_str(integer_to_poly(int(element), cls.characteristic))

        # Define print helper functions
        if len(prim) > 1:
            print_power = lambda power: "0" if power is None else f"({prim})^{power}"
        else:
            print_power = lambda power: "0" if power is None else f"{prim}^{power}"
        print_poly = lambda x: poly_to_str(integer_to_poly(int(x), cls.characteristic))
        print_vec = lambda x: str(integer_to_poly(int(x), cls.characteristic, degree=cls.degree-1))
        print_int = lambda x: str(int(x))

        # Determine column widths
        N_power = max([len(print_power(max(degrees))), len("Power")]) + 2
        N_poly = max([len(print_poly(e)) for e in x] + [len("Polynomial")]) + 2
        N_vec = max([len(print_vec(e)) for e in x] + [len("Vector")]) + 2
        N_int = max([len(print_int(e)) for e in x] + [len("Integer")]) + 2

        string = "Power".center(N_power) + " " + "Polynomial".center(N_poly) + " " + "Vector".center(N_vec) + " " + "Integer".center(N_int)
        string += "\n" + "-"*N_power + " " + "-"*N_poly + " " + "-"*N_vec + " " + "-"*N_int

        for i in range(x.size):
            d = None if i == 0 else degrees[i - 1]
            string += "\n" + print_power(d).center(N_power) + " " + poly_to_str(integer_to_poly(int(x[i]), cls.characteristic)).center(N_poly) + " " + str(integer_to_poly(int(x[i]), cls.characteristic, degree=cls.degree-1)).center(N_vec) + " " + cls._print_int(x[i]).center(N_int) + " "

        return string

    @classmethod
    def arithmetic_table(
        cls,
        operation: Literal["+", "-", "*", "/"],
        x: Optional[FieldArray] = None,
        y: Optional[FieldArray] = None
    ) -> str:
        r"""
        Generates the specified arithmetic table for the finite field.

        Parameters
        ----------
        operation
            The arithmetic operation.
        x
            Optionally specify the :math:`x` values for the arithmetic table. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}`.
        y
            Optionally specify the :math:`y` values for the arithmetic table. The default is `None`
            which represents :math:`\{0, \dots, p^m - 1\}` for addition, subtraction, and multiplication and
            :math:`\{1, \dots, p^m - 1\}` for division.

        Returns
        -------
        :
            A string representation of the arithmetic table.

        Examples
        --------
        Arithmetic tables can be displayed using any element representation.

        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    print(GF.arithmetic_table("+"))

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    print(GF.arithmetic_table("+"))

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    print(GF.arithmetic_table("+"))
                    @suppress
                    GF.display()

        An arithmetic table may also be constructed from arbitrary :math:`x` and :math:`y`.

        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    x = GF([7, 2, 8]); x
                    y = GF([1, 4, 5, 3]); y
                    print(GF.arithmetic_table("+", x=x, y=y))

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    x = GF([7, 2, 8]); x
                    y = GF([1, 4, 5, 3]); y
                    print(GF.arithmetic_table("+", x=x, y=y))

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    x = GF([7, 2, 8]); x
                    y = GF([1, 4, 5, 3]); y
                    print(GF.arithmetic_table("+", x=x, y=y))
                    @suppress
                    GF.display()
        """
        if not operation in ["+", "-", "*", "/"]:
            raise ValueError(f"Argument `operation` must be in ['+', '-', '*', '/'], not {operation!r}.")

        if cls.display_mode == "power":
            # Order elements by powers of the primitive element
            x_default = np.concatenate((np.atleast_1d(cls(0)), cls.primitive_element**np.arange(0, cls.order - 1, dtype=cls.dtypes[-1])))
        else:
            x_default = cls.elements
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
            print_element = cls._print_power

        operation_str = f"x {operation} y"

        N = max(len(print_element(e)) for e in x) + 1
        N_left = max(N, len(operation_str) + 1)

        string = operation_str.rjust(N_left - 1) + " |"
        for j in range(y.size):
            string += print_element(y[j]).rjust(N) + " "
        string += "\n" + "-"*N_left + "|" + "-"*(N + 1)*y.size

        for i in range(x.size):
            string += "\n" + print_element(x[i]).rjust(N_left - 1) + " |"
            for j in range(y.size):
                string += print_element(Z[i,j]).rjust(N) + " "

        return string

    @classmethod
    def primitive_root_of_unity(cls, n: int) -> FieldArray:
        r"""
        Finds a primitive :math:`n`-th root of unity in the finite field.

        Parameters
        ----------
        n
            The root of unity.

        Returns
        -------
        :
            The primitive :math:`n`-th root of unity, a 0-D scalar array.

        Raises
        ------
        ValueError
            If no primitive :math:`n`-th roots of unity exist. This happens when :math:`n` is not a
            divisor of :math:`p^m - 1`.

        Notes
        -----
        A primitive :math:`n`-th root of unity :math:`\omega_n` is such that :math:`\omega_n^n = 1` and :math:`\omega_n^k \ne 1`
        for all :math:`1 \le k \lt n`.

        In :math:`\mathrm{GF}(p^m)`, a primitive :math:`n`-th root of unity exists when :math:`n` divides :math:`p^m - 1`.
        Then, the primitive root is :math:`\omega_n = \alpha^{(p^m - 1)/n}` where :math:`\alpha` is a primitive
        element of the field.

        Examples
        --------
        In :math:`\mathrm{GF}(31)`, primitive roots exist for all divisors of 30.

        .. ipython:: python

            GF = galois.GF(31)
            GF.primitive_root_of_unity(2)
            GF.primitive_root_of_unity(5)
            GF.primitive_root_of_unity(15)

        However, they do not exist for :math:`n` that do not divide 30.

        .. ipython:: python
            :okexcept:

            GF.primitive_root_of_unity(7)

        For :math:`\omega_5`, one can see that :math:`\omega_5^5 = 1` and :math:`\omega_5^k \ne 1` for :math:`1 \le k \lt 5`.

        .. ipython:: python

            root = GF.primitive_root_of_unity(5); root
            powers = np.arange(1, 5 + 1); powers
            root ** powers
        """
        if not isinstance(n, (int, np.ndarray)):
            raise TypeError(f"Argument `n` must be an int, not {type(n)!r}.")
        if not 1 <= n < cls.order:
            raise ValueError(f"Argument `n` must be in [1, {cls.order}), not {n}.")
        if not (cls.order - 1) % n == 0:
            raise ValueError(f"There are no primitive {n}-th roots of unity in {cls.name}.")

        return cls.primitive_element ** ((cls.order - 1) // n)

    @classmethod
    def primitive_roots_of_unity(cls, n: int) -> FieldArray:
        r"""
        Finds all primitive :math:`n`-th roots of unity in the finite field.

        Parameters
        ----------
        n
            The root of unity.

        Returns
        -------
        :
            All primitive :math:`n`-th roots of unity, a 1-D array. The roots are sorted in lexicographically-increasing
            order.

        Raises
        ------
        ValueError
            If no primitive :math:`n`-th roots of unity exist. This happens when :math:`n` is not a
            divisor of :math:`p^m - 1`.

        Notes
        -----
        A primitive :math:`n`-th root of unity :math:`\omega_n` is such that :math:`\omega_n^n = 1` and :math:`\omega_n^k \ne 1`
        for all :math:`1 \le k \lt n`.

        In :math:`\mathrm{GF}(p^m)`, a primitive :math:`n`-th root of unity exists when :math:`n` divides :math:`p^m - 1`.
        Then, the primitive root is :math:`\omega_n = \alpha^{(p^m - 1)/n}` where :math:`\alpha` is a primitive
        element of the field.

        Examples
        --------
        In :math:`\mathrm{GF}(31)`, primitive roots exist for all divisors of 30.

        .. ipython:: python

            GF = galois.GF(31)
            GF.primitive_roots_of_unity(2)
            GF.primitive_roots_of_unity(5)
            GF.primitive_roots_of_unity(15)

        However, they do not exist for :math:`n` that do not divide 30.

        .. ipython:: python
            :okexcept:

            GF.primitive_roots_of_unity(7)

        For :math:`\omega_5`, one can see that :math:`\omega_5^5 = 1` and :math:`\omega_5^k \ne 1` for :math:`1 \le k \lt 5`.

        .. ipython:: python

            root = GF.primitive_roots_of_unity(5); root
            powers = np.arange(1, 5 + 1); powers
            np.power.outer(root, powers)
        """
        if not isinstance(n, (int, np.ndarray)):
            raise TypeError(f"Argument `n` must be an int, not {type(n)!r}.")
        if not (cls.order - 1) % n == 0:
            raise ValueError(f"There are no primitive {n}-th roots of unity in {cls.name}.")

        roots = np.unique(cls.primitive_elements ** ((cls.order - 1) // n))
        roots = np.sort(roots)

        return roots

    ###############################################################################
    # Instance methods
    ###############################################################################

    def additive_order(self) -> Union[np.integer, np.ndarray]:
        r"""
        Computes the additive order of each element in :math:`x`.

        Returns
        -------
        :
            An integer array of the additive order of each element in :math:`x`. The return value is a single integer if the
            input array :math:`x` is a scalar.

        Notes
        -----
        The additive order :math:`a` of :math:`x` in :math:`\mathrm{GF}(p^m)` is the smallest integer :math:`a`
        such that :math:`x a = 0`. With the exception of 0, the additive order of every element is
        the finite field's characteristic.

        Examples
        --------
        Compute the additive order of each element of :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.elements; x
            order = x.additive_order(); order
            x * order
            @suppress
            GF.display()
        """
        x = self
        field = type(self)

        if x.ndim == 0:
            order = np.int64(1) if x == 0 else np.int64(field.characteristic)
        else:
            order = field.characteristic * np.ones(x.shape, dtype=np.int64)
            order[np.where(x == 0)] = 1

        return order

    def multiplicative_order(self) -> Union[np.integer, np.ndarray]:
        r"""
        Computes the multiplicative order :math:`\textrm{ord}(x)` of each element in :math:`x`.

        Returns
        -------
        :
            An integer array of the multiplicative order of each element in :math:`x`. The return value is a single integer if the
            input array :math:`x` is a scalar.

        Raises
        ------
        ArithmeticError
            If zero is provided as an input. The multiplicative order of 0 is not defined. There is no power of 0 that ever
            results in 1.

        Notes
        -----
        The multiplicative order :math:`\textrm{ord}(x) = a` of :math:`x` in :math:`\mathrm{GF}(p^m)` is the smallest power :math:`a`
        such that :math:`x^a = 1`. If :math:`a = p^m - 1`, :math:`a` is said to be a generator of the multiplicative group
        :math:`\mathrm{GF}(p^m)^\times`.

        Note, :func:`multiplicative_order` should not be confused with :obj:`order`. The former returns the multiplicative order of
        :obj:`~galois.FieldArray` elements. The latter is a property of the field, namely the finite field's order or size.

        Examples
        --------
        Compute the multiplicative order of each non-zero element of :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.units; x
            order = x.multiplicative_order(); order
            x ** order

        The elements with :math:`\textrm{ord}(x) = 8` are multiplicative generators of :math:`\mathrm{GF}(3^2)^\times`,
        which are also called primitive elements.

        .. ipython:: python

            GF.primitive_elements
            @suppress
            GF.display()
        """
        if not np.count_nonzero(self) == self.size:
            raise ArithmeticError("The multiplicative order of 0 is not defined.")

        x = self
        field = type(self)

        if field.ufunc_mode == "jit-lookup":
            # This algorithm is faster if np.log() has a lookup table
            # β = α^k
            # ord(α) = p^m - 1
            # ord(β) = (p^m - 1) / gcd(p^m - 1, k)
            k = np.log(x)  # x as an exponent of α
            order = (field.order - 1) // np.gcd(field.order - 1, k)
        else:
            d = np.array(divisors(field.order - 1))  # Divisors d such that d | p^m - 1
            y = np.power.outer(x, d)  # x^d -- the first divisor d for which x^d == 1 is the order of x
            idxs = np.argmin(y, axis=-1)  # First index of divisors, which is the order of x
            order = d[idxs]  # The order of each element of x

        return order

    def is_quadratic_residue(self) -> Union[np.bool_, np.ndarray]:
        r"""
        Determines if the elements of :math:`x` are quadratic residues in the finite field.

        Returns
        -------
        :
            A boolean array indicating if each element in :math:`x` is a quadratic residue. The return value is a single boolean if the
            input array :math:`x` is a scalar.

        See Also
        --------
        quadratic_residues, quadratic_non_residues

        Notes
        -----
        An element :math:`x` in :math:`\mathrm{GF}(p^m)` is a *quadratic residue* if there exists a :math:`y` such that
        :math:`y^2 = x` in the field.

        In fields with characteristic 2, every element is a quadratic residue. In fields with characteristic greater than 2,
        exactly half of the nonzero elements are quadratic residues (and they have two unique square roots).

        References
        ----------
        * Section 3.5.1 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.

        Examples
        --------
        Since :math:`\mathrm{GF}(2^3)` has characteristic 2, every element has a square root.

        .. ipython:: python

            GF = galois.GF(2**3, display="poly")
            x = GF.elements; x
            x.is_quadratic_residue()
            @suppress
            GF.display()

        In :math:`\mathrm{GF}(11)`, the characteristic is greater than 2 so only half of the elements have square
        roots.

        .. ipython:: python

            GF = galois.GF(11)
            x = GF.elements; x
            x.is_quadratic_residue()
        """
        x = self
        field = type(self)

        if field.characteristic == 2:
            # All elements are quadratic residues if the field's characteristic is 2
            return np.ones(x.shape, dtype=bool) if x.ndim > 0 else np.bool_(True)
        else:
            # Compute the Legendre symbol on each element
            return x ** ((field.order - 1)//2) != field.characteristic - 1

    def vector(self, dtype: Optional[DTypeLike] = None) -> FieldArray:
        r"""
        Converts an array over :math:`\mathrm{GF}(p^m)` to length-:math:`m` vectors over the prime subfield :math:`\mathrm{GF}(p)`.

        This function is the inverse operation of the :func:`Vector` constructor. For an array with shape `(n1, n2)`, the output shape
        is `(n1, n2, m)`. By convention, the vectors are ordered from degree :math:`m-1` to degree 0.

        Parameters
        ----------
        dtype
            The :obj:`numpy.dtype` of the array elements. The default is `None` which represents the smallest unsigned
            data type for this :obj:`~galois.FieldArray` subclass (the first element in :obj:`~galois.FieldArray.dtypes`).

        Returns
        -------
        :
            An array over :math:`\mathrm{GF}(p)` with last dimension :math:`m`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**3, display="poly")
            a = GF([11, 7]); a
            vec = a.vector(); vec
            GF.Vector(vec)
            @suppress
            GF.display()
        """
        field = type(self)
        subfield = field.prime_subfield
        order = subfield.order
        degree = field.degree

        x = np.array(self)  # The original array as an integer array
        shape = list(self.shape) + [degree,]  # The new shape
        y = subfield.Zeros(shape, dtype=dtype)

        if self.dtype == np.object_:
            # Need a separate "if" statement because divmod() does not work with dtype=object input and integer dtype outputs
            for i in range(degree - 1, -1, -1):
                q, r = x // order, x % order
                y[..., i] = r
                x = q
        else:
            for i in range(degree - 1, -1, -1):
                q, r = divmod(x, order)
                y[..., i] = r
                x = q

        return y

    def row_reduce(self, ncols: Optional[int] = None) -> FieldArray:
        r"""
        Performs Gaussian elimination on the matrix to achieve reduced row echelon form (RREF).

        Parameters
        ----------
        ncols
            The number of columns to perform Gaussian elimination over. The default is `None` which represents
            the number of columns of the matrix.

        Returns
        -------
        :
            The reduced row echelon form of the input matrix.

        Notes
        -----

        The elementary row operations in Gaussian elimination are:

        1. Swap the position of any two rows.
        2. Multiply any row by a non-zero scalar.
        3. Add any row to a scalar multiple of another row.

        Examples
        --------
        Perform Gaussian elimination to get the reduced row echelon form of :math:`\mathbf{A}`.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF([[16, 12, 1, 25], [1, 10, 27, 29], [1, 0, 3, 19]]); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        Or only perform Gaussian elimination over 2 columns.

        .. ipython:: python

            A.row_reduce(ncols=2)
        """
        A_rre, _ = _linalg.row_reduce_jit(type(self))(self, ncols=ncols)
        return A_rre

    def lu_decompose(self) -> Tuple[FieldArray, FieldArray]:
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices.

        Returns
        -------
        L :
            The lower triangular matrix.
        U :
            The upper triangular matrix.

        Notes
        -----
        The LU decomposition of :math:`\mathbf{A}` is defined as :math:`\mathbf{A} = \mathbf{L} \mathbf{U}`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            # Not every square matrix has an LU decomposition
            A = GF([[22, 11, 25, 11], [30, 27, 10, 3], [21, 16, 29, 7]]); A
            L, U = A.lu_decompose()
            L
            U
            np.array_equal(A, L @ U)
        """
        field = type(self)
        A = self
        L, U = _linalg.lu_decompose_jit(field)(A)
        return L, U

    def plu_decompose(self) -> Tuple[FieldArray, FieldArray, FieldArray]:
        r"""
        Decomposes the input array into the product of lower and upper triangular matrices using partial pivoting.

        Returns
        -------
        P :
            The column permutation matrix.
        L :
            The lower triangular matrix.
        U :
            The upper triangular matrix.

        Notes
        -----
        The PLU decomposition of :math:`\mathbf{A}` is defined as :math:`\mathbf{A} = \mathbf{P} \mathbf{L} \mathbf{U}`. This is equivalent to
        :math:`\mathbf{P}^T \mathbf{A} = \mathbf{L} \mathbf{U}`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            A = GF([[0, 29, 2, 9], [20, 24, 5, 1], [2, 24, 1, 7]]); A
            P, L, U = A.plu_decompose()
            P
            L
            U
            np.array_equal(A, P @ L @ U)
            np.array_equal(P.T @ A, L @ U)
        """
        field = type(self)
        A = self
        P, L, U, _ = _linalg.plu_decompose_jit(field)(A)
        return P, L, U

    def row_space(self) -> FieldArray:
        r"""
        Computes the row space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The row space basis matrix. The rows of the basis matrix are the basis vectors that span the row space.
            The number of rows of the basis matrix is the dimension of the row space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *row space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^n\}` defined by all linear combinations of the rows
        of :math:`\mathbf{A}`. The row space has at most dimension :math:`\textrm{min}(m, n)`.

        The row space has properties :math:`\mathcal{R}(\mathbf{A}) = \mathcal{C}(\mathbf{A}^T)` and
        :math:`\textrm{dim}(\mathcal{R}(\mathbf{A})) + \textrm{dim}(\mathcal{LN}(\mathbf{A})) = m`.

        Examples
        --------
        The :func:`row_space` method defines basis vectors (its rows) that span the row space of :math:`\mathbf{A}`.

        .. ipython:: python

            m, n = 5, 3
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            R = A.row_space(); R

        The dimension of the row space and left null space sum to :math:`m`.

        .. ipython:: python

            LN = A.left_null_space(); LN
            R.shape[0] + LN.shape[0] == m
        """
        A = self
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a row space, not {A.ndim}-D.")

        A_rre = A.row_reduce()
        rank = np.sum(~np.all(A_rre == 0, axis=1))
        R = A_rre[0:rank,:]

        return R

    def column_space(self) -> FieldArray:
        r"""
        Computes the column space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The column space basis matrix. The rows of the basis matrix are the basis vectors that span the column space.
            The number of rows of the basis matrix is the dimension of the column space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *column space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^m\}` defined by all linear combinations of the columns
        of :math:`\mathbf{A}`. The column space has at most dimension :math:`\textrm{min}(m, n)`.

        The column space has properties :math:`\mathcal{C}(\mathbf{A}) = \mathcal{R}(\mathbf{A}^T)`  and
        :math:`\textrm{dim}(\mathcal{C}(\mathbf{A})) + \textrm{dim}(\mathcal{N}(\mathbf{A})) = n`.

        Examples
        --------
        The :func:`column_space` method defines basis vectors (its rows) that span the column space of :math:`\mathbf{A}`.

        .. ipython:: python

            m, n = 3, 5
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            C = A.column_space(); C

        The dimension of the column space and null space sum to :math:`n`.

        .. ipython:: python

            N = A.null_space(); N
            C.shape[0] + N.shape[0] == n
        """
        A = self
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a column space, not {A.ndim}-D.")

        return (A.T).row_space()  # pylint: disable=no-member

    def left_null_space(self) -> FieldArray:
        r"""
        Computes the left null space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The left null space basis matrix. The rows of the basis matrix are the basis vectors that span the left null space.
            The number of rows of the basis matrix is the dimension of the left null space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *left null space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^m\}` that annihilates the rows of :math:`\mathbf{A}`, i.e.
        :math:`\mathbf{x}\mathbf{A} = \mathbf{0}`.

        The left null space has properties :math:`\mathcal{LN}(\mathbf{A}) = \mathcal{N}(\mathbf{A}^T)` and
        :math:`\textrm{dim}(\mathcal{R}(\mathbf{A})) + \textrm{dim}(\mathcal{LN}(\mathbf{A})) = m`.

        Examples
        --------
        The :func:`left_null_space` method defines basis vectors (its rows) that span the left null space of :math:`\mathbf{A}`.

        .. ipython:: python

            m, n = 5, 3
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            LN = A.left_null_space(); LN

        The left null space is the set of vectors that sum the rows to 0.

        .. ipython:: python

            LN @ A

        The dimension of the row space and left null space sum to :math:`m`.

        .. ipython:: python

            R = A.row_space(); R
            R.shape[0] + LN.shape[0] == m
        """
        field = type(self)
        A = self
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a left null space, not {A.ndim}-D.")

        m, n = A.shape
        I = field.Identity(m, dtype=A.dtype)

        # Concatenate A and I to get the matrix AI = [A | I]
        AI = np.concatenate((A, I), axis=-1)

        # Perform Gaussian elimination to get the reduced row echelon form AI_rre = [I | A^-1]
        AI_rre, p = _linalg.row_reduce_jit(field)(AI, ncols=n)

        # Row reduce the left null space so that it begins with an I
        LN = AI_rre[p:,n:]
        LN = LN.row_reduce()

        return LN

    def null_space(self) -> FieldArray:
        r"""
        Computes the null space of the matrix :math:`\mathbf{A}`.

        Returns
        -------
        :
            The null space basis matrix. The rows of the basis matrix are the basis vectors that span the null space.
            The number of rows of the basis matrix is the dimension of the null space.

        Notes
        -----
        Given an :math:`m \times n` matrix :math:`\mathbf{A}` over :math:`\mathrm{GF}(q)`, the *null space* of :math:`\mathbf{A}`
        is the vector space :math:`\{\mathbf{x} \in \mathrm{GF}(q)^n\}` that annihilates the columns of :math:`\mathbf{A}`, i.e.
        :math:`\mathbf{A}\mathbf{x} = \mathbf{0}`.

        The null space has properties :math:`\mathcal{N}(\mathbf{A}) = \mathcal{LN}(\mathbf{A}^T)` and
        :math:`\textrm{dim}(\mathcal{C}(\mathbf{A})) + \textrm{dim}(\mathcal{N}(\mathbf{A})) = n`.

        Examples
        --------
        The :func:`null_space` method defines basis vectors (its rows) that span the null space of :math:`\mathbf{A}`.

        .. ipython:: python

            m, n = 3, 5
            GF = galois.GF(31)
            A = GF.Random((m, n)); A
            N = A.null_space(); N

        The null space is the set of vectors that sum the columns to 0.

        .. ipython:: python

            A @ N.T

        The dimension of the column space and null space sum to :math:`n`.

        .. ipython:: python

            C = A.column_space(); C
            C.shape[0] + N.shape[0] == n
        """
        A = self
        if not A.ndim == 2:
            raise ValueError(f"Only 2-D matrices have a null space, not {A.ndim}-D.")

        return (A.T).left_null_space()  # pylint: disable=no-member

    def field_trace(self) -> FieldArray:
        r"""
        Computes the field trace :math:`\mathrm{Tr}_{L / K}(x)` of the elements of :math:`x`.

        Returns
        -------
        :
            The field trace of :math:`x` in the prime subfield :math:`\mathrm{GF}(p)`.

        Notes
        -----
        The `self` array :math:`x` is over the extension field :math:`L = \mathrm{GF}(p^m)`. The field trace of :math:`x` is
        over the subfield :math:`K = \mathrm{GF}(p)`. In other words, :math:`\mathrm{Tr}_{L / K}(x) : L \rightarrow K`.

        For finite fields, since :math:`L` is a Galois extension of :math:`K`, the field trace of :math:`x` is defined as a sum
        of the Galois conjugates of :math:`x`.

        .. math::
            \mathrm{Tr}_{L / K}(x) = \sum_{i=0}^{m-1} x^{p^i}

        References
        ----------
        * https://en.wikipedia.org/wiki/Field_trace

        Examples
        --------
        Compute the field trace of the elements of :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.elements; x
            y = x.field_trace(); y
            @suppress
            GF.display()
        """
        field = type(self)
        x = self

        if field.is_prime_field:
            return x.copy()
        else:
            subfield = field.prime_subfield
            p = field.characteristic
            m = field.degree
            conjugates = np.power.outer(x, p**np.arange(0, m, dtype=field.dtypes[-1]))
            trace = np.add.reduce(conjugates, axis=-1)
            return subfield._view(trace)

    def field_norm(self) -> FieldArray:
        r"""
        Computes the field norm :math:`\mathrm{N}_{L / K}(x)` of the elements of :math:`x`.

        Returns
        -------
        :
            The field norm of :math:`x` in the prime subfield :math:`\mathrm{GF}(p)`.

        Notes
        -----
        The `self` array :math:`x` is over the extension field :math:`L = \mathrm{GF}(p^m)`. The field norm of :math:`x` is
        over the subfield :math:`K = \mathrm{GF}(p)`. In other words, :math:`\mathrm{N}_{L / K}(x) : L \rightarrow K`.

        For finite fields, since :math:`L` is a Galois extension of :math:`K`, the field norm of :math:`x` is defined as a product
        of the Galois conjugates of :math:`x`.

        .. math::
            \mathrm{N}_{L / K}(x) = \prod_{i=0}^{m-1} x^{p^i} = x^{(p^m - 1) / (p - 1)}

        References
        ----------
        * https://en.wikipedia.org/wiki/Field_norm

        Examples
        --------
        Compute the field norm of the elements of :math:`\mathrm{GF}(3^2)`.

        .. ipython:: python

            GF = galois.GF(3**2, display="poly")
            x = GF.elements; x
            y = x.field_norm(); y
            @suppress
            GF.display()
        """
        field = type(self)
        x = self

        if field.is_prime_field:
            return x.copy()
        else:
            subfield = field.prime_subfield
            p = field.characteristic
            m = field.degree
            norm = x**((p**m - 1) // (p - 1))
            return subfield._view(norm)

    def characteristic_poly(self) -> Poly:
        r"""
        Computes the characteristic polynomial of a finite field element :math:`a` or a square matrix :math:`\mathbf{A}`.

        Important
        ---------
        This function may only be invoked on a single finite field element (scalar 0-D array) or a square :math:`n \times n`
        matrix (2-D array).

        Returns
        -------
        :
            For scalar inputs, the degree-:math:`m` characteristic polynomial :math:`c_a(x)` of :math:`a` over :math:`\mathrm{GF}(p)`.
            For square :math:`n \times n` matrix inputs, the degree-:math:`n` characteristic polynomial :math:`c_A(x)` of
            :math:`\mathbf{A}` over :math:`\mathrm{GF}(p^m)`.

        Notes
        -----
        An element :math:`a` of :math:`\mathrm{GF}(p^m)` has characteristic polynomial :math:`c_a(x)` over :math:`\mathrm{GF}(p)`.
        The characteristic polynomial when evaluated in :math:`\mathrm{GF}(p^m)` annihilates :math:`a`, that is :math:`c_a(a) = 0`.
        In prime fields :math:`\mathrm{GF}(p)`, the characteristic polynomial of :math:`a` is simply :math:`c_a(x) = x - a`.

        An :math:`n \times n` matrix :math:`\mathbf{A}` has characteristic polynomial
        :math:`c_A(x) = \textrm{det}(x\mathbf{I} - \mathbf{A})` over :math:`\mathrm{GF}(p^m)`. The constant coefficient of the
        characteristic polynomial is :math:`\textrm{det}(-\mathbf{A})`. The :math:`x^{n-1}` coefficient of the characteristic
        polynomial is :math:`-\textrm{Tr}(\mathbf{A})`. The characteristic polynomial annihilates :math:`\mathbf{A}`, that is
        :math:`c_A(\mathbf{A}) = \mathbf{0}`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Characteristic_polynomial

        Examples
        --------
        The characteristic polynomial of the element :math:`a`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = GF.Random(); a
            poly = a.characteristic_poly(); poly
            # The characteristic polynomial annihilates a
            poly(a, field=GF)

        The characteristic polynomial of the square matrix :math:`\mathbf{A}`.

        .. ipython:: python

            GF = galois.GF(3**5)
            A = GF.Random((3,3)); A
            poly = A.characteristic_poly(); poly
            # The x^0 coefficient is det(-A)
            poly.coeffs[-1] == np.linalg.det(-A)
            # The x^n-1 coefficient is -Tr(A)
            poly.coeffs[1] == -np.trace(A)
            # The characteristic polynomial annihilates the matrix A
            poly(A, elementwise=False)
        """
        if self.ndim == 0:
            return _characteristic_poly_element(self)
        elif self.ndim == 2:
            return _characteristic_poly_matrix(self)
        else:
            raise ValueError(f"The array must be either 0-D to return the characteristic polynomial of a single element or 2-D to return the characteristic polynomial of a square matrix, not have shape {self.shape}.")

    def minimal_poly(self) -> Poly:
        r"""
        Computes the minimal polynomial of a finite field element :math:`a`.

        Important
        ---------
        This function may only be invoked on a single finite field element (scalar 0-D array).

        Returns
        -------
        :
            For scalar inputs, the minimal polynomial :math:`m_a(x)` of :math:`a` over :math:`\mathrm{GF}(p)`.

        Notes
        -----
        An element :math:`a` of :math:`\mathrm{GF}(p^m)` has minimal polynomial :math:`m_a(x)` over :math:`\mathrm{GF}(p)`.
        The minimal polynomial when evaluated in :math:`\mathrm{GF}(p^m)` annihilates :math:`a`, that is :math:`m_a(a) = 0`.
        The minimal polynomial always divides the characteristic polynomial. In prime fields :math:`\mathrm{GF}(p)`, the
        minimal polynomial of :math:`a` is simply :math:`m_a(x) = x - a`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Minimal_polynomial_(field_theory)
        * https://en.wikipedia.org/wiki/Minimal_polynomial_(linear_algebra)

        Examples
        --------
        The minimal polynomial of the element :math:`a`.

        .. ipython:: python

            GF = galois.GF(3**5)
            a = GF.Random(); a
            poly = a.minimal_poly(); poly
            # The minimal polynomial annihilates a
            poly(a, field=GF)
            # The minimal polynomial always divides the characteristic polynomial
            divmod(a.characteristic_poly(), poly)
        """
        if self.ndim == 0:
            return _minimal_poly_element(self)
        # elif self.ndim == 2:
        #     return _minimal_poly_matrix(self)
        else:
            raise ValueError(f"The array must be either 0-D to return the minimal polynomial of a single element or 2-D to return the minimal polynomial of a square matrix, not have shape {self.shape}.")

    ###############################################################################
    # Display methods
    ###############################################################################

    def __repr__(self) -> str:
        """
        Displays the array specifying the class and finite field order.

        This function prepends `GF(` and appends `, order=p^m)`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    x = GF([4, 2, 7, 5])
                    x

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    x = GF([4, 2, 7, 5])
                    x

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    x = GF([4, 2, 7, 5])
                    x
                    @suppress
                    GF.display()
        """
        return self._display("repr")

    def __str__(self) -> str:
        """
        Displays the array without specifying the class or finite field order.

        This function does not prepend `GF(` and or append `, order=p^m)`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Integer

                .. ipython:: python

                    GF = galois.GF(3**2)
                    x = GF([4, 2, 7, 5])
                    print(x)

            .. tab-item:: Polynomial

                .. ipython:: python

                    GF = galois.GF(3**2, display="poly")
                    x = GF([4, 2, 7, 5])
                    print(x)

            .. tab-item:: Power

                .. ipython:: python

                    GF = galois.GF(3**2, display="power")
                    x = GF([4, 2, 7, 5])
                    print(x)
        """
        return self._display("str")

    def _display(self, mode: Literal["repr", "str"]) -> str:
        # View the array as an ndarray so that the scalar -> 0-D array conversion in __array_finalize__() for Galois field
        # arrays isn't continually invoked. This improves performance slightly.
        x = self.view(np.ndarray)
        field = type(self)

        separator = ", "
        prefix = "GF(" if mode == "repr" else ""
        order = field._order_str if mode == "repr" else ""
        suffix = ")" if mode == "repr" else ""
        formatter = field._formatter(self)

        field._element_fixed_width = None  # Do not print with fixed-width
        field._element_fixed_width_counter = 0  # Reset element width counter

        string = np.array2string(x, separator=separator, prefix=prefix, suffix=suffix, formatter=formatter)

        if formatter != {}:
            # We are using special print methods and must perform element alignment ourselves. We will print each element
            # a second time use the max width of any element observed on the first array2string() call.
            field._element_fixed_width = field._element_fixed_width_counter

            string = np.array2string(x, separator=separator, prefix=prefix, suffix=suffix, formatter=formatter)

        field._element_fixed_width = None
        field._element_fixed_width_counter = 0

        # Determine the width of the last line in the string
        if mode == "repr":
            idx = string.rfind("\n") + 1
            last_line_width = len(string[idx:] + ", " + order + suffix)

            if last_line_width <= np.get_printoptions()["linewidth"]:
                return prefix + string + ", " + order + suffix
            else:
                return prefix + string + ",\n" + " "*len(prefix) + order + suffix
        else:
            return prefix + string + suffix

    @classmethod
    def _formatter(cls, array):
        """
        Returns a NumPy printoptions "formatter" dictionary.
        """
        formatter = {}

        if cls.display_mode == "poly" and cls.is_extension_field:
            # The "poly" display mode for prime field's is the same as the integer representation
            formatter["int"] = cls._print_poly
            formatter["object"] = cls._print_poly
        elif cls.display_mode == "power":
            formatter["int"] = cls._print_power
            formatter["object"] = cls._print_power
        elif array.dtype == np.object_:
            formatter["object"] = cls._print_int

        return formatter

    @classmethod
    def _print_int(cls, element):
        """
        Prints a single element in the integer representation. This is only needed for dtype=object arrays.
        """
        s = f"{int(element)}"

        if cls._element_fixed_width:
            s = s.rjust(cls._element_fixed_width)
        else:
            cls._element_fixed_width_counter = max(len(s), cls._element_fixed_width_counter)

        return s

    @classmethod
    def _print_poly(cls, element):
        """
        Prints a single element in the polynomial representation.
        """
        poly = integer_to_poly(int(element), cls.characteristic)
        poly_var = "α" if cls.primitive_element == cls.characteristic else "x"
        s = poly_to_str(poly, poly_var=poly_var)

        if cls._element_fixed_width:
            s = s.rjust(cls._element_fixed_width)
        else:
            cls._element_fixed_width_counter = max(len(s), cls._element_fixed_width_counter)

        return s

    @classmethod
    def _print_power(cls, element):
        """
        Prints a single element in the power representation.
        """
        if element in [0, 1]:
            s = f"{int(element)}"
        elif element == cls.primitive_element:
            s = "α"
        else:
            power = cls._log.ufunc(element, cls._primitive_element)
            s = f"α^{power}"

        if cls._element_fixed_width:
            s = s.rjust(cls._element_fixed_width)
        else:
            cls._element_fixed_width_counter = max(len(s), cls._element_fixed_width_counter)

        return s


def _poly_det(A: np.ndarray) -> Poly:
    """
    Computes the determinant of a matrix of `Poly` objects.
    """
    field = A.flatten()[0].field

    if A.shape == (2, 2):
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]

    n = A.shape[0]  # Size of the n x n matrix
    det = Poly.Zero(field)
    for i in range(n):
        idxs = np.delete(np.arange(0, n), i)
        if i % 2 == 0:
            det += A[0,i] * _poly_det(A[1:,idxs])
        else:
            det -= A[0,i] * _poly_det(A[1:,idxs])

    return det


def _characteristic_poly_element(a: FieldArray) -> Poly:
    """
    Computes the characteristic polynomial of the Galois field element `a`.
    """
    field = type(a)
    x = Poly.Identity(field)

    if field.is_prime_field:
        return x - a
    else:
        powers = a**(field.characteristic**np.arange(0, field.degree, dtype=field.dtypes[-1]))
        poly = Poly.Roots(powers, field=field)
        poly = Poly(poly.coeffs, field=field.prime_subfield)
        return poly


def _characteristic_poly_matrix(A: FieldArray) -> Poly:
    """
    Computes the characteristic polynomial of the Galois field matrix `A`.
    """
    if not A.shape[0] == A.shape[1]:
        raise ValueError(f"The 2-D array must be square to compute its characteristic polynomial, not have shape {A.shape}.")
    field = type(A)

    # Compute P = xI - A
    P = np.zeros(A.shape, dtype=object)
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            if i == j:
                P[i,j] = Poly([1, -A[i,j]], field=field)
            else:
                P[i,j] = Poly([-A[i,j]], field=field)

    # Compute det(P)
    return _poly_det(P)


def _minimal_poly_element(a: FieldArray) -> Poly:
    """
    Computes the minimal polynomial of the Galois field element `a`.
    """
    field = type(a)
    x = Poly.Identity(field)

    if field.is_prime_field:
        return x - a
    else:
        conjugates = np.unique(a**(field.characteristic**np.arange(0, field.degree, dtype=field.dtypes[-1])))
        poly = Poly.Roots(conjugates, field=field)
        poly = Poly(poly.coeffs, field=field.prime_subfield)
        return poly
