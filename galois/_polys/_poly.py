"""
A module containing a class for univariate polynomials over finite fields.
"""
from __future__ import annotations

from typing import Tuple, Sequence, Optional, Union, Type, overload
from typing_extensions import Literal

import numpy as np

from .._domains._array import Array, ElementLike, ArrayLike, DEFAULT_FIELD_ARRAY
from .._overrides import set_module

from . import _binary, _dense, _sparse
from ._conversions import integer_to_poly, integer_to_degree, poly_to_integer, poly_to_str, sparse_poly_to_integer, sparse_poly_to_str, str_to_sparse_poly

__all__ = ["Poly"]

# Values were obtained by running scripts/sparse_poly_performance_test.py
SPARSE_VS_DENSE_POLY_FACTOR = 0.00_125  # 1.25% density
SPARSE_VS_DENSE_POLY_MIN_COEFFS = int(1 / SPARSE_VS_DENSE_POLY_FACTOR)

PolyLike = Union[int, str, ArrayLike, "Poly"]
PolyLike.__doc__ = """
A :obj:`~typing.Union` representing objects that can be coerced into a polynomial.

.. rubric:: Union

- :obj:`int`: A polynomial in its integer representation, see :func:`~galois.Poly.Int`. The Galois field must be known from context.

.. ipython:: python

    # Known from context
    GF = galois.GF(3)
    galois.Poly.Int(19, field=GF)

- :obj:`str`: A polynomial in its string representation, see :func:`~galois.Poly.Str`. The Galois field must be known from context.

.. ipython:: python

    galois.Poly.Str("2x^2 + 1", field=GF)

- :obj:`~galois.typing.ArrayLike`: An array of polynomial coefficients in degree-descending order. If the coefficients are not
  :obj:`~galois.Array`, then the Galois field must be known from context.

.. ipython:: python

    galois.Poly([2, 0, 1], field=GF)
    galois.Poly(GF([2, 0, 1]))

- :obj:`~galois.Poly`: A previously-created :obj:`~galois.Poly` object. No coercion is necessary.

.. rubric:: Alias
"""


@set_module("galois")
class Poly:
    r"""
    A univariate polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

    Examples
    --------
    Create a polynomial over :math:`\mathrm{GF}(2)`.

    .. ipython:: python

        galois.Poly([1, 0, 1, 1])

    Create a polynomial over :math:`\mathrm{GF}(3^5)`.

    .. ipython:: python

        GF = galois.GF(3**5)
        galois.Poly([124, 0, 223, 0, 0, 15], field=GF)

    See :doc:`/basic-usage/poly` and :doc:`/basic-usage/poly-arithmetic` for more examples.
    """
    __slots__ = ["_field", "_degrees", "_coeffs", "_nonzero_degrees", "_nonzero_coeffs", "_integer", "_degree", "_type"]

    # Special private attributes that are once computed. There are three arithmetic types for polynomials: "dense", "binary",
    # and "sparse". All types define _field, "dense" defines _coeffs, "binary" defines "_integer", and "sparse" defines
    # _nonzero_degrees and _nonzero_coeffs. The other properties are created when needed.
    _field: Type[Array]
    _degrees: np.ndarray
    _coeffs: Array
    _nonzero_degrees: np.ndarray
    _nonzero_coeffs: Array
    _integer: int
    _degree: int
    _type: Literal["dense", "binary", "sparse"]

    # Increase my array priority so numpy will call my __radd__ instead of its own __add__
    __array_priority__ = 100

    def __init__(self, coeffs: ArrayLike, field: Optional[Type[Array]] = None, order: Literal["desc", "asc"] = "desc"):
        r"""
        Creates a polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)`.

        The polynomial :math:`f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0` with degree :math:`d` has coefficients
        :math:`\{a_{d}, a_{d-1}, \dots, a_1, a_0\}` in :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        coeffs
            The polynomial coefficients :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}`.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the coefficients are a :obj:`~galois.Array`, they won't be modified. If the coefficients
              are not explicitly in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using
              `galois.GF2(coeffs)`.
            * :obj:`~galois.Array` subclass: The coefficients are explicitly converted to this Galois field `field(coeffs)`.

        order
            The interpretation of the coefficient degrees.

            - `"desc"` (default): The first element of `coeffs` is the highest degree coefficient, i.e. :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}`.
            - `"asc"`: The first element of `coeffs` is the lowest degree coefficient, i.e. :math:`\{a_0, a_1, \dots,  a_{d-1}, a_d\}`.
        """
        if not isinstance(coeffs, (list, tuple, np.ndarray, Array)):
            raise TypeError(f"Argument `coeffs` must array-like, not {type(coeffs)}.")
        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {field}.")
        if not isinstance(order, str):
            raise TypeError(f"Argument `order` must be a str, not {type(order)}.")
        if isinstance(coeffs, (Array, np.ndarray)) and not coeffs.ndim <= 1:
            raise ValueError(f"Argument `coeffs` can have dimension at most 1, not {coeffs.ndim}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order!r}.")

        self._coeffs, self._field = self._convert_coeffs(coeffs, field)

        if self._coeffs.ndim == 0:
            self._coeffs = np.atleast_1d(self._coeffs)
        if order == "asc":
            self._coeffs = np.flip(self._coeffs)  # Ensure it's in descending-degree order
        if self._coeffs[0] == 0:
            self._coeffs = np.trim_zeros(self._coeffs, "f")  # Remove leading zeros
        if self._coeffs.size == 0:
            self._coeffs = self._field([0])

        if self._field == DEFAULT_FIELD_ARRAY:
            # Binary arithmetic is always faster than dense arithmetic
            self._type = "binary"
            # Compute the integer value so we're ready for arithmetic computations
            int(self)
        else:
            self._type = "dense"

    @classmethod
    def _convert_coeffs(cls, coeffs: ArrayLike, field: Optional[Type[Array]] = None) -> Tuple[Array, Type[Array]]:
        if isinstance(coeffs, Array):
            if field is None:
                # Infer the field from the coefficients provided
                field = type(coeffs)
            elif type(coeffs) is not field:  # pylint: disable=unidiomatic-typecheck
                # Convert coefficients into the specified field
                coeffs = field(coeffs)
        else:
            # Convert coefficients into the specified field (or GF2 if unspecified)
            if field is None:
                field = DEFAULT_FIELD_ARRAY
            coeffs = np.array(coeffs, dtype=field._dtypes[-1])
            sign = np.sign(coeffs)
            coeffs = sign * field(np.abs(coeffs))

        return coeffs, field

    @classmethod
    def _PolyLike(cls, poly_like: PolyLike, field: Optional[Type[Array]] = None) -> Poly:
        """
        A private alternate constructor that converts a poly-like object into a polynomial, given a finite field.
        """
        if isinstance(poly_like, int):
            poly = Poly.Int(poly_like, field=field)
        elif isinstance(poly_like, str):
            poly = Poly.Str(poly_like, field=field)
        elif isinstance(poly_like, (tuple, list, np.ndarray)):
            poly = Poly(poly_like, field=field)
        elif isinstance(poly_like, Poly):
            poly = poly_like
        else:
            raise TypeError(f"A 'poly-like' object must be an int, str, tuple, list, np.ndarray, or galois.Poly, not {type(poly_like)}.")

        return poly

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zero(cls, field: Optional[Type[Array]] = None) -> Poly:
        r"""
        Constructs the polynomial :math:`f(x) = 0` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which corresponds to :obj:`~galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x) = 0`.

        Examples
        --------
        Construct the zero polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Zero()

        Construct the zero polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Zero(GF)
        """
        return Poly([0], field=field)

    @classmethod
    def One(cls, field: Optional[Type[Array]] = None) -> Poly:
        r"""
        Constructs the polynomial :math:`f(x) = 1` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which corresponds to :obj:`~galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x) = 1`.

        Examples
        --------
        Construct the one polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.One()

        Construct the one polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.One(GF)
        """
        return Poly([1], field=field)

    @classmethod
    def Identity(cls, field: Optional[Type[Array]] = None) -> Poly:
        r"""
        Constructs the polynomial :math:`f(x) = x` over :math:`\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which corresponds to :obj:`~galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x) = x`.

        Examples
        --------
        Construct the identity polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Identity()

        Construct the identity polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Identity(GF)
        """
        return Poly([1, 0], field=field)

    @classmethod
    def Random(cls, degree: int, seed: Optional[Union[int, np.random.Generator]] = None, field: Optional[Type[Array]] = None) -> Poly:
        r"""
        Constructs a random polynomial over :math:`\mathrm{GF}(p^m)` with degree :math:`d`.

        Parameters
        ----------
        degree
            The degree of the polynomial.
        seed
            Non-negative integer used to initialize the PRNG. The default is `None` which means that unpredictable
            entropy will be pulled from the OS to be used as the seed. A :obj:`numpy.random.Generator` can also be passed.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which corresponds to :obj:`~galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a random degree-:math:`5` polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Random(5)

        Construct a random degree-:math:`5` polynomial over :math:`\mathrm{GF}(3^5)` with a given seed. This produces repeatable results.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Random(5, seed=123456789, field=GF)
            galois.Poly.Random(5, seed=123456789, field=GF)

        Construct multiple polynomials with one global seed.

        .. ipython:: python

            rng = np.random.default_rng(123456789)
            galois.Poly.Random(5, seed=rng, field=GF)
            galois.Poly.Random(5, seed=rng, field=GF)
        """
        field = DEFAULT_FIELD_ARRAY if field is None else field
        if not isinstance(degree, (int, np.integer)):
            raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
        if seed is not None:
            if not isinstance(seed, (int, np.integer, np.random.Generator)):
                raise ValueError("Seed must be an integer, a numpy.random.Generator or None.")
            if isinstance(seed, (int, np.integer)) and seed < 0:
                raise ValueError("Seed must be non-negative.")
        if not issubclass(field, Array):
            raise TypeError(f"Argument `field` must be a Galois field class, not {type(field)}.")
        if not degree >= 0:
            raise ValueError(f"Argument `degree` must be non-negative, not {degree}.")

        rng = np.random.default_rng(seed)  # Make the seed a PRNG object so it can "step" its state if the below "if" statement is invoked
        coeffs = field.Random(degree + 1, seed=rng)
        if coeffs[0] == 0:
            coeffs[0] = field.Random(low=1, seed=rng)  # Ensure leading coefficient is non-zero

        return Poly(coeffs)

    @classmethod
    def Str(cls, string: str, field: Optional[Type[Array]] = None) -> Poly:
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its string representation.

        :func:`~galois.Poly.Str` and :func:`~galois.Poly.__str__` are inverse operations.

        Parameters
        ----------
        string
            The string representation of the polynomial :math:`f(x)`.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which corresponds to :obj:`~galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Notes
        -----
        The string parsing rules include:

        * Either `^` or `**` may be used for indicating the polynomial degrees. For example, `"13x^3 + 117"` or `"13x**3 + 117"`.
        * Multiplication operators `*` may be used between coefficients and the polynomial indeterminate `x`, but are not required. For example,
          `"13x^3 + 117"` or `"13*x^3 + 117"`.
        * Polynomial coefficients of 1 may be specified or omitted. For example, `"x^3 + 117"` or `"1*x^3 + 117"`.
        * The polynomial indeterminate can be any single character, but must be consistent. For example, `"13x^3 + 117"` or `"13y^3 + 117"`.
        * Spaces are not required between terms. For example, `"13x^3 + 117"` or `"13x^3+117"`.
        * Any combination of the above rules is acceptable.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from its string representation.

        .. ipython:: python

            f = galois.Poly.Str("x^2 + 1"); f
            str(f)

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` from its string representation.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly.Str("13x^3 + 117", field=GF); f
            str(f)
        """
        if not isinstance(string, str):
            raise TypeError(f"Argument `string` be a string, not {type(string)}")

        degrees, coeffs = str_to_sparse_poly(string)

        return Poly.Degrees(degrees, coeffs, field=field)

    @classmethod
    def Int(cls, integer: int, field: Optional[Type[Array]] = None) -> Poly:
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its integer representation.

        :func:`~galois.Poly.Int` and :func:`~galois.Poly.__int__` are inverse operations.

        Parameters
        ----------
        integer
            The integer representation of the polynomial :math:`f(x)`.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which corresponds to :obj:`~galois.GF2`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        .. tab-set::

            .. tab-item:: Integer

                Construct a polynomial over :math:`\mathrm{GF}(2)` from its integer representation.

                .. ipython:: python

                    f = galois.Poly.Int(5); f
                    int(f)

                Construct a polynomial over :math:`\mathrm{GF}(3^5)` from its integer representation.

                .. ipython:: python

                    GF = galois.GF(3**5)
                    f = galois.Poly.Int(186535908, field=GF); f
                    int(f)
                    # The polynomial/integer equivalence
                    int(f) == 13*GF.order**3 + 117

            .. tab-item:: Binary string

                Construct a polynomial over :math:`\mathrm{GF}(2)` from its binary string.

                .. ipython:: python

                    f = galois.Poly.Int(int("0b1011", 2)); f
                    bin(f)

            .. tab-item:: Octal string

                Construct a polynomial over :math:`\mathrm{GF}(2^3)` from its octal string.

                .. ipython:: python

                    GF = galois.GF(2**3)
                    f = galois.Poly.Int(int("0o5034", 8), field=GF); f
                    oct(f)

            .. tab-item:: Hex string

                Construct a polynomial over :math:`\mathrm{GF}(2^8)` from its hexadecimal string.

                .. ipython:: python

                    GF = galois.GF(2**8)
                    f = galois.Poly.Int(int("0xf700a275", 16), field=GF); f
                    hex(f)
        """
        field = DEFAULT_FIELD_ARRAY if field is None else field
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Argument `integer` be an integer, not {type(integer)}")
        if not issubclass(field, Array):
            raise TypeError(f"Argument `field` must be a Galois field class, not {type(field)}.")
        if not integer >= 0:
            raise ValueError(f"Argument `integer` must be non-negative, not {integer}.")

        obj = object.__new__(cls)
        obj._integer = integer
        obj._field = field

        if field == DEFAULT_FIELD_ARRAY:
            obj._type = "binary"
        else:
            obj._type = "dense"

        return obj

    @classmethod
    def Degrees(
        cls,
        degrees: Union[Sequence[int], np.ndarray],
        coeffs: Optional[ArrayLike] = None,
        field: Optional[Type[Array]] = None
    ) -> Poly:
        r"""
        Constructs a polynomial over :math:`\mathrm{GF}(p^m)` from its non-zero degrees.

        Parameters
        ----------
        degrees
            The polynomial degrees with non-zero coefficients.
        coeffs
            The corresponding non-zero polynomial coefficients. The default is `None` which corresponds to all ones.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the coefficients are a :obj:`~galois.Array`, they won't be modified. If the coefficients are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(coeffs)`.
            * :obj:`~galois.Array` subclass: The coefficients are explicitly converted to this Galois field `field(coeffs)`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            galois.Poly.Degrees([3, 1, 0])

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Degrees([3, 1, 0], coeffs=[214, 73, 185], field=GF)
        """
        if not isinstance(degrees, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `degrees` must array-like, not {type(degrees)}.")
        if not isinstance(coeffs, (type(None), list, tuple, np.ndarray, Array)):
            raise TypeError(f"Argument `coeffs` must array-like, not {type(coeffs)}.")
        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {type(field)}.")

        degrees = np.array(degrees, dtype=np.int64)
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        coeffs, field = cls._convert_coeffs(coeffs, field)

        if not degrees.ndim <= 1:
            raise ValueError(f"Argument `degrees` can have dimension at most 1, not {degrees.ndim}.")
        if not degrees.size == np.unique(degrees).size:
            raise ValueError(f"Argument `degrees` must have unique entries, not {degrees}.")
        if not np.all(degrees >= 0):
            raise ValueError(f"Argument `degrees` must have non-negative values, not {degrees}.")
        if not coeffs.ndim <= 1:
            raise ValueError(f"Argument `coeffs` can have dimension at most 1, not {coeffs.ndim}.")
        if not degrees.size == coeffs.size:
            raise ValueError(f"Arguments `degrees` and `coeffs` must have the same length, not {degrees.size} and {coeffs.size}.")

        # Only keep non-zero coefficients
        idxs = np.nonzero(coeffs)
        degrees = degrees[idxs]
        coeffs = coeffs[idxs]

        # Sort by descending degrees
        idxs = np.argsort(degrees)[::-1]
        degrees = degrees[idxs]
        coeffs = coeffs[idxs]

        obj = object.__new__(cls)
        obj._nonzero_degrees = degrees
        obj._nonzero_coeffs = coeffs
        obj._field = field

        if obj._field == DEFAULT_FIELD_ARRAY:
            # Binary arithmetic is always faster than dense arithmetic
            obj._type = "binary"
            # Compute the integer value so we're ready for arithmetic computations
            int(obj)
        elif len(degrees) > 0 and len(degrees) < SPARSE_VS_DENSE_POLY_FACTOR*max(degrees):
            obj._type = "sparse"
        else:
            obj._type = "dense"

        return obj

    @classmethod
    def Roots(
        cls,
        roots: ArrayLike,
        multiplicities: Optional[Union[Sequence[int], np.ndarray]] = None,
        field: Optional[Type[Array]] = None
    ) -> Poly:
        r"""
        Constructs a monic polynomial over :math:`\mathrm{GF}(p^m)` from its roots.

        Parameters
        ----------
        roots
            The roots of the desired polynomial.
        multiplicities
            The corresponding root multiplicities. The default is `None` which corresponds to all ones.
        field
            The Galois field :math:`\mathrm{GF}(p^m)` the polynomial is over.

            * :obj:`None` (default): If the roots are a :obj:`~galois.Array`, they won't be modified. If the roots are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(roots)`.
            * :obj:`~galois.Array` subclass: The roots are explicitly converted to this Galois field `field(roots)`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Notes
        -----
        The polynomial :math:`f(x)` with :math:`k` roots :math:`\{r_1, r_2, \dots, r_k\}` with multiplicities
        :math:`\{m_1, m_2, \dots, m_k\}` is

        .. math::
            f(x) &= (x - r_1)^{m_1} (x - r_2)^{m_2} \dots (x - r_k)^{m_k} \\
                 &= a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0

        with degree :math:`d = \sum_{i=1}^{k} m_i`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` from a list of its roots.

        .. ipython:: python

            roots = [0, 0, 1]
            f = galois.Poly.Roots(roots); f
            # Evaluate the polynomial at its roots
            f(roots)

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` from a list of its roots with specific multiplicities.

        .. ipython:: python

            GF = galois.GF(3**5)
            roots = [121, 198, 225]
            f = galois.Poly.Roots(roots, multiplicities=[1, 2, 1], field=GF); f
            # Evaluate the polynomial at its roots
            f(roots)
        """
        multiplicities = [1,]*len(roots) if multiplicities is None else multiplicities
        if not isinstance(roots, (tuple, list, np.ndarray, Array)):
            raise TypeError(f"Argument `roots` must be array-like, not {type(roots)}.")
        if not isinstance(multiplicities, (tuple, list, np.ndarray)):
            raise TypeError(f"Argument `multiplicities` must be array-like, not {type(multiplicities)}.")
        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {field}.")

        roots, field = cls._convert_coeffs(roots, field)

        roots = field(roots).flatten()
        if not len(roots) == len(multiplicities):
            raise ValueError(f"Arguments `roots` and `multiplicities` must have the same length, not {len(roots)} and {len(multiplicities)}.")

        poly = Poly.One(field=field)
        x = Poly.Identity(field=field)
        for root, multiplicity in zip(roots, multiplicities):
            poly *= (x - root)**multiplicity

        return poly

    ###############################################################################
    # Methods
    ###############################################################################

    def coefficients(
        self,
        size: Optional[int] = None,
        order: Literal["desc", "asc"] = "desc"
    ) -> Array:
        """
        Returns the polynomial coefficients in the order and size specified.

        Parameters
        ----------
        size
            The fixed size of the coefficient array. Zeros will be added for higher-order terms. This value must be
            at least `degree + 1` or a :obj:`ValueError` will be raised. The default is `None` which corresponds
            to `degree + 1`.

        order
            The order of the coefficient degrees, either descending (default) or ascending.

        Returns
        -------
        :
            An array of the polynomial coefficients with length `size`, either in descending order or ascending order.

        Notes
        -----
        This accessor is similar to the :obj:`coeffs` property, but it has more settings. By default, `Poly.coeffs == Poly.coefficients()`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            f.coeffs
            f.coefficients()
            # Return the coefficients in ascending order
            f.coefficients(order="asc")
            # Return the coefficients in ascending order with size 8
            f.coefficients(8, order="asc")
        """
        if not isinstance(size, (type(None), int, np.integer)):
            raise TypeError(f"Argument `size` must be an integer, not {type(size)}.")
        if not isinstance(order, str):
            raise TypeError(f"Argument `order` must be a str, not {type(order)}.")
        size = len(self) if size is None else size
        if not size >= len(self):
            raise ValueError(f"Argument `size` must be at least `degree + 1` which is {len(self)}, not {size}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order!r}.")

        coeffs = self.field.Zeros(size)
        coeffs[-len(self):] = self.coeffs
        if order == "asc":
            coeffs = np.flip(coeffs)

        return coeffs

    def reverse(self) -> Poly:
        r"""
        Returns the :math:`d`-th reversal :math:`x^d f(\frac{1}{x})` of the polynomial :math:`f(x)` with degree :math:`d`.

        Returns
        -------
        :
            The :math:`n`-th reversal :math:`x^n f(\frac{1}{x})`.

        Notes
        -----
        For a polynomial :math:`f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0` with degree :math:`d`, the :math:`d`-th
        reversal is equivalent to reversing the coefficients.

        .. math::
            \textrm{rev}_d f(x) = x^d f(x^{-1}) = a_0 x^d + a_{1} x^{d-1} + \dots + a_{d-1} x + a_d

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([5, 0, 3, 4], field=GF); f
            f.reverse()
        """
        if self._type == "sparse":
            return Poly.Degrees(self.degree - self._nonzero_degrees, self._nonzero_coeffs)
        else:
            return Poly(self.coeffs[::-1])

    @overload
    def roots(self, multiplicity: Literal[False]) -> Array:
        ...
    @overload
    def roots(self, multiplicity: Literal[True]) -> Tuple[Array, np.ndarray]:
        ...
    def roots(self, multiplicity=False):
        r"""
        Calculates the roots :math:`r` of the polynomial :math:`f(x)`, such that :math:`f(r) = 0`.

        Parameters
        ----------
        multiplicity
            Optionally return the multiplicity of each root. The default is `False` which only returns the unique
            roots.

        Returns
        -------
        :
            An array of roots of :math:`f(x)`. The roots are ordered in increasing order.
        :
            The multiplicity of each root. This is only returned if `multiplicity=True`.

        Notes
        -----
        This implementation uses Chien's search to find the roots :math:`\{r_1, r_2, \dots, r_k\}` of the degree-:math:`d`
        polynomial

        .. math::
            f(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \dots + a_1x + a_0,

        where :math:`k \le d`. Then, :math:`f(x)` can be factored as

        .. math::
            f(x) = (x - r_1)^{m_1} (x - r_2)^{m_2} \dots (x - r_k)^{m_k},

        where :math:`m_i` is the multiplicity of root :math:`r_i` and :math:`d = \sum_{i=1}^{k} m_i`.

        The Galois field elements can be represented as :math:`\mathrm{GF}(p^m) = \{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m-2}\}`,
        where :math:`\alpha` is a primitive element of :math:`\mathrm{GF}(p^m)`.

        :math:`0` is a root of :math:`f(x)` if :math:`a_0 = 0`. :math:`1` is a root of :math:`f(x)` if :math:`\sum_{j=0}^{d} a_j = 0`. The
        remaining elements of :math:`\mathrm{GF}(p^m)` are powers of :math:`\alpha`. The following equations calculate :math:`f(\alpha^i)`,
        where :math:`\alpha^i` is a root of :math:`f(x)` if :math:`f(\alpha^i) = 0`.

        .. math::
            f(\alpha^i) &= a_{d}(\alpha^i)^{d} + a_{d-1}(\alpha^i)^{d-1} + \dots + a_1(\alpha^i) + a_0 \\
                        &\overset{\Delta}{=} \lambda_{i,d} + \lambda_{i,d-1} + \dots + \lambda_{i,1} + \lambda_{i,0} \\
                        &= \sum_{j=0}^{d} \lambda_{i,j}

        The next power of :math:`\alpha` can be easily calculated from the previous calculation.

        .. math::
            f(\alpha^{i+1}) &= a_{d}(\alpha^{i+1})^{d} + a_{d-1}(\alpha^{i+1})^{d-1} + \dots + a_1(\alpha^{i+1}) + a_0 \\
                            &= a_{d}(\alpha^i)^{d}\alpha^d + a_{d-1}(\alpha^i)^{d-1}\alpha^{d-1} + \dots + a_1(\alpha^i)\alpha + a_0 \\
                            &= \lambda_{i,d}\alpha^d + \lambda_{i,d-1}\alpha^{d-1} + \dots + \lambda_{i,1}\alpha + \lambda_{i,0} \\
                            &= \sum_{j=0}^{d} \lambda_{i,j}\alpha^j

        Examples
        --------
        Find the roots of a polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            f = galois.Poly.Roots([1, 0], multiplicities=[7, 3]); f
            f.roots()
            f.roots(multiplicity=True)

        Find the roots of a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly.Roots([18, 227, 153], multiplicities=[5, 7, 3], field=GF); f
            f.roots()
            f.roots(multiplicity=True)
        """
        if not isinstance(multiplicity, bool):
            raise TypeError(f"Argument `multiplicity` must be a bool, not {type(multiplicity)}.")

        roots = self.field._poly_roots(self.nonzero_degrees, self.nonzero_coeffs)

        if not multiplicity:
            return roots
        else:
            multiplicities = np.array([self._root_multiplicity(root) for root in roots])
            return roots, multiplicities

    def _root_multiplicity(self, root):
        poly = self
        multiplicity = 1

        while True:
            # If the root is also a root of the derivative, then its a multiple root.
            poly = poly.derivative()

            if poly == 0:
                # Cannot test whether p'(root) = 0 because p'(x) = 0. We've exhausted the non-zero derivatives. For
                # any Galois field, taking `characteristic` derivatives results in p'(x) = 0. For a root with multiplicity
                # greater than the field's characteristic, we need factor to the polynomial. Here we factor out (x - root)^m,
                # where m is the current multiplicity.
                poly = self // (Poly([1, -root], field=self.field)**multiplicity)

            if poly(root) == 0:
                multiplicity += 1
            else:
                break

        return multiplicity

    def derivative(self, k: int = 1) -> Poly:
        r"""
        Computes the :math:`k`-th formal derivative :math:`\frac{d^k}{dx^k} f(x)` of the polynomial :math:`f(x)`.

        Parameters
        ----------
        k
            The number of derivatives to compute. 1 corresponds to :math:`p'(x)`, 2 corresponds to :math:`p''(x)`, etc.
            The default is 1.

        Returns
        -------
        :
            The :math:`k`-th formal derivative of the polynomial :math:`f(x)`.

        Notes
        -----
        For the polynomial

        .. math::
            f(x) = a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0

        the first formal derivative is defined as

        .. math::
            f'(x) = (d) \cdot a_{d} x^{d-1} + (d-1) \cdot a_{d-1} x^{d-2} + \dots + (2) \cdot a_{2} x + a_1

        where :math:`\cdot` represents scalar multiplication (repeated addition), not finite field multiplication.
        The exponent that is "brought down" and multiplied by the coefficient is an integer, not a finite field element.
        For example, :math:`3 \cdot a = a + a + a`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Formal_derivative

        Examples
        --------
        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            f = galois.Poly.Random(7); f
            f.derivative()
            # p derivatives of a polynomial, where p is the field's characteristic, will always result in 0
            f.derivative(GF.characteristic)

        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(7)`.

        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly.Random(11, field=GF); f
            f.derivative()
            f.derivative(2)
            f.derivative(3)
            # p derivatives of a polynomial, where p is the field's characteristic, will always result in 0
            f.derivative(GF.characteristic)

        Compute the derivatives of a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly.Random(7, field=GF); f
            f.derivative()
            f.derivative(2)
            # p derivatives of a polynomial, where p is the field's characteristic, will always result in 0
            f.derivative(GF.characteristic)
        """
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
        if not k > 0:
            raise ValueError(f"Argument `k` must be a positive integer, not {k}.")

        if 0 in self.nonzero_degrees:
            # Cut off the 0th degree
            degrees = self.nonzero_degrees[:-1] - 1
            coeffs = self.nonzero_coeffs[:-1] * self.nonzero_degrees[:-1]  # Scalar multiplication
        else:
            degrees = self.nonzero_degrees - 1
            coeffs = self.nonzero_coeffs * self.nonzero_degrees  # Scalar multiplication

        p_prime = Poly.Degrees(degrees, coeffs, field=self.field)

        k -= 1
        if k > 0:
            return p_prime.derivative(k)
        else:
            return p_prime

    ###############################################################################
    # Overridden dunder methods
    ###############################################################################

    def __repr__(self) -> str:
        """
        A representation of the polynomial and the finite field it's over.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            f
        """
        return f"Poly({self!s}, {self.field._name})"

    def __str__(self) -> str:
        """
        The string representation of the polynomial, without specifying the finite field it's over.

        :func:`~galois.Poly.Str` and :func:`~galois.Poly.__str__` are inverse operations.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            str(f)
            print(f)
        """
        if self._type == "sparse":
            return sparse_poly_to_str(self._nonzero_degrees, self._nonzero_coeffs)
        else:
            return poly_to_str(self.coeffs)

    def __index__(self) -> int:
        # Define __index__ to enable use of bin(), oct(), and hex()
        if not hasattr(self, "_integer"):
            if hasattr(self, "_coeffs"):
                self._integer = poly_to_integer(self._coeffs.tolist(), self._field.order)
            elif hasattr(self, "_nonzero_coeffs"):
                self._integer = sparse_poly_to_integer(self._nonzero_degrees.tolist(), self._nonzero_coeffs.tolist(), self._field.order)

        return self._integer

    def __int__(self) -> int:
        r"""
        The integer representation of the polynomial.

        :func:`~galois.Poly.Int` and :func:`~galois.Poly.__int__` are inverse operations.

        Notes
        -----
        For the polynomial :math:`f(x) =  a_d x^d + a_{d-1} x^{d-1} + \dots + a_1 x + a_0` over the field :math:`\mathrm{GF}(p^m)`,
        the integer representation is :math:`i = a_d (p^m)^{d} + a_{d-1} (p^m)^{d-1} + \dots + a_1 (p^m) + a_0` using integer arithmetic,
        not finite field arithmetic.

        Said differently, the polynomial coefficients :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}` are considered as the :math:`d` "digits" of a radix-:math:`p^m`
        value. The polynomial's integer representation is that value in decimal (radix-:math:`10`).

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            int(f)
            int(f) == 3*GF.order**3 + 5*GF.order**1 + 2*GF.order**0
        """
        return self.__index__()

    def __hash__(self):
        t = tuple([self.field.order,] + self.nonzero_degrees.tolist() + self.nonzero_coeffs.tolist())
        return hash(t)

    def __call__(
        self,
        x: Union[ElementLike, ArrayLike],
        field: Optional[Type[Array]] = None,
        elementwise: bool = True
    ) -> Array:
        r"""
        Evaluates the polynomial :math:`f(x)` at `x`.

        Parameters
        ----------
        x
            A finite field scalar or array to evaluate the polynomial at.
        field
            The Galois field to evaluate the polynomial over. The default is `None` which represents
            the polynomial's current field, i.e. :obj:`field`.
        elementwise
            Indicates whether to evaluate :math:`x` elementwise. The default is `True`. If `False` (only valid
            for square matrices), the polynomial indeterminate :math:`x` is exponentiated using matrix powers
            (repeated matrix multiplication).

        Returns
        -------
        :
            The result of the polynomial evaluation :math:`f(x)`. The resulting array has the same shape as `x`.

        Examples
        --------
        Create a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly([37, 123, 0, 201], field=GF); f

        Evaluate the polynomial elementwise at :math:`x`.

        .. ipython:: python

            x = GF([185, 218, 84, 163]); x
            f(x)
            # The equivalent calculation
            GF(37)*x**3 + GF(123)*x**2 + GF(201)

        Evaluate the polynomial at the square matrix :math:`X`.

        .. ipython:: python

            X = GF([[185, 218], [84, 163]]); X
            f(X, elementwise=False)
            # The equivalent calculation
            GF(37)*np.linalg.matrix_power(X,3) + GF(123)*np.linalg.matrix_power(X,2) + GF(201)*GF.Identity(2)

        :meta public:
        """
        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {type(field)}.")

        field = self.field if field is None else field
        coeffs = field(self.coeffs)
        x = field(x)

        if elementwise:
            return field._poly_evaluate(coeffs, x)
        else:
            if not (x.ndim == 2 and x.shape[0] == x.shape[1]):
                raise ValueError(f"Argument `x` must be a square matrix when evaluating the polynomial not elementwise, not have shape {x.shape}.")
            return field._poly_evaluate_matrix(coeffs, x)

    def __len__(self) -> int:
        """
        Returns the length of the coefficient array `Poly.degree + 1`.

        Returns
        -------
        :
            The length of the coefficient array.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly([37, 123, 0, 201], field=GF); f
            f.coeffs
            len(f)
            f.degree + 1
        """
        return self.degree + 1

    def __eq__(self, other: PolyLike) -> bool:
        r"""
        Determines if two polynomials are equal.

        Parameters
        ----------
        other
            The polynomial to compare against.

        Returns
        -------
        :
            `True` if the two polynomials have the same coefficients and are over the same finite field.

        Examples
        --------
        Compare two polynomials over the same field.

        .. ipython:: python

            a = galois.Poly([3, 0, 5], field=galois.GF(7)); a
            b = galois.Poly([3, 0, 5], field=galois.GF(7)); b
            a == b
            # They are still two distinct objects, however
            a is b

        Compare two polynomials with the same coefficients but over different fields.

        .. ipython:: python

            a = galois.Poly([3, 0, 5], field=galois.GF(7)); a
            b = galois.Poly([3, 0, 5], field=galois.GF(7**2)); b
            a == b

        Comparison with :obj:`~galois.typing.PolyLike` objects is allowed for convenience.

        .. ipython:: python

            GF = galois.GF(7)
            a = galois.Poly([3, 0, 2], field=GF); a
            a == GF([3, 0, 2])
            a == [3, 0, 2]
            a == "3x^2 + 2"
            a == 3*7**2 + 2
        """
        # Coerce to a polynomial object
        if not isinstance(other, (Poly, Array)):
            field = self.field
        else:
            field = None
        other = Poly._PolyLike(other, field=field)

        return self.field is other.field and np.array_equal(self.nonzero_degrees, other.nonzero_degrees) and np.array_equal(self.nonzero_coeffs, other.nonzero_coeffs)

    ###############################################################################
    # Arithmetic
    ###############################################################################

    def _check_input_is_poly(self, a):
        """
        Verify polynomial arithmetic operands are either galois.Poly or scalars in a finite field.
        """
        if isinstance(a, Poly):
            field = a.field
        elif isinstance(a, Array):
            if not a.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {a.size}.")
            field = type(a)
        else:
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {self.field._name}, not {type(a)}.")

        if not field is self.field:
            raise TypeError(f"Both polynomial operands must be over the same field, not {field._name} and {self.field._name}.")

    def _check_input_is_poly_or_int(self, a):
        """
        Verify polynomial arithmetic operands are either galois.Poly, scalars in a finite field, or an integer scalar.
        """
        if isinstance(a, (int)):
            pass
        else:
            self._check_input_is_poly(a)

    def _check_input_is_poly_or_none(self, a):
        """
        Verify polynomial arithmetic operands are either galois.Poly, scalars in a finite field, or None.
        """
        if isinstance(a, type(None)):
            pass
        else:
            self._check_input_is_poly(a)

    def _convert_to_coeffs(self, a: Union[Poly, Array, int]) -> Array:
        """
        Convert the polynomial or finite field scalar into a coefficient array.
        """
        if isinstance(a, Poly):
            return a.coeffs
        elif isinstance(a, int):
            # Scalar multiplication
            return np.atleast_1d(self.field(a % self.field._characteristic))
        else:
            return np.atleast_1d(a)

    def _convert_to_integer(self, a: Union[Poly, Array, int]) -> int:
        """
        Convert the polynomial or finite field scalar into its integer representation.
        """
        if isinstance(a, int):
            # Scalar multiplication
            return a % self.field._characteristic
        else:
            return int(a)

    def _convert_to_sparse_coeffs(self, a: Union[Poly, Array, int]) -> Tuple[np.ndarray, Array]:
        """
        Convert the polynomial or finite field scalar into its non-zero degrees and coefficients.
        """
        if isinstance(a, Poly):
            return a.nonzero_degrees, a.nonzero_coeffs
        elif isinstance(a, int):
            return np.array([0]), np.atleast_1d(self.field(a % self.field._characteristic))
        else:
            return np.array([0]), np.atleast_1d(a)

    def __add__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(other)
            c = _binary.add(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(self)
            b_degrees, b_coeffs = self._convert_to_sparse_coeffs(other)
            c_degrees, c_coeffs = _sparse.add(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(other)
            c = _dense.add(a, b)
            return Poly(c, field=self.field)

    def __radd__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(other)
            b = self._convert_to_integer(self)
            c = _binary.add(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(other)
            b_degrees, b_coeffs = self._convert_to_sparse_coeffs(self)
            c_degrees, c_coeffs = _sparse.add(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(other)
            b = self._convert_to_coeffs(self)
            c = _dense.add(a, b)
            return Poly(c, field=self.field)

    def __neg__(self):
        if self._type == "binary":
            a = self._convert_to_integer(self)
            c = _binary.neg(a)
            return Poly.Int(c, field=self.field)
        elif self._type == "sparse":
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(self)
            c_degrees, c_coeffs = _sparse.neg(a_degrees, a_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            c = _dense.neg(a)
            return Poly(c, field=self.field)

    def __sub__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(other)
            c = _binary.sub(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(self)
            b_degrees, b_coeffs = self._convert_to_sparse_coeffs(other)
            c_degrees, c_coeffs = _sparse.sub(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(other)
            c = _dense.sub(a, b)
            return Poly(c, field=self.field)

    def __rsub__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(other)
            b = self._convert_to_integer(self)
            c = _binary.sub(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(other)
            b_degrees, b_coeffs = self._convert_to_sparse_coeffs(self)
            c_degrees, c_coeffs = _sparse.sub(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(other)
            b = self._convert_to_coeffs(self)
            c = _dense.sub(a, b)
            return Poly(c, field=self.field)

    def __mul__(self, other: Union[Poly, Array, int]) -> Poly:
        self._check_input_is_poly_or_int(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(other)
            c = _binary.mul(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(self)
            b_degrees, b_coeffs = self._convert_to_sparse_coeffs(other)
            c_degrees, c_coeffs = _sparse.mul(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(other)
            c = _dense.mul(a, b)
            return Poly(c, field=self.field)

    def __rmul__(self, other: Union[Poly, Array, int]) -> Poly:
        self._check_input_is_poly_or_int(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(other)
            b = self._convert_to_integer(self)
            c = _binary.mul(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = self._convert_to_sparse_coeffs(other)
            b_degrees, b_coeffs = self._convert_to_sparse_coeffs(self)
            c_degrees, c_coeffs = _sparse.mul(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = self._convert_to_coeffs(other)
            b = self._convert_to_coeffs(self)
            c = _dense.mul(a, b)
            return Poly(c, field=self.field)

    def __divmod__(self, other: Union[Poly, Array]) -> Tuple[Poly, Poly]:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(other)
            q, r = _binary.divmod(a, b)
            return Poly.Int(q, field=self.field), Poly.Int(r, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(other)
            q, r = _dense.divmod(a, b)
            return Poly(q, field=self.field), Poly(r, field=self.field)

    def __rdivmod__(self, other: Union[Poly, Array]) -> Tuple[Poly, Poly]:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(other)
            b = self._convert_to_integer(self)
            q, r = _binary.divmod(a, b)
            return Poly.Int(q, field=self.field), Poly.Int(r, field=self.field)
        else:
            a = self._convert_to_coeffs(other)
            b = self._convert_to_coeffs(self)
            q, r = _dense.divmod(a, b)
            return Poly(q, field=self.field), Poly(r, field=self.field)

    def __truediv__(self, other):
        raise NotImplementedError("Polynomial true division is not supported because fractional polynomials are not yet supported. Use floor division //, modulo %, and/or divmod() instead.")

    def __rtruediv__(self, other):
        raise NotImplementedError("Polynomial true division is not supported because fractional polynomials are not yet supported. Use floor division //, modulo %, and/or divmod() instead.")

    def __floordiv__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(other)
            q = _binary.floordiv(a, b)
            return Poly.Int(q, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(other)
            q = _dense.floordiv(a, b)
            return Poly(q, field=self.field)

    def __rfloordiv__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(other)
            b = self._convert_to_integer(self)
            q = _binary.floordiv(a, b)
            return Poly.Int(q, field=self.field)
        else:
            a = self._convert_to_coeffs(other)
            b = self._convert_to_coeffs(self)
            q = _dense.floordiv(a, b)
            return Poly(q, field=self.field)

    def __mod__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(other)
            r = _binary.mod(a, b)
            return Poly.Int(r, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(other)
            r = _dense.mod(a, b)
            return Poly(r, field=self.field)

    def __rmod__(self, other: Union[Poly, Array]) -> Poly:
        self._check_input_is_poly(other)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = self._convert_to_integer(other)
            b = self._convert_to_integer(self)
            r = _binary.mod(a, b)
            return Poly.Int(r, field=self.field)
        else:
            a = self._convert_to_coeffs(other)
            b = self._convert_to_coeffs(self)
            r = _dense.mod(a, b)
            return Poly(r, field=self.field)

    def __pow__(self, exponent: int, modulus: Optional[Poly] = None) -> Poly:
        self._check_input_is_poly_or_none(modulus)
        types = [getattr(self, "_type", None), getattr(modulus, "_type", None)]

        if not isinstance(exponent, (int, np.integer)):
            raise TypeError(f"For polynomial exponentiation, the second argument must be an int, not {exponent}.")
        if not exponent >= 0:
            raise ValueError(f"Can only exponentiate polynomials to non-negative integers, not {exponent}.")

        if "binary" in types:
            a = self._convert_to_integer(self)
            b = self._convert_to_integer(modulus) if modulus is not None else None
            q = _binary.pow(a, exponent, b)
            return Poly.Int(q, field=self.field)
        else:
            a = self._convert_to_coeffs(self)
            b = self._convert_to_coeffs(modulus) if modulus is not None else None
            q = _dense.pow(a, exponent, b)
            return Poly(q, field=self.field)

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self) -> Type["Array"]:
        """
        The :obj:`~galois.Array` subclass for the finite field the coefficients are over.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            a.field

        .. ipython:: python

            GF = galois.GF(2**8)
            b = galois.Poly.Random(5, field=GF); b
            b.field
        """
        return self._field

    @property
    def degree(self) -> int:
        """
        The degree of the polynomial. The degree of a polynomial is the highest degree with a non-zero coefficient.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.degree
        """
        if not hasattr(self, "_degree"):
            if hasattr(self, "_coeffs"):
                self._degree = self._coeffs.size - 1
            elif hasattr(self, "_nonzero_degrees"):
                if self._nonzero_degrees.size == 0:
                    self._degree = 0
                else:
                    self._degree = max(self._nonzero_degrees)
            elif hasattr(self, "_integer"):
                if self._integer == 0:
                    self._degree = 0
                else:
                    self._degree = integer_to_degree(self._integer, self._field.order)

        return self._degree

    @property
    def coeffs(self) -> "Array":
        """
        The coefficients of the polynomial in degree-descending order. The entries of :obj:`coeffs` are paired with :obj:`degrees`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.coeffs
        """
        if not hasattr(self, "_coeffs"):
            if hasattr(self, "_nonzero_coeffs"):
                degree = self.degree
                self._coeffs = self._field.Zeros(degree + 1)
                self._coeffs[degree - self._nonzero_degrees] = self._nonzero_coeffs
            elif hasattr(self, "_integer"):
                self._coeffs = self._field(integer_to_poly(self._integer, self._field.order))

        return self._coeffs.copy()

    @property
    def degrees(self) -> np.ndarray:
        """
        An array of the polynomial degrees in descending order. The entries of :obj:`coeffs` are paired with :obj:`degrees`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.degrees
        """
        if not hasattr(self, "_degrees"):
            self._degrees = np.arange(self.degree, -1, -1, dtype=int)

        return self._degrees.copy()

    @property
    def nonzero_coeffs(self) -> "Array":
        """
        The non-zero coefficients of the polynomial in degree-descending order. The entries of :obj:`nonzero_coeffs`
        are paired with :obj:`nonzero_degrees`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.nonzero_coeffs
        """
        if not hasattr(self, "_nonzero_coeffs"):
            coeffs = self.coeffs
            self._nonzero_coeffs = coeffs[np.nonzero(coeffs)]

        return self._nonzero_coeffs.copy()

    @property
    def nonzero_degrees(self) -> np.ndarray:
        """
        An array of the polynomial degrees that have non-zero coefficients in descending order. The entries of
        :obj:`nonzero_coeffs` are paired with :obj:`nonzero_degrees`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.nonzero_degrees
        """
        if not hasattr(self, "_nonzero_degrees"):
            degrees = self.degrees
            coeffs = self.coeffs
            self._nonzero_degrees = degrees[np.nonzero(coeffs)]

        return self._nonzero_degrees.copy()
