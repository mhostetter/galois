"""
A module containing a class for univariate polynomials over finite fields.
"""
from __future__ import annotations

from typing import Tuple, List, Sequence, Optional, Union, Type, overload
from typing_extensions import Literal

import numpy as np

from .._domains import Array, _factory
from .._helper import export, verify_isinstance, verify_issubclass
from .._prime import factors
from ..typing import ElementLike, ArrayLike, PolyLike

from . import _binary, _dense, _sparse
from ._conversions import integer_to_poly, integer_to_degree, poly_to_integer, poly_to_str, sparse_poly_to_integer, sparse_poly_to_str, str_to_sparse_poly

# Values were obtained by running scripts/sparse_poly_performance_test.py
SPARSE_VS_DENSE_POLY_FACTOR = 0.00_125  # 1.25% density
SPARSE_VS_DENSE_POLY_MIN_COEFFS = int(1 / SPARSE_VS_DENSE_POLY_FACTOR)

# Functions that will be monkey-patched in _polys/__init__.py to the actual implementations. This is required because the
# implementations are in a different module that needs to import Poly. This monkey patching avoids the circular import.
# If you have a better solution, please open a GitHub issue or pull request.
GCD = lambda x, y: x


@export
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

    :group: polys
    """
    # pylint: disable=too-many-public-methods

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

            * :obj:`None` (default): If the coefficients are an :obj:`~galois.Array`, they won't be modified. If the coefficients
              are not explicitly in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using
              `galois.GF2(coeffs)`.
            * :obj:`~galois.Array` subclass: The coefficients are explicitly converted to this Galois field using `field(coeffs)`.

        order
            The interpretation of the coefficient degrees.

            - `"desc"` (default): The first element of `coeffs` is the highest degree coefficient, i.e. :math:`\{a_d, a_{d-1}, \dots, a_1, a_0\}`.
            - `"asc"`: The first element of `coeffs` is the lowest degree coefficient, i.e. :math:`\{a_0, a_1, \dots,  a_{d-1}, a_d\}`.
        """
        verify_isinstance(coeffs, (list, tuple, np.ndarray, Array))
        verify_issubclass(field, Array, optional=True)
        verify_isinstance(order, str)
        if isinstance(coeffs, (Array, np.ndarray)) and not coeffs.ndim <= 1:
            raise ValueError(f"Argument `coeffs` can have dimension at most 1, not {coeffs.ndim}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order!r}.")

        self._coeffs, self._field = _convert_coeffs(coeffs, field)

        if self._coeffs.ndim == 0:
            self._coeffs = np.atleast_1d(self._coeffs)
        if order == "asc":
            self._coeffs = np.flip(self._coeffs)  # Ensure it's in descending-degree order
        if self._coeffs[0] == 0:
            self._coeffs = np.trim_zeros(self._coeffs, "f")  # Remove leading zeros
        if self._coeffs.size == 0:
            self._coeffs = self._field([0])

        if self._field == _factory.DEFAULT_ARRAY:
            # Binary arithmetic is always faster than dense arithmetic
            self._type = "binary"
            # Compute the integer value so we're ready for arithmetic computations
            int(self)
        else:
            self._type = "dense"

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
    def Random(cls, degree: int, seed: Optional[Union[int, np.integer, np.random.Generator]] = None, field: Optional[Type[Array]] = None) -> Poly:
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
        Construct a random degree-5 polynomial over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Random(5)

        Construct a random degree-5 polynomial over :math:`\mathrm{GF}(3^5)` with a given seed. This produces repeatable results.

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
        verify_isinstance(degree, int)
        verify_isinstance(seed, (int, np.integer, np.random.Generator), optional=True)
        verify_issubclass(field, Array, optional=True)

        field = _factory.DEFAULT_ARRAY if field is None else field
        if seed is not None:
            if isinstance(seed, int) and not seed >= 0:
                raise ValueError(f"Argument `seed` must be non-negative, not {seed}.")
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
        verify_isinstance(string, str)

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

        Construct a polynomial over :math:`\mathrm{GF}(2)` from its binary string.

        .. ipython:: python

            f = galois.Poly.Int(int("0b1011", 2)); f
            bin(f)

        Construct a polynomial over :math:`\mathrm{GF}(2^3)` from its octal string.

        .. ipython:: python

            GF = galois.GF(2**3)
            f = galois.Poly.Int(int("0o5034", 8), field=GF); f
            oct(f)

        Construct a polynomial over :math:`\mathrm{GF}(2^8)` from its hexadecimal string.

        .. ipython:: python

            GF = galois.GF(2**8)
            f = galois.Poly.Int(int("0xf700a275", 16), field=GF); f
            hex(f)
        """
        verify_isinstance(integer, int)
        verify_issubclass(field, Array, optional=True)

        field = _factory.DEFAULT_ARRAY if field is None else field
        if not integer >= 0:
            raise ValueError(f"Argument `integer` must be non-negative, not {integer}.")

        obj = object.__new__(cls)
        obj._integer = integer
        obj._field = field

        if field == _factory.DEFAULT_ARRAY:
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

            * :obj:`None` (default): If the coefficients are an :obj:`~galois.Array`, they won't be modified. If the coefficients are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(coeffs)`.
            * :obj:`~galois.Array` subclass: The coefficients are explicitly converted to this Galois field using `field(coeffs)`.

        Returns
        -------
        :
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\mathrm{GF}(2)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            galois.Poly.Degrees([3, 1, 0])

        Construct a polynomial over :math:`\mathrm{GF}(3^5)` by specifying the degrees with non-zero coefficients
        and their coefficient values.

        .. ipython:: python

            GF = galois.GF(3**5)
            galois.Poly.Degrees([3, 1, 0], coeffs=[214, 73, 185], field=GF)
        """
        verify_isinstance(degrees, (list, tuple, np.ndarray))
        verify_isinstance(coeffs, (list, tuple, np.ndarray, Array), optional=True)
        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {type(field)}.")

        degrees = np.array(degrees, dtype=np.int64)
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        coeffs, field = _convert_coeffs(coeffs, field)

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

        if obj._field == _factory.DEFAULT_ARRAY:
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

            * :obj:`None` (default): If the roots are an :obj:`~galois.Array`, they won't be modified. If the roots are not explicitly
              in a Galois field, they are assumed to be from :math:`\mathrm{GF}(2)` and are converted using `galois.GF2(roots)`.
            * :obj:`~galois.Array` subclass: The roots are explicitly converted to this Galois field using `field(roots)`.

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
        verify_isinstance(roots, (tuple, list, np.ndarray, Array))
        verify_isinstance(multiplicities, (tuple, list, np.ndarray))
        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {field}.")

        roots, field = _convert_coeffs(roots, field)

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

        Return the coefficients in ascending order.

        .. ipython:: python

            f.coefficients(order="asc")

        Return the coefficients in ascending order with size 8.

        .. ipython:: python

            f.coefficients(8, order="asc")
        """
        verify_isinstance(size, int, optional=True)
        verify_isinstance(order, str)
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
    def roots(self, multiplicity: Literal[False] = False) -> Array:
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

        0 is a root of :math:`f(x)` if :math:`a_0 = 0`. 1 is a root of :math:`f(x)` if :math:`\sum_{j=0}^{d} a_j = 0`. The
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
        verify_isinstance(multiplicity, bool)

        roots = _dense.roots_jit(self.field)(self.nonzero_degrees, self.nonzero_coeffs)

        if not multiplicity:
            return roots
        else:
            multiplicities = np.array([_root_multiplicity(self, root) for root in roots])
            return roots, multiplicities

    def square_free_factors(self) -> Tuple[List[Poly], List[int]]:
        r"""
        Factors the monic polynomial :math:`f(x)` into a product of square-free polynomials.

        Returns
        -------
        :
            The list of non-constant, square-free polynomials :math:`h_j(x)` in the factorization.
        :
            The list of corresponding multiplicities :math:`j`.

        Raises
        ------
        ValueError
            If :math:`f(x)` is not monic or has degree 0.

        Notes
        -----
        The Square-Free Factorization algorithm factors :math:`f(x)` into a product of :math:`m` square-free polynomials :math:`h_j(x)`
        with multiplicity :math:`j`.

        .. math::
            f(x) = \prod_{j=1}^{m} h_j(x)^j

        Some :math:`h_j(x) = 1`, but those polynomials are not returned by this function.

        A complete polynomial factorization is implemented in :func:`~Poly.factors`.

        References
        ----------
        * Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.1.7.
        * Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

        Examples
        --------
        Suppose :math:`f(x) = x(x^3 + 2x + 4)(x^2 + 4x + 1)^3` over :math:`\mathrm{GF}(5)`. Each polynomial :math:`x`, :math:`x^3 + 2x + 4`,
        and :math:`x^2 + 4x + 1` are all irreducible over :math:`\mathrm{GF}(5)`.

        .. ipython:: python

            GF = galois.GF(5)
            a = galois.Poly([1,0], field=GF); a, a.is_irreducible()
            b = galois.Poly([1,0,2,4], field=GF); b, b.is_irreducible()
            c = galois.Poly([1,4,1], field=GF); c, c.is_irreducible()
            f = a * b * c**3; f

        The square-free factorization is :math:`\{x(x^3 + 2x + 4), x^2 + 4x + 1\}` with multiplicities :math:`\{1, 3\}`.

        .. ipython:: python

            f.square_free_factors()
            [a*b, c], [1, 3]
        """
        if not self.degree >= 1:
            raise ValueError(f"The polynomial must be non-constant, not {self}.")
        if not self.is_monic:
            raise ValueError(f"The polynomial must be monic, not {self}.")

        field = self.field
        p = field.characteristic
        one = Poly.One(field=field)

        factors_ = []
        multiplicities = []

        # w is the product (without multiplicity) of all factors of f that have multiplicity not divisible by p
        d = GCD(self, self.derivative())
        w = self // d

        # Step 1: Find all factors in w
        i = 1
        while w != one:
            y = GCD(w, d)
            z = w // y
            if z != one and i % p != 0:
                factors_.append(z)
                multiplicities.append(i)
            w = y
            d = d // y
            i = i + 1
        # d is now the product (with multiplicity) of the remaining factors of f

        # Step 2: Find all remaining factors (their multiplicities are divisible by p)
        if d != one:
            degrees = [degree // p for degree in d.nonzero_degrees]
            coeffs = d.nonzero_coeffs ** (field.characteristic**(field.degree - 1))  # The inverse Frobenius automorphism of the coefficients
            delta = Poly.Degrees(degrees, coeffs=coeffs, field=field)  # The p-th root of d(x)
            f, m = delta.square_free_factors()
            factors_.extend(f)
            multiplicities.extend([mi*p for mi in m])

        # Sort the factors in increasing-multiplicity order
        factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: item[1]))

        return list(factors_), list(multiplicities)

    def distinct_degree_factors(self) -> Tuple[List[Poly], List[int]]:
        r"""
        Factors the monic, square-free polynomial :math:`f(x)` into a product of polynomials whose irreducible factors all have
        the same degree.

        Returns
        -------
        :
            The list of polynomials :math:`f_i(x)` whose irreducible factors all have degree :math:`i`.
        :
            The list of corresponding distinct degrees :math:`i`.

        Raises
        ------
        ValueError
            If :math:`f(x)` is not monic, has degree 0, or is not square-free.

        Notes
        -----
        The Distinct-Degree Factorization algorithm factors a square-free polynomial :math:`f(x)` with degree :math:`d` into a product of :math:`d` polynomials
        :math:`f_i(x)`, where :math:`f_i(x)` is the product of all irreducible factors of :math:`f(x)` with degree :math:`i`.

        .. math::
            f(x) = \prod_{i=1}^{d} f_i(x)

        For example, suppose :math:`f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)` over :math:`\mathrm{GF}(2)`, then the distinct-degree
        factorization is

        .. math::
            f_1(x) &= x(x + 1) = x^2 + x \\
            f_2(x) &= x^2 + x + 1 \\
            f_3(x) &= (x^3 + x + 1)(x^3 + x^2 + 1) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 \\
            f_i(x) &= 1\ \textrm{for}\ i = 4, \dots, 10.

        Some :math:`f_i(x) = 1`, but those polynomials are not returned by this function. In this example, the function returns
        :math:`\{f_1(x), f_2(x), f_3(x)\}` and :math:`\{1, 2, 3\}`.

        The Distinct-Degree Factorization algorithm is often applied after the Square-Free Factorization algorithm, see :func:`~Poly.square_free_factors`.
        A complete polynomial factorization is implemented in :func:`~Poly.factors`.

        References
        ----------
        * Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.2.2.
        * Section 2.2 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

        Examples
        --------
        From the example in the notes, suppose :math:`f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)` over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            a = galois.Poly([1, 0]); a, a.is_irreducible()
            b = galois.Poly([1, 1]); b, b.is_irreducible()
            c = galois.Poly([1, 1, 1]); c, c.is_irreducible()
            d = galois.Poly([1, 0, 1, 1]); d, d.is_irreducible()
            e = galois.Poly([1, 1, 0, 1]); e, e.is_irreducible()
            f = a * b * c * d * e; f

        The distinct-degree factorization is :math:`\{x(x + 1), x^2 + x + 1, (x^3 + x + 1)(x^3 + x^2 + 1)\}` whose irreducible factors
        have degrees :math:`\{1, 2, 3\}`.

        .. ipython:: python

            f.distinct_degree_factors()
            [a*b, c, d*e], [1, 2, 3]
        """
        if not self.degree >= 1:
            raise ValueError(f"The polynomial must be non-constant, not {self}.")
        if not self.is_monic:
            raise ValueError(f"The polynomial must be monic, not {self}.")
        if not self.is_square_free():
            raise ValueError(f"The polynomial must be square-free, not {self}.")

        field = self.field
        q = field.order
        n = self.degree
        one = Poly.One(field=field)
        x = Poly.Identity(field=field)

        factors_ = []
        degrees = []

        a = self
        h = x

        l = 1
        while l <= n // 2 and a != one:
            h = pow(h, q, a)
            z = GCD(a, h - x)
            if z != one:
                factors_.append(z)
                degrees.append(l)
                a = a // z
                h = h % a
            l += 1

        if a != one:
            factors_.append(a)
            degrees.append(a.degree)

        return factors_, degrees

    def equal_degree_factors(self, degree: int) -> List[Poly]:
        r"""
        Factors the monic, square-free polynomial :math:`f(x)` of degree :math:`rd` into a product of :math:`r` irreducible factors with
        degree :math:`d`.

        Parameters
        ----------
        degree
            The degree :math:`d` of each irreducible factor of :math:`f(x)`.

        Returns
        -------
        :
            The list of :math:`r` irreducible factors :math:`\{g_1(x), \dots, g_r(x)\}` in lexicographically-increasing order.

        Raises
        ------
        ValueError
            If :math:`f(x)` is not monic, has degree 0, or is not square-free.

        Notes
        -----
        The Equal-Degree Factorization algorithm factors a square-free polynomial :math:`f(x)` with degree :math:`rd` into a product of :math:`r`
        irreducible polynomials each with degree :math:`d`. This function implements the Cantor-Zassenhaus algorithm, which is probabilistic.

        The Equal-Degree Factorization algorithm is often applied after the Distinct-Degree Factorization algorithm, see :func:`~Poly.distinct_degree_factors`.
        A complete polynomial factorization is implemented in :func:`~Poly.factors`.

        References
        ----------
        * Section 2.3 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf
        * Section 1 from https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec8.pdf

        Examples
        --------
        Factor a product of degree-1 irreducible polynomials over :math:`\mathrm{GF}(2)`.

        .. ipython:: python

            a = galois.Poly([1, 0]); a, a.is_irreducible()
            b = galois.Poly([1, 1]); b, b.is_irreducible()
            f = a * b; f
            f.equal_degree_factors(1)

        Factor a product of degree-3 irreducible polynomials over :math:`\mathrm{GF}(5)`.

        .. ipython:: python

            GF = galois.GF(5)
            a = galois.Poly([1, 0, 2, 1], field=GF); a, a.is_irreducible()
            b = galois.Poly([1, 4, 4, 4], field=GF); b, b.is_irreducible()
            f = a * b; f
            f.equal_degree_factors(3)
        """
        verify_isinstance(degree, int)
        if not self.degree >= 1:
            raise ValueError(f"The polynomial must be non-constant, not {self}.")
        if not self.is_monic:
            raise ValueError(f"The polynomial must be monic, not {self}.")
        if not self.degree % degree == 0:
            raise ValueError(f"Argument `degree` must be divide the degree of the polynomial, {degree} does not divide {self.degree}.")
        if not self.is_square_free():
            raise ValueError(f"The polynomial must be square-free, not {self}.")

        field = self.field
        q = field.order
        r = self.degree // degree
        one = Poly.One(field)

        factors_ = [self]
        while len(factors_) < r:
            h = Poly.Random(degree, field=field)
            g = GCD(self, h)
            if g == one:
                g = pow(h, (q**degree - 1)//2, self) - one
            i = 0
            for u in list(factors_):
                if u.degree <= degree:
                    continue
                d = GCD(g, u)
                if d not in [one, u]:
                    factors_.remove(u)
                    factors_.append(d)
                    factors_.append(u // d)
                i += 1

        # Sort the factors in lexicographically-increasing order
        factors_ = sorted(factors_, key=int)

        return factors_

    def factors(self) -> Tuple[List[Poly], List[int]]:
        r"""
        Computes the irreducible factors of the non-constant, monic polynomial :math:`f(x)`.

        Returns
        -------
        :
            Sorted list of irreducible factors :math:`\{g_1(x), g_2(x), \dots, g_k(x)\}` of :math:`f(x)` sorted in
            lexicographically-increasing order.
        :
            List of corresponding multiplicities :math:`\{e_1, e_2, \dots, e_k\}`.

        Raises
        ------
        ValueError
            If :math:`f(x)` is not monic or has degree 0.

        Notes
        -----
        This function factors a monic polynomial :math:`f(x)` into its :math:`k` irreducible factors such that
        :math:`f(x) = g_1(x)^{e_1} g_2(x)^{e_2} \dots g_k(x)^{e_k}`.

        Steps:

        1. Apply the Square-Free Factorization algorithm to factor the monic polynomial into square-free polynomials.
           See :func:`~Poly.square_free_factors`.
        2. Apply the Distinct-Degree Factorization algorithm to factor each square-free polynomial into a product of factors
           with the same degree. See :func:`~Poly.distinct_degree_factors`.
        3. Apply the Equal-Degree Factorization algorithm to factor the product of factors of equal degree into their irreducible
           factors. See :func:`~Poly.equal_degree_factors`.

        References
        ----------
        * Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.1.7.
        * Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

        Examples
        --------
        Generate irreducible polynomials over :math:`\mathrm{GF}(3)`.

        .. ipython:: python

            GF = galois.GF(3)
            g1 = galois.irreducible_poly(3, 3); g1
            g2 = galois.irreducible_poly(3, 4); g2
            g3 = galois.irreducible_poly(3, 5); g3

        Construct a composite polynomial.

        .. ipython:: python

            e1, e2, e3 = 5, 4, 3
            f = g1**e1 * g2**e2 * g3**e3; f

        Factor the polynomial into its irreducible factors over :math:`\mathrm{GF}(3)`.

        .. ipython:: python

            f.factors()
        """
        if not self.degree >= 1:
            raise ValueError(f"The polynomial must be non-constant, not {self}.")
        if not self.is_monic:
            raise ValueError(f"The polynomial must be monic, not {self}.")

        factors_, multiplicities = [], []

        # Step 1: Find all the square-free factors
        sf_factors, sf_multiplicities = self.square_free_factors()

        # Step 2: Find all the factors with distinct degree
        for sf_factor, sf_multiplicity in zip(sf_factors, sf_multiplicities):
            df_factors, df_degrees = sf_factor.distinct_degree_factors()

            # Step 3: Find all the irreducible factors with degree d
            for df_factor, df_degree in zip(df_factors, df_degrees):
                f = df_factor.equal_degree_factors(df_degree)
                factors_.extend(f)
                multiplicities.extend([sf_multiplicity,]*len(f))

        # Sort the factors in increasing-multiplicity order
        factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: int(item[0])))

        return list(factors_), list(multiplicities)

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
        verify_isinstance(k, int)
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

    def is_irreducible(self) -> bool:
        r"""
        Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(p^m)` is irreducible.

        Returns
        -------
        :
            `True` if the polynomial is irreducible.

        Important
        ---------
        This is a method, not a property, to indicate this test is computationally expensive.

        See Also
        --------
        irreducible_poly, irreducible_polys

        Notes
        -----
        A polynomial :math:`f(x) \in \mathrm{GF}(p^m)[x]` is *reducible* over :math:`\mathrm{GF}(p^m)` if it can
        be represented as :math:`f(x) = g(x) h(x)` for some :math:`g(x), h(x) \in \mathrm{GF}(p^m)[x]` of strictly
        lower degree. If :math:`f(x)` is not reducible, it is said to be *irreducible*. Since Galois fields are not algebraically
        closed, such irreducible polynomials exist.

        This function implements Rabin's irreducibility test. It says a degree-:math:`m` polynomial :math:`f(x)`
        over :math:`\mathrm{GF}(q)` for prime power :math:`q` is irreducible if and only if :math:`f(x)\ |\ (x^{q^m} - x)`
        and :math:`\textrm{gcd}(f(x),\ x^{q^{m_i}} - x) = 1` for :math:`1 \le i \le k`, where :math:`m_i = m/p_i` for
        the :math:`k` prime divisors :math:`p_i` of :math:`m`.

        References
        ----------
        * Rabin, M. Probabilistic algorithms in finite fields. SIAM Journal on Computing (1980), 273-280. https://apps.dtic.mil/sti/pdfs/ADA078416.pdf
        * Gao, S. and Panarino, D. Tests and constructions of irreducible polynomials over finite fields. https://www.math.clemson.edu/~sgao/papers/GP97a.pdf
        * Section 4.5.1 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf
        * https://en.wikipedia.org/wiki/Factorization_of_polynomials_over_finite_fields

        Examples
        --------
        .. ipython:: python

            # Conway polynomials are always irreducible (and primitive)
            f = galois.conway_poly(2, 5); f

            # f(x) has no roots in GF(2), a necessary but not sufficient condition of being irreducible
            f.roots()

            f.is_irreducible()

        .. ipython:: python

            g = galois.irreducible_poly(2**4, 2, method="random"); g
            h = galois.irreducible_poly(2**4, 3, method="random"); h
            f = g * h; f

            f.is_irreducible()
        """
        # pylint: disable=too-many-return-statements
        if self.degree == 0:
            # Over fields, f(x) = 0 is the zero element of GF(p^m)[x] and f(x) = c are the units of GF(p^m)[x]. Both the
            # zero element and the units are not irreducible over the polynomial ring GF(p^m)[x].
            return False

        if self.degree == 1:
            # f(x) = x + a (even a = 0) in any Galois field is irreducible
            return True

        if self.coeffs[-1] == 0:
            # g(x) = x can be factored, therefore it is not irreducible
            return False

        if self.field.order == 2 and self.nonzero_coeffs.size % 2 == 0:
            # Polynomials over GF(2) with degree at least 2 and an even number of terms satisfy f(1) = 0, hence
            # g(x) = x + 1 can be factored. Section 4.5.2 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf.
            return False

        field = self.field
        q = field.order
        m = self.degree
        x = Poly.Identity(field)

        primes, _ = factors(m)
        h0 = Poly.Identity(field)
        n0 = 0
        for ni in sorted([m // pi for pi in primes]):
            # The GCD of f(x) and (x^(q^(m/pi)) - x) must be 1 for f(x) to be irreducible, where pi are the prime factors of m
            hi = pow(h0, q**(ni - n0), self)
            g = GCD(self, hi - x)
            if g != 1:
                return False
            h0, n0 = hi, ni

        # f(x) must divide (x^(q^m) - x) to be irreducible
        h = pow(h0, q**(m - n0), self)
        g = (h - x) % self
        if g != 0:
            return False

        return True

    def is_primitive(self) -> bool:
        r"""
        Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is primitive.

        Returns
        -------
        :
            `True` if the polynomial is primitive.

        Important
        ---------
        This is a method, not a property, to indicate this test is computationally expensive.

        See Also
        --------
        primitive_poly, primitive_polys, conway_poly, matlab_primitive_poly

        Notes
        -----
        A degree-:math:`m` polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is *primitive* if it is irreducible and
        :math:`f(x)\ |\ (x^k - 1)` for :math:`k = q^m - 1` and no :math:`k` less than :math:`q^m - 1`.

        References
        ----------
        * Algorithm 4.77 from https://cacr.uwaterloo.ca/hac/about/chap4.pdf

        Examples
        --------
        All Conway polynomials are primitive.

        .. ipython:: python

            f = galois.conway_poly(2, 8); f
            f.is_primitive()

            f = galois.conway_poly(3, 5); f
            f.is_primitive()

        The irreducible polynomial of :math:`\mathrm{GF}(2^8)` for AES is not primitive.

        .. ipython:: python

            f = galois.Poly.Degrees([8, 4, 3, 1, 0]); f
            f.is_irreducible()
            f.is_primitive()
        """
        if self.degree == 0:
            # Over fields, f(x) = 0 is the zero element of GF(p^m)[x] and f(x) = c are the units of GF(p^m)[x]. Both the
            # zero element and the units are not irreducible over the polynomial ring GF(p^m)[x], and therefore cannot
            # be primitive.
            return False

        if self.field.order == 2 and self.degree == 1:
            # There is only one primitive polynomial in GF(2)
            return self == Poly([1, 1])

        if self.coeffs[-1] == 0:
            # A primitive polynomial cannot have zero constant term
            # TODO: Why isn't f(x) = x primitive? It's irreducible and passes the primitivity tests.
            return False

        if not self.is_irreducible():
            # A polynomial must be irreducible to be primitive
            return False

        field = self.field
        q = field.order
        m = self.degree
        one = Poly.One(field)

        primes, _ = factors(q**m - 1)
        x = Poly.Identity(field)
        for ki in sorted([(q**m - 1) // pi for pi in primes]):
            # f(x) must not divide (x^((q^m - 1)/pi) - 1) for f(x) to be primitive, where pi are the prime factors of q**m - 1
            h = pow(x, ki, self)
            g = (h - one) % self
            if g == 0:
                return False

        return True

    def is_square_free(self) -> bool:
        r"""
        Determines whether the polynomial :math:`f(x)` over :math:`\mathrm{GF}(q)` is square-free.

        Returns
        -------
        :
            `True` if the polynomial is square-free.

        Important
        ---------
        This is a method, not a property, to indicate this test is computationally expensive.

        Notes
        -----
        A square-free polynomial :math:`f(x)` has no irreducible factors with multiplicity greater than one. Therefore,
        its canonical factorization is

        .. math::
            f(x) = \prod_{i=1}^{k} g_i(x)^{e_i} = \prod_{i=1}^{k} g_i(x) .

        Examples
        --------
        Generate irreducible polynomials over :math:`\mathrm{GF}(3)`.

        .. ipython:: python

            GF = galois.GF(3)
            f1 = galois.irreducible_poly(3, 3); f1
            f2 = galois.irreducible_poly(3, 4); f2

        Determine if composite polynomials are square-free over :math:`\mathrm{GF}(3)`.

        .. ipython:: python

            (f1 * f2).is_square_free()
            (f1**2 * f2).is_square_free()
        """
        if not self.is_monic:
            self //= self.coeffs[0]

        # Constant polynomials are square-free
        if self.degree == 0:
            return True

        _, multiplicities = self.square_free_factors()

        return multiplicities == [1,]

    ###############################################################################
    # Overridden dunder methods
    ###############################################################################

    def __repr__(self) -> str:
        """
        A representation of the polynomial and the finite field it's over.

        Tip
        ---
        Use :func:`~galois.set_printoptions` to display the polynomial coefficients in degree-ascending order.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            f = galois.Poly([3, 0, 5, 2], field=GF); f
            repr(f)
        """
        return f"Poly({self!s}, {self.field.name})"

    def __str__(self) -> str:
        """
        The string representation of the polynomial, without specifying the finite field it's over.

        :func:`~galois.Poly.Str` and :func:`~galois.Poly.__str__` are inverse operations.

        Tip
        ---
        Use :func:`~galois.set_printoptions` to display the polynomial coefficients in degree-ascending order.

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
        value. The polynomial's integer representation is that value in decimal (radix-10).

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

    @overload
    def __call__(self, at: Union[ElementLike, ArrayLike], field: Optional[Type[Array]] = None, elementwise: bool = True) -> Array:
        ...
    @overload
    def __call__(self, at: Poly) -> Poly:
        ...
    def __call__(self, at, field=None, elementwise=True):
        r"""
        Evaluates the polynomial :math:`f(x)` at :math:`x_0` or the polynomial composition :math:`f(g(x))`.

        Parameters
        ----------
        at
            A finite field scalar or array :math:`x_0` to evaluate the polynomial at or the polynomial
            :math:`g(x)` to evaluate the polynomial composition :math:`f(g(x))`.
        field
            The Galois field to evaluate the polynomial over. The default is `None` which represents
            the polynomial's current field, i.e. :obj:`field`.
        elementwise
            Indicates whether to evaluate :math:`x_0` element-wise. The default is `True`. If `False` (only valid
            for square matrices), the polynomial indeterminate :math:`x` is exponentiated using matrix powers
            (repeated matrix multiplication).

        Returns
        -------
        :
            The result of the polynomial evaluation :math:`f(x_0)`. The resulting array has the same shape as :math:`x_0`.
            Or the polynomial composition :math:`f(g(x))`.

        Examples
        --------
        Create a polynomial over :math:`\mathrm{GF}(3^5)`.

        .. ipython:: python

            GF = galois.GF(3**5)
            f = galois.Poly([37, 123, 0, 201], field=GF); f

        Evaluate the polynomial element-wise at :math:`x_0`.

        .. ipython:: python

            x0 = GF([185, 218, 84, 163])
            f(x0)
            # The equivalent calculation
            GF(37)*x0**3 + GF(123)*x0**2 + GF(201)

        Evaluate the polynomial at the square matrix :math:`X_0`.

        .. ipython:: python

            X0 = GF([[185, 218], [84, 163]])
            # This is performed element-wise. Notice the values are equal to the vector x0.
            f(X0)
            f(X0, elementwise=False)
            # The equivalent calculation
            GF(37)*np.linalg.matrix_power(X0, 3) + GF(123)*np.linalg.matrix_power(X0, 2) + GF(201)*GF.Identity(2)

        Evaluate the polynomial :math:`f(x)` at the polynomial :math:`g(x)`.

        .. ipython:: python

            g = galois.Poly([55, 0, 1], field=GF); g
            f(g)
            # The equivalent calculation
            GF(37)*g**3 + GF(123)*g**2 + GF(201)
        """
        if isinstance(at, Poly):
            return _evaluate_poly(self, at)

        if not (field is None or issubclass(field, Array)):
            raise TypeError(f"Argument `field` must be a Array subclass, not {type(field)}.")

        field = self.field if field is None else field
        coeffs = field(self.coeffs)
        x = field(at)  # An array of finite field elements

        if elementwise:
            return _dense.evaluate_elementwise_jit(field)(coeffs, x)
        else:
            if not (x.ndim == 2 and x.shape[0] == x.shape[1]):
                raise ValueError(f"Argument `x` must be a square matrix when evaluating the polynomial not element-wise, not have shape {x.shape}.")
            return _evaluate_matrix(coeffs, x)

    def __len__(self) -> int:
        """
        Returns the length of the coefficient array, which is equivalent to `Poly.degree + 1`.

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

    def __add__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(other, self.field)
            c = _binary.add(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(self, self.field)
            b_degrees, b_coeffs = _convert_to_sparse_coeffs(other, self.field)
            c_degrees, c_coeffs = _sparse.add(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(other, self.field)
            c = _dense.add(a, b)
            return Poly(c, field=self.field)

    def __radd__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(other, self.field)
            b = _convert_to_integer(self, self.field)
            c = _binary.add(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(other, self.field)
            b_degrees, b_coeffs = _convert_to_sparse_coeffs(self, self.field)
            c_degrees, c_coeffs = _sparse.add(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(other, self.field)
            b = _convert_to_coeffs(self, self.field)
            c = _dense.add(a, b)
            return Poly(c, field=self.field)

    def __neg__(self):
        if self._type == "binary":
            a = _convert_to_integer(self, self.field)
            c = _binary.negative(a)
            return Poly.Int(c, field=self.field)
        elif self._type == "sparse":
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(self, self.field)
            c_degrees, c_coeffs = _sparse.negative(a_degrees, a_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            c = _dense.negative(a)
            return Poly(c, field=self.field)

    def __sub__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(other, self.field)
            c = _binary.subtract(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(self, self.field)
            b_degrees, b_coeffs = _convert_to_sparse_coeffs(other, self.field)
            c_degrees, c_coeffs = _sparse.subtract(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(other, self.field)
            c = _dense.subtract(a, b)
            return Poly(c, field=self.field)

    def __rsub__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(other, self.field)
            b = _convert_to_integer(self, self.field)
            c = _binary.subtract(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(other, self.field)
            b_degrees, b_coeffs = _convert_to_sparse_coeffs(self, self.field)
            c_degrees, c_coeffs = _sparse.subtract(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(other, self.field)
            b = _convert_to_coeffs(self, self.field)
            c = _dense.subtract(a, b)
            return Poly(c, field=self.field)

    def __mul__(self, other: Union[Poly, Array, int]) -> Poly:
        _check_input_is_poly_or_int(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(other, self.field)
            c = _binary.multiply(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(self, self.field)
            b_degrees, b_coeffs = _convert_to_sparse_coeffs(other, self.field)
            c_degrees, c_coeffs = _sparse.multiply(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(other, self.field)
            c = _dense.multiply(a, b)
            return Poly(c, field=self.field)

    def __rmul__(self, other: Union[Poly, Array, int]) -> Poly:
        _check_input_is_poly_or_int(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(other, self.field)
            b = _convert_to_integer(self, self.field)
            c = _binary.multiply(a, b)
            return Poly.Int(c, field=self.field)
        elif "sparse" in types:
            a_degrees, a_coeffs = _convert_to_sparse_coeffs(other, self.field)
            b_degrees, b_coeffs = _convert_to_sparse_coeffs(self, self.field)
            c_degrees, c_coeffs = _sparse.multiply(a_degrees, a_coeffs, b_degrees, b_coeffs)
            return Poly.Degrees(c_degrees, c_coeffs, field=self.field)
        else:
            a = _convert_to_coeffs(other, self.field)
            b = _convert_to_coeffs(self, self.field)
            c = _dense.multiply(a, b)
            return Poly(c, field=self.field)

    def __divmod__(self, other: Union[Poly, Array]) -> Tuple[Poly, Poly]:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(other, self.field)
            q, r = _binary.divmod(a, b)
            return Poly.Int(q, field=self.field), Poly.Int(r, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(other, self.field)
            q, r = _dense.divmod_jit(self.field)(a, b)
            return Poly(q, field=self.field), Poly(r, field=self.field)

    def __rdivmod__(self, other: Union[Poly, Array]) -> Tuple[Poly, Poly]:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(other, self.field)
            b = _convert_to_integer(self, self.field)
            q, r = _binary.divmod(a, b)
            return Poly.Int(q, field=self.field), Poly.Int(r, field=self.field)
        else:
            a = _convert_to_coeffs(other, self.field)
            b = _convert_to_coeffs(self, self.field)
            q, r = _dense.divmod_jit(self.field)(a, b)
            return Poly(q, field=self.field), Poly(r, field=self.field)

    def __truediv__(self, other):
        raise NotImplementedError("Polynomial true division is not supported because fractional polynomials are not yet supported. Use floor division //, modulo %, and/or divmod() instead.")

    def __rtruediv__(self, other):
        raise NotImplementedError("Polynomial true division is not supported because fractional polynomials are not yet supported. Use floor division //, modulo %, and/or divmod() instead.")

    def __floordiv__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(other, self.field)
            q = _binary.floordiv(a, b)
            return Poly.Int(q, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(other, self.field)
            q = _dense.floordiv_jit(self.field)(a, b)
            return Poly(q, field=self.field)

    def __rfloordiv__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(other, self.field)
            b = _convert_to_integer(self, self.field)
            q = _binary.floordiv(a, b)
            return Poly.Int(q, field=self.field)
        else:
            a = _convert_to_coeffs(other, self.field)
            b = _convert_to_coeffs(self, self.field)
            q = _dense.floordiv_jit(self.field)(a, b)
            return Poly(q, field=self.field)

    def __mod__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(other, self.field)
            r = _binary.mod(a, b)
            return Poly.Int(r, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(other, self.field)
            r = _dense.mod_jit(self.field)(a, b)
            return Poly(r, field=self.field)

    def __rmod__(self, other: Union[Poly, Array]) -> Poly:
        _check_input_is_poly(other, self.field)
        types = [getattr(self, "_type", None), getattr(other, "_type", None)]

        if "binary" in types:
            a = _convert_to_integer(other, self.field)
            b = _convert_to_integer(self, self.field)
            r = _binary.mod(a, b)
            return Poly.Int(r, field=self.field)
        else:
            a = _convert_to_coeffs(other, self.field)
            b = _convert_to_coeffs(self, self.field)
            r = _dense.mod_jit(self.field)(a, b)
            return Poly(r, field=self.field)

    def __pow__(self, exponent: int, modulus: Optional[Poly] = None) -> Poly:
        _check_input_is_poly_or_none(modulus, self.field)
        types = [getattr(self, "_type", None), getattr(modulus, "_type", None)]

        verify_isinstance(exponent, int)
        if not exponent >= 0:
            raise ValueError(f"Can only exponentiate polynomials to non-negative integers, not {exponent}.")

        if "binary" in types:
            a = _convert_to_integer(self, self.field)
            b = _convert_to_integer(modulus, self.field) if modulus is not None else None
            q = _binary.pow(a, exponent, b)
            return Poly.Int(q, field=self.field)
        else:
            a = _convert_to_coeffs(self, self.field)
            b = _convert_to_coeffs(modulus, self.field) if modulus is not None else None
            q = _dense.pow_jit(self.field)(a, exponent, b)
            return Poly(q, field=self.field)

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self) -> Type[Array]:
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
                    self._degree = int(max(self._nonzero_degrees))
            elif hasattr(self, "_integer"):
                if self._integer == 0:
                    self._degree = 0
                else:
                    self._degree = integer_to_degree(self._integer, self._field.order)

        return self._degree

    @property
    def coeffs(self) -> Array:
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
    def nonzero_coeffs(self) -> Array:
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

    @property
    def is_monic(self) -> bool:
        r"""
        Returns whether the polynomial is monic, meaning its highest-degree coefficient is one.

        Examples
        --------
        A monic polynomial over :math:`\mathrm{GF}(7)`.

        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([1, 0, 4, 5], field=GF); p
            p.is_monic

        A non-monic polynomial over :math:`\mathrm{GF}(7)`.

        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 4, 5], field=GF); p
            p.is_monic
        """
        return self.nonzero_coeffs[0] == 1


def _convert_coeffs(coeffs: ArrayLike, field: Optional[Type[Array]] = None) -> Tuple[Array, Type[Array]]:
    """
    Converts the coefficient-like input into a Galois field array based on the `field` keyword argument.
    """
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
            field = _factory.DEFAULT_ARRAY
        coeffs = np.array(coeffs, dtype=field.dtypes[-1])
        sign = np.sign(coeffs)
        coeffs = sign * field(np.abs(coeffs))

    return coeffs, field


def _root_multiplicity(poly: Poly, root: Array) -> int:
    """
    Determines the multiplicity of each root of the polynomial.

    TODO: Process all roots simultaneously.
    """
    field = poly.field
    multiplicity = 1

    p = poly
    while True:
        # If the root is also a root of the derivative, then its a multiple root.
        p = p.derivative()

        if p == 0:
            # Cannot test whether p'(root) = 0 because p'(x) = 0. We've exhausted the non-zero derivatives. For
            # any Galois field, taking `characteristic` derivatives results in p'(x) = 0. For a root with multiplicity
            # greater than the field's characteristic, we need factor to the polynomial. Here we factor out (x - root)^m,
            # where m is the current multiplicity.
            p = poly // (Poly([1, -root], field=field)**multiplicity)

        if p(root) == 0:
            multiplicity += 1
        else:
            break

    return multiplicity


def _evaluate_matrix(coeffs: Array, X: Array) -> Array:
    """
    Evaluates the polynomial f(x) at the square matrix X.
    """
    assert X.ndim == 2 and X.shape[0] == X.shape[1]
    field = type(coeffs)
    I = field.Identity(X.shape[0])

    y = coeffs[0]*I
    for j in range(1, coeffs.size):
        y = coeffs[j]*I + y @ X

    return y


def _evaluate_poly(f: Poly, g: Poly) -> Poly:
    """
    Evaluates the polynomial f(x) at the polynomial g(x). This is polynomial composition.
    """
    assert f.field is g.field
    coeffs = f.coeffs

    h = Poly(coeffs[0])
    for j in range(1, coeffs.size):
        h = coeffs[j] + h*g

    return h


def _check_input_is_poly(a: Union[Poly, Array], field: Type[Array]):
    """
    Verify polynomial arithmetic operands are either galois.Poly or scalars in a finite field.
    """
    if isinstance(a, Poly):
        a_field = a.field
    elif isinstance(a, Array):
        if not a.size == 1:
            raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {a.size}.")
        a_field = type(a)
    else:
        raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {field.name}, not {type(a)}.")

    if not a_field is field:
        raise TypeError(f"Both polynomial operands must be over the same field, not {a_field.name} and {field.name}.")


def _check_input_is_poly_or_int(a: Union[Poly, Array, int], field: Type[Array]):
    """
    Verify polynomial arithmetic operands are either galois.Poly, scalars in a finite field, or an integer scalar.
    """
    if isinstance(a, int):
        return
    _check_input_is_poly(a, field)


def _check_input_is_poly_or_none(a: Optional[Union[Poly, Array]], field: Type[Array]):
    """
    Verify polynomial arithmetic operands are either galois.Poly, scalars in a finite field, or None.
    """
    if isinstance(a, type(None)):
        return
    _check_input_is_poly(a, field)


def _convert_to_coeffs(a: Union[Poly, Array, int], field: Type[Array]) -> Array:
    """
    Convert the polynomial or finite field scalar into a coefficient array.
    """
    if isinstance(a, Poly):
        return a.coeffs
    elif isinstance(a, int):
        # Scalar multiplication
        return np.atleast_1d(field(a % field.characteristic))
    else:
        return np.atleast_1d(a)


def _convert_to_integer(a: Union[Poly, Array, int], field: Type[Array]) -> int:
    """
    Convert the polynomial or finite field scalar into its integer representation.
    """
    if isinstance(a, int):
        # Scalar multiplication
        return a % field.characteristic
    else:
        return int(a)


def _convert_to_sparse_coeffs(a: Union[Poly, Array, int], field: Type[Array]) -> Tuple[np.ndarray, Array]:
    """
    Convert the polynomial or finite field scalar into its non-zero degrees and coefficients.
    """
    if isinstance(a, Poly):
        return a.nonzero_degrees, a.nonzero_coeffs
    elif isinstance(a, int):
        # Scalar multiplication
        return np.array([0]), np.atleast_1d(field(a % field.characteristic))
    else:
        return np.array([0]), np.atleast_1d(a)
