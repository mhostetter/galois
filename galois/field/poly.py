import math

import numpy as np

from ..overrides import set_module

from .array import FieldArray
from .gf2 import GF2
from .poly_conversion import integer_to_poly, poly_to_integer, sparse_poly_to_integer, sparse_poly_to_str, str_to_sparse_poly

__all__ = ["Poly"]

# Values were obtained by running scripts/sparse_poly_performance_test.py
SPARSE_VS_BINARY_POLY_FACTOR = 0.00_05
SPARSE_VS_BINARY_POLY_MIN_COEFFS = int(1 / SPARSE_VS_BINARY_POLY_FACTOR)
SPARSE_VS_DENSE_POLY_FACTOR = 0.00_5
SPARSE_VS_DENSE_POLY_MIN_COEFFS = int(1 / SPARSE_VS_DENSE_POLY_FACTOR)


@set_module("galois")
class Poly:
    """
    Create a polynomial :math:`f(x)` over :math:`\\mathrm{GF}(p^m)`.

    The polynomial :math:`f(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0` has coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}`
    in :math:`\\mathrm{GF}(p^m)`.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}` with type :obj:`galois.FieldArray`, :obj:`numpy.ndarray`,
        :obj:`list`, or :obj:`tuple`. The first element is the highest-degree element if `order="desc"` or the first element is
        the 0-th degree element if `order="asc"`.
    field : galois.FieldArray, optional
        The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is `None` which represents :obj:`galois.GF2`. If `coeffs`
        is a Galois field array, then that field is used and the `field` argument is ignored.
    order : str, optional
        The interpretation of the coefficient degrees, either `"desc"` (default) or `"asc"`. For `"desc"`,
        the first element of `coeffs` is the highest degree coefficient :math:`x^{d}`) and the last element is
        the 0-th degree element :math:`x^0`.

    Returns
    -------
    galois.Poly
        The polynomial :math:`f(x)`.

    Examples
    --------
    Create a polynomial over :math:`\\mathrm{GF}(2)`.

    .. ipython:: python

        galois.Poly([1,0,1,1])
        galois.Poly.Degrees([3,1,0])

    Create a polynomial over :math:`\\mathrm{GF}(2^8)`.

    .. ipython:: python

        GF = galois.GF(2**8)
        galois.Poly([124,0,223,0,0,15], field=GF)

        # Alternate way of constructing the same polynomial
        galois.Poly.Degrees([5,3,0], coeffs=[124,223,15], field=GF)

    Polynomial arithmetic using binary operators.

    .. ipython:: python

        a = galois.Poly([117,0,63,37], field=GF); a
        b = galois.Poly([224,0,21], field=GF); b

        a + b
        a - b

        # Compute the quotient of the polynomial division
        a / b

        # True division and floor division are equivalent
        a / b == a // b

        # Compute the remainder of the polynomial division
        a % b

        # Compute both the quotient and remainder in one pass
        divmod(a, b)
    """

    # Increase my array priority so numpy will call my __radd__ instead of its own __add__
    __array_priority__ = 100

    def __new__(cls, coeffs, field=None, order="desc"):
        if not (field is None or issubclass(field, FieldArray)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")
        if not isinstance(coeffs, (int, np.integer, list, tuple, np.ndarray, FieldArray)):
            raise TypeError(f"Argument `coeffs` must 'array-like', not {type(coeffs)}.")
        if isinstance(coeffs, (FieldArray, np.ndarray)) and not coeffs.ndim <= 1:
            raise ValueError(f"Argument `coeffs` can have dimension at most 1, not {coeffs.ndim}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order}.")

        if isinstance(coeffs, (int, np.integer)):
            coeffs = [coeffs,]  # Ensure it's iterable
        if isinstance(coeffs, (FieldArray, np.ndarray)):
            coeffs = np.atleast_1d(coeffs)

        if order == "asc":
            coeffs = coeffs[::-1]  # Ensure it's in descending-degree order

        coeffs, field = cls._convert_coeffs(coeffs, field)

        if field is GF2:
            if len(coeffs) >= SPARSE_VS_BINARY_POLY_MIN_COEFFS and np.count_nonzero(coeffs) <= SPARSE_VS_BINARY_POLY_FACTOR*len(coeffs):
                degrees = np.arange(coeffs.size - 1, -1, -1)
                return SparsePoly(degrees, coeffs, field=field)
            else:
                integer = poly_to_integer(coeffs, 2)
                return BinaryPoly(integer)
        else:
            if len(coeffs) >= SPARSE_VS_DENSE_POLY_MIN_COEFFS and np.count_nonzero(coeffs) <= SPARSE_VS_DENSE_POLY_FACTOR*len(coeffs):
                degrees = np.arange(coeffs.size - 1, -1, -1)
                return SparsePoly(degrees, coeffs, field=field)
            else:
                return DensePoly(coeffs, field=field)

    @classmethod
    def _convert_coeffs(cls, coeffs, field):
        if isinstance(coeffs, FieldArray) and field is None:
            # Use the field of the coefficients
            field = type(coeffs)
        else:
            # Convert coefficients to the specified field (or GF2 if unspecified)
            field = GF2 if field is None else field
            if isinstance(coeffs, np.ndarray):
                # Ensure coeffs is an iterable
                coeffs = coeffs.tolist()
            coeffs = field([int(-field(abs(c))) if c < 0 else c for c in coeffs])

        return coeffs, field

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zero(cls, field=GF2):
        """
        Constructs the zero polynomial :math:`f(x) = 0` over :math:`\\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct the zero polynomial over :math:`\\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Zero()

        Construct the zero polynomial over :math:`\\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Zero(field=GF)
        """
        return Poly([0], field=field)

    @classmethod
    def One(cls, field=GF2):
        """
        Constructs the one polynomial :math:`f(x) = 1` over :math:`\\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct the one polynomial over :math:`\\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.One()

        Construct the one polynomial over :math:`\\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.One(field=GF)
        """
        return Poly([1], field=field)

    @classmethod
    def Identity(cls, field=GF2):
        """
        Constructs the identity polynomial :math:`f(x) = x` over :math:`\\mathrm{GF}(p^m)`.

        Parameters
        ----------
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct the identity polynomial over :math:`\\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Identity()

        Construct the identity polynomial over :math:`\\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Identity(field=GF)
        """
        return Poly([1, 0], field=field)

    @classmethod
    def Random(cls, degree, field=GF2):
        """
        Constructs a random polynomial over :math:`\\mathrm{GF}(p^m)` with degree :math:`d`.

        Parameters
        ----------
        degree : int
            The degree of the polynomial.
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a random degree-:math:`5` polynomial over :math:`\\mathrm{GF}(2)`.

        .. ipython:: python

            galois.Poly.Random(5)

        Construct a random degree-:math:`5` polynomial over :math:`\\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Random(5, field=GF)
        """
        if not isinstance(degree, (int, np.integer)):
            raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
        if not degree >= 0:
            raise TypeError(f"Argument `degree` must be at least 0, not {degree}.")
        coeffs = field.Random(degree + 1)
        coeffs[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero
        return Poly(coeffs, field=field)

    @classmethod
    def Integer(cls, integer, field=GF2):
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(p^m)` from its integer representation.

        The integer value :math:`i` represents the polynomial :math:`f(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`
        over field :math:`\\mathrm{GF}(p^m)` if :math:`i = a_{d}(p^m)^{d} + a_{d-1}(p^m)^{d-1} + \\dots + a_1(p^m) + a_0` using integer arithmetic,
        not finite field arithmetic.

        Parameters
        ----------
        integer : int
            The integer representation of the polynomial :math:`f(x)`.
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)` from its integer representation.

        .. ipython:: python

            galois.Poly.Integer(5)

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)` from its integer representation.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Integer(13*256**3 + 117, field=GF)
        """
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Polynomial creation must have `integer` be an integer, not {type(integer)}")

        if field is GF2:
            # Explicitly create a binary poly
            return BinaryPoly(integer)
        else:
            coeffs = integer_to_poly(integer, field.order)
            return Poly(coeffs, field=field)

    @classmethod
    def String(cls, string, field=GF2):
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(p^m)` from its string representation.

        Parameters
        ----------
        string : str
            The string representation of the polynomial :math:`f(x)`.
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)` from its string representation.

        .. ipython:: python

            galois.Poly.String("x^2 + 1")

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)` from its string representation.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.String("13x^3 + 117", field=GF)
        """
        if not isinstance(string, str):
            raise TypeError(f"Polynomial creation must have `string` be an str, not {type(string)}")

        return Poly.Degrees(*str_to_sparse_poly(string), field=field)


    @classmethod
    def Degrees(cls, degrees, coeffs=None, field=None):  # pylint: disable=too-many-branches
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(p^m)` from its non-zero degrees.

        Parameters
        ----------
        degrees : list
            List of polynomial degrees with non-zero coefficients.
        coeffs : array_like, optional
            List of corresponding non-zero coefficients. The default is `None` which corresponds to all one
            coefficients, i.e. `[1,]*len(degrees)`.
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is`None` which represents :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            galois.Poly.Degrees([3,1,0])

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            GF = galois.GF(2**8)
            galois.Poly.Degrees([3,1,0], coeffs=[251,73,185], field=GF)
        """
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        if not isinstance(degrees, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `degrees` must 'array-like', not {type(degrees)}.")
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `coeffs` must 'array-like', not {type(coeffs)}.")
        if not len(degrees) == len(coeffs):
            raise ValueError(f"Arguments `degrees` and `coeffs` must have the same length, not {len(degrees)} and {len(coeffs)}.")
        if not all(degree >= 0 for degree in degrees):
            raise ValueError(f"Argument `degrees` must have non-negative values, not {degrees}.")

        if len(degrees) == 0:
            degrees = [0]
            coeffs = [0]
        dtype = np.int64 if max(degrees) <= np.iinfo(np.int64).max else np.object_
        degrees = np.array(degrees, dtype=dtype)
        coeffs, field = cls._convert_coeffs(coeffs, field)

        if field is GF2:
            if len(degrees) < SPARSE_VS_BINARY_POLY_FACTOR*max(degrees):
                # Explicitly create a sparse poly over GF(2)
                return SparsePoly(degrees, coeffs=coeffs, field=field)
            else:
                integer = sparse_poly_to_integer(degrees, coeffs, 2)
                return BinaryPoly(integer)
        else:
            if len(degrees) < SPARSE_VS_DENSE_POLY_FACTOR*max(degrees):
                # Explicitly create a sparse poly over GF(p^m)
                return SparsePoly(degrees, coeffs=coeffs, field=field)
            else:
                degree = max(degrees)  # The degree of the polynomial
                all_coeffs = type(coeffs).Zeros(degree + 1)
                all_coeffs[degree - degrees] = coeffs
                return DensePoly(all_coeffs)

    @classmethod
    def Roots(cls, roots, multiplicities=None, field=None):
        """
        Constructs a monic polynomial in :math:`\\mathrm{GF}(p^m)[x]` from its roots.

        The polynomial :math:`f(x)` with :math:`d` roots :math:`\\{r_0, r_1, \\dots, r_{d-1}\\}` is:

        .. math::
            f(x) &= (x - r_0) (x - r_1) \\dots (x - r_{d-1})

            f(x) &= a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0

        Parameters
        ----------
        roots : array_like
            List of roots in :math:`\\mathrm{GF}(p^m)` of the desired polynomial.
        multiplicities : array_like, optional
            List of multiplicity of each root. The default is `None` which corresponds to all ones.
        field : galois.FieldArray, optional
            The field :math:`\\mathrm{GF}(p^m)` the polynomial is over. The default is`None` which represents :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`f(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)` from a list of its roots.

        .. ipython:: python

            roots = [0, 0, 1]
            p = galois.Poly.Roots(roots); p
            p(roots)

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)` from a list of its roots.

        .. ipython:: python

            GF = galois.GF(2**8)
            roots = [121, 198, 225]
            p = galois.Poly.Roots(roots, field=GF); p
            p(roots)
        """
        if not (field is None or issubclass(field, FieldArray)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")
        if not isinstance(roots, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `roots` must 'array-like', not {type(roots)}.")
        if not isinstance(multiplicities, (type(None), list, tuple, np.ndarray)):
            raise TypeError(f"Argument `multiplicities` must 'array-like', not {type(multiplicities)}.")

        field = GF2 if field is None else field
        roots = field(roots).flatten().tolist()
        multiplicities = [1,]*len(roots) if multiplicities is None else multiplicities
        if not len(roots) == len(multiplicities):
            raise ValueError(f"Arguments `roots` and `multiplicities` must have the same length, not {len(roots)} and {len(multiplicities)}.")

        poly = Poly.One(field=field)
        for root, multiplicity in zip(roots, multiplicities):
            poly *= Poly([1, -int(root)], field=field)**multiplicity

        return poly

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        raise NotImplementedError

    def roots(self, multiplicity=False):
        """
        Calculates the roots :math:`r` of the polynomial :math:`f(x)`, such that :math:`f(r) = 0`.

        This implementation uses Chien's search to find the roots :math:`\\{r_0, r_1, \\dots, r_{k-1}\\}` of the degree-:math:`d`
        polynomial

        .. math::
            f(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0,

        where :math:`k \\le d`. Then, :math:`f(x)` can be factored as

        .. math::
            f(x) = (x - r_0)^{m_0} (x - r_1)^{m_1} \\dots (x - r_{k-1})^{m_{k-1}},

        where :math:`m_i` is the multiplicity of root :math:`r_i` and

        .. math::
            \\sum_{i=0}^{k-1} m_i = d.

        The Galois field elements can be represented as :math:`\\mathrm{GF}(p^m) = \\{0, 1, \\alpha, \\alpha^2, \\dots, \\alpha^{p^m-2}\\}`,
        where :math:`\\alpha` is a primitive element of :math:`\\mathrm{GF}(p^m)`.

        :math:`0` is a root of :math:`f(x)` if:

        .. math::
            a_0 = 0

        :math:`1` is a root of :math:`f(x)` if:

        .. math::
            \\sum_{j=0}^{d} a_j = 0

        The remaining elements of :math:`\\mathrm{GF}(p^m)` are powers of :math:`\\alpha`. The following
        equations calculate :math:`p(\\alpha^i)`, where :math:`\\alpha^i` is a root of :math:`f(x)` if :math:`p(\\alpha^i) = 0`.

        .. math::
            p(\\alpha^i) &= a_{d}(\\alpha^i)^{d} + a_{d-1}(\\alpha^i)^{d-1} + \\dots + a_1(\\alpha^i) + a_0

            p(\\alpha^i) &\\overset{\\Delta}{=} \\lambda_{i,d} + \\lambda_{i,d-1} + \\dots + \\lambda_{i,1} + \\lambda_{i,0}

            p(\\alpha^i) &= \\sum_{j=0}^{d} \\lambda_{i,j}

        The next power of :math:`\\alpha` can be easily calculated from the previous calculation.

        .. math::
            p(\\alpha^{i+1}) &= a_{d}(\\alpha^{i+1})^{d} + a_{d-1}(\\alpha^{i+1})^{d-1} + \\dots + a_1(\\alpha^{i+1}) + a_0

            p(\\alpha^{i+1}) &= a_{d}(\\alpha^i)^{d}\\alpha^d + a_{d-1}(\\alpha^i)^{d-1}\\alpha^{d-1} + \\dots + a_1(\\alpha^i)\\alpha + a_0

            p(\\alpha^{i+1}) &= \\lambda_{i,d}\\alpha^d + \\lambda_{i,d-1}\\alpha^{d-1} + \\dots + \\lambda_{i,1}\\alpha + \\lambda_{i,0}

            p(\\alpha^{i+1}) &= \\sum_{j=0}^{d} \\lambda_{i,j}\\alpha^j

        Parameters
        ----------
        multiplicity : bool, optional
            Optionally return the multiplicity of each root. The default is `False`, which only returns the unique
            roots.

        Returns
        -------
        galois.FieldArray
            Galois field array of roots of :math:`f(x)`.
        np.ndarray
            The multiplicity of each root. Only returned if `multiplicity=True`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Chien_search

        Examples
        --------
        Find the roots of a polynomial over :math:`\\mathrm{GF}(2)`.

        .. ipython:: python

            p = galois.Poly.Roots([0,]*7 + [1,]*13); p
            p.roots()
            p.roots(multiplicity=True)

        Find the roots of a polynomial over :math:`\\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            p = galois.Poly.Roots([18,]*7 + [155,]*13 + [227,]*9, field=GF); p
            p.roots()
            p.roots(multiplicity=True)
        """
        lambda_vector = self.nonzero_coeffs
        alpha_vector = self.field.primitive_element ** self.nonzero_degrees
        roots = []
        multiplicities = []

        # Test if 0 is a root
        if 0 not in self.nonzero_degrees:
            root = 0
            roots.append(root)
            multiplicities.append(self._root_multiplicity(root) if multiplicity else 1)

        # Test if 1 is a root
        if np.sum(lambda_vector) == 0:
            root = 1
            roots.append(root)
            multiplicities.append(self._root_multiplicity(root) if multiplicity else 1)

        # Test if the powers of alpha are roots
        for i in range(1, self.field.order - 1):
            lambda_vector *= alpha_vector
            if np.sum(lambda_vector) == 0:
                root = int(self.field.primitive_element**i)
                roots.append(root)
                multiplicities.append(self._root_multiplicity(root) if multiplicity else 1)
            if sum(multiplicities) == self.degree:
                # We can exit early once we have `d` roots for a degree-d polynomial
                break

        idxs = np.argsort(roots)
        if not multiplicity:
            return self.field(roots)[idxs]
        else:
            return self.field(roots)[idxs], np.array(multiplicities)[idxs]

    def _root_multiplicity(self, root):
        zero = Poly.Zero(self.field)
        poly = self.copy()
        multiplicity = 1

        while True:
            # If the root is also a root of the derivative, then its a multiple root.
            poly = poly.derivative()

            if poly == zero:
                # Cannot test whether p'(root) = 0 because p'(x) = 0. We've exhausted the non-zero derivatives. For
                # any Galois field, taking `characteristic` derivatives results in p'(x) = 0. For a root with multiplicity
                # greater than the field's characteristic, we need factor the polynomial. Here we factor out (x - root)^m,
                # where m is the current multiplicity.
                poly = self.copy() // (Poly([1, -root], field=self.field)**multiplicity)

            if poly(root) == 0:
                multiplicity += 1
            else:
                break

        return multiplicity

    def derivative(self, k=1):
        """
        Computes the :math:`k`-th formal derivative :math:`\\frac{d^k}{dx^k} f(x)` of the polynomial :math:`f(x)`.

        For the polynomial

        .. math::
            f(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0

        the first formal derivative is defined as

        .. math::
            p'(x) = (d) \\cdot a_{d} x^{d-1} + (d-1) \\cdot a_{d-1} x^{d-2} + \\dots + (2) \\cdot a_{2} x + a_1

        where :math:`\\cdot` represents scalar multiplication (repeated addition), not finite field multiplication,
        e.g. :math:`3 \\cdot a = a + a + a`.

        Parameters
        ----------
        k : int, optional
            The number of derivatives to compute. 1 corresponds to :math:`p'(x)`, 2 corresponds to :math:`p''(x)`, etc.
            The default is 1.

        Returns
        -------
        galois.Poly
            The :math:`k`-th formal derivative of the polynomial :math:`f(x)`.

        References
        ----------
        * https://en.wikipedia.org/wiki/Formal_derivative

        Examples
        --------
        Compute the derivatives of a polynomial over :math:`\\mathrm{GF}(2)`.

        .. ipython:: python

            p = galois.Poly.Random(7); p
            p.derivative()

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(2)

        Compute the derivatives of a polynomial over :math:`\\mathrm{GF}(7)`.

        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly.Random(11, field=GF); p
            p.derivative()
            p.derivative(2)
            p.derivative(3)

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(7)

        Compute the derivatives of a polynomial over :math:`\\mathrm{GF}(2^8)`.

        .. ipython:: python

            GF = galois.GF(2**8)
            p = galois.Poly.Random(7, field=GF); p
            p.derivative()

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(2)
        """
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
        if not k > 0:
            raise ValueError(f"Argument `k` must be a positive integer, not {k}.")

        if 0 in self.nonzero_degrees:
            # Cut off the 0th degree
            degrees = self.nonzero_degrees[:-1] - 1
            coeffs = self.nonzero_coeffs[:-1] * self.nonzero_degrees[:-1]
        else:
            degrees = self.nonzero_degrees - 1
            coeffs = self.nonzero_coeffs * self.nonzero_degrees

        p_prime = Poly.Degrees(degrees, coeffs)

        k -= 1
        if k > 0:
            return p_prime.derivative(k)
        else:
            return p_prime

    def __repr__(self):
        return f"Poly({self.string}, {self.field.name})"

    ###############################################################################
    # Overridden dunder methods
    ###############################################################################

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        t = tuple([self.field.order,] + self.nonzero_degrees.tolist() + self.nonzero_coeffs.tolist())
        return hash(t)

    def __call__(self, x, field=None):
        """
        Evaluate the polynomial.

        Parameters
        ----------
        x : galois.FieldArray
            An array (or 0-dim array) of field element to evaluate the polynomial over.
        field : galois.FieldMeta, optional
            The Galois field to evaluate the polynomial over. The default is `None` which represents
            the polynomial's current field, i.e. :obj:`field`.

        Returns
        -------
        galois.FieldArray
            The result of the polynomial evaluation of the same shape as `x`.
        """
        if field is None:
            field = self.field
            coeffs = self.coeffs
        else:
            assert issubclass(field, FieldArray)
            coeffs = field(self.coeffs)
        if not isinstance(x, field):
            x = field(x)
        return field._poly_evaluate(coeffs, x)

    def _check_inputs_are_polys(self, a, b):
        if not isinstance(a, (Poly, self.field)):
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {b.field.name}, not {type(a)}.")
        if not isinstance(b, (Poly, self.field)):
            raise TypeError(f"Both operands must be a galois.Poly or a single element of its field {a.field.name}, not {type(b)}.")

        # Promote a single field element to a 0-degree polynomial
        if not isinstance(a, Poly):
            if not a.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {a.size}.")
            a = Poly(np.atleast_1d(a))
        if not isinstance(b, Poly):
            if not b.size == 1:
                raise ValueError(f"Arguments that are Galois field elements must have size 1 (equivalently a 0-degree polynomial), not size {b.size}.")
            b = Poly(np.atleast_1d(b))

        if not a.field is b.field:
            raise TypeError(f"Both polynomial operands must be over the same field, not {str(a.field)} and {str(b.field)}.")

        if isinstance(a, SparsePoly) or isinstance(b, SparsePoly):
            return SparsePoly, a, b
        elif isinstance(a, BinaryPoly) or isinstance(b, BinaryPoly):
            return BinaryPoly, a, b
        else:
            return DensePoly, a, b

    def __add__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._add(a, b)

    def __radd__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._add(b, a)

    def __sub__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._sub(a, b)

    def __rsub__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._sub(b, a)

    def __mul__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mul(a, b)

    def __rmul__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mul(b, a)

    def __divmod__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(a, b)

    def __rdivmod__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(b, a)

    def __truediv__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(a, b)[0]

    def __rtruediv__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(b, a)[0]

    def __floordiv__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(a, b)[0]

    def __rfloordiv__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._divmod(b, a)[0]

    def __mod__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mod(a, b)

    def __rmod__(self, other):
        cls, a, b = self._check_inputs_are_polys(self, other)
        return cls._mod(b, a)

    def __pow__(self, other):
        if not isinstance(other, (int, np.integer)):
            raise TypeError(f"For polynomial exponentiation, argument 1 must be of type int. Argument 1 is of type {type(other)}. Argument 1 = {other}.")
        if not other >= 0:
            raise ValueError(f"Can only exponentiate polynomials to non-negative integers, not {other}.")
        field, a, power = self.field, self, other

        # c(x) = a(x) ** power
        if power == 0:
            return Poly.One(field)

        c_square = a  # The "squaring" part
        c_mult = Poly.One(field)  # The "multiplicative" part

        while power > 1:
            if power % 2 == 0:
                c_square *= c_square
                power //= 2
            else:
                c_mult *= c_square
                power -= 1
        c = c_mult * c_square

        return c

    def __neg__(self):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, Poly):
            if other == 0:
                addendum = " If you are trying to compare against 0, use `galois.Poly.Zero(GF)` or `galois.Poly([0], field=GF)`."
            elif other == 1:
                addendum = " If you are trying to compare against 1, use `galois.Poly.One(GF)` or `galois.Poly([1], field=GF)`."
            else:
                addendum = ""
            raise TypeError(f"Can't compare Poly and non-Poly objects, {other} is not a Poly object.{addendum}")

        return self.field is other.field and np.array_equal(self.nonzero_degrees, other.nonzero_degrees) and np.array_equal(self.nonzero_coeffs, other.nonzero_coeffs)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _add(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _sub(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _mul(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _divmod(cls, a, b):
        raise NotImplementedError

    @classmethod
    def _mod(cls, a, b):
        raise NotImplementedError

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        """
        galois.FieldMeta: The Galois field array class to which the coefficients belong.

        Examples
        --------
        .. ipython:: python

            a = galois.Poly.Random(5); a
            a.field
            b = galois.Poly.Random(5, field=galois.GF(2**8)); b
            b.field
        """
        raise NotImplementedError

    @property
    def degree(self):
        """
        int: The degree of the polynomial, i.e. the highest degree with non-zero coefficient.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.degree
        """
        raise NotImplementedError

    @property
    def nonzero_degrees(self):
        """
        numpy.ndarray: An array of the polynomial degrees that have non-zero coefficients, in degree-descending order. The entries of
        :obj:`galois.Poly.nonzero_degrees` are paired with :obj:`galois.Poly.nonzero_coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.nonzero_degrees
        """
        raise NotImplementedError

    @property
    def nonzero_coeffs(self):
        """
        galois.FieldArray: The non-zero coefficients of the polynomial in degree-descending order. The entries of :obj:`galois.Poly.nonzero_degrees`
        are paired with :obj:`galois.Poly.nonzero_coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.nonzero_coeffs
        """
        raise NotImplementedError

    @property
    def degrees(self):
        """
        numpy.ndarray: An array of the polynomial degrees in degree-descending order. The entries of :obj:`galois.Poly.degrees`
        are paired with :obj:`galois.Poly.coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.degrees
        """
        raise NotImplementedError

    @property
    def coeffs(self):
        """
        galois.FieldArray: The coefficients of the polynomial in degree-descending order. The entries of :obj:`galois.Poly.degrees` are
        paired with :obj:`galois.Poly.coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.coeffs
        """
        raise NotImplementedError

    @property
    def integer(self):
        """
        int: The integer representation of the polynomial. For polynomial :math:`f(x) =  a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`
        with elements in :math:`a_k \\in \\mathrm{GF}(p^m)`, the integer representation is :math:`i = a_{d} (p^m)^{d} + a_{d-1} (p^m)^{d-1} + \\dots + a_1 (p^m) + a_0`
        (using integer arithmetic, not finite field arithmetic).

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.integer
            p.integer == 3*7**3 + 5*7**1 + 2*7**0
        """
        return sparse_poly_to_integer(self.nonzero_degrees, self.nonzero_coeffs, self.field.order)

    @property
    def string(self):
        """
        str: The string representation of the polynomial, without specifying the Galois field.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF); p
            p.string
        """
        return sparse_poly_to_str(self.nonzero_degrees, self.nonzero_coeffs)


class DensePoly(Poly):
    """
    Implementation of dense polynomials over Galois fields.
    """

    __slots__ = ["_coeffs"]

    def __new__(cls, coeffs, field=None):  # pylint: disable=signature-differs
        # Arguments aren't verified in Poly.__new__()
        obj = object.__new__(cls)
        obj._coeffs = coeffs

        if obj._coeffs.size > 1:
            # Remove leading zero coefficients
            idxs = np.nonzero(obj._coeffs)[0]
            if idxs.size > 0:
                obj._coeffs = obj._coeffs[idxs[0]:]
            else:
                obj._coeffs = obj._coeffs[-1]

        # Ensure the coefficient array isn't 0-dimension
        obj._coeffs = np.atleast_1d(obj._coeffs)

        return obj

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        return DensePoly(np.copy(self._coeffs))

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __neg__(self):
        return DensePoly(-self._coeffs)

    @classmethod
    def _add(cls, a, b):
        field = a.field

        # c(x) = a(x) + b(x)
        c_coeffs = field.Zeros(max(a.coeffs.size, b.coeffs.size))
        c_coeffs[-a.coeffs.size:] = a.coeffs
        c_coeffs[-b.coeffs.size:] += b.coeffs

        return Poly(c_coeffs)

    @classmethod
    def _sub(cls, a, b):
        field = a.field

        # c(x) = a(x) + b(x)
        c_coeffs = field.Zeros(max(a.coeffs.size, b.coeffs.size))
        c_coeffs[-a.coeffs.size:] = a.coeffs
        c_coeffs[-b.coeffs.size:] -= b.coeffs

        return Poly(c_coeffs)

    @classmethod
    def _mul(cls, a, b):
        # c(x) = a(x) * b(x)
        c_coeffs = np.convolve(a.coeffs, b.coeffs)

        return Poly(c_coeffs)

    @classmethod
    def _divmod(cls, a, b):
        field = a.field
        zero = Poly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            return Poly(a.coeffs // b.coeffs), zero

        elif a == zero:
            return zero, zero

        elif a.degree < b.degree:
            return zero, a.copy()

        else:
            q_coeffs, r_coeffs = field._poly_divmod(a.coeffs, b.coeffs)  # pylint: disable=protected-access
            return Poly(q_coeffs), Poly(r_coeffs)

    @classmethod
    def _mod(cls, a, b):
        return cls._divmod(a, b)[1]

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        return type(self._coeffs)

    @property
    def degree(self):
        return self._coeffs.size - 1

    @property
    def nonzero_degrees(self):
        return self.degree - np.nonzero(self._coeffs)[0]

    @property
    def nonzero_coeffs(self):
        return self._coeffs[np.nonzero(self._coeffs)[0]]

    @property
    def degrees(self):
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        return self._coeffs


class BinaryPoly(Poly):
    """
    Implementation of polynomials over GF(2).
    """

    __slots__ = ["_integer", "_coeffs"]

    def __new__(cls, integer):  # pylint: disable=signature-differs
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Argument `integer` must be an integer, not {type(integer)}.")
        if not integer >= 0:
            raise ValueError(f"Argument `integer` must be non-negative, not {integer}.")

        obj = object.__new__(cls)
        obj._integer = integer
        obj._coeffs = None  # Only compute these if requested

        return obj

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        return BinaryPoly(self._integer)

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __neg__(self):
        return self.copy()

    @classmethod
    def _add(cls, a, b):
        return BinaryPoly(a.integer ^ b.integer)

    @classmethod
    def _sub(cls, a, b):
        return BinaryPoly(a.integer ^ b.integer)

    @classmethod
    def _mul(cls, a, b):
        # Re-order operands such that a > b so the while loop has less loops
        a = a.integer
        b = b.integer
        if b > a:
            a, b = b, a

        c = 0
        while b > 0:
            if b & 0b1:
                c ^= a  # Add a(x) to c(x)
            b >>= 1  # Divide b(x) by x
            a <<= 1  # Multiply a(x) by x

        return BinaryPoly(c)

    @classmethod
    def _divmod(cls, a, b):
        deg_a = a.degree
        deg_q = a.degree - b.degree
        deg_r = b.degree - 1
        a = a.integer
        b = b.integer

        q = 0
        mask = 1 << deg_a
        for i in range(deg_q, -1, -1):
            q <<= 1
            if a & mask:
                a ^= b << i
                q ^= 1  # Set the LSB then left shift
            assert a & mask == 0
            mask >>= 1

        # q = a >> deg_r
        mask = (1 << (deg_r + 1)) - 1  # The last deg_r + 1 bits of a
        r = a & mask

        return BinaryPoly(q), BinaryPoly(r)

    @classmethod
    def _mod(cls, a, b):
        return cls._divmod(a, b)[1]

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        return GF2

    @property
    def degree(self):
        if self._integer == 0:
            return 0
        else:
            return int(math.floor(math.log2(self._integer)))

    @property
    def nonzero_degrees(self):
        return self.degree - np.nonzero(self.coeffs)[0]

    @property
    def nonzero_coeffs(self):
        return self.coeffs[np.nonzero(self.coeffs)[0]]

    @property
    def degrees(self):
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        if self._coeffs is None:
            binstr = bin(self._integer)[2:]
            self._coeffs = GF2([int(b) for b in binstr])
        return self._coeffs

    @property
    def integer(self):
        return self._integer


class SparsePoly(Poly):
    """
    Implementation of sparse polynomials over Galois fields.
    """

    __slots__ = ["_degrees", "_coeffs"]

    def __new__(cls, degrees, coeffs=None, field=None):  # pylint: disable=signature-differs
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        if not isinstance(degrees, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `degrees` must 'array-like', not {type(degrees)}.")
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `coeffs` must 'array-like', not {type(coeffs)}.")
        if not len(degrees) == len(coeffs):
            raise ValueError(f"Arguments `degrees` and `coeffs` must have the same length, not {len(degrees)} and {len(coeffs)}.")
        if not all(degree >= 0 for degree in degrees):
            raise ValueError(f"Argument `degrees` must have non-negative values, not {degrees}.")

        obj = object.__new__(cls)

        if isinstance(coeffs, FieldArray) and field is None:
            obj._degrees = np.array(degrees)
            obj._coeffs = coeffs
        else:
            field = GF2 if field is None else field
            if isinstance(coeffs, np.ndarray):
                # Ensure coeffs is an iterable
                coeffs = coeffs.tolist()
            obj._degrees = np.array(degrees)
            obj._coeffs = field([-field(abs(c)) if c < 0 else field(c) for c in coeffs])

        # Sort the degrees and coefficients in descending order
        idxs = np.argsort(degrees)[::-1]
        obj._degrees = obj._degrees[idxs]
        obj._coeffs = obj._coeffs[idxs]

        # Remove zero coefficients
        idxs = np.nonzero(obj._coeffs)[0]
        obj._degrees = obj._degrees[idxs]
        obj._coeffs = obj._coeffs[idxs]

        return obj

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        return SparsePoly(np.copy(self._degrees), np.copy(self._coeffs))

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __neg__(self):
        return SparsePoly(self._degrees, -self._coeffs)

    @classmethod
    def _add(cls, a, b):
        field = a.field

        # c(x) = a(x) + b(x)
        cc = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
        for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
            cc[b_degree] = cc.get(b_degree, field(0)) + b_coeff

        return Poly.Degrees(list(cc.keys()), list(cc.values()), field=field)

    @classmethod
    def _sub(cls, a, b):
        field = a.field

        # c(x) = a(x) - b(x)
        cc = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
        for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
            cc[b_degree] = cc.get(b_degree, field(0)) - b_coeff

        return Poly.Degrees(list(cc.keys()), list(cc.values()), field=field)

    @classmethod
    def _mul(cls, a, b):
        field = a.field

        # c(x) = a(x) * b(x)
        cc = {}
        for a_degree, a_coeff in zip(a.nonzero_degrees, a.nonzero_coeffs):
            for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
                cc[a_degree + b_degree] = cc.get(a_degree + b_degree, field(0)) + a_coeff*b_coeff

        return Poly.Degrees(list(cc.keys()), list(cc.values()), field=field)

    @classmethod
    def _divmod(cls, a, b):
        field = a.field
        zero = Poly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            q_degrees = a.nonzero_degrees
            q_coeffs = [a_coeff // b.coeffs[0] for a_coeff in a.nonzero_coeffs]
            return Poly.Degrees(q_degrees, q_coeffs, field=field), zero

        elif a == zero:
            return zero, zero

        elif a.degree < b.degree:
            return zero, a.copy()

        else:
            aa = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
            b_coeffs = b.coeffs

            q_degree = a.degree - b.degree
            r_degree = b.degree  # One larger than final remainder
            qq = {}
            r_coeffs = field.Zeros(r_degree + 1)

            # Preset remainder so we can rotate at the start of loop
            for i in range(0, b.degree):
                r_coeffs[1 + i] = aa.get(a.degree - i, 0)

            for i in range(0, q_degree + 1):
                r_coeffs = np.roll(r_coeffs, -1)
                r_coeffs[-1] = aa.get(a.degree - (i + b.degree), 0)

                if r_coeffs[0] > 0:
                    q = r_coeffs[0] // b_coeffs[0]
                    r_coeffs -= q*b_coeffs
                    qq[q_degree - i] = q

            return Poly.Degrees(list(qq.keys()), list(qq.values()), field=field), Poly(r_coeffs[1:])

    @classmethod
    def _mod(cls, a, b):
        field = a.field
        zero = Poly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            return zero

        elif a == zero:
            return zero

        elif a.degree < b.degree:
            return a.copy()

        else:
            aa = dict(zip(a.nonzero_degrees, a.nonzero_coeffs))
            b_coeffs = b.coeffs

            q_degree = a.degree - b.degree
            r_degree = b.degree  # One larger than final remainder
            r_coeffs = field.Zeros(r_degree + 1)

            # Preset remainder so we can rotate at the start of loop
            for i in range(0, b.degree):
                r_coeffs[1 + i] = aa.get(a.degree - i, 0)

            for i in range(0, q_degree + 1):
                r_coeffs = np.roll(r_coeffs, -1)
                r_coeffs[-1] = aa.get(a.degree - (i + b.degree), 0)

                if r_coeffs[0] > 0:
                    q = r_coeffs[0] // b_coeffs[0]
                    r_coeffs -= q*b_coeffs

            return Poly(r_coeffs[1:])

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self):
        return type(self._coeffs)

    @property
    def degree(self):
        return 0 if self._degrees.size == 0 else int(np.max(self._degrees))

    @property
    def nonzero_degrees(self):
        return self._degrees

    @property
    def nonzero_coeffs(self):
        return self._coeffs

    @property
    def degrees(self):
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        # Assemble a full list of coefficients, including zeros
        coeffs = self.field.Zeros(self.degree + 1)
        if self.nonzero_degrees.size > 0:
            coeffs[self.degree - self.nonzero_degrees] = self.nonzero_coeffs
        return coeffs
