import numpy as np

from .conversion import integer_to_poly, poly_to_integer, poly_to_str
from .gf import GF
from .gf2 import GF2


class Poly:
    """
    Create a polynomial over a Galois field, :math:`p(x) \\in \\mathrm{GF}(q)[x]`.

    The polynomial :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0` has coefficients :math:`\\{a_{N-1}, \\dots, a_1, a_0\\}`
    in :math:`\\mathrm{GF}(q)`.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients :math:`\\{a_{N-1}, \\dots, a_1, a_0\\}` with type :obj:`galois.GF`, :obj:`numpy.ndarray`,
        :obj:`list`, or :obj:`tuple`. The first element is the highest-degree element if `order="desc"` or the first element is
        the 0-th degree element if `order="asc"`.
    field : galois.GF, optional
        The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`. If `coeffs`
        is a Galois field array, then that field is used and the `field` argument is ignored.
    order : str, optional
        The interpretation of the coefficient degrees, either `"desc"` (default) or `"asc"`. For `"desc"`,
        the first element of `coeffs` is the highest degree coefficient :math:`x^{N-1}`) and the last element is
        the 0-th degree element :math:`x^0`.

    Returns
    -------
    galois.Poly
        The polynomial :math:`p(x)`.

    Examples
    --------

    Create a polynomial over :math:`\\mathrm{GF}(2)[x]`.

    .. ipython:: python

        galois.Poly([1,0,1,1])
        galois.Poly.Degrees([3,1,0])

    Create a polynomial over :math:`\\mathrm{GF}(7)[x]`.

    .. ipython:: python

        GF7 = galois.GF_factory(7, 1)
        galois.Poly([4,0,3,0,0,2], field=GF7)
        galois.Poly.Degrees([5,3,0], coeffs=[4,3,2], field=GF7)

    Polynomial arithmetic using binary operators.

    .. ipython:: python

        a = galois.Poly([1,0,6,3], field=GF7); a
        b = galois.Poly([2,0,2], field=GF7); b

        a + b
        a - b
        # Compute the quotient of the polynomial division
        a / b
        # True division and floor division are equivalent
        a / b == a // b
        # Compute the remainder of the polynomial division
        a % b
    """

    def __init__(self, coeffs, field=None, order="desc"):
        if not (field is None or issubclass(field, GF)):
            raise TypeError(f"The Galois field `field` must be a subclass of GF, not {field}")
        self.order = order

        if isinstance(coeffs, GF) and field is None:
            self.coeffs = coeffs
        else:
            field = GF2 if field is None else field

            # Convert list or np.ndarray of integers into the specified `field`. Apply negation
            # operator to any negative integers. For instance, `coeffs=[1, -1]` represents
            # `x - 1` in GF2. However, the `-1` element does not exist in GF2, but the operation
            # `-1` (the additive inverse of the `1` element) does exist.
            c = np.array(coeffs, dtype=field.dtypes[-1])
            c = np.atleast_1d(c)
            assert c.ndim == 1, "Polynomials must only have one dimension"
            assert np.all(np.abs(c) < field.order)
            neg_idxs = np.where(c < 0)
            c = np.abs(c)
            c = field(c)
            c[neg_idxs] *= -1
            self.coeffs = c

    @classmethod
    def Zero(cls, field=GF2):
        """
        Create the zero polynomial, :math:`p(x) = 0 \\in \\mathrm{GF}(q)[x]`.

        Parameters
        ----------
        field : galois.GF, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct the zero polynomial over :math:`\\mathrm{GF}(2)[x]`.

        .. ipython:: python

            galois.Poly.Zero()

        Construct the zero polynomial over :math:`\\mathrm{GF}(7)[x]`.

        .. ipython:: python

            GF7 = galois.GF_factory(7, 1)
            galois.Poly.Zero(field=GF7)
        """
        return cls([0], field=field)

    @classmethod
    def One(cls, field=GF2):
        """
        Create the one polynomial, :math:`p(x) = 1 \\in \\mathrm{GF}(q)[x]`.

        Parameters
        ----------
        field : galois.GF, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct the one polynomial over :math:`\\mathrm{GF}(2)[x]`.

        .. ipython:: python

            galois.Poly.One()

        Construct the one polynomial over :math:`\\mathrm{GF}(7)[x]`.

        .. ipython:: python

            GF7 = galois.GF_factory(7, 1)
            galois.Poly.One(field=GF7)
        """
        return cls([1], field=field)

    @classmethod
    def Identity(cls, field=GF2):
        """
        Create the identity polynomial, :math:`p(x) = x \\in \\mathrm{GF}(q)[x]`.

        Parameters
        ----------
        field : galois.GF, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct the identity polynomial over :math:`\\mathrm{GF}(2)[x]`.

        .. ipython:: python

            galois.Poly.Identity()

        Construct the identity polynomial over :math:`\\mathrm{GF}(7)[x]`.

        .. ipython:: python

            GF7 = galois.GF_factory(7, 1)
            galois.Poly.Identity(field=GF7)
        """
        return cls([1, 0], field=field)

    @classmethod
    def Integer(cls, integer, field=GF2, order="desc"):
        """
        Create a polynomial over :math:`\\mathrm{GF}(q)[x]` from its integer representation.

        The integer value :math:`d` represents polynomial :math:`p(x) =  a_{N-1}x^{N-1} + \\dots + a_1x + a_0`
        over field :math:`\\mathrm{GF}(q)` if :math:`d = a_{N-1} q^{N-1} + \\dots + a_1 q + a_0` using integer arithmetic,
        not field arithmetic.

        Parameters
        ----------
        integer : int
            The integer representation of the polynomial :math:`p(x)`.
        field : galois.GF, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)[x]` from its integer representation.

        .. ipython:: python

            galois.Poly.Integer(5)

        Construct a polynomial over :math:`\\mathrm{GF}(7)[x]` from its integer representation.

        .. ipython:: python

            GF7 = galois.GF_factory(7, 1)
            galois.Poly.Integer(9, field=GF7)
        """
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Polynomial creation must have `integer` be an integer, not {type(integer)}")
        c = integer_to_poly(integer, field.order)
        if order == "desc":
            c = np.flip(c)
        return cls(c, field=field, order=order)

    @classmethod
    def Degrees(cls, degrees, coeffs=None, field=GF2):
        """
        Create a polynomial over :math:`\\mathrm{GF}(q)[x]` from its non-zero degrees.

        Parameters
        ----------
        degrees : list
            The polynomial degrees with non-zero coefficients.
        coeffs : array_like, optional
            List of corresponding non-zero coefficients. The default is `None` which corresponds to all one
            coefficients, i.e. `[1,]*len(degrees)`.
        roots : array_like
            List of roots in :math:`\\mathrm{GF}(q)` of the desired polynomial.
        field : galois.GF, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)[x]` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            galois.Poly.Degrees([3,1,0])

        Construct a polynomial over :math:`\\mathrm{GF}(7)[x]` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            GF7 = galois.GF_factory(7, 1)
            galois.Poly.Degrees([3,1,0], coeffs=[5,2,1], field=GF7)
        """
        if coeffs is None:
            coeffs = [1,]*len(degrees)
        assert len(coeffs) == len(degrees)
        degrees = np.array(degrees)
        assert np.issubdtype(degrees.dtype, np.integer) and np.all(degrees >= 0)
        degree = np.max(degrees)  # The degree of the polynomial
        all_coeffs = np.zeros(degree + 1, dtype=np.int64)
        all_coeffs[degree - degrees] = coeffs
        return cls(all_coeffs, field=field)

    @classmethod
    def Roots(cls, roots, field=GF2):
        """
        Create a monic polynomial in :math:`\\mathrm{GF}(q)[x]` from its roots.

        The polynomial :math:`p(x)` with roots :math:`\\{r_0, r_1, \\dots, r_{N-1}\\}` is:

        .. math::
            p(x) = (x - r_0) (x - r_1) \\dots (x - r_{N-1})

        Parameters
        ----------
        roots : array_like
            List of roots in :math:`\\mathrm{GF}(q)` of the desired polynomial.
        field : galois.GF, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)[x]` from a list of its roots.

        .. ipython:: python

            roots = [0,0,1]
            p = galois.Poly.Roots(roots); p
            p(roots)

        Construct a polynomial over :math:`\\mathrm{GF}(7)[x]` from a list of its roots.

        .. ipython:: python

            GF7 = galois.GF_factory(7, 1)
            roots = [2,6,1]
            p = galois.Poly.Roots(roots, field=GF7); p
            p(roots)
        """
        field._check_values(roots)  # pylint: disable=protected-access
        p = cls.One(field=field)
        for root in roots:
            p = p * cls([1, -int(root)], field=field)
        return p

    def __repr__(self):
        poly_str = poly_to_str(self.coeffs_asc)
        if self.field.degree == 1:
            order = "{}".format(self.field.characteristic)
        else:
            order = "{}^{}".format(self.field.characteristic, self.field.degree)
        return f"Poly({poly_str}, GF({order}))"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _verify_inputs(input1, input2):
        a = input1.coeffs

        # Verify type of second polynomial arithmetic argument and convert to a GF field type
        if isinstance(input2, Poly):
            b = input2.coeffs
        elif isinstance(input2, input1.field):
            b = input2
        elif isinstance(input2, int):
            assert 0 <= input2 < input1.field.order
            b = input1.field(input2)
        else:
            raise AssertionError("Can only perform polynomial arithmetic with Poly, GF, or int classes")
        assert type(a) is type(b), "Can only perform polynomial arthimetic between two polynomials with coefficients in the same field"

        return a, b

    # TODO: Speed this up with numba
    @staticmethod
    def divmod(dividend, divisor):
        # q(x)*b(x) + r(x) = a(x)
        a, b = Poly._verify_inputs(dividend, divisor)
        field = dividend.field
        a_degree = dividend.degree
        b_degree = divisor.degree

        if np.array_equal(a, [0]):
            quotient = Poly([0], field=field)
            remainder = Poly([0], field=field)
        elif a_degree < b_degree:
            quotient = Poly([0], field=field)
            remainder = Poly(a, field=field)
        else:
            deg_q = a_degree - b_degree
            deg_r = b_degree - 1
            aa = field(np.append(a, field.Zeros(deg_r + 1)))
            for i in range(0, deg_q + 1):
                if aa[i] != 0:
                    val = aa[i] / b[0]
                    aa[i:i+b.size] -= val*b
                else:
                    val = 0
                aa[i] = val
            quotient = Poly(aa[0:deg_q + 1], field=field)
            remainder = Poly(aa[deg_q + 1:deg_q + 1 + deg_r + 1], field=field)

        return quotient, remainder

    def __call__(self, x):
        return self.field._poly_eval(self.coeffs, x)

    def __add__(self, other):
        # c(x) = a(x) + b(x)
        a, b = Poly._verify_inputs(self, other)
        c = self.field.Zeros(max(a.size, b.size))
        c[-a.size:] = a
        c[-b.size:] += b
        return Poly(c, field=self.field)

    def __sub__(self, other):
        # c(x) = a(x) - b(x)
        a, b = Poly._verify_inputs(self, other)
        c = self.field.Zeros(max(a.size, b.size))
        c[-a.size:] = a
        c[-b.size:] -= b
        return Poly(c, field=self.field)

    def __mul__(self, other):
        # c(x) = a(x) * b(x)
        a, b = Poly._verify_inputs(self, other)
        a_degree = a.size - 1
        b_degree = b.size - 1
        c = self.field.Zeros(a_degree + b_degree + 1)
        for i in np.nonzero(b)[0]:
            c[i:i + a.size] += a*b[i]
        return Poly(c, field=self.field)

    def __neg__(self):
        return Poly(-self.coeffs, field=self.field)

    def __truediv__(self, other):
        return Poly.divmod(self, other)[0]

    def __floordiv__(self, other):
        return Poly.divmod(self, other)[0]

    def __mod__(self, other):
        return Poly.divmod(self, other)[1]

    def __pow__(self, other):
        assert isinstance(other, (int, np.integer)) and other >= 0
        # c(x) = a(x) ** b
        a = self
        b = other  # An integer
        if b == 0:
            return Poly([1], field=self.field)
        else:
            c = Poly(a.coeffs, field=self.field)
            for _ in range(b-1):
                c *= a
            return c

    def __eq__(self, other):
        return isinstance(other, Poly) and (self.field is other.field) and np.array_equal(self.coeffs_asc, other.coeffs_asc)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def order(self):
        """
        str: The interpretation of the ordering of the polynomial coefficients. `coeffs` are in degree-descending order
        if `order="desc"` and in degree-ascending order if `order="asc"`.
        """
        return self._order

    @order.setter
    def order(self, order):
        if order not in ["desc", "asc"]:
            raise ValueError(f"The coefficient degree ordering `order` must be either 'desc' or 'asc', not {order}")
        self._order = order

    @property
    def coeffs(self):
        """
        galois.GF: The polynomial coefficients as a Galois field array. Coefficients are :math:`\\{a_{N-1}, \\dots, a_1, a_0\\}` if `order="desc"` or
        :math:`\\{a_0, a_1, \\dots, a_{N-1}\\}` if `order="asc"`, where :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0`.
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        if not isinstance(coeffs, GF):
            raise TypeError(f"Galois field polynomials must have coefficients in a valid Galois field class (i.e. subclasses of GF), not {type(coeffs)}")
        if coeffs.ndim != 1:
            raise ValueError(f"Galois field polynomial coefficients must be arrays with dimension 1, not {coeffs.ndim}")
        idxs = np.nonzero(coeffs)[0]  # Non-zero indices

        if idxs.size > 0:
            # Trim leading non-zero powers
            coeffs = coeffs[:idxs[-1]+1] if self.order == "asc" else coeffs[idxs[0]:]
        else:
            # All coefficients are zero, only return the x^0 place
            field = coeffs.__class__
            coeffs = field([0])

        self._coeffs = coeffs

    @property
    def coeffs_asc(self):
        """
        galois.GF: The polynomial coefficients :math:`\\{a_0, a_1, \\dots, a_{N-1}\\}` as a Galois field array
        in degree-ascending order, where :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0`.
        """
        return self.coeffs if self.order == "asc" else np.flip(self.coeffs)

    @property
    def coeffs_desc(self):
        """
        galois.GF: The polynomial coefficients :math:`\\{a_{N-1}, \\dots, a_1, a_0\\}` as a Galois field array
        in degree-ascending order, where :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0`.
        """
        return self.coeffs if self.order == "desc" else np.flip(self.coeffs)

    @property
    def degree(self):
        """
        int: The degree of the polynomial, i.e. the highest degree with non-zero coefficient.
        """
        return self.coeffs.size - 1

    @property
    def field(self):
        """
        galois.GF: The finite field to which the coefficients belong.
        """
        return self.coeffs.__class__

    @property
    def integer(self):
        """
        int: The integer representation of the polynomial. For :math:`p(x) =  a_{N-1}x^{N-1} + \\dots + a_1x + a_0`
        with elements in :math:`\\mathrm{GF}(q)`, the integer representation is :math:`d = a_{N-1} q^{N-1} + \\dots + a_1 q + a_0`
        (using integer arithmetic, not field arithmetic) where :math:`q` is the field order.
        """
        c = self.coeffs_asc
        c = c.view(np.ndarray)  # We want to do integer math, not Galois field math
        return poly_to_integer(c, self.field.order)
