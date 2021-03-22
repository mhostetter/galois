import numpy as np

from .gf import GFBase
from .gf2 import GF2


class Poly:
    """
    A polynomial class with coefficients in any Galois field.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients of type Galois field array, `np.ndarray`, list, or tuple. The first
        element is the highest-degree element if `order="desc"` or the first element is the 0-th degree element
        if `order="asc"`.
    field : galois.GFBase, optional
        Optionally specify the field to which the coefficients belong. The default field is `galois.GF2`. If
        `coeffs` is a Galois field array, then that field is used and the `field` parameter is ignored.
    order : str, optional
        The interpretation of the coefficient degrees, either `"desc"` (default) or `"asc"`. For `"desc"`,
        the first element of `coeffs` is the highest degree coefficient (`x^(N-1)`) and the last element is
        the 0-th degree element (`x^0`).

    Examples
    --------

    Create polynomials over GF(2)

    .. ipython:: python

        # Construct a polynominal over GF(2)
        a = galois.Poly([1,0,1,1]); a

        # Construct the same polynomial by only specifying its non-zero coefficients
        b = galois.Poly.NonZero([1,1,1], [3,1,0]); b

    Create polynomials over GF(7)

    .. ipython:: python

        # Construct the GF(7) field
        GF = galois.GF_factory(7, 1)

        # Construct a polynominal over GF(7)
        galois.Poly([4,0,3,0,0,2], field=GF)

        # Construct the same polynomial by only specifying its non-zero coefficients
        galois.Poly.NonZero([4,3,2], [5,3,0], field=GF)

    Polynomial arithmetic

    .. ipython:: python

        a = galois.Poly([1,0,6,3], field=GF); a
        b = galois.Poly([2,0,2], field=GF); b

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
        if not (field is None or issubclass(field, GFBase)):
            raise TypeError(f"The Galois field `field` must be a subclass of GFBase, not {field}")
        self.order = order

        if isinstance(coeffs, GFBase) and field is None:
            self.coeffs = coeffs
        else:
            field = GF2 if field is None else field

            # Convert list or np.ndarray of integers into the specified `field`. Apply negation
            # operator to any negative integers. For instance, `coeffs=[1, -1]` represents
            # `x - 1` in GF2. However, the `-1` element does not exist in GF2, but the operation
            # `-1` (the additive inverse of the `1` element) does exist.
            c = np.array(coeffs)
            c = np.atleast_1d(c)
            assert c.ndim == 1, "Polynomials must only have one dimension"
            assert np.all(np.abs(c) < field.order)
            neg_idxs = np.where(c < 0)
            c = np.abs(c)
            c = field(c)
            c[neg_idxs] *= -1
            self.coeffs = c

    @classmethod
    def NonZero(cls, coeffs, degrees, field=GF2):
        """
        Examples
        --------

        .. ipython:: python

            # Construct a polynomial over GF2 only specifying the non-zero terms
            a = galois.Poly.NonZero([1,1,1], [3,1,0]); a
        """
        assert len(coeffs) == len(degrees)
        degrees = np.array(degrees)
        assert np.issubdtype(degrees.dtype, np.integer) and np.all(degrees >= 0)
        degree = np.max(degrees)  # The degree of the polynomial
        all_coeffs = np.zeros(degree + 1, dtype=np.int64)
        all_coeffs[degree - degrees] = coeffs
        return cls(all_coeffs, field=field)

    @classmethod
    def Decimal(cls, decimal, field=GF2, order="desc"):
        if not isinstance(decimal, (int, np.integer)):
            raise TypeError(f"Polynomial creation must have `decimal` be an integer, not {type(decimal)}")

        # NOTE: log_b(n) = log(n) / log(b)
        degree = int(np.floor(np.log(decimal) / np.log(field.order)))

        c = []  # Coefficients in descending order
        for d in range(degree, -1, -1):
            c += [decimal // field.order**d]
            decimal = decimal % field.order**d

        if order == "asc":
            c = np.flip(c)

        return cls(c, field=field, order=order)

    def __repr__(self):
        return "Poly({}, {})".format(self.str, self.field.__name__)

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
        # y[:] = p(x[:])
        x = self.field(x)
        scalar = x.shape == ()
        x = np.atleast_1d(x)
        y = self.field.Zeros(x.shape)
        y = self.field._numba_ufunc_poly_eval(self.coeffs, x, y)
        y = self.field(y)
        return y if not scalar else y[0]

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
        return isinstance(other, Poly) and (self.field is other.field) and (self.coeffs.shape == other.coeffs.shape) and np.all(self.coeffs_asc == other.coeffs_asc)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def order(self):
        """
        str: The interpretation of the ordering of the polynomial coefficients. `coeffs` are in exponent-descending order
        if `order="desc"` and in exponent-ascending order if `order="asc"`.
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
        galois.GF2, galois.GF2m, galois.GFp, galois.GFpm: The polynomial coefficients as a Galois field array. Coefficients are :math:`[a_{N-1}, \\dots, a_1, a_0]` if `order="desc"` or
        :math:`[a_0, a_1, \\dots, a_{N-1}]` if `order="asc"`, where :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0`.
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        if not isinstance(coeffs, GFBase):
            raise TypeError(f"Galois field polynomials must have coefficients in a valid Galois field class (i.e. subclasses of GFBase), not {type(coeffs)}")
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
        galois.GF2, galois.GF2m, galois.GFp, galois.GFpm: The polynomial coefficients :math:`[a_0, a_1, \\dots, a_{N-1}]` as a Galois field array
        in exponent-ascending order, where :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0`.
        """
        return self.coeffs if self.order == "asc" else np.flip(self.coeffs)

    @property
    def coeffs_desc(self):
        """
        galois.GF2, galois.GF2m, galois.GFp, galois.GFpm: The polynomial coefficients :math:`[a_{N-1}, \\dots, a_1, a_0]` as a Galois field array
        in exponent-ascending order, where :math:`p(x) = a_{N-1}x^{N-1} + \\dots + a_1x + a_0`.
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
        galois.GF2, galois.GF2m, galois.GFp, galois.GFpm: The finite field to which the coefficients belong.
        """
        return self.coeffs.__class__

    @property
    def decimal(self):
        """
        int: The integer representation of the polynomial. For :math:`p(x) =  a_{N-1}x^{N-1} + \\dots + a_1x + a_0`
        with elements in :math:`\\mathrm{GF}(q)`, the decimal representation is :math:`d = a_{N-1} q^{N-1} + \\dots + a_1 q + a_0`
        (using integer arithmetic, not field arithmetic) where :math:`q` is the field order.
        """
        c = self.coeffs_asc
        c = c.view(np.ndarray)  # We want to do integer math, not Galois field math
        decimal = 0
        for i in range(c.size):
            decimal += c[i] * self.field.order**i
        return decimal

    @property
    def str(self):
        """
        str: The string representation of the polynomial.
        """
        c = self.coeffs_asc

        x = []
        if self.degree >= 0 and c[0] != 0:
            x += ["{}".format(c[0])]
        if self.degree >= 1 and c[1] != 0:
            x += ["{}x".format(c[1] if c[1] != 1 else "")]
        if self.degree >= 2:
            idxs = np.nonzero(c[2:])[0]  # Indices with non-zeros coefficients
            x += ["{}x^{}".format(c[2+i] if c[2+i] != 1 else "", 2+i) for i in idxs]

        poly_str = " + ".join(x[::-1]) if x else "0"

        return poly_str
