from functools import partial
import numpy as np

from .gf import _GF
from .gf2 import GF2


@partial(np.vectorize, excluded=[0])
def _evaluate(coeffs, x0):
    result = coeffs[0]
    for i in range(1, coeffs.size):
        result = coeffs[i] + result*x0
    return result


class Poly:
    """
    asdf

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
        a = galois.Poly([4,0,3,0,0,2], field=GF); a

        # Construct the same polynomial by only specifying its non-zero coefficients
        b = galois.Poly.NonZero([4,3,2], [5,3,0], field=GF); b

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

    def __init__(self, coeffs, field=GF2):
        assert issubclass(field, _GF)
        if isinstance(coeffs, _GF):
            self.coeffs = coeffs
        else:
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
        all_coeffs = np.zeros(degree + 1, dtype=int)
        all_coeffs[degree - degrees] = coeffs
        return cls(all_coeffs, field=field)

    def __repr__(self):
        return "Poly({} , {})".format(self.str, type(self.coeffs).__name__)

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
            AssertionError("Can only perform polynomial arithmetic with Poly, GF, or int classes")
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

    def __call__(self, x0):
        x0 = self.field._verify_and_convert(x0)
        return _evaluate(self.coeffs, x0)

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
        return isinstance(other, Poly) and (self.field is other.field) and (self.coeffs.shape == other.coeffs.shape) and np.all(self.coeffs == other.coeffs)

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def coeffs(self):
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs):
        assert isinstance(coeffs, _GF), "Galois field polynomials must have coefficients belonging to a valid Galois field class (i.e. subclasses of _GF)"
        assert coeffs.ndim == 1, "Polynomial coefficients must be arrays of 1 dimension"
        idxs = np.nonzero(coeffs)[0]  # Non-zero indices
        if idxs.size > 0:
            # Trim leading non-zero powers
            coeffs = coeffs[idxs[0]:]
        else:
            # All coefficients are zero, only return the x^0 place
            field = coeffs.__class__
            coeffs = field([0])
        self._coeffs = coeffs

    @property
    def degree(self):
        """
        int: The degree of the polynomial, i.e. the highest degree with non-zero coefficient.
        """
        return self.coeffs.size - 1

    @property
    def field(self):
        """
        galois.GF2 or galois.GFp: The finite field to which the coefficients belong.
        """
        return self.coeffs.__class__

    @property
    def str(self):
        c = self.coeffs
        x = []
        if self.degree >= 0 and c[-1] != 0:
            x = ["{}".format(c[-1])] + x
        if self.degree >= 1 and c[-2] != 0:
            x = ["{}x".format(c[-2] if c[-2] != 1 else "")] + x
        if self.degree >= 2:
            idxs = np.nonzero(c[0:-2])[0]  # Indices with non-zeros coefficients
            x = ["{}x^{}".format(c[i] if c[i] != 1 else "", self.degree - i) for i in idxs] + x
        poly_str = " + ".join(x) if x else "0"
        return poly_str
