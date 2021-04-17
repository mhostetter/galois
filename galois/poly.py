import numpy as np

from .array import GFArray
from .gf2 import GF2
from .poly_conversion import integer_to_poly, sparse_poly_to_integer, sparse_poly_to_str


class Poly:
    """
    Create a polynomial :math:`p(x)` over :math:`\\mathrm{GF}(q)`.

    The polynomial :math:`p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0` has coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}`
    in :math:`\\mathrm{GF}(q)`.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}` with type :obj:`galois.GFArray`, :obj:`numpy.ndarray`,
        :obj:`list`, or :obj:`tuple`. The first element is the highest-degree element if `order="desc"` or the first element is
        the 0-th degree element if `order="asc"`.
    field : galois.GFArray, optional
        The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is `None` which represents :obj:`galois.GF2`. If `coeffs`
        is a Galois field array, then that field is used and the `field` argument is ignored.
    order : str, optional
        The interpretation of the coefficient degrees, either `"desc"` (default) or `"asc"`. For `"desc"`,
        the first element of `coeffs` is the highest degree coefficient :math:`x^{d}`) and the last element is
        the 0-th degree element :math:`x^0`.

    Returns
    -------
    galois.Poly
        The polynomial :math:`p(x)`.

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

    :special-members: __call__
    """

    __slots__ = ["_degrees", "_coeffs"]

    def __new__(cls, coeffs, field=None, order="desc"):
        if not isinstance(coeffs, int) and len(coeffs) > 1000 and np.count_nonzero(coeffs) < 0.001*len(coeffs):
            return SparsePoly.Coeffs(coeffs, field=field, order=order)
        else:
            return DensePoly(coeffs, field=field, order=order)

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zero(cls, field=GF2):
        """
        Constructs the zero polynomial :math:`p(x) = 0` over :math:`\\mathrm{GF}(q)`.

        Parameters
        ----------
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        return DensePoly.Zero(field=field)

    @classmethod
    def One(cls, field=GF2):
        """
        Constructs the one polynomial :math:`p(x) = 1` over :math:`\\mathrm{GF}(q)`.

        Parameters
        ----------
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        return DensePoly.One(field=field)

    @classmethod
    def Identity(cls, field=GF2):
        """
        Constructs the identity polynomial :math:`p(x) = x` over :math:`\\mathrm{GF}(q)`.

        Parameters
        ----------
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        return DensePoly.Identity(field=field)

    @classmethod
    def Random(cls, degree, field=GF2):
        """
        Constructs a random polynomial over :math:`\\mathrm{GF}(q)` with degree :math:`d`.

        Parameters
        ----------
        degree : int
            The degree of the polynomial.
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        return DensePoly.Random(degree, field=field)

    @classmethod
    def Integer(cls, integer, field=GF2):
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(q)` from its integer representation.

        The integer value :math:`i` represents the polynomial :math:`p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`
        over field :math:`\\mathrm{GF}(q)` if :math:`i = a_{d}q^{d} + a_{d-1}q^{d-1} + \\dots + a_1q + a_0` using integer arithmetic,
        not finite field arithmetic.

        Parameters
        ----------
        integer : int
            The integer representation of the polynomial :math:`p(x)`.
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        return DensePoly.Integer(integer, field=field)

    @classmethod
    def Degrees(cls, degrees, coeffs=None, field=None):
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(q)` from its non-zero degrees.

        Parameters
        ----------
        degrees : list
            List of polynomial degrees with non-zero coefficients.
        coeffs : array_like, optional
            List of corresponding non-zero coefficients. The default is `None` which corresponds to all one
            coefficients, i.e. `[1,]*len(degrees)`.
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is`None` which represents :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        degree = max(degrees)
        if len(degrees) < 0.001*degree:
            return SparsePoly.Degrees(degrees, coeffs=coeffs, field=field)
        else:
            return DensePoly.Degrees(degrees, coeffs=coeffs, field=field)

    @classmethod
    def Coeffs(cls, coeffs, field=None, order="desc"):
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(q)` from its coefficients.

        Alias of :obj:`galois.Poly` constructor.

        Parameters
        ----------
        coeffs : array_like
            List of polynomial coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}` with type :obj:`galois.GFArray`, :obj:`numpy.ndarray`,
            :obj:`list`, or :obj:`tuple`. The first element is the highest-degree element if `order="desc"` or the first element is
            the 0-th degree element if `order="asc"`.
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is `None` which represents :obj:`galois.GF2`. If `coeffs`
            is a Galois field array, then that field is used and the `field` argument is ignored.
        order : str, optional
            The interpretation of the coefficient degrees, either `"desc"` (default) or `"asc"`. For `"desc"`,
            the first element of `coeffs` is the highest degree coefficient :math:`x^{d}`) and the last element is
            the 0-th degree element :math:`x^0`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.
        """
        return DensePoly.Coeffs(coeffs, field=field, order=order)

    @classmethod
    def Roots(cls, roots, multiplicities=None, field=None):
        """
        Constructs a monic polynomial in :math:`\\mathrm{GF}(q)[x]` from its roots.

        The polynomial :math:`p(x)` with :math:`d` roots :math:`\\{r_0, r_1, \\dots, r_{d-1}\\}` is:

        .. math::
            p(x) &= (x - r_0) (x - r_1) \\dots (x - r_{d-1})

            p(x) &= a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0

        Parameters
        ----------
        roots : array_like
            List of roots in :math:`\\mathrm{GF}(q)` of the desired polynomial.
        multiplicities : array_like, optional
            List of multiplicity of each root. The default is `None` which corresponds to all ones.
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default is`None` which represents :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

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
        return DensePoly.Roots(roots, multiplicities=multiplicities, field=field)

    ###############################################################################
    # Methods
    ###############################################################################

    def copy(self):
        if isinstance(self, DensePoly):
            return DensePoly(self.coeffs)
        elif isinstance(self, SparsePoly):
            return SparsePoly(self.nonzero_degrees, self.nonzero_coeffs)
        else:
            raise NotImplementedError

    def roots(self, multiplicity=False):
        """
        Calculates the roots :math:`r` of the polynomial :math:`p(x)`, such that :math:`p(r) = 0`.

        This implementation uses Chien's search to find the roots :math:`\\{r_0, r_1, \\dots, r_{k-1}\\}` of the degree-:math:`d`
        polynomial

        .. math::
            p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0,

        where :math:`k \\le d`. Then, :math:`p(x)` can be factored as

        .. math::
            p(x) = (x - r_0)^{m_0} (x - r_1)^{m_1} \\dots (x - r_{k-1})^{m_{k-1}},

        where :math:`m_i` is the multiplicity of root :math:`r_i` and

        .. math::
            \\sum_{i=0}^{k-1} m_i = d.

        The Galois field elements can be represented as :math:`\\mathrm{GF}(q) = \\{0, 1, \\alpha, \\alpha^2, \\dots, \\alpha^{q-2}\\}`,
        where :math:`\\alpha` is a primitive element of :math:`\\mathrm{GF}(q)`.

        :math:`0` is a root of :math:`p(x)` if:

        .. math::
            a_0 = 0

        :math:`1` is a root of :math:`p(x)` if:

        .. math::
            \\sum_{j=0}^{d} a_j = 0

        The remaining elements of :math:`\\mathrm{GF}(q)` are powers of :math:`\\alpha`. The following
        equations calculate :math:`p(\\alpha^i)`, where :math:`\\alpha^i` is a root of :math:`p(x)` if :math:`p(\\alpha^i) = 0`.

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
        galois.GFArray
            Galois field array of roots of :math:`p(x)`.
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
        Computes the :math:`k`-th formal derivative :math:`\\frac{d^k}{dx^k} p(x)` of the polynomial :math:`p(x)`.

        For the polynomial

        .. math::
            p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0

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
            The :math:`k`-th formal derivative of the polynomial :math:`p(x)`.

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

        cls = self.__class__
        p_prime = cls.Degrees(degrees, coeffs)

        k -= 1
        if k > 0:
            return p_prime.derivative(k)
        else:
            return p_prime

    def __repr__(self):
        poly_str = sparse_poly_to_str(self.nonzero_degrees, self.nonzero_coeffs)
        return f"Poly({poly_str}, {self.field.name})"

    ###############################################################################
    # Overridden dunder methods
    ###############################################################################

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        t = tuple([self.field.order,] + self.nonzero_degrees.tolist() + self.nonzero_coeffs.tolist())
        return hash(t)

    @classmethod
    def _check_inputs_are_polys(cls, a, b):
        if not isinstance(a, Poly):
            raise TypeError(f"Both operands must be a galois.Poly, not {type(a)}.")
        if not isinstance(b, Poly):
            raise TypeError(f"Both operands must be a galois.Poly, not {type(b)}.")
        if not a.field is b.field:
            raise TypeError(f"Both polynomial operands must be over the same field, not {str(a.field)} and {str(b.field)}.")

        # If only one input is sparse, convert the other to sparse
        if isinstance(a, SparsePoly) and not isinstance(b, SparsePoly):
            b = SparsePoly.Coeffs(b.coeffs)
        if isinstance(b, SparsePoly) and not isinstance(a, SparsePoly):
            a = SparsePoly.Coeffs(a.coeffs)

        return a, b

    def __add__(self, other):
        a, b = self._check_inputs_are_polys(self, other)
        if isinstance(a, DensePoly):
            return DensePoly._add(a, b)
        else:
            return SparsePoly._add(a, b)

    def __sub__(self, other):
        a, b = self._check_inputs_are_polys(self, other)
        if isinstance(a, DensePoly):
            return DensePoly._sub(a, b)
        else:
            return SparsePoly._sub(a, b)

    def __mul__(self, other):
        a, b = self._check_inputs_are_polys(self, other)
        if isinstance(a, DensePoly):
            return DensePoly._mul(a, b)
        else:
            return SparsePoly._mul(a, b)

    def __divmod__(self, other):
        a, b = self._check_inputs_are_polys(self, other)
        if isinstance(a, DensePoly):
            return DensePoly._divmod(a, b)
        else:
            return SparsePoly._divmod(a, b)

    def __truediv__(self, other):
        return self.__divmod__(other)[0]

    def __floordiv__(self, other):
        return self.__divmod__(other)[0]

    def __mod__(self, other):
        a, b = self._check_inputs_are_polys(self, other)
        if isinstance(a, DensePoly):
            return DensePoly._mod(a, b)
        else:
            return SparsePoly._mod(a, b)

    def __pow__(self, other):
        if not isinstance(self, Poly):
            raise TypeError(f"For polynomial exponentiation, argument 0 must be of type galois.Poly. Argument 0 is of type {type(self)}. Argument 0 = {self}.")
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
        self._coeffs = -self._coeffs
        return self

    def __eq__(self, other):
        return isinstance(other, Poly) and self.field is other.field and np.array_equal(self.nonzero_degrees, other.nonzero_degrees) and np.array_equal(self.nonzero_coeffs, other.nonzero_coeffs)

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
        type: The Galois field to which the coefficients belong. The :obj:`galois.Poly.field` property is a
        subclass of :obj:`galois.GFArray`. This property is settable.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(2**8)

            # The primitive polynomial of the field GF(p^m) is degree-m over GF(p)[x]
            prim_poly = GF.irreducible_poly; prim_poly
            prim_poly.field

            # Convert the primitive polynomial from GF(p)[x] to GF(p^m)[x]
            prim_poly.field = GF; prim_poly

            # The primitive element alpha is a root of the primitive polynomial in GF(p^m)
            prim_poly(GF.primitive_element)
        """
        return type(self._coeffs)

    @field.setter
    def field(self, field):
        if not issubclass(field, GFArray):
            raise TypeError(f"Property `field` must be a subclass of galois.GFArray, not {field}.")
        self._coeffs = field(self._coeffs)

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
        # pylint: disable=no-member
        if isinstance(self, DensePoly):
            return self._coeffs.size - 1
        elif isinstance(self, SparsePoly):
            return 0 if self._degrees.size == 0 else int(np.max(self._degrees))
        else:
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
        # pylint: disable=no-member
        if isinstance(self, DensePoly):
            return self.degree - np.nonzero(self._coeffs)[0]
        elif isinstance(self, SparsePoly):
            return self._degrees
        else:
            raise NotImplementedError

    @property
    def nonzero_coeffs(self):
        """
        galois.GFArray: The non-zero coefficients of the polynomial in degree-descending order. The entries of :obj:`galois.Poly.nonzero_degrees`
        are paired with :obj:`galois.Poly.nonzero_coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.nonzero_coeffs
        """
        if isinstance(self, DensePoly):
            return self._coeffs[np.nonzero(self._coeffs)[0]]
        elif isinstance(self, SparsePoly):
            return self._coeffs
        else:
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
        return np.arange(self.degree, -1, -1)

    @property
    def coeffs(self):
        """
        galois.GFArray: The coefficients of the polynomial in degree-descending order. The entries of :obj:`galois.Poly.degrees` are
        paired with :obj:`galois.Poly.coeffs`.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(7)
            p = galois.Poly([3, 0, 5, 2], field=GF)
            p.coeffs
        """
        if isinstance(self, DensePoly):
            return self._coeffs
        elif isinstance(self, SparsePoly):
            # Assemble a full list of coefficients, including zeros
            coeffs = self.field.Zeros(self.degree + 1)
            if self.nonzero_degrees.size > 0:
                coeffs[self.degree - self.nonzero_degrees] = self.nonzero_coeffs
            return coeffs
        else:
            raise NotImplementedError

    @property
    def integer(self):
        """
        int: The integer representation of the polynomial. For polynomial :math:`p(x) =  a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`
        with elements in :math:`a_k \\in \\mathrm{GF}(q)`, the integer representation is :math:`i = a_{d} q^{d} + a_{d-1} q^{d-1} + \\dots + a_1 q + a_0`
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


class DensePoly(Poly):
    """
    Galois field polynomial implementation using dense polynomials.
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        return object.__new__(cls)

    def __init__(self, coeffs, field=None, order="desc"):
        if not (field is None or issubclass(field, GFArray)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")
        if not isinstance(coeffs, (list, tuple, np.ndarray, GFArray)):
            raise TypeError(f"Argument `coeffs` must 'array-like', not {type(coeffs)}.")
        if not len(coeffs) > 0:
            raise ValueError(f"Argument `coeffs` must have non-zero length, not {len(coeffs)}.")
        if not order in ["desc", "asc"]:
            raise ValueError(f"Argument `order` must be either 'desc' or 'asc', not {order}.")

        if isinstance(coeffs, GFArray) and field is None:
            pass
        else:
            field = GF2 if field is None else field
            if isinstance(coeffs, np.ndarray):
                # Ensure coeffs is an iterable
                coeffs = coeffs.tolist()
            coeffs = field([-field(abs(c)) if c < 0 else field(c) for c in coeffs])  # pylint: disable=invalid-unary-operand-type

        if order == "desc":
            self._coeffs = coeffs
        else:
            self._coeffs = coeffs[::-1]

        # Remove leading zero coefficients
        idxs = np.nonzero(self._coeffs)[0]
        if idxs.size > 0:
            self._coeffs = self._coeffs[idxs[0]:]
        else:
            self._coeffs = self._coeffs[-1]

        # Ensure the coefficient array isn't 0-dimension
        self._coeffs = np.atleast_1d(self._coeffs)

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zero(cls, field=GF2):
        return DensePoly([0], field=field)

    @classmethod
    def One(cls, field=GF2):
        return DensePoly([1], field=field)

    @classmethod
    def Identity(cls, field=GF2):
        return DensePoly([1, 0], field=field)

    @classmethod
    def Random(cls, degree, field=GF2):
        if not isinstance(degree, (int, np.integer)):
            raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
        if not degree >= 0:
            raise TypeError(f"Argument `degree` must be at least 0, not {degree}.")

        coeffs = field.Random(degree + 1)
        coeffs[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero

        return DensePoly(coeffs, field=field)

    @classmethod
    def Integer(cls, integer, field=GF2):
        if not isinstance(integer, (int, np.integer)):
            raise TypeError(f"Polynomial creation must have `integer` be an integer, not {type(integer)}")
        c = integer_to_poly(integer, field.order)
        return DensePoly(c, field=field)

    @classmethod
    def Degrees(cls, degrees, coeffs=None, field=None):
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
            # The zero polynomial p(x) = 0
            degrees = [0]
            coeffs = [0]

        degree = np.max(degrees)  # The degree of the polynomial
        if isinstance(coeffs, GFArray):
            # Preserve coefficient field if a Galois field array was specified
            all_coeffs = type(coeffs).Zeros(degree + 1)
            all_coeffs[degree - degrees] = coeffs
        else:
            all_coeffs = [0]*(degree + 1)
            for d, c in zip(degrees, coeffs):
                all_coeffs[degree - d] = c

        return DensePoly(all_coeffs, field=field)

    @classmethod
    def Coeffs(cls, coeffs, field=None, order="desc"):
        # Alias of Coeffs()
        return DensePoly(coeffs, field=field, order=order)

    @classmethod
    def Roots(cls, roots, multiplicities=None, field=None):
        if not (field is None or issubclass(field, GFArray)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")
        if not isinstance(roots, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `roots` must 'array-like', not {type(roots)}.")
        if not isinstance(multiplicities, (type(None), list, tuple, np.ndarray)):
            raise TypeError(f"Argument `multiplicities` must 'array-like', not {type(multiplicities)}.")

        field = GF2 if field is None else field
        multiplicities = [1,]*len(roots) if multiplicities is None else multiplicities
        roots = field(roots).flatten().tolist()

        p = DensePoly.One(field=field)
        for root, multiplicity in zip(roots, multiplicities):
            p *= DensePoly([1, -int(root)], field=field)**multiplicity

        return p

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    def __call__(self, x, field=None):
        """
        Evaluate the polynomial.

        Parameters
        ----------
        x : galois.GFArray
            An array (or 0-dim array) of field element to evaluate the polynomial over.
        field : galois.GFMeta, optional
            The Galois field to evaluate the polynomial over. The default is `None` which represents
            the polynomial's current field, i.e. :obj:`field`.

        Returns
        -------
        galois.GFArray
            The result of the polynomial evaluation of the same shape as `x`.
        """
        if field is None:
            return self.field._poly_eval(self.coeffs, x)
        else:
            assert issubclass(field, GFArray)
            return field._poly_eval(self.coeffs, x)

    @classmethod
    def _check_inputs_are_dense_polys(cls, a, b):
        if not isinstance(a, DensePoly):
            raise TypeError(f"Both arguments must be of type galois.DensePoly, not {type(a)} and {a}.")
        if not isinstance(b, DensePoly):
            raise TypeError(f"Both arguments must be of type galois.DensePoly, not {type(b)} and {b}.")
        if not a.field is b.field:
            raise TypeError(f"Both polynomials must be over the same field, not {str(a.field)} and {str(b.field)}.")

    @classmethod
    def _add(cls, a, b):
        cls._check_inputs_are_dense_polys(a, b)
        field = a.field

        # c(x) = a(x) + b(x)
        c_coeffs = field.Zeros(max(a.coeffs.size, b.coeffs.size))
        c_coeffs[-a.coeffs.size:] = a.coeffs
        c_coeffs[-b.coeffs.size:] += b.coeffs

        return DensePoly(c_coeffs)

    @classmethod
    def _sub(cls, a, b):
        cls._check_inputs_are_dense_polys(a, b)
        field = a.field

        # c(x) = a(x) + b(x)
        c_coeffs = field.Zeros(max(a.coeffs.size, b.coeffs.size))
        c_coeffs[-a.coeffs.size:] = a.coeffs
        c_coeffs[-b.coeffs.size:] -= b.coeffs

        return DensePoly(c_coeffs)

    @classmethod
    def _mul(cls, a, b):
        cls._check_inputs_are_dense_polys(a, b)
        field = a.field

        # c(x) = a(x) * b(x)
        a_degree = a.coeffs.size - 1
        b_degree = b.coeffs.size - 1
        c_coeffs = field.Zeros(a_degree + b_degree + 1)
        for i in np.nonzero(b.coeffs)[0]:
            c_coeffs[i:i + a.coeffs.size] += a.coeffs*b.coeffs[i]

        return DensePoly(c_coeffs)

    @classmethod
    def _divmod(cls, a, b):
        cls._check_inputs_are_dense_polys(a, b)
        field = a.field
        zero = DensePoly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            return DensePoly(a.coeffs // b.coeffs), zero

        elif a == zero:
            return zero, zero

        elif a.degree < b.degree:
            return zero, a.copy()

        else:
            q_degree = a.degree - b.degree
            r_degree = b.degree  # One degree larger than final remainder
            q_coeffs = field.Zeros(q_degree + 1)
            r_coeffs = field.Zeros(r_degree + 1)

            # Preset remainder so we can rotate at the start of loop
            r_coeffs[1:] = a.coeffs[0:b.degree]

            for i in range(0, q_degree + 1):
                r_coeffs = np.roll(r_coeffs, -1)
                r_coeffs[-1] = a.coeffs[i + b.degree]

                if r_coeffs[0] > 0:
                    q_coeffs[i] = r_coeffs[0] // b.coeffs[0]
                    r_coeffs -= q_coeffs[i]*b.coeffs

            return DensePoly(q_coeffs), DensePoly(r_coeffs[1:])

    @classmethod
    def _mod(cls, a, b):
        cls._check_inputs_are_dense_polys(a, b)
        field = a.field
        zero = DensePoly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            return zero

        elif a == zero:
            return zero

        elif a.degree < b.degree:
            return a.copy()

        else:
            q_degree = a.degree - b.degree
            r_degree = b.degree  # One degree larger than final remainder
            r_coeffs = field.Zeros(r_degree + 1)

            # Preset remainder so we can rotate at the start of loop
            r_coeffs[1:] = a.coeffs[0:b.degree]

            for i in range(0, q_degree + 1):
                r_coeffs = np.roll(r_coeffs, -1)
                r_coeffs[-1] = a.coeffs[i + b.degree]

                if r_coeffs[0] > 0:
                    q = r_coeffs[0] // b.coeffs[0]
                    r_coeffs -= q*b.coeffs

            return DensePoly(r_coeffs[1:])


class SparsePoly(Poly):
    """
    Galois field polynomial implementation using dense polynomials.
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        return object.__new__(cls)

    def __init__(self, degrees, coeffs=None, field=None):
        coeffs = [1,]*len(degrees) if coeffs is None else coeffs
        if not isinstance(degrees, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `degrees` must 'array-like', not {type(degrees)}.")
        if not isinstance(coeffs, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `coeffs` must 'array-like', not {type(coeffs)}.")
        if not len(degrees) == len(coeffs):
            raise ValueError(f"Arguments `degrees` and `coeffs` must have the same length, not {len(degrees)} and {len(coeffs)}.")
        if not all(degree >= 0 for degree in degrees):
            raise ValueError(f"Argument `degrees` must have non-negative values, not {degrees}.")

        if isinstance(coeffs, GFArray) and field is None:
            self._degrees = np.array(degrees)
            self._coeffs = coeffs
        else:
            field = GF2 if field is None else field
            if isinstance(coeffs, np.ndarray):
                # Ensure coeffs is an iterable
                coeffs = coeffs.tolist()
            self._degrees = np.array(degrees)
            self._coeffs = field([-field(abs(c)) if c < 0 else field(c) for c in coeffs])  # pylint: disable=invalid-unary-operand-type

        # Sort the degrees and coefficients in descending order
        idxs = np.argsort(degrees)[::-1]
        self._degrees = self._degrees[idxs]
        self._coeffs = self._coeffs[idxs]

        # Remove zero coefficients
        idxs = np.nonzero(self._coeffs)[0]
        self._degrees = self._degrees[idxs]
        self._coeffs = self._coeffs[idxs]

    ###############################################################################
    # Alternate constructors
    ###############################################################################

    @classmethod
    def Zero(cls, field=GF2):
        return SparsePoly.Coeffs([0], field=field)

    @classmethod
    def One(cls, field=GF2):
        return SparsePoly.Coeffs([1], field=field)

    @classmethod
    def Identity(cls, field=GF2):
        return SparsePoly.Coeffs([1, 0], field=field)

    @classmethod
    def Random(cls, degree, field=GF2):
        coeffs = DensePoly.Random(degree, field=field).coeffs
        return SparsePoly.Coeffs(coeffs)

    @classmethod
    def Integer(cls, integer, field=GF2):
        coeffs = DensePoly.Integer(integer, field=field).coeffs
        return SparsePoly.Coeffs(coeffs)

    @classmethod
    def Degrees(cls, degrees, coeffs=None, field=None):
        # Alias of SparsePoly()
        return SparsePoly(degrees, coeffs=coeffs, field=field)

    @classmethod
    def Coeffs(cls, coeffs, field=None, order="desc"):
        coeffs = DensePoly(coeffs, field=field, order=order).coeffs
        degrees = np.arange(coeffs.size - 1, -1, -1)
        return SparsePoly(degrees, coeffs, field=field)

    @classmethod
    def Roots(cls, roots, multiplicities=None, field=None):
        coeffs = DensePoly.Roots(roots, multiplicities=multiplicities, field=field).coeffs
        return SparsePoly.Coeffs(coeffs)

    ###############################################################################
    # Arithmetic methods
    ###############################################################################

    @classmethod
    def _check_inputs_are_sparse_polys(cls, a, b):
        if not isinstance(a, SparsePoly):
            raise TypeError(f"Both arguments must be of type galois.SparsePoly, not {type(a)} and {a}.")
        if not isinstance(b, SparsePoly):
            raise TypeError(f"Both arguments must be of type galois.SparsePoly, not {type(b)} and {b}.")
        if not a.field is b.field:
            raise TypeError(f"Both polynomials must be over the same field, not {str(a.field)} and {str(b.field)}.")

    @classmethod
    def _return_poly(cls, d, field):
        degree = 0 if len(d.keys()) == 0 else max(d.keys())
        nonzero_degrees = list(d.keys())
        nonzero_coeffs = list(d.values())
        if len(nonzero_degrees) < 0.001*(degree + 1):
            return SparsePoly(nonzero_degrees, nonzero_coeffs, field=field)
        else:
            return DensePoly.Degrees(nonzero_degrees, nonzero_coeffs, field=field)

    @classmethod
    def _add(cls, a, b):
        cls._check_inputs_are_sparse_polys(a, b)
        field = a.field

        # c(x) = a(x) + b(x)
        d = {}
        for a_degree, a_coeff in zip(a.nonzero_degrees, a.nonzero_coeffs):
            d[a_degree] = a_coeff
        for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
            d[b_degree] = d.get(b_degree, field(0)) + b_coeff

        return cls._return_poly(d, field)

    @classmethod
    def _sub(cls, a, b):
        cls._check_inputs_are_sparse_polys(a, b)
        field = a.field

        # c(x) = a(x) - b(x)
        d = {}
        for a_degree, a_coeff in zip(a.nonzero_degrees, a.nonzero_coeffs):
            d[a_degree] = a_coeff
        for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
            d[b_degree] = d.get(b_degree, field(0)) - b_coeff

        return cls._return_poly(d, field)

    @classmethod
    def _mul(cls, a, b):
        cls._check_inputs_are_sparse_polys(a, b)
        field = a.field

        # c(x) = a(x) * b(x)
        d = {}
        for a_degree, a_coeff in zip(a.nonzero_degrees, a.nonzero_coeffs):
            for b_degree, b_coeff in zip(b.nonzero_degrees, b.nonzero_coeffs):
                d[a_degree + b_degree] = d.get(a_degree + b_degree, field(0)) + a_coeff*b_coeff

        return cls._return_poly(d, field)

    @classmethod
    def _divmod(cls, a, b):
        cls._check_inputs_are_sparse_polys(a, b)
        field = a.field
        zero = DensePoly.Zero(field)

        # q(x)*b(x) + r(x) = a(x)
        if b.degree == 0:
            q_degrees = a.nonzero_degrees
            q_coeffs = [a_coeff // b.coeffs[0] for a_coeff in a.nonzero_coeffs]
            qq = dict(zip(q_degrees, q_coeffs))
            return cls._return_poly(qq, field), zero

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

            return cls._return_poly(qq, field), DensePoly(r_coeffs[1:])

    @classmethod
    def _mod(cls, a, b):
        cls._check_inputs_are_sparse_polys(a, b)
        field = a.field
        zero = DensePoly.Zero(field)

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

            return DensePoly(r_coeffs[1:])
