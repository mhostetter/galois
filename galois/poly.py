import numpy as np

from .conversion import integer_to_poly, poly_to_integer, poly_to_str
from .gf import GFArray
from .gf2 import GF2


class Poly:
    """
    Create a polynomial :math:`p(x)` over :math:`\\mathrm{GF}(q)[x]`.

    The polynomial :math:`p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0` has coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}`
    in :math:`\\mathrm{GF}(q)`.

    Parameters
    ----------
    coeffs : array_like
        List of polynomial coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}` with type :obj:`galois.GFArray`, :obj:`numpy.ndarray`,
        :obj:`list`, or :obj:`tuple`. The first element is the highest-degree element if `order="desc"` or the first element is
        the 0-th degree element if `order="asc"`.
    field : galois.GFArray, optional
        The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`. If `coeffs`
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
    Create a polynomial over :math:`\\mathrm{GF}(2)[x]`.

    .. ipython:: python

        galois.Poly([1,0,1,1])
        galois.Poly.Degrees([3,1,0])

    Create a polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

    .. ipython:: python

        GF256 = galois.GF(2**8)
        galois.Poly([124,0,223,0,0,15], field=GF256)

        # Alternate way of constructing the same polynomial
        galois.Poly.Degrees([5,3,0], coeffs=[124,223,15], field=GF256)

    Polynomial arithmetic using binary operators.

    .. ipython:: python

        a = galois.Poly([117,0,63,37], field=GF256); a
        b = galois.Poly([224,0,21], field=GF256); b

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
        if not (field is None or issubclass(field, GFArray)):
            raise TypeError(f"Argument `field` must be a Galois field array class, not {field}.")

        self.order = order

        if isinstance(coeffs, GFArray) and field is None:
            self.coeffs = coeffs
        else:
            field = GF2 if field is None else field

            # Convert list or np.ndarray of integers into the specified `field`. Apply negation
            # operator to any negative integers. For instance, `coeffs=[1, -1]` represents
            # `x - 1` in GF2. However, the `-1` element does not exist in GF2, but the operation
            # `-1` (the additive inverse of the `1` element) does exist.
            c = np.array(coeffs, dtype=field.dtypes[-1], copy=True, ndmin=1)
            assert c.ndim == 1, "Polynomials must only have one dimension"
            assert np.all(np.abs(c) < field.order)
            neg_idxs = np.where(c < 0)
            c = field(np.abs(c))
            c[neg_idxs] *= -1
            self.coeffs = c

    @classmethod
    def Zero(cls, field=GF2):
        """
        Constructs the zero polynomial :math:`p(x) = 0` over :math:`\\mathrm{GF}(q)[x]`.

        Parameters
        ----------
        field : galois.GFArray, optional
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

        Construct the zero polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            galois.Poly.Zero(field=GF256)
        """
        return cls([0], field=field)

    @classmethod
    def One(cls, field=GF2):
        """
        Constructs the one polynomial :math:`p(x) = 1` over :math:`\\mathrm{GF}(q)[x]`.

        Parameters
        ----------
        field : galois.GFArray, optional
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

        Construct the one polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            galois.Poly.One(field=GF256)
        """
        return cls([1], field=field)

    @classmethod
    def Identity(cls, field=GF2):
        """
        Constructs the identity polynomial :math:`p(x) = x` over :math:`\\mathrm{GF}(q)[x]`.

        Parameters
        ----------
        field : galois.GFArray, optional
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

        Construct the identity polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            galois.Poly.Identity(field=GF256)
        """
        return cls([1, 0], field=field)

    @classmethod
    def Random(cls, degree, field=GF2):
        """
        Constructs a random polynomial over :math:`\\mathrm{GF}(q)[x]` with degree :math:`d`.

        Parameters
        ----------
        degree : int
            The degree of the polynomial.
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct a random degree-:math:`5` polynomial over :math:`\\mathrm{GF}(2)[x]`.

        .. ipython:: python

            galois.Poly.Random(5)

        Construct a random degree-:math:`5` polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            galois.Poly.Random(5, field=GF256)
        """
        if not isinstance(degree, (int, np.integer)):
            raise TypeError(f"Argument `degree` must be an integer, not {type(degree)}.")
        if not degree >= 0:
            raise TypeError(f"Argument `degree` must be at least 0, not {degree}.")

        coeffs = field.Random(degree + 1)
        coeffs[0] = field.Random(low=1)  # Ensure leading coefficient is non-zero

        return cls(coeffs)

    @classmethod
    def Integer(cls, integer, field=GF2, order="desc"):
        """
        Constructs a polynomial over :math:`\\mathrm{GF}(q)[x]` from its integer representation.

        The integer value :math:`i` represents the polynomial :math:`p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`
        over field :math:`\\mathrm{GF}(q)` if :math:`i = a_{d}q^{d} + a_{d-1}q^{d-1} + \\dots + a_1q + a_0` using integer arithmetic,
        not finite field arithmetic.

        Parameters
        ----------
        integer : int
            The integer representation of the polynomial :math:`p(x)`.
        field : galois.GFArray, optional
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

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)[x]` from its integer representation.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            galois.Poly.Integer(13*256**3 + 117, field=GF256)
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
        Constructs a polynomial over :math:`\\mathrm{GF}(q)[x]` from its non-zero degrees.

        Parameters
        ----------
        degrees : list
            The polynomial degrees with non-zero coefficients.
        coeffs : array_like, optional
            List of corresponding non-zero coefficients. The default is `None` which corresponds to all one
            coefficients, i.e. `[1,]*len(degrees)`.
        roots : array_like
            List of roots in :math:`\\mathrm{GF}(q)` of the desired polynomial.
        field : galois.GFArray, optional
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

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)[x]` by specifying the degrees with non-zero coefficients.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            galois.Poly.Degrees([3,1,0], coeffs=[251,73,185], field=GF256)
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

        degree = np.max(degrees)  # The degree of the polynomial
        if isinstance(coeffs, GFArray):
            # Preserve coefficient field if a Galois field array was specified
            all_coeffs = type(coeffs).Zeros(degree + 1)
            all_coeffs[degree - degrees] = coeffs
        else:
            all_coeffs = [0]*(degree + 1)
            for d, c in zip(degrees, coeffs):
                all_coeffs[degree - d] = c

        return cls(all_coeffs, field=field)

    @classmethod
    def Roots(cls, roots, field=GF2):
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
        field : galois.GFArray, optional
            The field :math:`\\mathrm{GF}(q)` the polynomial is over. The default :obj:`galois.GF2`.

        Returns
        -------
        galois.Poly
            The polynomial :math:`p(x)`.

        Examples
        --------
        Construct a polynomial over :math:`\\mathrm{GF}(2)[x]` from a list of its roots.

        .. ipython:: python

            roots = [0, 0, 1]
            p = galois.Poly.Roots(roots); p
            p(roots)

        Construct a polynomial over :math:`\\mathrm{GF}(2^8)[x]` from a list of its roots.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            roots = [121, 198, 225]
            p = galois.Poly.Roots(roots, field=GF256); p
            p(roots)
        """
        if not isinstance(roots, (list, tuple, np.ndarray)):
            raise TypeError(f"Argument `roots` must 'array-like', not {type(roots)}.")

        roots = field(roots).flatten().tolist()

        p = cls.One(field=field)
        for root in roots:
            p = p * cls([1, -int(root)], field=field)

        return p

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
        Find the roots of a polynomial over :math:`\\mathrm{GF}(2)[x]`.

        .. ipython:: python

            p = galois.Poly.Roots([0,]*7 + [1,]*13); p
            p.roots()
            p.roots(multiplicity=True)

        Find the roots of a polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            p = galois.Poly.Roots([18,]*7 + [155,]*13 + [227,]*9, field=GF256); p
            p.roots()
            p.roots(multiplicity=True)
        """
        lambda_vector = self.coeffs_asc
        alpha_vector = self.field.alpha ** np.arange(0, self.degree + 1)
        roots = []
        multiplicities = []

        # Test if 0 is a root
        if lambda_vector[0] == 0:
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
                root = int(self.field.alpha**i)
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
        poly = Poly(self.coeffs_desc)
        multiplicity = 1

        while True:
            # If the root is also a root of the derivative, then its a multiple root.
            poly = poly.derivative()

            if poly == zero:
                # Cannot test whether p'(root) = 0 because p'(x) = 0. We've exhausted the non-zero derivatives. For
                # any Galois field, taking `characteristic` derivatives results in p'(x) = 0. For a root with multiplicity
                # greater than the field's characteristic, we need factor the polynomial. Here we factor out (x - root)^m,
                # where m is the current multiplicity.
                poly = Poly(self.coeffs_desc) // (Poly([1, -root], field=self.field)**multiplicity)

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
        Compute the derivatives of a polynomial over :math:`\\mathrm{GF}(2)[x]`.

        .. ipython:: python

            p = galois.Poly.Random(7); p
            p.derivative()

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(2)

        Compute the derivatives of a polynomial over :math:`\\mathrm{GF}(7)[x]`.

        .. ipython:: python

            GF7 = galois.GF(7)
            p = galois.Poly.Random(11, field=GF7); p
            p.derivative()
            p.derivative(2)
            p.derivative(3)

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(7)

        Compute the derivatives of a polynomial over :math:`\\mathrm{GF}(2^8)[x]`.

        .. ipython:: python

            GF256 = galois.GF(2**8)
            p = galois.Poly.Random(7, field=GF256); p
            p.derivative()

            # k derivatives of a polynomial where k is the Galois field's characteristic will always result in 0
            p.derivative(2)
        """
        if not isinstance(k, (int, np.integer)):
            raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
        if not k > 0:
            raise ValueError(f"Argument `k` must be a positive integer, not {k}.")

        coeffs = self.coeffs_desc[0:-1] * np.arange(self.degree, 0, -1)
        p_prime = self.__class__(coeffs)

        k -= 1
        if k > 0:
            return p_prime.derivative(k)
        else:
            return p_prime

    def __repr__(self):
        poly_str = poly_to_str(self.coeffs_asc)
        if self.field.degree == 1:
            order = "{}".format(self.field.characteristic)
        else:
            order = "{}^{}".format(self.field.characteristic, self.field.degree)
        return f"Poly({poly_str}, GF({order}))"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        t = tuple([self.field.order,] + self.coeffs_desc.tolist())
        return hash(t)

    def __call__(self, x):
        return self.field._poly_eval(self.coeffs_desc, x)

    def __add__(self, other):
        if not isinstance(self, Poly):
            raise TypeError(f"For polynomial addition, both arguments must be of type galois.Poly. Argument 0 is of type {type(self)}. Argument 0 = {self}.")
        if not isinstance(other, Poly):
            raise TypeError(f"For polynomial addition, both arguments must be of type galois.Poly. Argument 1 is of type {type(other)}. Argument 1 = {other}.")
        field, a, b = self.field, self.coeffs_desc, other.coeffs_desc

        # c(x) = a(x) + b(x)
        c = field.Zeros(max(a.size, b.size))
        c[-a.size:] = a
        c[-b.size:] += b

        return Poly(c, field=field)

    def __sub__(self, other):
        if not isinstance(self, Poly):
            raise TypeError(f"For polynomial subtraction, both arguments must be of type galois.Poly. Argument 0 is of type {type(self)}. Argument 0 = {self}.")
        if not isinstance(other, Poly):
            raise TypeError(f"For polynomial subtraction, both arguments must be of type galois.Poly. Argument 1 is of type {type(other)}. Argument 1 = {other}.")
        field, a, b = self.field, self.coeffs_desc, other.coeffs_desc

        # c(x) = a(x) - b(x)
        c = field.Zeros(max(a.size, b.size))
        c[-a.size:] = a
        c[-b.size:] -= b

        return Poly(c, field=field)

    def __mul__(self, other):
        if not isinstance(self, Poly):
            raise TypeError(f"For polynomial multiplication, both arguments must be of type galois.Poly. Argument 0 is of type {type(self)}. Argument 0 = {self}.")
        if not isinstance(other, Poly):
            raise TypeError(f"For polynomial multiplication, both arguments must be of type galois.Poly. Argument 1 is of type {type(other)}. Argument 1 = {other}.")
        field, a, b = self.field, self.coeffs_desc, other.coeffs_desc

        # c(x) = a(x) * b(x)
        a_degree = a.size - 1
        b_degree = b.size - 1
        c = field.Zeros(a_degree + b_degree + 1)
        for i in np.nonzero(b)[0]:
            c[i:i + a.size] += a*b[i]

        return Poly(c, field=field)

    def __neg__(self):
        return Poly(-self.coeffs_desc, field=self.field)

    def __divmod__(self, other):
        if not isinstance(self, Poly):
            raise TypeError(f"For polynomial divmod, both arguments must be of type galois.Poly. Argument 0 is of type {type(self)}. Argument 0 = {self}.")
        if not isinstance(other, Poly):
            raise TypeError(f"For polynomial divmod, both arguments must be of type galois.Poly. Argument 1 is of type {type(other)}. Argument 1 = {other}.")
        field, a, b = self.field, self.coeffs_desc, other.coeffs_desc

        # q(x)*b(x) + r(x) = a(x)
        field = self.field
        a_degree = self.degree
        b_degree = other.degree

        if np.array_equal(a, [0]):
            quotient = Poly.Zero(field)
            remainder = Poly.Zero(field)
        elif a_degree < b_degree:
            quotient = Poly.Zero(field)
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

    def __truediv__(self, other):
        return self.__divmod__(other)[0]

    def __floordiv__(self, other):
        return self.__divmod__(other)[0]

    def __mod__(self, other):
        return self.__divmod__(other)[1]

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

        c_square = Poly(a.coeffs_desc, field=field)  # The "squaring" part
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

    def __eq__(self, other):
        return isinstance(other, Poly) and (self.field is other.field) and np.array_equal(self.coeffs_desc, other.coeffs_desc)

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
        galois.GFArray: The polynomial coefficients as a Galois field array. Coefficients are :math:`\\{a_{d}, \\dots, a_1, a_0\\}` if `order="desc"` or
        :math:`\\{a_0, a_1, \\dots, a_{d}\\}` if `order="asc"`, where :math:`p(x) = a_{d}x^{d} + \\dots + a_1x + a_0`.
        """
        return self.field(self._coeffs)

    @coeffs.setter
    def coeffs(self, coeffs):
        if not isinstance(coeffs, GFArray):
            raise TypeError(f"Galois field polynomials must have coefficients in a valid Galois field class (i.e. subclasses of GFArray), not {type(coeffs)}")
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
        self._field = coeffs.__class__

    @property
    def coeffs_asc(self):
        """
        galois.GFArray: The polynomial coefficients :math:`\\{a_0, a_1, \\dots, a_{d-1}, a_{d}\\}` as a Galois field array
        in degree-ascending order, where :math:`p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`.
        """
        if self.order == "asc":
            return self.field(self._coeffs)
        else:
            return self.field(np.flip(self._coeffs))

    @property
    def coeffs_desc(self):
        """
        galois.GFArray: The polynomial coefficients :math:`\\{a_{d}, a_{d-1}, \\dots, a_1, a_0\\}` as a Galois field array
        in degree-ascending order, where :math:`p(x) = a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`.
        """
        if self.order == "desc":
            return self.field(self._coeffs)
        else:
            return self.field(np.flip(self._coeffs))

    @property
    def degree(self):
        """
        int: The degree of the polynomial, i.e. the highest degree with non-zero coefficient.
        """
        return self.coeffs.size - 1

    @property
    def field(self):
        """
        type: The Galois field to which the coefficients belong. The `field` property is a type that is a
        subclass of :obj:`galois.GFArray`.
        """
        return self._field

    @property
    def integer(self):
        """
        int: The integer representation of the polynomial. For :math:`p(x) =  a_{d}x^{d} + a_{d-1}x^{d-1} + \\dots + a_1x + a_0`
        with elements in :math:`\\mathrm{GF}(q)`, the integer representation is :math:`i = a_{d} q^{d} + a_{d-1} q^{d-1} + \\dots + a_1 q + a_0`
        (using integer arithmetic, not finite field arithmetic) where :math:`q` is the field order.
        """
        c = self.coeffs_asc
        c = c.view(np.ndarray)  # We want to do integer math, not Galois field math
        return poly_to_integer(c, self.field.order)


def poly_gcd(a, b):
    """
    Finds the greatest common divisor of two polynomials :math:`a(x)` and :math:`b(x)`
    over :math:`\\mathrm{GF}(q)[x]`.

    This implementation uses the Extended Euclidean Algorithm.

    Parameters
    ----------
    a : galois.Poly
        A polynomial :math:`a(x)` over :math:`\\mathrm{GF}(q)[x]`.
    b : galois.Poly
        A polynomial :math:`b(x)` over :math:`\\mathrm{GF}(q)[x]`.

    Returns
    -------
    galois.Poly
        Polynomial greatest common divisor of :math:`a(x)` and :math:`b(x)`.
    galois.Poly
        Polynomial :math:`x(x)`, such that :math:`a x + b y = gcd(a, b)`.
    galois.Poly
        Polynomial :math:`y(x)`, such that :math:`a x + b y = gcd(a, b)`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(7)
        a = galois.Poly.Roots([2,2,2,3,6], field=GF); a

        # a(x) and b(x) only share the root 2 in common
        b = galois.Poly.Roots([1,2], field=GF); b

        gcd, x, y = galois.poly_gcd(a, b)

        # The GCD has only 2 as a root with multiplicity 1
        gcd.roots(multiplicity=True)

        a*x + b*y == gcd
    """
    if not isinstance(a, Poly):
        raise TypeError(f"Argument `a` must be of type galois.Poly, not {type(a)}.")
    if not isinstance(b, Poly):
        raise TypeError(f"Argument `b` must be of type galois.Poly, not {type(b)}.")
    if not a.field == b.field:
        raise ValueError(f"Polynomials `a` and `b` must be over the same Galois field, not {str(a.field)} and {str(b.field)}.")

    field = a.field
    zero = Poly.Zero(field)
    one = Poly.One(field)

    if a == zero:
        return b, 0, 1
    if b == zero:
        return a, 1, 0

    r = [a, b]
    s = [one, zero]
    t = [zero, one]

    while True:
        qi = r[-2] // r[-1]
        ri = r[-2] % r[-1]
        r.append(ri)
        s.append(s[-2] - qi*s[-1])
        t.append(t[-2] - qi*t[-1])
        if ri == zero:
            break

    return r[-2], s[-2], t[-2]
