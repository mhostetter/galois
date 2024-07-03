"""
A module containing functions to factor univariate polynomials over finite fields.
"""

from __future__ import annotations

from .._helper import method_of, verify_isinstance
from ._functions import gcd
from ._poly import Poly

__all__ = []


@method_of(Poly)
def is_square_free(f) -> bool:
    r"""
    Determines whether the polynomial $f(x)$ over $\mathrm{GF}(q)$ is square-free.

    .. question:: Why is this a method and not a property?
        :collapsible:

        This is a method to indicate it is a computationally expensive task.

    Returns:
        `True` if the polynomial is square-free.

    Notes:
        A square-free polynomial $f(x)$ has no irreducible factors with multiplicity greater than one.
        Therefore, its canonical factorization is

        $$f(x) = \prod_{i=1}^{k} g_i(x)^{e_i} = \prod_{i=1}^{k} g_i(x) .$$

    Examples:
        Generate irreducible polynomials over $\mathrm{GF}(3)$.

        .. ipython:: python

            GF = galois.GF(3)
            f1 = galois.irreducible_poly(3, 3); f1
            f2 = galois.irreducible_poly(3, 4); f2

        Determine if composite polynomials are square-free over $\mathrm{GF}(3)$.

        .. ipython:: python

            (f1 * f2).is_square_free()
            (f1**2 * f2).is_square_free()
    """
    if not f.is_monic:
        f //= f.coeffs[0]

    # Constant polynomials are square-free
    if f.degree == 0:
        return True

    _, multiplicities = square_free_factors(f)

    return multiplicities == [1]


@method_of(Poly)
def square_free_factors(f: Poly) -> tuple[list[Poly], list[int]]:
    r"""
    Factors the monic polynomial $f(x)$ into a product of square-free polynomials.

    Returns:
        - The list of non-constant, square-free polynomials $h_j(x)$ in the factorization.
        - The list of corresponding multiplicities $j$.

    Raises:
        ValueError: If $f(x)$ is not monic or has degree 0.

    Notes:
        The Square-Free Factorization algorithm factors $f(x)$ into a product of $m$ square-free
        polynomials $h_j(x)$ with multiplicity $j$.

        $$f(x) = \prod_{j=1}^{m} h_j(x)^j$$

        Some $h_j(x) = 1$, but those polynomials are not returned by this function.

        A complete polynomial factorization is implemented in :func:`~Poly.factors`.

    References:
        - Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.1.7.
        - Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples:
        Suppose $f(x) = x(x^3 + 2x + 4)(x^2 + 4x + 1)^3$ over $\mathrm{GF}(5)$. Each polynomial
        $x$, $x^3 + 2x + 4$, and $x^2 + 4x + 1$ are all irreducible over $\mathrm{GF}(5)$.

        .. ipython:: python

            GF = galois.GF(5)
            a = galois.Poly([1,0], field=GF); a, a.is_irreducible()
            b = galois.Poly([1,0,2,4], field=GF); b, b.is_irreducible()
            c = galois.Poly([1,4,1], field=GF); c, c.is_irreducible()
            f = a * b * c**3; f

        The square-free factorization is $\{x(x^3 + 2x + 4), x^2 + 4x + 1\}$ with multiplicities
        $\{1, 3\}$.

        .. ipython:: python

            f.square_free_factors()
            [a*b, c], [1, 3]

    Group:
        Factorization methods

    Order:
        51
    """
    if not f.degree >= 1:
        raise ValueError(f"The polynomial must be non-constant, not {f}.")
    if not f.is_monic:
        raise ValueError(f"The polynomial must be monic, not {f}.")

    field = f.field
    p = field.characteristic
    one = Poly([1], field=field)

    factors_ = []
    multiplicities = []

    # w is the product (without multiplicity) of all factors of f that have multiplicity not divisible by p
    f_prime = f.derivative()
    d = gcd(f, f_prime)
    w = f // d

    # Step 1: Find all factors in w
    i = 1
    while w != one:
        y = gcd(w, d)
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
        # The inverse Frobenius automorphism of the coefficients
        coeffs = d.nonzero_coeffs ** (field.characteristic ** (field.degree - 1))
        delta = Poly.Degrees(degrees, coeffs=coeffs, field=field)  # The p-th root of d(x)
        g, m = square_free_factors(delta)
        factors_.extend(g)
        multiplicities.extend([mi * p for mi in m])

    # Sort the factors in increasing-multiplicity order
    factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: item[1]))

    return list(factors_), list(multiplicities)


@method_of(Poly)
def distinct_degree_factors(f: Poly) -> tuple[list[Poly], list[int]]:
    r"""
    Factors the monic, square-free polynomial $f(x)$ into a product of polynomials whose irreducible factors
    all have the same degree.

    Returns:
        - The list of polynomials $f_i(x)$ whose irreducible factors all have degree $i$.
        - The list of corresponding distinct degrees $i$.

    Raises:
        ValueError: If $f(x)$ is not monic, has degree 0, or is not square-free.

    Notes:
        The Distinct-Degree Factorization algorithm factors a square-free polynomial $f(x)$ with degree
        $d$ into a product of $d$ polynomials $f_i(x)$, where $f_i(x)$ is the product of
        all irreducible factors of $f(x)$ with degree $i$.

        $$f(x) = \prod_{i=1}^{d} f_i(x)$$

        For example, suppose $f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)$ over
        $\mathrm{GF}(2)$, then the distinct-degree factorization is

        $$
        f_1(x) &= x(x + 1) = x^2 + x \\
        f_2(x) &= x^2 + x + 1 \\
        f_3(x) &= (x^3 + x + 1)(x^3 + x^2 + 1) = x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 \\
        f_i(x) &= 1\ \textrm{for}\ i = 4, \dots, 10.
        $$

        Some $f_i(x) = 1$, but those polynomials are not returned by this function. In this example,
        the function returns $\{f_1(x), f_2(x), f_3(x)\}$ and $\{1, 2, 3\}$.

        The Distinct-Degree Factorization algorithm is often applied after the Square-Free Factorization algorithm,
        see :func:`~Poly.square_free_factors`. A complete polynomial factorization is implemented in
        :func:`~Poly.factors`.

    References:
        - Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.2.2.
        - Section 2.2 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples:
        From the example in the notes, suppose $f(x) = x(x + 1)(x^2 + x + 1)(x^3 + x + 1)(x^3 + x^2 + 1)$
        over $\mathrm{GF}(2)$.

        .. ipython:: python

            a = galois.Poly([1, 0]); a, a.is_irreducible()
            b = galois.Poly([1, 1]); b, b.is_irreducible()
            c = galois.Poly([1, 1, 1]); c, c.is_irreducible()
            d = galois.Poly([1, 0, 1, 1]); d, d.is_irreducible()
            e = galois.Poly([1, 1, 0, 1]); e, e.is_irreducible()
            f = a * b * c * d * e; f

        The distinct-degree factorization is $\{x(x + 1), x^2 + x + 1, (x^3 + x + 1)(x^3 + x^2 + 1)\}$
        whose irreducible factors have degrees $\{1, 2, 3\}$.

        .. ipython:: python

            f.distinct_degree_factors()
            [a*b, c, d*e], [1, 2, 3]

    Group:
        Factorization methods

    Order:
        51
    """
    if not f.degree >= 1:
        raise ValueError(f"The polynomial must be non-constant, not {f}.")
    if not f.is_monic:
        raise ValueError(f"The polynomial must be monic, not {f}.")
    if not f.is_square_free():
        raise ValueError(f"The polynomial must be square-free, not {f}.")

    field = f.field
    q = field.order
    n = f.degree
    one = Poly([1], field=field)
    x = Poly([1, 0], field=field)

    factors_ = []
    degrees = []

    a = f
    h = x

    l = 1
    while l <= n // 2 and a != one:
        h = pow(h, q, a)
        z = gcd(a, h - x)
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


@method_of(Poly)
def equal_degree_factors(f: Poly, degree: int) -> list[Poly]:
    r"""
    Factors the monic, square-free polynomial $f(x)$ of degree $rd$ into a product of $r$
    irreducible factors with degree $d$.

    Arguments:
        degree: The degree $d$ of each irreducible factor of $f(x)$.

    Returns:
        The list of $r$ irreducible factors $\{g_1(x), \dots, g_r(x)\}$ in lexicographical order.

    Raises:
        ValueError: If $f(x)$ is not monic, has degree 0, or is not square-free.

    Notes:
        The Equal-Degree Factorization algorithm factors a square-free polynomial $f(x)$ with degree
        $rd$ into a product of $r$ irreducible polynomials each with degree $d$. This function
        implements the Cantor-Zassenhaus algorithm, which is probabilistic.

        The Equal-Degree Factorization algorithm is often applied after the Distinct-Degree Factorization algorithm,
        see :func:`~Poly.distinct_degree_factors`. A complete polynomial factorization is implemented in
        :func:`~Poly.factors`.

    References:
        - Section 2.3 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf
        - Section 1 from https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec8.pdf

    Examples:
        Factor a product of degree-1 irreducible polynomials over $\mathrm{GF}(2)$.

        .. ipython:: python

            a = galois.Poly([1, 0]); a, a.is_irreducible()
            b = galois.Poly([1, 1]); b, b.is_irreducible()
            f = a * b; f
            f.equal_degree_factors(1)

        Factor a product of degree-3 irreducible polynomials over $\mathrm{GF}(5)$.

        .. ipython:: python

            GF = galois.GF(5)
            a = galois.Poly([1, 0, 2, 1], field=GF); a, a.is_irreducible()
            b = galois.Poly([1, 4, 4, 4], field=GF); b, b.is_irreducible()
            f = a * b; f
            f.equal_degree_factors(3)

    Group:
        Factorization methods

    Order:
        51
    """
    verify_isinstance(degree, int)
    if not f.degree >= 1:
        raise ValueError(f"The polynomial must be non-constant, not {f}.")
    if not f.is_monic:
        raise ValueError(f"The polynomial must be monic, not {f}.")
    if not f.degree % degree == 0:
        raise ValueError(
            f"Argument 'degree' must divide the degree of the polynomial, {degree} does not divide {f.degree}."
        )
    if not f.is_square_free():
        raise ValueError(f"The polynomial must be square-free, not {f}.")

    field = f.field
    q = field.order
    r = f.degree // degree
    one = Poly([1], field=field)

    factors_ = [f]
    while len(factors_) < r:
        h = Poly.Random(degree, field=field)
        g = gcd(f, h)
        if g == one:
            g = pow(h, (q**degree - 1) // 2, f) - one
        i = 0
        for u in list(factors_):
            if u.degree <= degree:
                continue
            d = gcd(g, u)
            if d not in [one, u]:
                factors_.remove(u)
                factors_.append(d)
                factors_.append(u // d)
            i += 1

    # Sort the factors in lexicographical order
    factors_ = sorted(factors_, key=int)

    return factors_


@method_of(Poly)
def factors(f) -> tuple[list[Poly], list[int]]:
    r"""
    Computes the irreducible factors of the non-constant, monic polynomial $f(x)$.

    Returns:
        - Sorted list of irreducible factors $\{g_1(x), g_2(x), \dots, g_k(x)\}$ of $f(x)$ sorted in
          lexicographical order.
        - List of corresponding multiplicities $\{e_1, e_2, \dots, e_k\}$.

    Raises:
        ValueError: If $f(x)$ is not monic or has degree 0.

    Notes:
        This function factors a monic polynomial $f(x)$ into its $k$ irreducible factors such that
        $f(x) = g_1(x)^{e_1} g_2(x)^{e_2} \dots g_k(x)^{e_k}$.

        Steps:

        1. Apply the Square-Free Factorization algorithm to factor the monic polynomial into square-free
           polynomials. See :func:`~Poly.square_free_factors`.
        2. Apply the Distinct-Degree Factorization algorithm to factor each square-free polynomial into a product
           of factors with the same degree. See :func:`~Poly.distinct_degree_factors`.
        3. Apply the Equal-Degree Factorization algorithm to factor the product of factors of equal degree into
           their irreducible factors. See :func:`~Poly.equal_degree_factors`.

    References:
        - Hachenberger, D. and Jungnickel, D. Topics in Galois Fields. Algorithm 6.1.7.
        - Section 2.1 from https://people.csail.mit.edu/dmoshkov/courses/codes/poly-factorization.pdf

    Examples:
        Generate irreducible polynomials over $\mathrm{GF}(3)$.

        .. ipython:: python

            GF = galois.GF(3)
            g1 = galois.irreducible_poly(3, 3); g1
            g2 = galois.irreducible_poly(3, 4); g2
            g3 = galois.irreducible_poly(3, 5); g3

        Construct a composite polynomial.

        .. ipython:: python

            e1, e2, e3 = 5, 4, 3
            f = g1**e1 * g2**e2 * g3**e3; f

        Factor the polynomial into its irreducible factors over $\mathrm{GF}(3)$.

        .. ipython:: python

            f.factors()

    Group:
        Factorization methods

    Order:
        51
    """
    if not f.degree >= 1:
        raise ValueError(f"The polynomial must be non-constant, not {f}.")
    if not f.is_monic:
        raise ValueError(f"The polynomial must be monic, not {f}.")

    factors_, multiplicities = [], []

    # Step 1: Find all the square-free factors
    sf_factors, sf_multiplicities = square_free_factors(f)

    # Step 2: Find all the factors with distinct degree
    for sf_factor, sf_multiplicity in zip(sf_factors, sf_multiplicities):
        df_factors, df_degrees = distinct_degree_factors(sf_factor)

        # Step 3: Find all the irreducible factors with degree d
        for df_factor, df_degree in zip(df_factors, df_degrees):
            f = equal_degree_factors(df_factor, df_degree)
            factors_.extend(f)
            multiplicities.extend([sf_multiplicity] * len(f))

    # Sort the factors in increasing-multiplicity order
    factors_, multiplicities = zip(*sorted(zip(factors_, multiplicities), key=lambda item: int(item[0])))

    return list(factors_), list(multiplicities)
