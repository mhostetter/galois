"""
A module containing functions to generate and test normal elements of finite fields.
"""

from __future__ import annotations

import random

import numpy as np
from typing_extensions import Literal

from .._helper import export, verify_isinstance
from .._polys import Poly
from ..typing import PolyLike


@export
def is_normal_element(element: PolyLike, irreducible_poly: Poly) -> bool:
    r"""
    Determines whether an element of the extension field $\mathrm{GF}(q^m)$ is normal.

    Let $f(x)$ be a degree-$m$ irreducible polynomial over $\mathrm{GF}(q)$ and let $\alpha$ denote
    the residue class of $x$ in the quotient

    $$
    \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
    $$

    Every field element can be written uniquely as $g(\alpha)$ with $\deg g < m$. This function
    determines whether the element represented by the input polynomial $g(x)$ is normal, i.e.,
    whether its Frobenius conjugates
    $\{g(\alpha), g(\alpha)^q, \dots, g(\alpha)^{q^{m-1}}\}$ form a basis of $\mathrm{GF}(q^m)$
    as a vector space over $\mathrm{GF}(q)$.

    Arguments:
        element:
            A polynomial $g(x)$ over $\mathrm{GF}(q)$ with degree less than $m$. This may be any
            :obj:`~galois.typing.PolyLike` and will be converted to a :class:`~galois.Poly` over the same field
            as `irreducible_poly`. The corresponding field element is the residue class
            $g(\alpha)$ in $\mathrm{GF}(q)[x] / (f(x))$, where $\alpha$ is the image of $x$ modulo
            $f(x)$.
        irreducible_poly:
            The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that defines the
            extension field

            $$
            \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
            $$

    Returns:
        `True` if the residue class $g(\alpha)$ is a normal element of $\mathrm{GF}(q^m)$ over
        $\mathrm{GF}(q)$.

    Notes:
        The field $\mathrm{GF}(q^m)$ is an $m$-dimensional vector space over $\mathrm{GF}(q)$.
        An element $a \in \mathrm{GF}(q^m)$ is **normal** (over $\mathrm{GF}(q)$) if its
        Frobenius conjugates

        $$
        \{a, a^q, a^{q^2}, \dots, a^{q^{m-1}}\}
        $$

        form a basis of $\mathrm{GF}(q^m)$ as a vector space over $\mathrm{GF}(q)$.

        Equivalently, an element $a = g(\alpha)$ is normal if and only if its Frobenius conjugates
        are linearly independent over $\mathrm{GF}(q)$, i.e., the only solution

        $$
        \sum_{i=0}^{m-1} c_i a^{q^i} = 0, \quad c_i \in \mathrm{GF}(q),
        $$

        is $c_0 = \dots = c_{m-1} = 0$.

        This function tests normality as follows.

        1. Compute the Frobenius conjugates of $a$:

           $$
           a^{q^0}, a^{q^1}, \dots, a^{q^{m-1}}.
           $$

        2. Express each conjugate as a polynomial over $\mathrm{GF}(q)$ of degree less than $m$:

           $$
           a^{q^i} \leftrightarrow \sum_{j=0}^{m-1} c_{i,j} \alpha^j.
           $$

           The coefficient vector $(c_{i,0}, \dots, c_{i,m-1})$ is the coordinate vector of
           $a^{q^i}$ in the power basis $\{1, \alpha, \dots, \alpha^{m-1}\}$.

        3. Form the $m \times m$ matrix over $\mathrm{GF}(q)$ whose columns are these coordinate
           vectors. The element is normal if and only if this matrix has full rank $m$.

        In this implementation, exponentiation in $\mathrm{GF}(q^m)$ is performed via polynomial
        exponentiation modulo `irreducible_poly` using the built-in `pow()`, and coefficient
        vectors are obtained from the polynomial representatives of the Frobenius conjugates.

    See Also:
        normal_element, normal_elements

    Examples:
        Construct the extension field $\mathrm{GF}(2^3)$ with irreducible polynomial
        $f(x) = x^3 + x + 1$.

        .. ipython:: python

            f = galois.Poly.Str("x^3 + x + 1"); f

            # x + 1 represents α + 1
            assert galois.is_normal_element("x + 1", f)

            # x represents α
            assert not galois.is_normal_element("x", f)

    Group:
        galois-fields-normal-elements
    """
    verify_isinstance(irreducible_poly, Poly)
    field = irreducible_poly.field

    # Convert element into a Poly object over the same base field as irreducible_poly
    element = Poly.Like(element, field=field)

    if element.field is not irreducible_poly.field:
        raise ValueError(
            f"Arguments 'element' and 'irreducible_poly' must be over the same field, "
            f"not {element.field.name} and {irreducible_poly.field.name}."
        )
    if not element.degree < irreducible_poly.degree:
        raise ValueError(
            f"Argument 'element' must have degree less than 'irreducible_poly', "
            f"not degrees {element.degree} and {irreducible_poly.degree}."
        )
    if not irreducible_poly.is_irreducible():
        raise ValueError(
            f"Argument 'irreducible_poly' must be irreducible, "
            f"{irreducible_poly} is reducible over {irreducible_poly.field.name}."
        )

    return _is_normal_element(element, irreducible_poly)


def _is_normal_element(element: Poly, irreducible_poly: Poly) -> bool:
    r"""
    A private version of :func:`is_normal_element` without type checking/conversion for internal use.

    Arguments:
        element:
            A polynomial $g(x)$ over $\mathrm{GF}(q)$ with $\deg g < m$.
        irreducible_poly:
            The irreducible polynomial $f(x)$ defining the field
            $\mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x))$.

    Returns:
        `True` if the residue class of $g(x)$ modulo $f(x)$ is normal over $\mathrm{GF}(q)$,
        otherwise `False`.
    """
    GF = irreducible_poly.field
    q = GF.order
    m = irreducible_poly.degree

    # Build the m x m matrix whose j-th column is the coordinate vector (over GF(q)) of a^{q^j}
    # in the power basis {1, α, ..., α^{m-1}}, where a = g(α).
    M = GF.Zeros((m, m))

    # a^{q^0}, a^{q^1}, ..., a^{q^{m-1}}
    a_power = element
    for j in range(m):
        # Reduce modulo f(x) to get representative of a^{q^j}
        a_power = Poly(a_power.coeffs, field=GF)  # Ensure canonical Poly type
        a_power = a_power % irreducible_poly

        # Extract coefficients c_0, ..., c_{m-1} in a fixed order (descending powers).
        # Poly.coefficients(m, order="desc") returns a length-m vector in GF(q).
        coeffs = a_power.coefficients(m, order="desc")
        M[:, j] = coeffs

        # Next Frobenius power: a_power <- a_power^q (in GF(q^m))
        a_power = pow(a_power, q, irreducible_poly)

    # Treating entries as reals preserves linear-independence tests in practice:
    # any GF(q)-linear dependency among these 0,1,...,q-1 coefficients induces an R-dependency.
    rank = np.linalg.matrix_rank(M)

    return rank == m


@export
def normal_element(irreducible_poly: Poly, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Finds a normal element of the extension field $\mathrm{GF}(q^m)$ defined by an
    irreducible polynomial.

    Let $f(x)$ be a degree-$m$ irreducible polynomial over $\mathrm{GF}(q)$ and let

    $$
    \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
    $$

    This function searches for a polynomial $g(x)$ over $\mathrm{GF}(q)$ with $\deg g < m$ whose
    residue class modulo $f(x)$ is normal, i.e., whose Frobenius conjugates
    $\{g(\alpha), g(\alpha)^q, \dots, g(\alpha)^{q^{m-1}}\}$ form a basis of $\mathrm{GF}(q^m)$
    over $\mathrm{GF}(q)$.

    Arguments:
        irreducible_poly:
            The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that defines the
            extension field

            $$
            \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
            $$
        method:
            The search method for finding a normal representative polynomial. Choices are:

            - `"min"`: Return a normal representative with the smallest integer encoding.
            - `"max"`: Return a normal representative with the largest integer encoding.
            - `"random"`: Return a random normal representative.

    Returns:
        A polynomial $g(x)$ over $\mathrm{GF}(q)$ with degree less than $m$ such that its residue
        class modulo $f(x)$ is a normal element of $\mathrm{GF}(q^m)$ over $\mathrm{GF}(q)$.

    Notes:
        Integers in the range $[0, q^m - 1]$ correspond to polynomials over $\mathrm{GF}(q)$ with
        degree less than $m$ via the usual base-$q$ expansion. Each such polynomial $g(x)$
        represents a unique residue class $g(\alpha)$ in $\mathrm{GF}(q)[x] / (f(x))$.

        Constant polynomials (integers $0, 1, \dots, q - 1$) represent elements of the base field
        $\mathrm{GF}(q)$ and are never normal in $\mathrm{GF}(q^m)$ for $m > 1$, since their
        Frobenius conjugates do not span the $m$-dimensional extension space.

        Therefore, this function searches over integers in the range $[q, q^m - 1]$, which
        correspond to non-constant polynomials $g(x)$ of degree at most $m - 1$.

    See Also:
        is_normal_element, normal_elements

    Examples:
        Construct the extension field $\mathrm{GF}(2^3)$ with irreducible polynomial
        $f(x) = x^3 + x + 1$.

        .. ipython:: python

            f = galois.Poly.Str("x^3 + x + 1"); f

        Find the smallest normal representative polynomial for $\mathrm{GF}(2^3)$.

        .. ipython:: python

            g = galois.normal_element(f); g

        Find a random normal representative polynomial for $\mathrm{GF}(2^3)$.

        .. ipython:: python

            g = galois.normal_element(f, method="random"); g

    Group:
        galois-fields-normal-elements
    """
    verify_isinstance(irreducible_poly, Poly)
    field = irreducible_poly.field

    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument 'irreducible_poly' must have degree greater than 1, not {irreducible_poly.degree}.")
    if not irreducible_poly.is_irreducible():
        raise ValueError(
            f"Argument 'irreducible_poly' must be irreducible, "
            f"{irreducible_poly} is reducible over {irreducible_poly.field.name}."
        )
    if method not in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    q = field.order
    m = irreducible_poly.degree

    # Skip constants 0..q-1 (elements of GF(q)), which cannot be normal in GF(q^m) for m > 1
    start = q
    stop = q**m

    if method == "min":
        for integer in range(start, stop):
            element = Poly.Int(integer, field=field)
            if _is_normal_element(element, irreducible_poly):
                return element
    elif method == "max":
        for integer in range(stop - 1, start - 1, -1):
            element = Poly.Int(integer, field=field)
            if _is_normal_element(element, irreducible_poly):
                return element
    else:
        while True:
            integer = random.randint(start, stop - 1)
            element = Poly.Int(integer, field=field)
            if _is_normal_element(element, irreducible_poly):
                return element

    raise RuntimeError(f"No normal elements in GF({q}^{m}) were found with irreducible polynomial {irreducible_poly}.")


@export
def normal_elements(irreducible_poly: Poly) -> list[Poly]:
    r"""
    Enumerates all normal elements of the extension field $\mathrm{GF}(q^m)$ defined by an
    irreducible polynomial.

    Let $f(x)$ be a degree-$m$ irreducible polynomial over $\mathrm{GF}(q)$ and let

    $$
    \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
    $$

    This function returns all polynomials $g(x)$ over $\mathrm{GF}(q)$ with $\deg g < m$ whose
    residue classes modulo $f(x)$ are normal elements of $\mathrm{GF}(q^m)$ over $\mathrm{GF}(q)$.

    Arguments:
        irreducible_poly:
            The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that defines the
            extension field

            $$
            \mathrm{GF}(q^m) \cong \mathrm{GF}(q)[x] / (f(x)).
            $$

    Returns:
        List of all polynomials $g(x)$ over $\mathrm{GF}(q)$ with degree less than $m$ whose residue
        classes modulo $f(x)$ are normal elements of $\mathrm{GF}(q^m)$ over $\mathrm{GF}(q)$.

    Notes:
        The number of normal elements depends only on $(q, m)$ and not on the particular choice
        of irreducible polynomial $f(x)$ used to construct the field.

        A convenient theoretical description is via the **polynomial Euler totient** over
        $\mathrm{GF}(q)$. If

        $$
        x^m - 1 = \prod_i P_i(x)^{e_i}
        $$

        is the factorization of $x^m - 1$ into monic irreducible polynomials over $\mathrm{GF}(q)$,
        then the number of normal elements of $\mathrm{GF}(q^m)$ over $\mathrm{GF}(q)$ is

        $$
        N(q, m) = q^m \prod_i \left(1 - \frac{1}{q^{\deg P_i}}\right)
        $$

        when $x^m - 1$ is squarefree (the general case can be expressed in terms of a polynomial
        totient function $\varphi_q$; see standard references on normal bases). In particular,
        $N(q, m)$ is independent of the irreducible polynomial $f(x)$ that defines the field.

        This function **does not** use the above product formula internally; instead, it enumerates
        normal elements explicitly:

        1. Iterate over integers $n$ in the range $[q, q^m - 1]$, which correspond to non-constant
           polynomials of degree less than $m$ via the base-$q$ encoding.
        2. Convert each integer to a polynomial $g(x)$ using :meth:`Poly.Int`.
        3. Test whether the residue class of $g(x)$ is normal using :func:`is_normal_element`.
        4. Collect all such $g(x)$.

        For large fields, this enumeration is expensive ($O(q^m)$ normality tests). For many
        applications, it is preferable to use :func:`normal_element` to obtain a single normal
        element efficiently via a (typically fast) randomized search.

    See Also:
        is_normal_element, normal_element

    Examples:
        Construct the extension field $\mathrm{GF}(2^3)$ with irreducible polynomial
        $f(x) = x^3 + x + 1$.

        .. ipython:: python

            f = galois.Poly.Str("x^3 + x + 1"); f

        Find all normal representative polynomials for $\mathrm{GF}(2^3)$.

        .. ipython:: python

            g = galois.normal_elements(f); g
            assert len(g) == 3

    Group:
        galois-fields-normal-elements
    """
    verify_isinstance(irreducible_poly, Poly)
    field = irreducible_poly.field

    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument 'irreducible_poly' must have degree greater than 1, not {irreducible_poly.degree}.")
    if not irreducible_poly.is_irreducible():
        raise ValueError(
            f"Argument 'irreducible_poly' must be irreducible, "
            f"{irreducible_poly} is reducible over {irreducible_poly.field.name}."
        )

    q = field.order
    m = irreducible_poly.degree

    elements: list[Poly] = []

    # Iterate over all non-constant representative polynomials g(x) (integers q..q^m-1)
    for integer in range(q, q**m):
        element = Poly.Int(integer, field=field)
        if _is_normal_element(element, irreducible_poly):
            elements.append(element)

    # Sort elements lexicographically by their integer representation
    elements = sorted(elements, key=int)

    return elements
