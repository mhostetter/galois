"""
A module containing functions to work with vector spaces over Galois fields.
"""

from __future__ import annotations

import galois

from .._helper import export


@export
def companion_matrix(poly: galois.Poly) -> galois.FieldArray:
    r"""
    Returns the companion matrix of a monic polynomial over a finite field.

    Arguments:
        poly:
            A monic polynomial $p(x)$ over $\mathrm{GF}(q)$ with degree $m \ge 1$.

    Returns:
        The companion matrix $C(p) \in \mathrm{GF}(q)^{m \times m}$.

    Notes:
        Let $p(x) \in \mathrm{GF}(q)[x]$ be a monic polynomial of degree $m \ge 1$,

        $$p(x) = x^m + c_{m-1} x^{m-1} + \cdots + c_1 x + c_0.$$

        This function returns the matrix $C(p) \in \mathrm{GF}(q)^{m \times m}$ that represents
        multiplication by $\alpha := x \bmod p(x)$ in the quotient ring
        $\mathrm{GF}(q)[x] / (p(x))$ when elements are represented as **row vectors** in the
        **descending power basis**

        $$\{\alpha^{m-1}, \alpha^{m-2}, \dots, \alpha, 1\}.$$

        Specifically, if

        $$e(\alpha) = e_{m-1}\alpha^{m-1} + \cdots + e_1\alpha + e_0$$

        is represented by the row vector

        $$e = [e_{m-1}, \dots, e_1, e_0],$$

        then

        $$e' = e C(p)$$

        is the row vector representing $\alpha \cdot e(\alpha) \bmod p(\alpha)$ in the same basis.

        Under these conventions, $C(p)$ has the form

        $$C(p) =
        \begin{pmatrix}
        -c_{m-1} & -c_{m-2} & \cdots & -c_1 & -c_0 \\
        1        & 0        & \cdots & 0    & 0 \\
        0        & 1        & \cdots & 0    & 0 \\
        \vdots   &          & \ddots &      & \vdots \\
        0        & 0        & \cdots & 1    & 0
        \end{pmatrix}.
        $$

        Note, this is the transpose of the common column-vector companion matrix convention.

        The companion matrix satisfies $p(C) = 0$ as a matrix identity. The characteristic and minimal polynomials of
        $C(p)$ are both equal to $p(x)$. So, the matrix $C(p)$ and polynomial $p(x)$ are companions in this sense.

        If $p(x)$ is irreducible of degree $m$, then the algebra $\mathrm{GF}(q)[C]$ is a field isomorphic to
        $\mathrm{GF}(q^m)$, and $C(p)$ represents multiplication by the field generator $\alpha$.

    References:
        - https://en.wikipedia.org/wiki/Companion_matrix

    See Also:
        FieldArray.characteristic_poly(), FieldArray.minimal_poly()

    Examples:
        .. ipython:: python

            p, m = 5, 4
            poly = galois.primitive_poly(p, m, method="max"); poly  # A monic irreducible (and primitive) polynomial of degree 4 over GF(5)
            C = galois.companion_matrix(poly); C
            assert np.all(poly(C, elementwise=False) == 0)
            assert C.characteristic_poly() == poly
            assert C.minimal_poly() == poly

            GF = galois.GF(p**m, irreducible_poly=poly)
            x = GF.Random(10, seed=0)
            y = x * GF("a")  # Multiply by alpha (the root of the irreducible polynomial)

            x_vec = x.vector()
            y_vec = x_vec @ C

            assert np.array_equal(y.vector(), y_vec)
            assert np.array_equal(y, GF.Vector(y_vec))

    Group:
        galois-fields-vector-spaces
    """
    if not isinstance(poly, galois.Poly):
        raise TypeError(f"Argument 'poly' must be a galois.Poly, not {type(poly)!r}.")
    if poly.degree < 1:
        raise ValueError(f"Argument 'poly' must have degree at least 1, not {poly.degree}.")
    if not poly.is_monic:
        raise ValueError(f"Argument 'poly' must be monic to construct its companion matrix, not {poly}.")

    field = poly.field
    m = poly.degree

    C = field.Zeros((m, m))

    # Subdiagonal ones (row-vector convention)
    if m > 1:
        C[1:, :-1] = field.Identity(m - 1)

    # Coefficients in descending degree order: [1, c_{m-1}, ..., c_0]
    coeffs = poly.coefficients(m + 1, order="desc")
    C[0, :] = -coeffs[1:]

    return C
