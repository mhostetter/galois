"""
A module containing functions to generate and test primitive elements of finite fields.
"""

from __future__ import annotations

import random

from typing_extensions import Literal

from .._helper import export, verify_isinstance
from .._modular import totatives
from .._polys import Poly
from .._prime import factors
from ..typing import PolyLike


@export
def is_primitive_element(element: PolyLike, irreducible_poly: Poly) -> bool:
    r"""
    Determines if $g$ is a primitive element of the Galois field $\mathrm{GF}(q^m)$ with
    degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$.

    Arguments:
        element: An element $g$ of $\mathrm{GF}(q^m)$ is a polynomial over $\mathrm{GF}(q)$ with
            degree less than $m$.
        irreducible_poly: The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that
            defines the extension field $\mathrm{GF}(q^m)$.

    Returns:
        `True` if $g$ is a primitive element of $\mathrm{GF}(q^m)$.

    See Also:
        primitive_element, FieldArray.primitive_element

    Examples:
        In the extension field $\mathrm{GF}(3^4)$, the element $x + 2$ is a primitive element whose
        order is $3^4 - 1 = 80$.

        .. ipython:: python

            GF = galois.GF(3**4)
            f = GF.irreducible_poly; f
            galois.is_primitive_element("x + 2", f)
            GF("x + 2").multiplicative_order()

        However, the element $x + 1$ is not a primitive element, as noted by its order being only 20.

        .. ipython:: python

            galois.is_primitive_element("x + 1", f)
            GF("x + 1").multiplicative_order()

    Group:
        galois-fields-primitive-elements
    """
    verify_isinstance(irreducible_poly, Poly)
    field = irreducible_poly.field

    # Convert element into a Poly object
    element = Poly._PolyLike(element, field=field)

    if not element.field == irreducible_poly.field:
        raise ValueError(
            f"Arguments 'element' and 'irreducible_poly' must be over the same field, "
            f"not {element.field.name} and {irreducible_poly.field.name}."
        )
    if not element.degree < irreducible_poly.degree:
        raise ValueError(
            f"Argument 'element' must have degree less than 'irreducible_poly', "
            f"not {element.degree} and {irreducible_poly.degree}."
        )
    if not irreducible_poly.is_irreducible():
        raise ValueError(
            f"Argument 'irreducible_poly' must be irreducible, "
            f"{irreducible_poly} is reducible over {irreducible_poly.field.name}."
        )

    return _is_primitive_element(element, irreducible_poly)


def _is_primitive_element(element: Poly, irreducible_poly: Poly) -> bool:
    """
    A private version of `is_primitive_element()` without type checking/conversion for internal use.
    """
    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    order = q**m - 1  # Multiplicative order of GF(q^m)
    primes, _ = factors(order)

    for k in sorted([order // pi for pi in primes]):
        g = pow(element, k, irreducible_poly)
        if g == 1:
            return False

    g = pow(element, order, irreducible_poly)
    if g != 1:
        return False

    return True


@export
def primitive_element(irreducible_poly: Poly, method: Literal["min", "max", "random"] = "min") -> Poly:
    r"""
    Finds a primitive element $g$ of the Galois field $\mathrm{GF}(q^m)$ with degree-$m$
    irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$.

    Arguments:
        irreducible_poly: The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that
            defines the extension field $\mathrm{GF}(q^m)$.
        method: The search method for finding the primitive element.

    Returns:
        A primitive element $g$ of $\mathrm{GF}(q^m)$ with irreducible polynomial $f(x)$.
        The primitive element $g$ is a polynomial over $\mathrm{GF}(q)$ with degree less than $m$.

    See Also:
        is_primitive_element, FieldArray.primitive_element

    Examples:
        Construct the extension field $\mathrm{GF}(7^5)$.

        .. ipython:: python

            f = galois.irreducible_poly(7, 5, method="max"); f
            GF = galois.GF(7**5, irreducible_poly=f, repr="poly")
            print(GF.properties)

        Find the smallest primitive element for the degree-5 extension of $\mathrm{GF}(7)$ with irreducible
        polynomial $f(x)$.

        .. ipython:: python

            g = galois.primitive_element(f); g
            # Convert the polynomial over GF(7) into an element of GF(7^5)
            g = GF(int(g)); g
            g.multiplicative_order() == GF.order - 1

        Find the largest primitive element for the degree-5 extension of $\mathrm{GF}(7)$ with irreducible
        polynomial $f(x)$.

        .. ipython:: python

            g = galois.primitive_element(f, method="max"); g
            # Convert the polynomial over GF(7) into an element of GF(7^5)
            g = GF(int(g)); g
            g.multiplicative_order() == GF.order - 1

        Find a random primitive element for the degree-5 extension of $\mathrm{GF}(7)$ with irreducible
        polynomial $f(x)$.

        .. ipython:: python

            g = galois.primitive_element(f, method="random"); g
            # Convert the polynomial over GF(7) into an element of GF(7^5)
            g = GF(int(g)); g
            g.multiplicative_order() == GF.order - 1
            @suppress
            GF.repr()

    Group:
        galois-fields-primitive-elements
    """
    verify_isinstance(irreducible_poly, Poly)
    if not irreducible_poly.degree > 1:
        raise ValueError(f"Argument 'irreducible_poly' must have degree greater than 1, not {irreducible_poly.degree}.")
    if not irreducible_poly.is_irreducible():
        raise ValueError(
            f"Argument 'irreducible_poly' must be irreducible, "
            f"{irreducible_poly} is reducible over {irreducible_poly.field.name}."
        )
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    field = irreducible_poly.field
    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    start = q
    stop = q**m

    if method == "min":
        for integer in range(start, stop):
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                return element
    elif method == "max":
        for integer in range(stop - 1, start - 1, -1):
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                return element
    else:
        while True:
            integer = random.randint(start, stop - 1)
            element = Poly.Int(integer, field=field)
            if _is_primitive_element(element, irreducible_poly):
                return element

    raise RuntimeError(
        f"No primitive elements in GF({q}^{m}) were found with irreducible polynomial {irreducible_poly}."
    )


@export
def primitive_elements(irreducible_poly: Poly) -> list[Poly]:
    r"""
    Finds all primitive elements $g$ of the Galois field $\mathrm{GF}(q^m)$ with
    degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$.

    Arguments:
        irreducible_poly: The degree-$m$ irreducible polynomial $f(x)$ over $\mathrm{GF}(q)$ that
            defines the extension field $\mathrm{GF}(q^m)$.

    Returns:
        List of all primitive elements of $\mathrm{GF}(q^m)$ with irreducible polynomial $f(x)$.
        Each primitive element $g$ is a polynomial over $\mathrm{GF}(q)$ with degree less than $m$.

    See Also:
        is_primitive_element, FieldArray.primitive_elements

    Notes:
        The number of primitive elements of $\mathrm{GF}(q^m)$ is $\phi(q^m - 1)$, where
        $\phi(n)$ is the Euler totient function. See :obj:`~galois.euler_phi`.

    Examples:
        Construct the extension field $\mathrm{GF}(3^4)$.

        .. ipython:: python

            f = galois.irreducible_poly(3, 4, method="max"); f
            GF = galois.GF(3**4, irreducible_poly=f, repr="poly")
            print(GF.properties)

        Find all primitive elements for the degree-4 extension of $\mathrm{GF}(3)$.

        .. ipython:: python

            g = galois.primitive_elements(f); g

        The number of primitive elements is given by $\phi(q^m - 1)$.

        .. ipython:: python

            phi = galois.euler_phi(3**4 - 1); phi
            len(g) == phi

        Shows that each primitive element has an order of $q^m - 1$.

        .. ipython:: python

            # Convert the polynomials over GF(3) into elements of GF(3^4)
            g = GF([int(gi) for gi in g]); g
            np.all(g.multiplicative_order() == GF.order - 1)
            @suppress
            GF.repr()

    Group:
        galois-fields-primitive-elements
    """
    # Find one primitive element first
    element = primitive_element(irreducible_poly)

    q = irreducible_poly.field.order
    m = irreducible_poly.degree

    elements = []
    for totative in totatives(q**m - 1):
        h = pow(element, totative, irreducible_poly)
        elements.append(h)

    elements = sorted(elements, key=int)  # Sort element lexicographically

    return elements
