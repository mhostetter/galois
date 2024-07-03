"""
A module containing functions for modular arithmetic.
"""

from __future__ import annotations

import functools
import math
import random
from typing import Iterator

from typing_extensions import Literal

from ._helper import export, verify_isinstance
from ._math import lcm
from ._prime import factors


@export
def totatives(n: int) -> list[int]:
    r"""
    Returns the positive integers (totatives) in $[1, n)$ that are coprime to $n$.

    Arguments:
        n: A positive integer.

    Returns:
        The totatives of $n$.

    See Also:
        euler_phi, carmichael_lambda, is_cyclic

    Notes:
        The totatives of $n$ form the multiplicative group $(\mathbb{Z}/n\mathbb{Z}){^\times}$.

    References:
        - Section 2.4.3 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - https://oeis.org/A000010

    Examples:
        Find the totatives that are coprime with $n = 20$.

        .. ipython:: python

            n = 20
            x = galois.totatives(n); x

        The number of totatives of $n$ is $\phi(n)$.

        .. ipython:: python

            phi = galois.euler_phi(n); phi
            len(x) == phi

    Group:
        number-theory-divisibility
    """
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        return [0]

    return [t for t in range(1, n) if math.gcd(n, t) == 1]


@export
def euler_phi(n: int) -> int:
    r"""
    Counts the positive integers (totatives) in $[1, n)$ that are coprime to $n$.

    Arguments:
        n: A positive integer.

    Returns:
        The number of totatives that are coprime to $n$.

    See Also:
        carmichael_lambda, totatives, is_cyclic

    Notes:
        This function implements the Euler totient function

        $$\phi(n) = n \prod_{p \mid n} \bigg(1 - \frac{1}{p}\bigg) = \prod_{i=1}^{k} p_i^{e_i-1} \big(p_i - 1\big)$$

        for prime $p$ and the prime factorization $n = p_1^{e_1} \dots p_k^{e_k}$.

    References:
        - Section 2.4.1 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - https://oeis.org/A000010

    Examples:
        Compute $\phi(20)$.

        .. ipython:: python

            n = 20
            phi = galois.euler_phi(n); phi

        Find the totatives that are coprime with $n = 20$. The number of totatives of $n$ is
        $\phi(n)$.

        .. ipython:: python

            x = galois.totatives(n); x
            len(x) == phi

        For prime $n$, $\phi(n) = n - 1$.

        .. ipython:: python

            n = 13
            galois.euler_phi(n)

    Group:
        number-theory-divisibility
    """
    return _euler_phi(n)


# NOTE: This is a separate function to hide the "lru_cache" from the public API
@functools.lru_cache(maxsize=64)
def _euler_phi(n: int) -> int:
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        return 1

    p, e = factors(n)

    phi = 1
    for pi, ei in zip(p, e):
        phi *= pi ** (ei - 1) * (pi - 1)

    return phi


@export
def carmichael_lambda(n: int) -> int:
    r"""
    Finds the smallest positive integer $m$ such that $a^m \equiv 1\ (\textrm{mod}\ n)$ for
    every integer $a$ in $[1, n)$ that is coprime to $n$.

    Arguments:
        n: A positive integer.

    Returns:
        The smallest positive integer $m$ such that $a^m \equiv 1\ (\textrm{mod}\ n)$ for
        every $a$ in $[1, n)$ that is coprime to $n$.

    See Also:
        euler_phi, totatives, is_cyclic

    Notes:
        This function implements the Carmichael function $\lambda(n)$.

    References:
        - https://oeis.org/A002322

    Examples:
        The Carmichael $\lambda(n)$ function and Euler $\phi(n)$ function are often equal. However,
        there are notable exceptions.

        .. ipython:: python

            [galois.euler_phi(n) for n in range(1, 20)]
            [galois.carmichael_lambda(n) for n in range(1, 20)]

        For prime $n$, $\phi(n) = \lambda(n) = n - 1$. And for most composite $n$,
        $\phi(n) = \lambda(n) < n - 1$.

        .. ipython:: python

            n = 9
            phi = galois.euler_phi(n); phi
            lambda_ = galois.carmichael_lambda(n); lambda_
            totatives = galois.totatives(n); totatives

            for power in range(1, phi + 1):
                y = [pow(a, power, n) for a in totatives]
                print("Power {}: {} (mod {})".format(power, y, n))

            galois.is_cyclic(n)

        When $\phi(n) \ne \lambda(n)$, the multiplicative group $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is
        not cyclic. See :func:`~galois.is_cyclic`.

        .. ipython:: python

            n = 8
            phi = galois.euler_phi(n); phi
            lambda_ = galois.carmichael_lambda(n); lambda_
            totatives = galois.totatives(n); totatives

            for power in range(1, phi + 1):
                y = [pow(a, power, n) for a in totatives]
                print("Power {}: {} (mod {})".format(power, y, n))

            galois.is_cyclic(n)

    Group:
        number-theory-congruences
    """
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        return 1

    p, e = factors(n)

    lambdas = []
    for pi, ei in zip(p, e):
        # Carmichael function for prime powers
        if pi == 2 and ei > 2:
            l = euler_phi(pi**ei) // 2
        else:
            l = euler_phi(pi**ei)
        lambdas.append(l)

    return lcm(*lambdas)


@export
def is_cyclic(n: int) -> bool:
    r"""
    Determines whether the multiplicative group $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is cyclic.

    Arguments:
        n: A positive integer.

    Returns:
        `True` if the multiplicative group $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is cyclic.

    See Also:
        euler_phi, carmichael_lambda, totatives

    Notes:
        The multiplicative group $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is the set of positive integers
        $1 \le a < n$ that are coprime with $n$. $(\mathbb{Z}/n\mathbb{Z}){^\times}$ being cyclic
        means that some primitive root of $n$, or generator, $g$ can generate the group
        $\{1, g, g^2, \dots, g^{\phi(n)-1}\}$, where $\phi(n)$ is Euler's totient function and calculates
        the order of the group. If $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is cyclic, the number of primitive roots
        is found by $\phi(\phi(n))$.

        $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is *cyclic* if and only if $n$ is 2, 4, $p^k$,
        or $2p^k$, where $p$ is an odd prime and $k$ is a positive integer.

    Examples:
        .. md-tab-set::

            .. md-tab-item:: n = 14

                The elements of $(\mathbb{Z}/14\mathbb{Z}){^\times} = \{1, 3, 5, 9, 11, 13\}$ are the totatives
                of 14.

                .. ipython:: python

                    n = 14
                    Znx = galois.totatives(n); Znx

                The Euler totient $\phi(n)$ function counts the totatives of $n$, which is equivalent to
                the order of $(\mathbb{Z}/n\mathbb{Z}){^\times}$.

                .. ipython:: python

                    phi = galois.euler_phi(n); phi
                    len(Znx) == phi

                Since 14 is of the form $2p^k$, the multiplicative group
                $(\mathbb{Z}/14\mathbb{Z}){^\times}$ is cyclic, meaning there exists at least one element that
                generates the group by its powers.

                .. ipython:: python

                    galois.is_cyclic(n)

                Find the smallest primitive root modulo 14. Observe that the powers of $g$ uniquely represent
                each element in $(\mathbb{Z}/14\mathbb{Z}){^\times}$.

                .. ipython:: python

                    g = galois.primitive_root(n); g
                    [pow(g, i, n) for i in range(0, phi)]

                Find the largest primitive root modulo 14. Observe that the powers of $g$ also uniquely represent
                each element in $(\mathbb{Z}/14\mathbb{Z}){^\times}$, although in a different order.

                .. ipython:: python

                    g = galois.primitive_root(n, method="max"); g
                    [pow(g, i, n) for i in range(0, phi)]

            .. md-tab-item:: n = 15

                A non-cyclic group is $(\mathbb{Z}/15\mathbb{Z}){^\times} = \{1, 2, 4, 7, 8, 11, 13, 14\}$.

                .. ipython:: python

                    n = 15
                    Znx = galois.totatives(n); Znx
                    phi = galois.euler_phi(n); phi

                Since 15 is not of the form 2, 4, $p^k$, or $2p^k$, the multiplicative group
                $(\mathbb{Z}/15\mathbb{Z}){^\times}$ is not cyclic, meaning no elements exist whose powers
                generate the group.

                .. ipython:: python

                    galois.is_cyclic(n)

                Below, every element is tested to see if it spans the group.

                .. ipython:: python

                    for a in Znx:
                        span = set([pow(a, i, n) for i in range(0, phi)])
                        primitive_root = span == set(Znx)
                        print("Element: {:2d}, Span: {:<13}, Primitive root: {}".format(a, str(span), primitive_root))

                The Carmichael $\lambda(n)$ function finds the maximum multiplicative order of any element, which
                is 4 and not 8.

                .. ipython:: python

                    galois.carmichael_lambda(n)

                Observe that no primitive roots modulo 15 exist and a `RuntimeError` is raised.

                .. ipython:: python
                    :okexcept:

                    galois.primitive_root(n)

            .. md-tab-item:: Prime fields

                For prime $n$, a primitive root modulo $n$ is also a primitive element of the Galois field
                $\mathrm{GF}(n)$.

                .. ipython:: python

                    n = 31
                    galois.is_cyclic(n)

                A primitive element is a generator of the multiplicative group
                $\mathrm{GF}(p)^{\times} = \{1, 2, \dots, p-1\} = \{1, g, g^2, \dots, g^{\phi(n)-1}\}$.

                .. ipython:: python

                    GF = galois.GF(n)
                    galois.primitive_root(n)
                    GF.primitive_element

                The number of primitive roots/elements is $\phi(\phi(n))$.

                .. ipython:: python

                    list(galois.primitive_roots(n))
                    GF.primitive_elements
                    galois.euler_phi(galois.euler_phi(n))

    Group:
        number-theory-congruences
    """
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        return True

    p, e = factors(n)

    if n in [2, 4]:
        return True

    if len(p) == 2 and 2 in p and e[p.index(2)] == 1:
        # n = 2 * p^e
        return True

    if len(p) == 1 and p[0] != 2:
        # n = p^e
        return True

    # n does not represent a cyclic group
    return False


@export
def primitive_root(
    n: int,
    start: int = 1,
    stop: int | None = None,
    method: Literal["min", "max", "random"] = "min",
) -> int:
    r"""
    Finds a primitive root modulo $n$ in the range $[\textrm{start}, \textrm{stop})$.

    Arguments:
        n: A positive integer.
        start: Starting value (inclusive) in the search for a primitive root.
        stop: Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds
            to $n$.
        method: The search method for finding the primitive root.

    Returns:
        A primitive root modulo $n$ in the specified range.

    Raises:
        RuntimeError: If no primitive roots exist in the specified range.

    See Also:
        primitive_roots, is_primitive_root, is_cyclic, totatives, euler_phi, carmichael_lambda

    Notes:
        The integer $g$ is a primitive root modulo $n$ if the totatives of $n$ can be generated by
        the powers of $g$. The totatives of $n$ are the positive integers in $[1, n)$ that are
        coprime with $n$.

        Alternatively said, $g$ is a primitive root modulo $n$ if and only if $g$ is a generator of
        the multiplicative group of integers modulo $n$
        $(\mathbb{Z}/n\mathbb{Z}){^\times} = \{1, g, g^2, \dots, g^{\phi(n)-1}\}$, where $\phi(n)$ is
        the order of the group.

        If $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is cyclic, the number of primitive roots modulo $n$ is
        given by $\phi(\phi(n))$.

    References:
        - Shoup, V. Searching for primitive roots in finite fields.
          https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
        - Hua, L.K. On the least primitive root of a prime.
          https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
        - http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples:
        .. md-tab-set::

            .. md-tab-item:: n = 14

                The elements of $(\mathbb{Z}/14\mathbb{Z}){^\times} = \{1, 3, 5, 9, 11, 13\}$ are the totatives
                of 14.

                .. ipython:: python

                    n = 14
                    Znx = galois.totatives(n); Znx

                The Euler totient $\phi(n)$ function counts the totatives of $n$, which is equivalent to
                the order of $(\mathbb{Z}/n\mathbb{Z}){^\times}$.

                .. ipython:: python

                    phi = galois.euler_phi(n); phi
                    len(Znx) == phi

                Since 14 is of the form $2p^k$, the multiplicative group
                $(\mathbb{Z}/14\mathbb{Z}){^\times}$ is cyclic, meaning there exists at least one element that
                generates the group by its powers.

                .. ipython:: python

                    galois.is_cyclic(n)

                Find the smallest primitive root modulo 14. Observe that the powers of $g$ uniquely represent
                each element in $(\mathbb{Z}/14\mathbb{Z}){^\times}$.

                .. ipython:: python

                    g = galois.primitive_root(n); g
                    [pow(g, i, n) for i in range(0, phi)]

                Find the largest primitive root modulo 14. Observe that the powers of $g$ also uniquely represent
                each element in $(\mathbb{Z}/14\mathbb{Z}){^\times}$, although in a different order.

                .. ipython:: python

                    g = galois.primitive_root(n, method="max"); g
                    [pow(g, i, n) for i in range(0, phi)]

            .. md-tab-item:: n = 15

                A non-cyclic group is $(\mathbb{Z}/15\mathbb{Z}){^\times} = \{1, 2, 4, 7, 8, 11, 13, 14\}$.

                .. ipython:: python

                    n = 15
                    Znx = galois.totatives(n); Znx
                    phi = galois.euler_phi(n); phi

                Since 15 is not of the form 2, 4, $p^k$, or $2p^k$, the multiplicative group
                $(\mathbb{Z}/15\mathbb{Z}){^\times}$ is not cyclic, meaning no elements exist whose powers
                generate the group.

                .. ipython:: python

                    galois.is_cyclic(n)

                Below, every element is tested to see if it spans the group.

                .. ipython:: python

                    for a in Znx:
                        span = set([pow(a, i, n) for i in range(0, phi)])
                        primitive_root = span == set(Znx)
                        print("Element: {:2d}, Span: {:<13}, Primitive root: {}".format(a, str(span), primitive_root))

                The Carmichael $\lambda(n)$ function finds the maximum multiplicative order of any element,
                which is 4 and not 8.

                .. ipython:: python

                    galois.carmichael_lambda(n)

                Observe that no primitive roots modulo 15 exist and a `RuntimeError` is raised.

                .. ipython:: python
                    :okexcept:

                    galois.primitive_root(n)

            .. md-tab-item:: Very large n

                The algorithm is also efficient for very large $n$.

                .. ipython:: python

                    n = 1000000000000000035000061
                    phi = galois.euler_phi(n); phi

                Find the smallest, the largest, and a random primitive root modulo $n$.

                .. ipython:: python

                    galois.primitive_root(n)
                    galois.primitive_root(n, method="max")
                    galois.primitive_root(n, method="random")

    Group:
        number-theory-primitive-roots
    """
    if n in [1, 2]:
        return n - 1

    verify_isinstance(n, int)
    verify_isinstance(start, int)
    verify_isinstance(stop, int, optional=True)

    stop = n if stop is None else stop
    if not 1 <= start < stop <= n:
        raise ValueError(f"Arguments must satisfy 1 <= start < stop <= n, not 1 <= {start} < {stop} <= {n}.")
    if not method in ["min", "max", "random"]:
        raise ValueError(f"Argument 'method' must be in ['min', 'max', 'random'], not {method!r}.")

    try:
        if method == "min":
            root = next(primitive_roots(n, start, stop=stop))
        elif method == "max":
            root = next(primitive_roots(n, start, stop=stop, reverse=True))
        else:
            root = _primitive_root_random_search(n, start, stop)
        return root
    except StopIteration as e:
        raise RuntimeError(f"No primitive roots modulo {n} exist in the range [{start}, {stop}).") from e


@export
def primitive_roots(
    n: int,
    start: int = 1,
    stop: int | None = None,
    reverse: bool = False,
) -> Iterator[int]:
    r"""
    Iterates through all primitive roots modulo $n$ in the range $[\textrm{start}, \textrm{stop})$.

    Arguments:
        n: A positive integer.
        start: Starting value (inclusive) in the search for a primitive root. The default is 1.
        stop: Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds
            to $n$.
        reverse: Indicates to return the primitive roots from largest to smallest. The default is `False`.

    Returns:
        An iterator over the primitive roots modulo $n$ in the specified range.

    See Also:
        primitive_root, is_primitive_root, is_cyclic, totatives, euler_phi, carmichael_lambda

    Notes:
        The integer $g$ is a primitive root modulo $n$ if the totatives of $n$ can be generated by
        the powers of $g$. The totatives of $n$ are the positive integers in $[1, n)$ that are
        coprime with $n$.

        Alternatively said, $g$ is a primitive root modulo $n$ if and only if $g$ is a generator of
        the multiplicative group of integers modulo $n$
        $(\mathbb{Z}/n\mathbb{Z}){^\times} = \{1, g, g^2, \dots, g^{\phi(n)-1}\}$, where $\phi(n)$ is
        the order of the group.

        If $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is cyclic, the number of primitive roots modulo $n$ is
        given by $\phi(\phi(n))$.

    References:
        - Shoup, V. Searching for primitive roots in finite fields.
          https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
        - Hua, L.K. On the least primitive root of a prime.
          https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
        - http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples:
        All primitive roots modulo 31. You may also use `tuple()` on the returned generator.

        .. ipython:: python

            list(galois.primitive_roots(31))

        There are no primitive roots modulo 30.

        .. ipython:: python

            list(galois.primitive_roots(30))

        Show the each primitive root modulo 22 generates the multiplicative group
        $(\mathbb{Z}/22\mathbb{Z}){^\times}$.

        .. ipython:: python

            n = 22
            Znx = galois.totatives(n); Znx
            phi = galois.euler_phi(n); phi
            for root in galois.primitive_roots(22):
                span = set(pow(root, i, n) for i in range(0, phi))
                print(f"Element: {root:>2}, Span: {span}")

        Find the three largest primitive roots modulo 31 in reversed order.

        .. ipython:: python

            generator = galois.primitive_roots(31, reverse=True); generator
            [next(generator) for _ in range(3)]

        Loop over all the primitive roots in reversed order, only finding them as needed. The search cost for the
        roots that would have been found after the `break` condition is never incurred.

        .. ipython:: python

            for root in galois.primitive_roots(31, reverse=True):
                print(root)
                if root % 7 == 0:  # Arbitrary early exit condition
                    break

    Group:
        number-theory-primitive-roots
    """
    if n in [1, 2]:
        yield n - 1
        return

    stop = n if stop is None else stop
    verify_isinstance(n, int)
    verify_isinstance(start, int)
    verify_isinstance(stop, int)
    verify_isinstance(reverse, bool)
    if not 1 <= start < stop <= n:
        raise ValueError(f"Arguments must satisfy `1 <= start < stop <= n`, not `1 <= {start} < {stop} <= {n}`.")

    # If the multiplicative group (Z/nZ)* is not cyclic, then it has no multiplicative generators
    if not is_cyclic(n):
        return

    phi = euler_phi(n)  # Number of non-zero elements in the multiplicative group (Z/nZ)*
    if phi == n - 1 or n % 2 == 1:
        # For prime n or odd n, we must test all elements
        step = 1
    else:
        # For even n, we only have to test odd elements
        if start % 2 == 0:
            start += 1  # Make start odd
        step = 2

    if reverse:
        start, stop, step = stop - 1, start - 1, -1

    while True:
        root = _primitive_root_deterministic_search(n, start, stop, step)
        if root is not None:
            start = root + step
            yield root
        else:
            break


# @functools.lru_cache(maxsize=4096)
def _primitive_root_deterministic_search(n, start, stop, step) -> int | None:
    """
    Searches for a primitive root in the range using the specified deterministic method.
    """
    for root in range(start, stop, step):
        if _is_primitive_root(root, n):
            return root

    return None


def _primitive_root_random_search(n, start, stop) -> int:
    """
    Searches for a random primitive root.
    """
    i = 0
    while True:
        root = random.randint(start, stop - 1)
        if _is_primitive_root(root, n):
            return root

        i += 1
        if i > 2 * (stop - start):
            # A primitive root should have been found given 2*N tries
            raise StopIteration


@export
def is_primitive_root(g: int, n: int) -> bool:
    r"""
    Determines if $g$ is a primitive root modulo $n$.

    Arguments:
        g: A positive integer.
        n: positive integer.

    Returns:
        `True` if $g$ is a primitive root modulo $n$.

    See Also:
        primitive_root, primitive_roots, is_cyclic, euler_phi

    Notes:
        The integer $g$ is a primitive root modulo $n$ if the totatives of $n$, the positive integers
        $1 \le a < n$ that are coprime with $n$, can be generated by powers of $g$.

        Alternatively said, $g$ is a primitive root modulo $n$ if and only if $g$ is a generator of
        the multiplicative group of integers modulo $n$,

        $$(\mathbb{Z}/n\mathbb{Z}){^\times} = \{1, g, g^2, \dots, g^{\phi(n)-1}\}$$

        where $\phi(n)$ is order of the group.

        If $(\mathbb{Z}/n\mathbb{Z}){^\times}$ is cyclic, the number of primitive roots modulo $n$ is
        given by $\phi(\phi(n))$.

    Examples:
        .. ipython:: python

            list(galois.primitive_roots(7))
            galois.is_primitive_root(2, 7)
            galois.is_primitive_root(3, 7)

    Group:
        number-theory-primitive-roots
    """
    verify_isinstance(g, int)
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")
    if not 0 < g < n:
        raise ValueError(f"Argument 'g' must be a positive integer less than 'n', not {g}.")

    return _is_primitive_root(g, n)


def _is_primitive_root(g: int, n: int) -> bool:
    """
    A private version of `is_primitive_root()` without type checking for internal use.
    """
    if n == 2:
        # Euler totient of 2 is 1. We cannot compute the prime factorization of 1. There is only one
        # primitive root modulo 2 and it's 1.
        return g == 1

    phi = euler_phi(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    primes, _ = factors(phi)

    return pow(g, phi, n) == 1 and all(pow(g, phi // p, n) != 1 for p in primes)
