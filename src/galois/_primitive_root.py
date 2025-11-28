"""
A module containing functions to generate and test primitive roots modulo n.
"""

from __future__ import annotations

import random
from typing import Iterator

from typing_extensions import Literal

from ._helper import export, verify_isinstance
from ._modular import euler_phi, is_cyclic
from ._prime import factors


@export
def is_primitive_root(g: int, n: int) -> bool:
    r"""
    Determines if $g$ is a primitive root modulo $n$.

    Arguments:
        g: An integer in $[1, n)$.
        n: A positive integer.

    Returns:
        `True` if $g$ is a primitive root modulo $n$.

    See Also:
        primitive_root, primitive_roots, is_cyclic, euler_phi

    Notes:
        Let $(\mathbb{Z}/n\mathbb{Z})^\times$ denote the multiplicative group of units modulo $n$:

        $$
        (\mathbb{Z}/n\mathbb{Z})^\times
        = \{ [a]_n : 1 \le a < n,\ \gcd(a, n) = 1 \},
        $$

        with group operation given by multiplication modulo $n$. Its order is Euler's totient:

        $$
        \left|(\mathbb{Z}/n\mathbb{Z})^\times\right| = \varphi(n).
        $$

        An integer $g$ is a **primitive root modulo $n$** if its residue class $[g]_n$ generates
        $(\mathbb{Z}/n\mathbb{Z})^\times$:

        $$
        (\mathbb{Z}/n\mathbb{Z})^\times = \{ [g^0]_n, [g^1]_n, \dots, [g^{\varphi(n)-1}]_n \}.
        $$

        Equivalently, the multiplicative order of $g$ modulo $n$ is $\varphi(n)$:

        $$
        \operatorname{ord}_n(g) = \varphi(n).
        $$

        If $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic, the number of primitive roots modulo $n$ is
        $\varphi(\varphi(n))$. See :func:`~galois.is_cyclic` for the classification of which $n$ admit
        primitive roots.

        This function uses the standard order test. Let $\varphi(n)$ have prime factorization

        $$
        \varphi(n) = \prod_{i=1}^k q_i^{f_i}.
        $$

        Then $g$ is a primitive root modulo $n$ if and only if

        $$
        g^{\varphi(n)} \equiv 1 \pmod{n}
        \quad\text{and}\quad
        g^{\varphi(n)/q_i} \not\equiv 1 \pmod{n}
        \quad\text{for all distinct primes } q_i.
        $$

        In particular, if $\gcd(g, n) \ne 1$, then $g$ cannot be a primitive root.

    Examples:
        Primitive roots modulo 7 and membership checks.

        .. ipython:: python

            list(galois.primitive_roots(7))
            assert not galois.is_primitive_root(2, 7)
            assert galois.is_primitive_root(3, 7)

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
    A private version of :func:`is_primitive_root` without type checking for internal use.
    """
    if n == 2:
        # (Z/2Z)^× = {1} has a single element of order 1, so 1 is the unique primitive root modulo 2.
        return g == 1

    phi = euler_phi(n)  # |(Z/nZ)^×|
    primes, _ = factors(phi)

    # g is primitive iff:
    #   - g^φ(n) ≡ 1 (mod n), and
    #   - for every prime q | φ(n), g^{φ(n)/q} ≢ 1 (mod n).
    return pow(g, phi, n) == 1 and all(pow(g, phi // q, n) != 1 for q in primes)


@export
def primitive_root(
    n: int,
    start: int = 1,
    stop: int | None = None,
    method: Literal["min", "max", "random"] = "min",
) -> int:
    r"""
    Finds a primitive root modulo $n$ in the range $[\texttt{start}, \texttt{stop})$.

    Arguments:
        n: A positive integer.
        start: Starting value (inclusive) in the search for a primitive root. The default is 1.
        stop: Stopping value (exclusive) in the search for a primitive root. The default is `None`,
            which corresponds to `n`.
        method: The search method for finding the primitive root. Choices are:

            - `"min"`: Returns the smallest primitive root in the range.
            - `"max"`: Returns the largest primitive root in the range.
            - `"random"`: Returns a random primitive root in the range.

    Returns:
        A primitive root modulo $n$ in the specified range.

    See Also:
        primitive_roots, is_primitive_root, is_cyclic, totatives, euler_phi, carmichael_lambda

    Notes:
        An integer $g$ is a primitive root modulo $n$ if and only if $g$ generates the multiplicative
        group of units $(\mathbb{Z}/n\mathbb{Z})^\times$:

        $$
        (\mathbb{Z}/n\mathbb{Z})^\times = \{ [g^0]_n, [g^1]_n, \dots, [g^{\varphi(n)-1}]_n \},
        $$

        where $\varphi(n)$ is Euler's totient function and equals the order of this group.

        Existence of primitive roots depends only on $n$. By a classical theorem, the group
        $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic (and hence has primitive roots) if and only if

        $$
        n \in \{1, 2, 4\} \quad \text{or} \quad
        n = p^k \text{ or } n = 2p^k
        $$

        for some odd prime $p$ and integer $k \ge 1$. See :func:`~galois.is_cyclic` for a convenience test.

        - If $(\mathbb{Z}/n\mathbb{Z})^\times$ is not cyclic, no primitive roots modulo $n$ exist and
          this function raises a `RuntimeError`.
        - If $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic, the number of primitive roots modulo $n$ is
          $\varphi(\varphi(n))$, and this function finds one in the specified range.

        The edge cases `n = 1` and `n = 2` are handled by convention:

        - For `n = 1`, the trivial group is considered cyclic and this function returns `0`.
        - For `n = 2`, the unique primitive root modulo 2 is `1`.

    References:
        - Shoup, V. "Searching for primitive roots in finite fields."
          https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/
        - Hua, L. K. "On the least primitive root of a prime."
          https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/
        - http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples:
        .. md-tab-set::

            .. md-tab-item:: n = 14

                The elements of $(\mathbb{Z}/14\mathbb{Z})^\times = \{1, 3, 5, 9, 11, 13\}$ are the totatives
                of 14.

                .. ipython:: python

                    n = 14
                    Znx = galois.totatives(n); Znx

                The Euler totient $\varphi(n)$ function counts the totatives of $n$, which is equivalent to
                the order of $(\mathbb{Z}/n\mathbb{Z})^\times$.

                .. ipython:: python

                    phi = galois.euler_phi(n); phi
                    assert len(Znx) == phi

                Since 14 is of the form $2p^k$, the multiplicative group
                $(\mathbb{Z}/14\mathbb{Z})^\times$ is cyclic, meaning there exists at least one element that
                generates the group by its powers.

                .. ipython:: python

                    assert galois.is_cyclic(n)

                Find the smallest primitive root modulo 14. Observe that the powers of `g` uniquely represent
                each element in $(\mathbb{Z}/14\mathbb{Z})^\times$.

                .. ipython:: python

                    g = galois.primitive_root(n); g
                    [pow(g, i, n) for i in range(0, phi)]

                Find the largest primitive root modulo 14. Observe that the powers of `g` also uniquely represent
                each element in $(\mathbb{Z}/14\mathbb{Z})^\times$, although in a different order.

                .. ipython:: python

                    g = galois.primitive_root(n, method="max"); g
                    [pow(g, i, n) for i in range(0, phi)]

            .. md-tab-item:: n = 15

                A non-cyclic group is $(\mathbb{Z}/15\mathbb{Z})^\times = \{1, 2, 4, 7, 8, 11, 13, 14\}$.

                .. ipython:: python

                    n = 15
                    Znx = galois.totatives(n); Znx
                    phi = galois.euler_phi(n); phi

                Since 15 is not of the form $2$, $4$, $p^k$, or $2p^k$, the multiplicative group
                $(\mathbb{Z}/15\mathbb{Z})^\times$ is not cyclic, meaning no elements exist whose powers
                generate the group.

                .. ipython:: python

                    assert not galois.is_cyclic(n)

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
    verify_isinstance(n, int)
    verify_isinstance(start, int)
    verify_isinstance(stop, int, optional=True)

    if n in [1, 2]:
        # By convention:
        #   n = 1 -> return 0 (the unique residue class),
        #   n = 2 -> return 1 (the unique primitive root modulo 2).
        return n - 1

    stop = n if stop is None else stop
    if not 1 <= start < stop <= n:
        raise ValueError(f"Arguments must satisfy 1 <= start < stop <= n, not 1 <= {start} < {stop} <= {n}.")
    if method not in ["min", "max", "random"]:
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
        # No primitive root found in the requested range (either non-cyclic group or range too small).
        raise RuntimeError(f"No primitive roots modulo {n} exist in the range [{start}, {stop}).") from e


@export
def primitive_roots(
    n: int,
    start: int = 1,
    stop: int | None = None,
    reverse: bool = False,
) -> Iterator[int]:
    r"""
    Iterates through all primitive roots modulo $n$ in the range $[\texttt{start}, \texttt{stop})$.

    Arguments:
        n: A positive integer.
        start: Starting value (inclusive) in the search for primitive roots. The default is 1.
        stop: Stopping value (exclusive) in the search for primitive roots. The default is `None`,
            which corresponds to `n`.
        reverse: Indicates whether to return the primitive roots from largest to smallest. The default is `False`.

    Returns:
        An iterator over the primitive roots modulo $n$ in the specified range.

    See Also:
        primitive_root, is_primitive_root, is_cyclic, totatives, euler_phi, carmichael_lambda

    Notes:
        An integer $g$ is a primitive root modulo $n$ if and only if it generates the multiplicative
        group of units $(\mathbb{Z}/n\mathbb{Z})^\times$:

        $$
        (\mathbb{Z}/n\mathbb{Z})^\times = \{ [g^0]_n, [g^1]_n, \dots, [g^{\varphi(n)-1}]_n \}.
        $$

        If $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic, the number of primitive roots modulo $n$ is
        $\varphi(\varphi(n))$, and this function yields all such roots (within the specified range)
        in increasing order by default, or decreasing order when `reverse=True`.

        If $(\mathbb{Z}/n\mathbb{Z})^\times$ is not cyclic (see :func:`~galois.is_cyclic`), then no primitive
        roots exist and this generator yields no values.

        For efficiency, when $n$ is even and the group is cyclic, the search automatically skips even
        candidates since they cannot be units modulo $n$.

        The special cases `n = 1` and `n = 2` are handled by convention:

        - For `n = 1`, the generator yields a single value `0`.
        - For `n = 2`, the generator yields a single value `1`.

    References:
        - Shoup, V. "Searching for primitive roots in finite fields."
          https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/
        - Hua, L. K. "On the least primitive root of a prime."
          https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/
        - http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples:
        All primitive roots modulo 31. You may also use `tuple()` on the returned generator.

        .. ipython:: python

            list(galois.primitive_roots(31))

        There are no primitive roots modulo 30.

        .. ipython:: python

            list(galois.primitive_roots(30))

        Show that each primitive root modulo 22 generates the multiplicative group
        $(\mathbb{Z}/22\mathbb{Z})^\times$.

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
    verify_isinstance(n, int)
    verify_isinstance(start, int)
    verify_isinstance(stop, int, optional=True)
    verify_isinstance(reverse, bool)

    if n in [1, 2]:
        # n = 1 -> yield 0; n = 2 -> yield 1.
        yield n - 1
        return

    stop = n if stop is None else stop
    if not 1 <= start < stop <= n:
        raise ValueError(f"Arguments must satisfy 1 <= start < stop <= n, not 1 <= {start} < {stop} <= {n}.")

    # If the multiplicative group (Z/nZ)^× is not cyclic, then it has no multiplicative generators.
    if not is_cyclic(n):
        return

    phi = euler_phi(n)  # |(Z/nZ)^×|
    if phi == n - 1 or n % 2 == 1:
        # For prime n or odd n, test all integers in [start, stop).
        step = 1
    else:
        # For even n, only odd integers can be units modulo n, so skip even candidates.
        if start % 2 == 0:
            start += 1
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


def _primitive_root_deterministic_search(n: int, start: int, stop: int, step: int) -> int | None:
    """
    Searches deterministically for a primitive root in the range [start, stop) with the given step.
    """
    for candidate in range(start, stop, step):
        if _is_primitive_root(candidate, n):
            return candidate

    return None


def _primitive_root_random_search(n: int, start: int, stop: int) -> int:
    """
    Searches for a primitive root by random sampling in the range [start, stop).
    """
    i = 0
    while True:
        root = random.randint(start, stop - 1)
        if _is_primitive_root(root, n):
            return root

        i += 1
        if i > 2 * (stop - start):
            # A primitive root should have been found with high probability after 2 * (stop - start) trials
            # if one exists in the range; fall back to the caller via StopIteration.
            raise StopIteration
