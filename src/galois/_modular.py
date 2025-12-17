"""
A module containing functions for modular arithmetic.
"""

from __future__ import annotations

import functools
import math

from ._helper import export, verify_isinstance
from ._math import lcm
from ._prime import factors


@export
def totatives(n: int) -> list[int]:
    r"""
    Returns the integers (totatives) in $[1, n)$ that are coprime to $n$.

    Arguments:
        n: A positive integer.

    Returns:
        A list of the totatives of $n$, sorted in increasing order. For $n > 1$, this is the set

        $$
        \{ t \in \mathbb{Z} : 1 \le t < n,\ \gcd(t, n) = 1 \}.
        $$

        For $n = 1$, the function returns `[0]`, representing the unique residue class modulo $1$
        (and ensuring the length of the list equals $\varphi(1) = 1$).

    See Also:
        euler_phi, carmichael_lambda, is_cyclic

    Notes:
        For $n > 1$, the totatives of $n$ are precisely the integers in $[1, n)$ that are coprime
        to $n$:

        $$
        \{ t \in \{1, 2, \dots, n - 1\} : \gcd(t, n) = 1 \}.
        $$

        Modulo $n$, these correspond to the units in the ring $\mathbb{Z}/n\mathbb{Z}$, and under
        multiplication modulo $n$ they form the multiplicative group of units $(\mathbb{Z}/n\mathbb{Z})^\times.$

        The number of totatives of $n$ is Euler's totient function $\varphi(n)$, so
        `len(totatives(n)) == euler_phi(n)` for all $n \ge 1$. For $n = 1$, the ring
        $\mathbb{Z}/1\mathbb{Z}$ has a single residue class, and this implementation represents
        that class by the integer `0`, hence the special case `[0]`.

    References:
        - Section 2.4.3 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - https://oeis.org/A000010

    Examples:
        Find the totatives that are coprime with $n = 20$.

        .. ipython:: python

            n = 20
            x = galois.totatives(n); x

        The number of totatives of $n$ is $\varphi(n)$.

        .. ipython:: python

            phi = galois.euler_phi(n); phi
            assert len(x) == phi

    Group:
        number-theory-divisibility
    """
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        # Represent the unique residue class modulo 1 by 0; ensures len(totatives(1)) == euler_phi(1) == 1.
        return [0]

    return [t for t in range(1, n) if math.gcd(n, t) == 1]


@export
def euler_phi(n: int) -> int:
    r"""
    Counts the positive integers in $[1, n]$ that are coprime to $n$.

    Arguments:
        n: A positive integer.

    Returns:
        The Euler totient $\varphi(n)$, the number of integers $k$ with $1 \le k \le n$ and
        $\gcd(k, n) = 1$.

    See Also:
        carmichael_lambda, totatives, is_cyclic

    Notes:
        The Euler totient function $\varphi(n)$ is defined by

        $$
        \varphi(n) = \left|\{ k \in \mathbb{Z} : 1 \le k \le n,\ \gcd(k, n) = 1 \}\right|.
        $$

        Equivalently, for $n > 1$, it counts the totatives of $n$ in $[1, n)$, so for all
        $n \ge 1$:

        $$
        \varphi(n) = \lvert \texttt{totatives}(n) \rvert.
        $$

        The function is multiplicative in the sense that if $\gcd(m, n) = 1$, then

        $$
        \varphi(mn) = \varphi(m)\,\varphi(n).
        $$

        Given the prime factorization

        $$
        n = p_1^{e_1} p_2^{e_2} \cdots p_k^{e_k},
        $$

        the totient can be computed as

        $$
        \varphi(n)
        = n \prod_{p \mid n}\left(1 - \frac{1}{p}\right)
        = \prod_{i=1}^{k} p_i^{e_i - 1}(p_i - 1),
        $$

        where the product over $p \mid n$ runs over the distinct primes dividing $n$.
        This implementation uses the latter formula.

        The special case $n = 1$ is defined by convention as $\varphi(1) = 1$,
        since the ring $\mathbb{Z}/1\mathbb{Z}$ has a single residue class, which is counted as
        one "unit" in this context.

        Group-theoretically, $\varphi(n)$ is the size of the multiplicative group of units

        $$
        \varphi(n) = \left|(\mathbb{Z}/n\mathbb{Z})^\times\right|.
        $$

    References:
        - Section 2.4.1 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
        - https://oeis.org/A000010

    Examples:
        Compute $\varphi(20)$.

        .. ipython:: python

            n = 20
            phi = galois.euler_phi(n); phi

        Find the totatives that are coprime with $n = 20$. The number of totatives of $n$ is
        $\varphi(n)$.

        .. ipython:: python

            x = galois.totatives(n); x
            assert len(x) == phi

        For prime $n$, $\varphi(n) = n - 1$.

        .. ipython:: python

            n = 13
            galois.euler_phi(n)

    Group:
        number-theory-divisibility
    """
    return _euler_phi(n)


# NOTE: This is a separate function to hide the "lru_cache" from the public API.
@functools.lru_cache(maxsize=64)
def _euler_phi(n: int) -> int:
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        # By convention, φ(1) = 1.
        return 1

    p, e = factors(n)

    phi = 1
    for pi, ei in zip(p, e):
        phi *= pi ** (ei - 1) * (pi - 1)

    return phi


@export
def mobius(n: int) -> int:
    r"""
    Evaluates the Möbius function $\mu(n)$.

    Arguments:
        n: A positive integer.

    Returns:
        The Möbius function $\mu(n)$, which takes values in $\{-1, 0, 1\}$ and is defined by

        .. math::

            \mu(n) = \begin{cases}
                1,       & \text{if } n = 1, \\
                0,       & \text{if } p^2 \mid n \text{ for some prime } p, \\
                (-1)^k,  & \text{if } n = p_1 p_2 \cdots p_k \text{ is a product of } k \text{ distinct primes}.
            \end{cases}

    See Also:
        euler_phi, carmichael_lambda, totatives, is_square_free

    Notes:
        The Möbius function $\mu(n)$ is a multiplicative arithmetic function that encodes the
        square-free structure of $n$:

        - $\mu(1) = 1$.
        - $\mu(n) = 0$ if $n$ is divisible by the square of a prime (i.e., $n$ is not square-free).
        - If $n$ is square-free with prime factorization $n = p_1 p_2 \cdots p_k$, then $\mu(n) = (-1)^k$,
          so $\mu(n) = -1$ when $n$ has an odd number of distinct prime factors, and
          $\mu(n) = 1$ when $n$ has an even number.

        The Möbius function plays a central role in Dirichlet convolution and inversion. If
        $f$ and $F$ are arithmetic functions related by

        $$
        F(n) = \sum_{d \mid n} f(d),
        $$

        then Möbius inversion recovers $f$ from $F$ via

        $$
        f(n) = \sum_{d \mid n} \mu(d)\, F\!\left(\frac{n}{d}\right).
        $$

        A classical identity characterizing $\mu$ is

        .. math::

            \sum_{d \mid n} \mu(d) =
            \begin{cases}
                1, & \text{if } n = 1, \\
                0, & \text{if } n > 1,
            \end{cases}

        meaning that $\mu$ is the Dirichlet inverse of the constant function $1(n) \equiv 1$.

        This implementation uses the prime factorization

        $$
        n = p_1^{e_1} p_2^{e_2} \cdots p_k^{e_k}.
        $$

        If any exponent $e_i > 1$, then $\mu(n) = 0$. Otherwise $n$ is square-free and
        $\mu(n) = (-1)^k$ where $k$ is the number of distinct prime factors.

    References:
        - https://oeis.org/A008683

    Examples:
        Basic values of the Möbius function.

        .. ipython:: python

            [galois.mobius(n) for n in range(1, 21)]

        Values for powers of primes and square-free products.

        .. ipython:: python

            # 1 by definition
            assert galois.mobius(1) == 1

            # 2, one prime factor
            assert galois.mobius(2) == -1

            # 2 * 3, two distinct primes
            assert galois.mobius(6) ==  1

            # 2^2 * 3, not square-free
            assert galois.mobius(12) ==  0

        Verify the identity $\sum_{d \mid n} \mu(d) = 0$ for $n > 1$.

        .. ipython:: python

            for n in range(1, 100):
                s = sum(galois.mobius(d) for d in range(1, n + 1) if n % d == 0)
                if n == 1:
                    assert s == 1
                else:
                    assert s == 0

    Group:
        number-theory-divisibility
    """
    return _mobius(n)


# NOTE: This is a separate function to hide the "lru_cache" from the public API.
@functools.lru_cache(maxsize=64)
def _mobius(n: int) -> int:
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        # By definition, μ(1) = 1.
        return 1

    p, e = factors(n)

    # If any exponent e_i > 1, then n is not square-free and μ(n) = 0.
    if any(ei > 1 for ei in e):
        return 0

    # n is square-free with k distinct prime factors -> μ(n) = (-1)^k.
    k = len(p)
    return -1 if k % 2 == 1 else 1


@export
def carmichael_lambda(n: int) -> int:
    r"""
    Finds the smallest positive integer $m$ such that $a^m \equiv 1 \pmod{n}$ for every integer
    $a$ in $[1, n)$ that is coprime to $n$.

    Arguments:
        n: A positive integer.

    Returns:
        The Carmichael function $\lambda(n)$, i.e., the smallest positive integer $m$ such that
        $a^m \equiv 1 \pmod{n}$ for every integer $a$ with $1 \le a < n$ and $\gcd(a, n) = 1$.

    See Also:
        euler_phi, totatives, is_cyclic

    Notes:
        The Carmichael function $\lambda(n)$ is defined as the least positive integer $m$ such that

        $$
        a^m \equiv 1 \pmod{n}
        $$

        for all integers $a$ with $\gcd(a, n) = 1$. In group-theoretic terms, if
        $(\mathbb{Z}/n\mathbb{Z})^\times$ denotes the multiplicative group of units modulo $n$,
        then $\lambda(n)$ is the **exponent** of this group:

        $$
        \lambda(n) = \min\{ m \ge 1 : g^m = 1 \text{ for all } g \in (\mathbb{Z}/n\mathbb{Z})^\times \}.
        $$

        This function is closely related to Euler's totient function $\varphi(n)$:

        - One always has $\lambda(n) \mid \varphi(n)$.
        - If $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic, then $\lambda(n) = \varphi(n)$, and there exists a
          generator (a primitive root modulo $n$).
        - If $(\mathbb{Z}/n\mathbb{Z})^\times$ is not cyclic, then $\lambda(n) < \varphi(n)$, and no single element
          generates all units.

        The Carmichael function can be computed from the prime factorization

        $$
        n = p_1^{e_1} p_2^{e_2} \cdots p_k^{e_k}
        $$

        as follows. First, compute $\lambda(p_i^{e_i})$ for each prime power; then

        $$
        \lambda(n) = \operatorname{lcm}\big(\lambda(p_1^{e_1}), \dots, \lambda(p_k^{e_k})\big).
        $$

        For each prime power $p^e$:

        - If $p$ is odd or $(p = 2$ and $e \le 2)$, then

          $$
          \lambda(p^e) = \varphi(p^e) = p^{e - 1}(p - 1).
          $$

        - If $p = 2$ and $e \ge 3$, then

          $$
          \lambda(2^e) = 2^{e - 2} = \frac{\varphi(2^e)}{2}.
          $$

        This implementation uses the above formulas:
        it computes $\lambda(p^e)$ from $\varphi(p^e)$, with the factor-of-two adjustment for
        $2^e$ when $e > 2$, and then returns the least common multiple of the resulting values.

        The special case $n = 1$ is defined by convention as $\lambda(1) = 1$, which is consistent with the view
        that $\mathbb{Z}/1\mathbb{Z}$ has a trivial (one-element) multiplicative structure.

    References:
        - https://oeis.org/A002322

    Examples:
        The Carmichael $\lambda(n)$ function and Euler $\varphi(n)$ function are often equal.
        However, there are notable exceptions.

        .. ipython:: python

            [galois.euler_phi(n) for n in range(1, 20)]
            [galois.carmichael_lambda(n) for n in range(1, 20)]

        For prime $n$, $\varphi(n) = \lambda(n) = n - 1$. And for many composite $n$,
        $\varphi(n) = \lambda(n) < n - 1$.

        .. ipython:: python

            n = 9
            phi = galois.euler_phi(n); phi
            lambda_ = galois.carmichael_lambda(n); lambda_
            totatives = galois.totatives(n); totatives

            for power in range(1, phi + 1):
                y = [pow(a, power, n) for a in totatives]
                print("Power {}: {} (mod {})".format(power, y, n))

            assert galois.is_cyclic(n)

        When $\varphi(n) \ne \lambda(n)$, the multiplicative group $(\mathbb{Z}/n\mathbb{Z})^\times$
        is not cyclic. See :func:`~galois.is_cyclic`.

        .. ipython:: python

            n = 8
            phi = galois.euler_phi(n); phi
            lambda_ = galois.carmichael_lambda(n); lambda_
            totatives = galois.totatives(n); totatives

            for power in range(1, phi + 1):
                y = [pow(a, power, n) for a in totatives]
                print("Power {}: {} (mod {})".format(power, y, n))

            assert not galois.is_cyclic(n)

    Group:
        number-theory-congruences
    """
    verify_isinstance(n, int)
    if not n > 0:
        raise ValueError(f"Argument 'n' must be a positive integer, not {n}.")

    if n == 1:
        # By convention, λ(1) = 1.
        return 1

    p, e = factors(n)

    lambdas = []
    for pi, ei in zip(p, e):
        # Carmichael function for prime powers:
        #   - λ(p^e) = φ(p^e) for odd p or for 2^e with e <= 2
        #   - λ(2^e) = φ(2^e)/2 for e > 2
        if pi == 2 and ei > 2:
            l = euler_phi(pi**ei) // 2
        else:
            l = euler_phi(pi**ei)
        lambdas.append(l)

    return lcm(*lambdas)


@export
def is_cyclic(n: int) -> bool:
    r"""
    Determines whether the multiplicative group $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic.

    Arguments:
        n: A positive integer.

    Returns:
        `True` if the multiplicative group $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic.

    See Also:
        euler_phi, carmichael_lambda, totatives

    Notes:
        For a positive integer $n$, the multiplicative group of units modulo $n$ is

        $$
        (\mathbb{Z}/n\mathbb{Z})^\times
        = \{ [a]_n : 1 \le a < n,\ \gcd(a, n) = 1 \},
        $$

        with multiplication induced by integer multiplication modulo $n$. Its order is Euler's
        totient:

        $$
        \left|(\mathbb{Z}/n\mathbb{Z})^\times\right| = \varphi(n).
        $$

        The group $(\mathbb{Z}/n\mathbb{Z})^\times$ is **cyclic** if there exists an element
        $g \in (\mathbb{Z}/n\mathbb{Z})^\times$ (a *primitive root modulo* $n$) such that

        $$
        (\mathbb{Z}/n\mathbb{Z})^\times = \{ g^0, g^1, \dots, g^{\varphi(n)-1} \}.
        $$

        In that case, the number of primitive roots modulo $n$ is
        $\varphi(\varphi(n))$, and every element of $(\mathbb{Z}/n\mathbb{Z})^\times$ is some
        power of $g$.

        A classical theorem completely characterizes when $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic:

        - The group $(\mathbb{Z}/n\mathbb{Z})^\times$ is cyclic **if and only if**

          $$
          n \in \{1, 2, 4\} \quad \text{or} \quad
          n = p^k \text{ or } n = 2p^k
          $$

          for some odd prime $p$ and integer $k \ge 1$.

        In particular:

        - For $n = p$ prime, $(\mathbb{Z}/p\mathbb{Z})^\times$ is always cyclic of order $p - 1$
          and admits primitive roots modulo $p$.
        - For $n = 2, 4, p^k$, or $2p^k$ (with $p$ odd), there are primitive roots modulo $n$.
        - For any other composite $n$, $(\mathbb{Z}/n\mathbb{Z})^\times$ is not cyclic, so no
          primitive roots modulo $n$ exist.
        - The trivial group $(\mathbb{Z}/1\mathbb{Z})^\times$ is cyclic by convention, and this
          function returns `True` for $n = 1$.

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
                    assert galois.is_cyclic(n)

                A primitive element is a generator of the multiplicative group
                $\mathrm{GF}(p)^\times = \{1, 2, \dots, p - 1\} = \{1, g, g^2, \dots, g^{\varphi(n)-1}\}$.

                .. ipython:: python

                    GF = galois.GF(n)
                    galois.primitive_root(n)
                    GF.primitive_element

                The number of primitive roots/elements is $\varphi(\varphi(n))$.

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
        # The trivial group (Z/1Z)^× is cyclic
        return True

    p, e = factors(n)

    if n in [2, 4]:
        return True

    if len(p) == 2 and 2 in p and e[p.index(2)] == 1:
        # n = 2 * p^k with p an odd prime (since 2 is one factor and has exponent 1)
        return True

    if len(p) == 1 and p[0] != 2:
        # n = p^k with p an odd prime
        return True

    # n does not represent a cyclic group
    return False
