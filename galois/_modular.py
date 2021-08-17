import math

import numpy as np

from ._factor import factors
from ._math import lcm
from ._overrides import set_module

__all__ = [
    "totatives", "euler_phi", "carmichael_lambda", "is_cyclic",
    "is_primitive_root", "primitive_root", "primitive_roots",
]


@set_module("galois")
def totatives(n):
    r"""
    Returns the positive integers (totatives) in :math:`[1, n]` that are coprime to :math:`n`.

    The totatives of :math:`n` form the multiplicative group :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    list
        The totatives of :math:`n`.

    References
    ----------
    * Section 2.4.3 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    * https://oeis.org/A000010

    Examples
    --------
    .. ipython:: python

        n = 20
        totatives = galois.totatives(n); totatives
        phi = galois.euler_phi(n); phi
        len(totatives) == phi
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return [0]
    else:
        return [t for t in range(1, n) if math.gcd(n, t) == 1]


@set_module("galois")
def euler_phi(n):
    r"""
    Counts the positive integers (totatives) in :math:`[1, n]` that are coprime to :math:`n`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The number of totatives that are coprime to :math:`n`.

    Notes
    -----
    This function implements the Euler totient function

    .. math:: \phi(n) = n \prod_{p\ |\ n} \bigg(1 - \frac{1}{p}\bigg) = \prod_{i=1}^{k} p_i^{e_i-1} \big(p_i - 1\big)

    for prime :math:`p` and the prime factorization :math:`n = p_1^{e_1} \dots p_k^{e_k}`.

    References
    ----------
    * Section 2.4.1 from https://cacr.uwaterloo.ca/hac/about/chap2.pdf
    * https://oeis.org/A000010

    Examples
    --------
    .. ipython:: python

        n = 20
        phi = galois.euler_phi(n); phi

        # Find the totatives that are coprime with n
        totatives = [k for k in range(n) if math.gcd(k, n) == 1]; totatives

        # The number of totatives is phi
        len(totatives) == phi

        # For prime n, ϕ(n) = n - 1
        galois.euler_phi(13)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return 1

    p, e = factors(n)

    phi = 1
    for pi, ei in zip(p, e):
        phi *= pi**(ei - 1) * (pi - 1)

    return phi


@set_module("galois")
def carmichael_lambda(n):
    r"""
    Finds the smallest positive integer :math:`m` such that :math:`a^m \equiv 1\ (\textrm{mod}\ n)` for
    every integer :math:`a` in :math:`[1, n]` that is coprime to :math:`n`.

    This function implements the Carmichael function :math:`\lambda(n)`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The smallest positive integer :math:`m` such that :math:`a^m \equiv 1 (\textrm{mod}\ n)` for
        every :math:`a` in :math:`[1, n]` that is coprime to :math:`n`.

    References
    ----------
    * https://oeis.org/A002322

    Examples
    --------
    The Carmichael lambda function and Euler totient function are often equal. However, there are notable exceptions.

    .. ipython:: python

        [galois.euler_phi(n) for n in range(1, 20)]
        [galois.carmichael_lambda(n) for n in range(1, 20)]

    For prime :math:`n`, :math:`\phi(n) = \lambda(n) = n - 1`. And for most composite :math:`n`, :math:`\phi(n) = \lambda(n) < n - 1`.

    .. ipython:: python

        n = 9
        phi = galois.euler_phi(n); phi
        lambda_ = galois.carmichael_lambda(n); lambda_
        totatives = galois.totatives(n); totatives

        for power in range(1, phi + 1):
            y = [pow(a, power, n) for a in totatives]
            print("Power {}: {} (mod {})".format(power, y, n))

        galois.is_cyclic(n)

    When :math:`\phi(n) \ne \lambda(n)`, the multiplicative group :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is not cyclic.
    See :func:`galois.is_cyclic`.

    .. ipython:: python

        n = 8
        phi = galois.euler_phi(n); phi
        lambda_ = galois.carmichael_lambda(n); lambda_
        totatives = galois.totatives(n); totatives

        for power in range(1, phi + 1):
            y = [pow(a, power, n) for a in totatives]
            print("Power {}: {} (mod {})".format(power, y, n))

        galois.is_cyclic(n)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return 1

    p, e = factors(n)

    lambdas = []
    for i in range(len(p)):
        # Carmichael function for prime powers
        if p[i] == 2 and e[i] > 2:
            l = euler_phi(p[i]**e[i]) // 2
        else:
            l = euler_phi(p[i]**e[i])
        lambdas.append(l)

    return lcm(*lambdas)


@set_module("galois")
def is_cyclic(n):
    r"""
    Determines whether the multiplicative group :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is cyclic.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool
        `True` if the multiplicative group :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is cyclic.

    Notes
    -----
    The multiplicative group :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is the set of positive integers :math:`1 \le a < n`
    that are coprime with :math:`n`. :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` being cyclic means that some primitive root of :math:`n`,
    or generator, :math:`g` can generate the group :math:`\{g^0, g^1, g^2, \dots, g^{\phi(n)-1}\}`, where
    :math:`\phi(n)` is Euler's totient function and calculates the order of the group. If :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is cyclic,
    the number of primitive roots is found by :math:`\phi(\phi(n))`.

    :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is *cyclic* if and only if :math:`n` is :math:`2`, :math:`4`, :math:`p^k`, or :math:`2p^k`,
    where :math:`p` is an odd prime and :math:`k` is a positive integer.

    Examples
    --------
    The elements of :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` are the positive integers less than :math:`n` that are coprime with :math:`n`.
    For example, :math:`(\mathbb{Z}/14\mathbb{Z}){^\times} = \{1, 3, 5, 9, 11, 13\}`.

    .. ipython:: python

        # n is of type 2*p^e, which is cyclic
        n = 14
        galois.is_cyclic(n)
        Znx = set(galois.totatives(n)); Znx
        phi = galois.euler_phi(n); phi
        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = galois.is_primitive_root(a, n)
            print("Element: {:2d}, Span: {:<20}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Euler's totient function ϕ(ϕ(n)) counts the primitive roots of n
        len(roots) == galois.euler_phi(phi)

    A counterexample is :math:`n = 15 = 3 \cdot 5`, which doesn't fit the condition for cyclicness.
    :math:`(\mathbb{Z}/15\mathbb{Z}){^\times} = \{1, 2, 4, 7, 8, 11, 13, 14\}`. Since the group is not cyclic, it has no primitive roots.

    .. ipython:: python

        # n is of type p1^e1 * p2^e2, which is not cyclic
        n = 15
        galois.is_cyclic(n)
        Znx = set(galois.totatives(n)); Znx
        phi = galois.euler_phi(n); phi
        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = galois.is_primitive_root(a, n)
            print("Element: {:2d}, Span: {:<13}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Note the max order of any element is 4, not 8, which is Carmichael's lambda function
        galois.carmichael_lambda(n)

    For prime :math:`n`, a primitive root modulo :math:`n` is also a primitive element of the Galois field :math:`\mathrm{GF}(n)`. A
    primitive element is a generator of the multiplicative group :math:`\mathrm{GF}(p)^{\times} = \{1, 2, \dots, p-1\} = \{g^0, g^1, g^2, \dots, g^{\phi(n)-1}\}`.

    .. ipython:: python

        # n is of type p, which is cyclic
        n = 7
        galois.is_cyclic(n)
        Znx = set(galois.totatives(n)); Znx
        phi = galois.euler_phi(n); phi
        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = galois.is_primitive_root(a, n)
            print("Element: {:2d}, Span: {:<18}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Euler's totient function ϕ(ϕ(n)) counts the primitive roots of n
        len(roots) == galois.euler_phi(phi)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return True

    p, e = factors(n)

    if n in [2, 4]:
        return True
    elif len(p) == 2 and 2 in p and e[p.index(2)] == 1:
        # n = 2 * p^e
        return True
    elif len(p) == 1 and p[0] != 2:
        # n = p^e
        return True
    else:
        # n does not represent a cyclic group
        return False


@set_module("galois")
def is_primitive_root(g, n):
    r"""
    Determines if :math:`g` is a primitive root modulo :math:`n`.

    Parameters
    ----------
    g : int
        A positive integer that may be a primitive root modulo :math:`n`.
    n : int
        A positive integer.

    Returns
    -------
    bool
        `True` if :math:`g` is a primitive root modulo :math:`n`.

    Notes
    -----
    :math:`g` is a primitive root if the totatives of :math:`n`, the positive integers :math:`1 \le a < n`
    that are coprime with :math:`n`, can be generated by powers of :math:`g`. Alternatively said, :math:`g` is a primitive root
    modulo :math:`n` if and only if :math:`g` is a generator of the multiplicative group of integers modulo :math:`n`,
    :math:`(\mathbb{Z}/n\mathbb{Z}){^\times} = \{g^0, g^1, g^2, \dots, g^{\phi(n)-1}\}` where :math:`\phi(n)` is order of the group.
    If :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is cyclic, the number of primitive roots modulo :math:`n` is given by :math:`\phi(\phi(n))`.

    Examples
    --------
    .. ipython:: python

        galois.is_primitive_root(2, 7)
        galois.is_primitive_root(3, 7)
        galois.primitive_roots(7)
    """
    if not isinstance(g, (int, np.integer)):
        raise TypeError(f"Argument `g` must be an integer, not {type(g)}.")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")
    if not 0 < g < n:
        raise ValueError(f"Argument `g` must be a positive integer less than `n`, not {g}.")

    if n == 2:
        # Euler totient of 2 is 1. We cannot compute the prime factorization of 1. There is only one
        # primitive root modulo 2 and it's 1.
        return g == 1

    phi = euler_phi(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    primes, _ = factors(phi)

    return pow(g, phi, n) == 1 and all(pow(g, phi // p, n) != 1 for p in primes)


@set_module("galois")
def primitive_root(n, start=1, stop=None, reverse=False):
    r"""
    Finds the smallest primitive root modulo :math:`n`.

    Parameters
    ----------
    n : int
        A positive integer.
    start : int, optional
        Starting value (inclusive) in the search for a primitive root. The default is 1. The resulting primitive
        root, if found, will be :math:`\textrm{start} \le g < \textrm{stop}`.
    stop : int, optional
        Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds to :math:`n`.
        The resulting primitive root, if found, will be :math:`\textrm{start} \le g < \textrm{stop}`.
    reverse : bool, optional
        Search for a primitive root in reverse order, i.e. find the largest primitive root first. Default is `False`.

    Returns
    -------
    int
        The smallest primitive root modulo :math:`n`. Returns `None` if no primitive roots exist.

    Notes
    -----
    :math:`g` is a primitive root if the totatives of :math:`n`, the positive integers :math:`1 \le a < n`
    that are coprime with :math:`n`, can be generated by powers of :math:`g`. Alternatively said, :math:`g` is a primitive root
    modulo :math:`n` if and only if :math:`g` is a generator of the multiplicative group of integers modulo :math:`n`,
    :math:`(\mathbb{Z}/n\mathbb{Z}){^\times} = \{g^0, g^1, g^2, \dots, g^{\phi(n)-1}\}` where :math:`\phi(n)` is order of the group.
    If :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is cyclic, the number of primitive roots modulo :math:`n` is given by :math:`\phi(\phi(n))`.

    References
    ----------
    * V. Shoup. Searching for primitive roots in finite fields. https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * L. K. Hua. On the least primitive root of a prime. https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
    * http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples
    --------
    The elements of :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` are the positive integers less than :math:`n` that are coprime with :math:`n`.
    For example, :math:`(\mathbb{Z}/14\mathbb{Z}){^\times} = \{1, 3, 5, 9, 11, 13\}`.

    .. ipython:: python

        # n is of type 2*p^k, which is cyclic
        n = 14
        galois.is_cyclic(n)

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_phi(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {:2d}, Span: {:<20}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Euler's totient function ϕ(ϕ(n)) counts the primitive roots of n
        len(roots) == galois.euler_phi(phi)

    A counterexample is :math:`n = 15 = 3 \cdot 5`, which doesn't fit the condition for cyclicness.
    :math:`(\mathbb{Z}/15\mathbb{Z}){^\times} = \{1, 2, 4, 7, 8, 11, 13, 14\}`.

    .. ipython:: python

        # n is of type p1^k1 * p2^k2, which is not cyclic
        n = 15
        galois.is_cyclic(n)

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_phi(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {:2d}, Span: {:<13}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Note the max order of any element is 4, not 8, which is Carmichael's lambda function
        galois.carmichael_lambda(n)

    The algorithm is also efficient for very large :math:`n`.

    .. ipython:: python

        n = 1000000000000000035000061
        galois.primitive_root(n)
    """
    try:
        return next(_primitive_roots(n, start=start, stop=stop, reverse=reverse))
    except StopIteration:
        return None


@set_module("galois")
def primitive_roots(n, start=1, stop=None, reverse=False):
    r"""
    Finds all primitive roots modulo :math:`n`.

    Parameters
    ----------
    n : int
        A positive integer.
    start : int, optional
        Starting value (inclusive) in the search for a primitive root. The default is 1. The resulting primitive
        roots, if found, will be :math:`\textrm{start} \le x < \textrm{stop}`.
    stop : int, optional
        Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds to `n`.
        The resulting primitive roots, if found, will be :math:`\textrm{start} \le x < \textrm{stop}`.
    reverse : bool, optional
        List all primitive roots in descending order, i.e. largest to smallest. Default is `False`.

    Returns
    -------
    list
        All the positive primitive :math:`n`-th roots of unity, :math:`x`.

    Notes
    -----
    :math:`g` is a primitive root if the totatives of :math:`n`, the positive integers :math:`1 \le a < n`
    that are coprime with :math:`n`, can be generated by powers of :math:`g`. Alternatively said, :math:`g` is a primitive root
    modulo :math:`n` if and only if :math:`g` is a generator of the multiplicative group of integers modulo :math:`n`,
    :math:`(\mathbb{Z}/n\mathbb{Z}){^\times} = \{g^0, g^1, g^2, \dots, g^{\phi(n)-1}\}` where :math:`\phi(n)` is order of the group.
    If :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` is cyclic, the number of primitive roots modulo :math:`n` is given by :math:`\phi(\phi(n))`.

    References
    ----------
    * V. Shoup. Searching for primitive roots in finite fields. https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples
    --------
    The elements of :math:`(\mathbb{Z}/n\mathbb{Z}){^\times}` are the positive integers less than :math:`n` that are coprime with :math:`n`.
    For example, :math:`(\mathbb{Z}/14\mathbb{Z}){^\times} = \{1, 3, 5, 9, 11, 13\}`.

    .. ipython:: python

        # n is of type 2*p^k, which is cyclic
        n = 14
        galois.is_cyclic(n)

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_phi(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {:2d}, Span: {:<20}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Euler's totient function ϕ(ϕ(n)) counts the primitive roots of n
        len(roots) == galois.euler_phi(phi)

    A counterexample is :math:`n = 15 = 3 \cdot 5`, which doesn't fit the condition for cyclicness.
    :math:`(\mathbb{Z}/15\mathbb{Z}){^\times} = \{1, 2, 4, 7, 8, 11, 13, 14\}`.

    .. ipython:: python

        # n is of type p1^k1 * p2^k2, which is not cyclic
        n = 15
        galois.is_cyclic(n)

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_phi(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {:2d}, Span: {:<13}, Primitive root: {}".format(a, str(span), primitive_root))

        # Find the smallest primitive root
        galois.primitive_root(n)
        # Find all primitive roots
        roots = galois.primitive_roots(n); roots

        # Note the max order of any element is 4, not 8, which is Carmichael's lambda function
        galois.carmichael_lambda(n)
    """
    return list(_primitive_roots(n, start=start, stop=stop, reverse=reverse))


def _primitive_roots(n, start=1, stop=None, reverse=False):
    if n in [1, 2]:
        yield n - 1
        return

    stop = n if stop is None else stop
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(start, (int, np.integer)):
        raise TypeError(f"Argument `start` must be an integer, not {type(start)}.")
    if not isinstance(stop, (int, np.integer)):
        raise TypeError(f"Argument `stop` must be an integer, not {type(stop)}.")
    if not isinstance(reverse, bool):
        raise TypeError(f"Argument `reverse` must be a bool, not {type(reverse)}.")
    if not 1 <= start < stop <= n:
        raise ValueError(f"Arguments must satisfy `1 <= start < stop <= n`, not `1 <= {start} < {stop} <= {n}`.")

    if not is_cyclic(n):
        return

    n = int(n)  # Needed for the pow() function
    phi = euler_phi(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    primes, _ = factors(phi)

    if phi == n - 1 or n % 2 == 1:
        # For prime n or odd n, must test all elements
        possible_roots = range(start, stop)
    else:
        # For even n, only have to test odd elements
        if start % 2 == 0:
            start += 1  # Make start odd
        possible_roots = range(start, stop, 2)

    if reverse:
        possible_roots = reversed(possible_roots)

    for r in possible_roots:
        if pow(r, phi, n) == 1 and all(pow(r, phi // p, n) != 1 for p in primes):
            yield r
