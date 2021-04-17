import math

import numpy as np

from .math_ import lcm
from .prime import prime_factors


def totatives(n):
    """
    Returns the positive integers (totatives) in :math:`1 \\le k < n` that are coprime with :math:`n`,
    i.e. :math:`gcd(n, k) = 1`.

    The totatives of :math:`n` form the multiplicative group :math:`\\mathbb{Z}{_n^\\times}`.

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
    * https://en.wikipedia.org/wiki/Totative
    * https://oeis.org/A000010

    Examples
    --------
    .. ipython:: python

        n = 20
        totatives = galois.totatives(n); totatives
        phi = galois.euler_totient(n); phi
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


def euler_totient(n):
    """
    Counts the positive integers (totatives) in :math:`1 \\le k < n` that are relatively prime to
    :math:`n`, i.e. :math:`gcd(n, k) = 1`.

    Implements the Euler Totient function :math:`\\phi(n)`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The number of totatives that are relatively prime to :math:`n`.

    References
    ----------
    * https://en.wikipedia.org/wiki/Euler%27s_totient_function
    * https://oeis.org/A000010

    Examples
    --------
    .. ipython:: python

        n = 20
        phi = galois.euler_totient(n); phi

        # Find the totatives that are coprime with n
        totatives = [k for k in range(n) if math.gcd(k, n) == 1]; totatives

        # The number of totatives is phi
        len(totatives) == phi

        # For prime n, phi is always n-1
        galois.euler_totient(13)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return 1

    p, k = prime_factors(n)

    phi = 1
    for i in range(len(p)):
        phi *= p[i]**(k[i] - 1) * (p[i] - 1)

    return phi


def _carmichael_prime_power(p, k):
    if p == 2 and k > 2:
        return euler_totient(p**k) // 2
    else:
        return euler_totient(p**k)


def carmichael(n):
    """
    Finds the smallest positive integer :math:`m` such that :math:`a^m \\equiv 1 (\\textrm{mod}\\ n)` for
    every integer :math:`a` in :math:`1 \\le a < n` that is coprime to :math:`n`.

    Implements the Carmichael function :math:`\\lambda(n)`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    int
        The smallest positive integer :math:`m` such that :math:`a^m \\equiv 1 (\\textrm{mod}\\ n)` for
        every :math:`a` in :math:`1 \\le a < n` that is coprime to :math:`n`.

    References
    ----------
    * https://en.wikipedia.org/wiki/Carmichael_function
    * https://oeis.org/A002322

    Examples
    --------
    .. ipython:: python

        n = 20
        lambda_ = galois.carmichael(n); lambda_

        # Find the totatives that are relatively coprime with n
        totatives = [i for i in range(n) if math.gcd(i, n) == 1]; totatives

        for a in totatives:
            result = pow(a, lambda_, n)
            print("{:2d}^{} = {} (mod {})".format(a, lambda_, result, n))

        # For prime n, phi and lambda are always n-1
        galois.euler_totient(13), galois.carmichael(13)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    if n == 1:
        return 1

    p, k = prime_factors(n)

    lambdas = []
    for i in range(len(p)):
        lambdas.append(_carmichael_prime_power(p[i], k[i]))

    lambda_ = lcm(*lambdas)

    return lambda_


def is_cyclic(n):
    """
    Determines whether the multiplicative group :math:`\\mathbb{Z}{_n^\\times}` is cyclic.

    The multiplicative group :math:`\\mathbb{Z}{_n^\\times}` is the set of positive integers :math:`1 \\le a < n`
    that are coprime with :math:`n`. :math:`\\mathbb{Z}{_n^\\times}` being cyclic means that some primitive root
    (or generator) :math:`g` can generate the group :math:`\\mathbb{Z}{_n^\\times} = \\{g, g^2, \\dots, g^k\\}`, where :math:`k` is order of the group.
    The order of the group is defined by Euler's totient function, :math:`\\phi(n) = k`. If :math:`\\mathbb{Z}{_n^\\times}` is cyclic,
    the number of primitive roots is found by :math:`\\phi(k)` or :math:`\\phi(\\phi(n))`.

    :math:`\\mathbb{Z}{_n^\\times}` is cyclic if and only if :math:`n` is :math:`2`, :math:`4`, :math:`p^k`, or :math:`2p^k`,
    where :math:`p` is an odd prime and :math:`k` is a positive integer.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    bool
        `True` if the multiplicative group :math:`\\mathbb{Z}{_n^\\times}` is cyclic.

    References
    ----------
    * https://en.wikipedia.org/wiki/Primitive_root_modulo_n

    Examples
    --------
    The elements of :math:`\\mathbb{Z}{_n^\\times}` are the positive integers less than :math:`n` that are coprime with :math:`n`.
    For example when :math:`n = 14`, then :math:`\\mathbb{Z}{_{14}^\\times} = \\{1, 3, 5, 9, 11, 13\\}`.

    .. ipython:: python

        # n is of type 2*p^k, which is cyclic
        n = 14
        galois.is_cyclic(n)

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {:2d}, Span: {:<20}, Primitive root: {}".format(a, str(span), primitive_root))

        roots = galois.primitive_roots(n); roots

        # Euler's totient function phi(phi(n)) counts the primitive roots of n
        len(roots) == galois.euler_totient(phi)

    A counterexample is :math:`n = 15 = 3*5`, which doesn't fit the condition for cyclicness.
    :math:`\\mathbb{Z}{_{15}^\\times} = \\{1, 2, 4, 7, 8, 11, 13, 14\\}`.

    .. ipython:: python

        # n is of type p1^k1 * p2^k2, which is not cyclic
        n = 15
        galois.is_cyclic(n)

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {:2d}, Span: {:<13}, Primitive root: {}".format(a, str(span), primitive_root))

        roots = galois.primitive_roots(n); roots

        # Note the max order of any element is 4, not 8, which is Carmichael's lambda function
        galois.carmichael(n)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")

    p, k = prime_factors(n)

    if n in [2, 4]:
        return True
    elif len(p) == 2 and 2 in p and k[p.index(2)] == 1:
        # n = 2 * p^k
        return True
    elif len(p) == 1 and p[0] != 2:
        # n = p^k
        return True
    else:
        # n does not represent a cyclic group
        return False


def is_primitive_root(g, n):
    """
    Determines if :math:`g` is a primitive root modulo :math:`n`.

    :math:`g` is a primitive root if the totatives of :math:`n`, the positive integers :math:`1 \\le a < n`
    that are coprime with :math:`n`, can be generated by powers of :math:`g`.

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

    phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    primes, _ = prime_factors(phi)

    return pow(g, phi, n) == 1 and all(pow(g, phi // p, n) != 1 for p in primes)


def primitive_root(n, start=1, stop=None, reverse=False):
    """
    Finds the smallest primitive root modulo :math:`n`.

    :math:`g` is a primitive root if the totatives of :math:`n`, the positive integers :math:`1 \\le a < n`
    that are coprime with :math:`n`, can be generated by powers of :math:`g`.

    Alternatively said, :math:`g` is a primitive root modulo :math:`n` if and only
    if :math:`g` is a generator of the multiplicative group of integers modulo :math:`n`, :math:`\\mathbb{Z}{_n^\\times}`.
    That is, :math:`\\mathbb{Z}{_n^\\times} = \\{g, g^2, \\dots, g^k\\}`, where :math:`k` is order of the group.
    The order of the group :math:`\\mathbb{Z}{_n^\\times}` is defined by Euler's totient function, :math:`\\phi(n) = k`.
    If :math:`\\mathbb{Z}{_n^\\times}` is cyclic, the number of primitive roots modulo :math:`n` is given by
    :math:`\\phi(k)` or :math:`\\phi(\\phi(n))`.

    See :obj:`galois.is_cyclic`.

    Parameters
    ----------
    n : int
        A positive integer.
    start : int, optional
        Starting value (inclusive) in the search for a primitive root. The default is `1`. The resulting primitive
        root, if found, will be :math:`\\textrm{start} \\le g < \\textrm{stop}`.
    stop : int, optional
        Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds to `n`.
        The resulting primitive root, if found, will be :math:`\\textrm{start} \\le g < \\textrm{stop}`.
    reverse : bool, optional
        Search for a primitive root in reverse order, i.e. find the largest primitive root first. Default is `False`.

    Returns
    -------
    int
        The smallest primitive root modulo :math:`n`. Returns `None` if no primitive roots exist.

    References
    ----------
    * V. Shoup. Searching for primitive roots in finite fields. https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * L. K. Hua. On the least primitive root of a prime. https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
    * https://en.wikipedia.org/wiki/Finite_field#Roots_of_unity
    * https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    * http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples
    --------
    Here is an example with one primitive root, :math:`n = 6 = 2 * 3^1`, which fits the definition
    of cyclicness, see :obj:`galois.is_cyclic`. Because :math:`n = 6` is not prime, the primitive root
    isn't a multiplicative generator of :math:`\\mathbb{Z}/\\textbf{n}\\mathbb{Z}`.

    .. ipython:: python

        n = 6
        root = galois.primitive_root(n); root

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {}, Span: {:<6}, Primitive root: {}".format(a, str(span), primitive_root))

    Here is an example with two primitive roots, :math:`n = 7 = 7^1`, which fits the definition
    of cyclicness, see :obj:`galois.is_cyclic`. Since :math:`n = 7` is prime, the primitive root
    is a multiplicative generator of :math:`\\mathbb{Z}/\\textbf{n}\\mathbb{Z}`.

    .. ipython:: python

        n = 7
        root = galois.primitive_root(n); root

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # The primitive roots are the elements in Znx that multiplicatively generate the group
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {}, Span: {:<18}, Primitive root: {}".format(a, str(span), primitive_root))

    The algorithm is also efficient for very large :math:`n`.

    .. ipython:: python

        n = 1000000000000000035000061
        galois.primitive_root(n)
        galois.primitive_root(n, reverse=True)

    Here is a counterexample with no primitive roots, :math:`n = 8 = 2^3`, which does not fit the definition
    of cyclicness, see :obj:`galois.is_cyclic`.

    .. ipython:: python

        n = 8
        root = galois.primitive_root(n); root

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # Test all elements for being primitive roots. The powers of a primitive span the congruence classes mod n.
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {}, Span: {:<6}, Primitive root: {}".format(a, str(span), primitive_root))

        # Note the max order of any element is 2, not 4, which is Carmichael's lambda function
        galois.carmichael(n)
    """
    try:
        return next(_primitive_roots(n, start=start, stop=stop, reverse=reverse))
    except StopIteration:
        return None


def primitive_roots(n, start=1, stop=None, reverse=False):
    """
    Finds all primitive roots modulo :math:`n`.

    :math:`g` is a primitive root if the totatives of :math:`n`, the positive integers :math:`1 \\le a < n`
    that are coprime with :math:`n`, can be generated by powers of :math:`g`.

    Alternatively said, :math:`g` is a primitive root modulo :math:`n` if and only
    if :math:`g` is a generator of the multiplicative group of integers modulo :math:`n`, :math:`\\mathbb{Z}{_n^\\times}`.
    That is, :math:`\\mathbb{Z}{_n^\\times} = \\{g, g^2, \\dots, g^k\\}`, where :math:`k` is order of the group.
    The order of the group :math:`\\mathbb{Z}{_n^\\times}` is defined by Euler's totient function, :math:`\\phi(n) = k`.
    If :math:`\\mathbb{Z}{_n^\\times}` is cyclic, the number of primitive roots modulo :math:`n` is given by
    :math:`\\phi(k)` or :math:`\\phi(\\phi(n))`.

    See :obj:`galois.is_cyclic`.

    Parameters
    ----------
    n : int
        A positive integer.
    start : int, optional
        Starting value (inclusive) in the search for a primitive root. The default is 1. The resulting primitive
        roots, if found, will be :math:`\\textrm{start} \\le x < \\textrm{stop}`.
    stop : int, optional
        Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds to `n`.
        The resulting primitive roots, if found, will be :math:`\\textrm{start} \\le x < \\textrm{stop}`.
    reverse : bool, optional
        List all primitive roots in descending order, i.e. largest to smallest. Default is `False`.

    Returns
    -------
    list
        All the positive primitive :math:`n`-th roots of unity, :math:`x`.

    References
    ----------
    * V. Shoup. Searching for primitive roots in finite fields. https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * https://en.wikipedia.org/wiki/Finite_field#Roots_of_unity
    * https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    * http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples
    --------
    Here is an example with one primitive root, :math:`n = 6 = 2 * 3^1`, which fits the definition
    of cyclicness, see :obj:`galois.is_cyclic`. Because :math:`n = 6` is not prime, the primitive root
    isn't a multiplicative generator of :math:`\\mathbb{Z}/\\textbf{n}\\mathbb{Z}`.

    .. ipython:: python

        n = 6
        roots = galois.primitive_roots(n); roots

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # Test all elements for being primitive roots. The powers of a primitive span the congruence classes mod n.
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {}, Span: {:<6}, Primitive root: {}".format(a, str(span), primitive_root))

        # Euler's totient function phi(phi(n)) counts the primitive roots of n
        len(roots) == galois.euler_totient(phi)

    Here is an example with two primitive roots, :math:`n = 7 = 7^1`, which fits the definition
    of cyclicness, see :obj:`galois.is_cyclic`. Since :math:`n = 7` is prime, the primitive root
    is a multiplicative generator of :math:`\\mathbb{Z}/\\textbf{n}\\mathbb{Z}`.

    .. ipython:: python

        n = 7
        roots = galois.primitive_roots(n); roots

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # Test all elements for being primitive roots. The powers of a primitive span the congruence classes mod n.
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {}, Span: {:<18}, Primitive root: {}".format(a, str(span), primitive_root))

        # Euler's totient function phi(phi(n)) counts the primitive roots of n
        len(roots) == galois.euler_totient(phi)

    The algorithm is also efficient for very large :math:`n`.

    .. ipython:: python

        n = 1000000000000000035000061

        # Euler's totient function phi(phi(n)) counts the primitive roots of n
        galois.euler_totient(galois.euler_totient(n))

        # Only find some of the primitive roots
        galois.primitive_roots(n, stop=100)

        # The generator can also be used in a for loop
        for r in galois.primitive_roots(n, stop=100):
            print(r, end=" ")

    Here is a counterexample with no primitive roots, :math:`n = 8 = 2^3`, which does not fit the definition
    of cyclicness, see :obj:`galois.is_cyclic`.

    .. ipython:: python

        n = 8
        roots = galois.primitive_roots(n); roots

        # The congruence class coprime with n
        Znx = set([a for a in range(1, n) if math.gcd(n, a) == 1]); Znx

        # Euler's totient function counts the "totatives", positive integers coprime with n
        phi = galois.euler_totient(n); phi

        len(Znx) == phi

        # Test all elements for being primitive roots. The powers of a primitive span the congruence classes mod n.
        for a in Znx:
            span = set([pow(a, i, n) for i in range(1, phi + 1)])
            primitive_root = span == Znx
            print("Element: {}, Span: {:<6}, Primitive root: {}".format(a, str(span), primitive_root))
    """
    return list(_primitive_roots(n, start=start, stop=stop, reverse=reverse))


def _primitive_roots(n, start=1, stop=None, reverse=False):
    # pylint: disable=too-many-branches
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
    phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    primes, _ = prime_factors(phi)

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
