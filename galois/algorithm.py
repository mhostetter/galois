import math
from itertools import combinations

import numba
import numpy as np

from .prime import prime_factors


@numba.jit(nopython=True)
def _numba_factors(n):  # pragma: no cover
    f = []  # Positive factors
    max_factor = int(np.ceil(np.sqrt(n)))
    for i in range(1, max_factor + 1):
        if n % i == 0:
            q = n // i
            f.extend([i, q])
    return f


def factors(n):
    """
    Returns the positive factors of the integer :math:`n`.

    Parameters
    ----------
    n : int
        An integer to be factored.

    Returns
    -------
    list
        Sorted array of factors of :math:`n`.

    Examples
    --------
    .. ipython:: python

        galois.factors(120)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    f = _numba_factors(n)
    return sorted(list(set(f)))  # Use set() to remove duplicates


def euclidean_algorithm(a, b):
    """
    Finds the greatest common divisor of two integers :math:`a` and :math:`b`.

    Parameters
    ----------
    a : int
        Any integer.
    b : int
        Any integer.

    Returns
    -------
    int
        Greatest common divisor of :math:`a` and :math:`b`, i.e. :math:`gcd(a,b)`.

    References
    ----------
    * T. Moon, "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181
    * https://en.wikipedia.org/wiki/Euclidean_algorithm

    Examples
    --------
    .. ipython:: python

        a = 2
        b = 13
        galois.euclidean_algorithm(a, b)
    """
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(b, (int, np.integer)):
        raise TypeError(f"Argument `b` must be an integer, not {type(b)}.")

    r = [a, b]

    while True:
        ri = r[-2] % r[-1]
        r.append(ri)
        if ri == 0:
            break

    return r[-2]


def extended_euclidean_algorithm(a, b):
    """
    Finds the integer multiplicands of :math:`a` and :math:`b` such that :math:`a x + b y = gcd(a,b)`.

    Parameters
    ----------
    a : int
        Any integer.
    b : int
        Any integer.

    Returns
    -------
    int
        Integer :math:`x`, such that :math:`a x + b y = gcd(a, b)`.
    int
        Integer :math:`y`, such that :math:`a x + b y = gcd(a, b)`.
    int
        Greatest common divisor of :math:`a` and :math:`b`.

    References
    ----------
    * T. Moon, "Error Correction Coding", Section 5.2.2: The Euclidean Algorithm and Euclidean Domains, p. 181
    * https://en.wikipedia.org/wiki/Euclidean_algorithm#Extended_Euclidean_algorithm

    Examples
    --------
    .. ipython:: python

        a = 2
        b = 13
        x, y, gcd = galois.extended_euclidean_algorithm(a, b)
        x, y, gcd
        a*x + b*y == gcd
    """
    if not isinstance(a, (int, np.integer)):
        raise TypeError(f"Argument `a` must be an integer, not {type(a)}.")
    if not isinstance(b, (int, np.integer)):
        raise TypeError(f"Argument `b` must be an integer, not {type(b)}.")

    r = [a, b]
    s = [1, 0]
    t = [0, 1]

    while True:
        qi = r[-2] // r[-1]
        ri = r[-2] % r[-1]
        r.append(ri)
        s.append(s[-2] - qi*s[-1])
        t.append(t[-2] - qi*t[-1])
        if ri == 0:
            break

    return s[-2], t[-2], r[-2]


@numba.jit("int64[:](int64, int64)", nopython=True)
def extended_euclidean_algorithm_jit(a, b):  # pragma: no cover
    r = [a, b]
    s = [1, 0]
    t = [0, 1]

    while True:
        qi = r[-2] // r[-1]
        ri = r[-2] % r[-1]
        r.append(ri)
        s.append(s[-2] - qi*s[-1])
        t.append(t[-2] - qi*t[-1])
        if ri == 0:
            break

    return np.array([s[-2], t[-2], r[-2]], dtype=np.int64)


def chinese_remainder_theorem(a, m):
    """
    Solves the simultaneous system of congruences for :math:`x`.

    .. math::
        x &\\equiv a_1\\ (\\textrm{mod}\\ m_1)

        x &\\equiv a_2\\ (\\textrm{mod}\\ m_2)

        x &\\equiv \\ldots

        x &\\equiv a_n\\ (\\textrm{mod}\\ m_n)

    Parameters
    ----------
    a : array_like
        The integer remainders :math:`a_i`.
    m : array_like
        The integer modulii :math:`m_i`.

    Returns
    -------
    int
        The simultaneous solution :math:`x` to the system of congruences.

    Examples
    --------
    .. ipython:: python

        a = [0, 3, 4]
        m = [3, 4, 5]
        x = galois.chinese_remainder_theorem(a, m); x

        for i in range(len(a)):
            ai = x % m[i]
            print(f"{x} = {ai} (mod {m[i]}), Valid congruence: {ai == a[i]}")
    """
    a = np.array(a)
    m = np.array(m)
    if not m.size == a.size:
        raise ValueError(f"Arguments `a` and `m` are not the same size, {a.size} != {m.size}.")
    for pair in combinations(m, 2):
        if not math.gcd(pair[0], pair[1]) == 1:
            raise ValueError(f"Elements of argument `m` must be pairwise coprime, {pair} are not.")

    # Iterate through the system of congruences reducing a pair of congruences into a
    # single one. The answer to the final congruence solves all the congruences.
    a1 = a[0]
    m1 = m[0]
    for i in range(1, m.size):
        a2 = a[i]
        m2 = m[i]

        # Use the Extended Euclidean Algorithm to determine: b1*m1 + b2*m2 = 1,
        # where 1 is the GCD(m1, m2) because m1 and m2 are pairwise relatively coprime
        b1, b2 = extended_euclidean_algorithm(m1, m2)[0:2]

        # Compute x through explicit construction
        x = a1*b2*m2 + a2*b1*m1

        m1 = m1 * m2  # The new modulus
        a1 = x % m1  # The new equivalent remainder

    # Align x to be within [0, prod(m))
    x = x % np.prod(m)

    return x


def totatives(n):
    """
    Returns the positive integers (totatives) in :math:`1 \\le k < n` that are relatively prime to
    :math:`n`, i.e. :math:`gcd(n, k) = 1`.

    Parameters
    ----------
    n : int
        A positive integer.

    Returns
    -------
    list
        The totatives of :math:`n`.

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
        totatives = [k for k in range(n) if galois.euclidean_algorithm(k, n) == 1]; totatives

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

    return int(phi)  # TODO: Needed until PRIMES is a list of ints


def _carmichael_prime_power(p, k):
    if p == 2 and k > 2:
        return 1/2 * euler_totient(p**k)
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
        totatives = [i for i in range(n) if galois.euclidean_algorithm(i, n) == 1]; totatives

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

    lambdas = np.zeros(len(p), dtype=np.int64)
    for i in range(len(p)):
        lambdas[i] = _carmichael_prime_power(p[i], k[i])

    lambda_ = int(np.lcm.reduce(lambdas))

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

        # To find the primitive roots 3 and 5, you can just call `primitive_roots()`
        roots = list(galois.primitive_roots(n)); roots

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

        roots = list(galois.primitive_roots(n)); roots

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


def primitive_root(n, start=1, stop=None, largest=False):
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
        Starting value (inclusive) for the search for a primitive root. The default is `1`. The resulting primitive
        root, if found, will be :math:`\\textrm{start} \\le g < \\textrm{stop}`.
    stop : int, optional
        Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds to `n`.
        The resulting primitive root, if found, will be :math:`\\textrm{start} \\le g < \\textrm{stop}`.
    largest : bool, optional
        Return the largest primitive root in the specified range, not the smallest. Default is `False`.

    Returns
    -------
    int
        The smallest primitive root modulo :math:`n`. Returns `None` if no primitive roots exist.

    References
    ----------
    * https://en.wikipedia.org/wiki/Finite_field#Roots_of_unity
    * https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    * https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
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
        galois.primitive_root(n, largest=True)

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
        return next(primitive_roots(n, start=start, stop=stop, reverse=largest))
    except StopIteration:
        return None


def primitive_roots(n, start=1, stop=None, reverse=False):
    """
    A generator that finds all primitive roots modulo :math:`n`.

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
        Starting value (inclusive) for the search for a primitive root. The default is 1. The resulting primitive
        roots, if found, will be :math:`\\textrm{start} \\le x < \\textrm{stop}`.
    stop : int, optional
        Stopping value (exclusive) in the search for a primitive root. The default is `None` which corresponds to `n`.
        The resulting primitive roots, if found, will be :math:`\\textrm{start} \\le x < \\textrm{stop}`.
    reverse : bool, optional
        List all primitive roots in descending order, i.e. largest to smallest. Default is `False`.

    Returns
    -------
    typing.Generator
        A generator of all the positive primitive :math:`n`-th roots of unity, :math:`x`. Use `list(galois.primitive_roots(n))` to retrieve
        them all instantly. Otherwise, you can access them in a `for` loop or list comprehension.

    References
    ----------
    * https://en.wikipedia.org/wiki/Finite_field#Roots_of_unity
    * https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    * https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
    * http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples
    --------
    Here is an example with one primitive root, :math:`n = 6 = 2 * 3^1`, which fits the definition
    of cyclicness, see :obj:`galois.is_cyclic`. Because :math:`n = 6` is not prime, the primitive root
    isn't a multiplicative generator of :math:`\\mathbb{Z}/\\textbf{n}\\mathbb{Z}`.

    .. ipython:: python

        n = 6
        roots = list(galois.primitive_roots(n)); roots

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
        roots = list(galois.primitive_roots(n)); roots

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

        # Only list some of the primitive roots
        list(galois.primitive_roots(n, stop=100))

        # The generator can also be used in a for loop
        for r in galois.primitive_roots(n, stop=100):
            print(r, end=" ")

    Here is a counterexample with no primitive roots, :math:`n = 8 = 2^3`, which does not fit the definition
    of cyclicness, see :obj:`galois.is_cyclic`.

    .. ipython:: python

        n = 8
        roots = list(galois.primitive_roots(n)); roots

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

    n = int(n)  # TODO: Need while primes is a numpy array
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
        list(galois.primitive_roots(7))
    """
    if not isinstance(g, (int, np.integer)):
        raise TypeError(f"Argument `g` must be an integer, not {type(g)}.")
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not n > 0:
        raise ValueError(f"Argument `n` must be a positive integer, not {n}.")
    if not 0 < g < n:
        raise ValueError(f"Argument `g` must be a positive integer less than `n`, not {g}.")

    n = int(n)  # TODO: Need while primes is a numpy array
    phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
    primes, _ = prime_factors(phi)

    return pow(g, phi, n) == 1 and all(pow(g, phi // p, n) != 1 for p in primes)
