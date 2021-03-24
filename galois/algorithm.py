from itertools import combinations
import math

import numba
import numpy as np

from .modular import modular_exp
from .prime import prime_factors, is_prime


@numba.jit(nopython=True)
def _numba_factors(x):
    f = []  # Positive factors
    for i in range(1, int(np.ceil(np.sqrt(x))) + 1):
        if x % i == 0:
            q = x // i
            f.extend([i, q])
    return f


def factors(x):
    """
    Computes the positive factors of the integer :math:`x`.

    Parameters
    ----------
    x : int
        An integer to be factored.

    Returns
    -------
    numpy.ndarray:
        Sorted array of factors of :math:`x`.

    Examples
    --------
    .. ipython:: python

        galois.factors(120)
    """
    f = _numba_factors(x)
    f = sorted(list(set(f)))  # Use set() to emove duplicates
    return np.array(f)


def euclidean_algorithm(a, b):
    """
    Finds the greatest common divisor of two integers.

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

        a, b = 2, 13
        galois.euclidean_algorithm(a, b)
    """
    assert isinstance(a, (int, np.integer))
    assert isinstance(b, (int, np.integer))
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

        a, b = 2, 13
        x, y, gcd = galois.extended_euclidean_algorithm(a, b)
        x, y, gcd
        a*x + b*y == gcd
    """
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
def extended_euclidean_algorithm_jit(a, b):
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
            print(f"{x} % {m[i]} = {ai}, Valid congruence: {ai == a[i]}")
    """
    a = np.array(a)
    m = np.array(m)
    assert m.size == a.size

    # Check that modulii are pairwise relatively coprime
    for pair in combinations(m, 2):
        assert math.gcd(pair[0], pair[1]) == 1, "{} and {} are not pairwise relatively coprime".format(pair[0], pair[1])

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
    assert n > 0
    if n == 1:
        return 1
    p, k = prime_factors(n)
    phi = np.multiply.reduce(p**(k - 1) * (p - 1))
    return phi


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
            result = galois.modular_exp(a, lambda_, n)
            print("{:2d}^{} = {} (mod {})".format(a, lambda_, result, n))

        # For prime n, phi and lambda are always n-1
        galois.euler_totient(13), galois.carmichael(13)
    """
    assert n > 0
    if n == 1:
        return 1

    p, k = prime_factors(n)

    lambdas = np.zeros(p.size, dtype=np.int64)
    for i in range(p.size):
        lambdas[i] = _carmichael_prime_power(p[i], k[i])

    lambda_ = np.lcm.reduce(lambdas)

    return lambda_


def primitive_root(n, start=1):
    """
    Finds the first, smallest primitive n-th root of unity :math:`x` that satisfies :math:`x^n \\equiv 1 (\\textrm{mod}\\ n)`.

    Parameters
    ----------
    n : int
        A positive integer :math:`n > 1`.
    start : int, optional
        Initial primitive root hypothesis. The resulting primitive root will be `root >= start`. The default is 1.

    Returns
    -------
    int
        The first, smallest primitive root of unity modulo :math:`n`.

    References
    ----------
    * https://en.wikipedia.org/wiki/Finite_field#Roots_of_unity
    * https://en.wikipedia.org/wiki/Primitive_root_modulo_n
    * https://www.ams.org/journals/mcom/1992-58-197/S0025-5718-1992-1106981-9/S0025-5718-1992-1106981-9.pdf
    * https://www.ams.org/journals/bull/1942-48-10/S0002-9904-1942-07767-6/S0002-9904-1942-07767-6.pdf
    * http://www.numbertheory.org/courses/MP313/lectures/lecture7/page1.html

    Examples
    --------
    .. ipython:: python

        n = 7
        root = galois.primitive_root(n); root

        # Test that the primitive root is a multiplicative generator of GF(n)
        for i in range(1, n):
            result = galois.modular_exp(root, i, n)
            print(f"{root}^{i} = {result} (mod {n})")

        # The algorithm is also efficient for very large prime n
        galois.primitive_root(1000000000000000035000061)
    """
    if n == 1:
        return 0
    if n == 2:
        return 1

    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"`n` must be an integer, not {type(n)}")
    if not n >= 1:
        raise ValueError(f"`n` must be a postive integer, not {n}")
    if not is_prime(n):
        raise ValueError(f"This algorithm is optimized for prime n, not {n}")
    if not 1 <= start < n:
        raise ValueError(f"The starting hypothesis must be in [1, {n}), not {start}")

    p, _ = prime_factors(n - 1)

    a = start  # Initial field element
    while a < n:
        tests = []  # Tests of primitivity
        tests.append(modular_exp(a, n-1, n) == 1)
        for prime_factor in p:
            tests.append(modular_exp(a, (n - 1)//prime_factor, n) != 1)

        if all(tests):
            return a
        else:
            a += 1

    return None


# def primitive_roots(n):
#     """
#     Finds all primitive n-th roots of unity :math:`x` that satisfy :math:`x^n \\equiv 1 (\\textrm{mod}\\ n)`.

#     Parameters
#     ----------
#     n : int
#         A positive integer :math:`n > 1`.

#     Returns
#     -------
#     list
#         A list of integer roots of unity modulo :math:`n`.

#     References
#     ----------
#     * https://en.wikipedia.org/wiki/Finite_field#Roots_of_unity
#     * https://en.wikipedia.org/wiki/Primitive_root_modulo_n

#     Examples
#     --------
#     .. ipython:: python

#         n = 7
#         roots = galois.primitive_roots(n); roots

#         # Test that each primitive root is a multiplicative generator of GF(n)
#         for root in roots:
#             print(f"\\nPrimitive root: {root}")
#             for i in range(1, n):
#                 result = galois.modular_exp(root, i, n)
#                 print(f"{root}^{i} = {result} (mod {n})")
#     """
#     assert n > 0
#     if n == 1:
#         return [0]
#     if n == 2:
#         return [1]
#     # phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
#     # p, k = prime_factors(phi)  # Prime factorization of phi(n)
#     # print("prime_factors", p, k)

#     # roots = []
#     # for m in range(1, n):
#     #     y = np.array([modular_exp(m, phi // pi, n) for pi in p])
#     #     print(m, y)
#     #     if np.all(y != 1):
#     #         roots.append(m)

#     # print(roots)

#     # if len(roots) > 0:
#     #     N_roots = euler_totient(phi)
#     #     assert len(roots) == N_roots, "The number of primitive roots ({} found), if there are any, is phi(phi(n)) = {}".format(len(roots), N_roots)

#     # return roots

#     phi = euler_totient(n)  # Number of non-zero elements in the multiplicative group Z/nZ
#     # lambda_ = carmichael(n)  # The smallest m such that a^m = 1 (mod n) for all a coprime to n

#     elements = np.arange(1, n)  # Non-zero integers less than n

#     # According to Euler's theorem, a**phi(n) = 1 (mod n) for every a coprime to n
#     congruenes = elements[np.where(modular_exp(elements, phi, n) == 1)]
#     assert len(congruenes) == phi, "The number of congruences ({} found) is phi(n) = {}".format(len(congruenes), phi)

#     roots = []
#     degrees = np.arange(1, phi+1)
#     for m in congruenes:
#         y = modular_exp(m, degrees, n)
#         if set(y) == set(congruenes):
#             roots.append(m)

#     if len(roots) > 0:
#         assert len(roots) == euler_totient(phi), "The number of primitive roots, if there are any, is phi(phi(n))"

#     return roots
