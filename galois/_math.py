"""
A module that contains some future features of the math stdlib for earlier Python versions.
"""
import math
import sys

import numpy as np

from ._overrides import set_module

__all__ = ["pow", "lcm", "prod"]


@set_module("galois")
def pow(base, exponent, modulus):  # pylint: disable=redefined-builtin
    r"""
    Efficiently exponentiates an integer :math:`a^k\ \textrm{mod}\ m`.

    The algorithm is more efficient than exponentiating first and then reducing modulo :math:`m`.

    Note
    ----
    This function is an alias of :func:`pow` in the standard library.

    Parameters
    ----------
    base : int
        The integer base :math:`a`.
    exponent : int
        The integer exponent :math:`k`.
    modulus : int
        The integer modulus :math:`m`.

    Returns
    -------
    int
        The modular exponentiation :math:`a^k\ \textrm{mod}\ m`.

    Examples
    --------
    .. ipython:: python

        galois.pow(3, 5, 7)
        (3**5) % 7
    """
    import builtins  # pylint: disable=import-outside-toplevel
    return builtins.pow(base, exponent, modulus)


@set_module("galois")
def lcm(*integers):
    r"""
    Computes the least common multiple of the integer arguments.

    Note
    ----
    This function is included for Python versions before 3.9. For Python 3.9 and later, this function
    calls :func:`math.lcm` from the standard library.

    Parameters
    ----------
    *integers : tuple of int
        Each argument must be an integer.

    Returns
    -------
    int
        The least common multiple of the integer arguments. If any argument is 0, the LCM is 0. If no
        arguments are provided, 1 is returned.

    Examples
    --------
    .. ipython:: python

        galois.lcm()
        galois.lcm(2, 4, 14)
        galois.lcm(3, 0, 9)

    This function also works on arbitrarily-large integers.

    .. ipython:: python

        prime1, prime2 = galois.mersenne_primes(100)[-2:]
        prime1, prime2
        lcm = galois.lcm(prime1, prime2); lcm
        lcm == prime1 * prime2
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 9:  # pragma: no cover
        return math.lcm(*integers)  # pylint: disable=no-member

    if not all(isinstance(a, (int, np.integer)) for a in integers):
        raise TypeError(f"Each argument must be an integer, not {integers}.")

    lcm_  = 1
    for integer in integers:
        lcm_ = (lcm_ * integer) // math.gcd(lcm_, integer)

    return lcm_


@set_module("galois")
def prod(iterable, start=1):
    r"""
    Computes the product of the integer arguments.

    Note
    ----
    This function is included for Python versions before 3.8. For Python 3.8 and later, this function
    calls :func:`math.prod` from the standard library.

    Returns
    -------
    int
        The product of the integer arguments.

    Examples
    --------
    .. ipython:: python

        galois.prod([2, 4, 14])
        galois.prod([2, 4, 14], start=2)
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:  # pragma: no cover
        return math.prod(iterable, start=start)  # pylint: disable=no-member

    result = start
    for integer in iterable:
        result *= integer

    return result
