import math

import numpy as np

from ..factor import prime_factors
from ..field import Field, Poly, GF2, primitive_poly as primitive_poly_
from ..overrides import set_module

__all__ = ["bch_valid_codes", "bch_generator_poly", "bch_generator_matrix"]


@set_module("galois")
def bch_valid_codes(n, t_min=1):
    """
    Returns a list of :math:`(n, k, t)` tuples of valid primitive binary BCH codes.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
    t_min : int, optional
        The minimum error-correcting capability. The default is 1.

    Returns
    -------
    list
        A list of :math:`(n, k, t)` tuples of valid primitive BCH codes.

    Examples
    --------
    .. ipython:: python

        galois.bch_valid_codes(31)
        galois.bch_valid_codes(31, t_min=3)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(t_min, (int, np.integer)):
        raise TypeError(f"Argument `t_min` must be an integer, not {type(t_min)}.")
    p, e = prime_factors(n + 1)
    if not (len(p) == 1 and p[0] == 2):
        raise ValueError(f"Argument `n` must have value 2^m - 1 for some positive m, not {n}.")
    if not t_min >= 1:
        raise ValueError(f"Argument `t_min` must be at least 1, not {t_min}.")

    m = e[0]
    GF = Field(2**m)
    alpha = GF.primitive_element

    codes = []
    t = t_min
    while True:
        c = 1
        roots = alpha**(c + np.arange(0, 2*t - 1))
        powers = GF.characteristic**np.arange(0, GF.degree)
        conjugates = np.unique(np.power.outer(roots, powers))
        g_degree = len(conjugates)
        k = n - g_degree

        if k <= 1:
            # There are no more valid codes
            break

        if len(codes) > 0 and codes[-1][1] == k:
            # If this code has the same size but more correcting power, replace it
            codes[-1] = (n, k, t)
        else:
            codes.append((n, k, t))

        t += 1

    return codes


@set_module("galois")
def bch_generator_poly(n, k, c=1, primitive_poly=None, primitive_element=None):
    """
    Returns the generator polynomial for the primitive binary :math:`\\textrm{BCH}(n, k)` code.

    The BCH generator polynomial :math:`g(x)` is defined as :math:`g(x) = \\textrm{LCM}(m_{c}(x), m_{c+1}(x), \\dots, m_{c+2t-2}(x))`,
    where :math:`m_c(x)` is the minimal polynomial of :math:`\\alpha^c` where :math:`\\alpha` is a primitive element of :math:`\\mathrm{GF}(2^m)`.
    If :math:`c = 1`, then the code is said to be *narrow-sense*.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
    k : int
        The message size :math:`k`.
    c : int, optional
        The first consecutive power of :math:`\\alpha`. The default is 1.
    primitive_poly : galois.Poly, optional
        Optionally specify the primitive polynomial that defines the extension field :math:`\\mathrm{GF}(2^m)`. The default is
        `None` which uses the lexicographically-smallest primitive polynomial, i.e. `galois.primitive_poly(2, m, method="smallest")`.
        The use of the lexicographically-smallest primitive polynomial, as opposed to a Conway polynomial, is most common for the
        default in textbooks, Matlab, and Octave.
    primitive_element : int, galois.Poly, optional
        Optionally specify the primitive element :math:`\\alpha` whose powers are roots of the generator polynomial :math:`g(x)`.
        The default is `None` which uses the lexicographically-smallest primitive element in :math:`\\mathrm{GF}(2^m)`, i.e.
        `galois.primitive_element(2, m)`.

    Returns
    -------
    galois.Poly
        The generator polynomial :math:`g(x)`.

    Raises
    ------
    ValueError
        If the :math:`\\textrm{BCH}(n, k)` code does not exist.

    Examples
    --------
    .. ipython:: python

        g = galois.bch_generator_poly(15, 7); g
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
    if not isinstance(c, (int, np.integer)):
        raise TypeError(f"Argument `c` must be an integer, not {type(c)}.")
    if not isinstance(primitive_poly, (type(None), int, Poly)):
        raise TypeError(f"Argument `primitive_poly` must be None, an int, or galois.Poly, not {type(primitive_poly)}.")
    p, _ = prime_factors(n + 1)
    if not (len(p) == 1 and p[0] == 2):
        raise ValueError(f"Argument `n` must have value 2^m - 1 for some positive m, not {n}.")
    if not c >= 1:
        raise ValueError(f"Argument `c` must be at least 1, not {c}.")

    return _compute_generator_poly(n, k, c=c, primitive_poly=primitive_poly, primitive_element=primitive_element)[0]


def _compute_generator_poly(n, k, c=1, primitive_poly=None, primitive_element=None):
    p, e = prime_factors(n + 1)
    assert len(p) == 1 and p[0] == 2
    m = e[0]

    if primitive_poly is None:
        primitive_poly = primitive_poly_(2, m, method="smallest")

    GF = Field(2**m, irreducible_poly=primitive_poly, primitive_element=primitive_element)
    alpha = GF.primitive_element

    t = int(math.ceil((n - k) / m))
    while True:
        # We want to find LCM(m_r1(x), m_r2(x), ...) with ri being an element of `roots`. Instead of computing each
        # minimal polynomial and then doing an LCM, we will compute all the unique conjugates of all the roots
        # and then compute (x - c1)*(x - c2)*...*(x -cn), which is equivalent.
        roots = alpha**(c + np.arange(0, 2*t - 1))
        powers = GF.characteristic**np.arange(0, GF.degree)
        conjugates = np.unique(np.power.outer(roots, powers))
        g = Poly.Roots(conjugates)
        if g.degree < n - k:
            t += 1
        elif g.degree == n - k:
            break  # This is a valid BCH code size and g(x) is its generator
        else:
            raise ValueError(f"The code BCH({n}, {k}) with c={c} does not exist.")

    g =  Poly(g.coeffs, field=GF2)  # Convert from GF(2^m) to GF(2)

    return g, roots, t


@set_module("galois")
def bch_generator_matrix(n, k, c=1, primitive_poly=None, primitive_element=None, systematic=True):
    """
    Returns the generator matrix for the primitive binary :math:`\\textrm{BCH}(n, k)` code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = 2^m - 1`.
    k : int
        The message size :math:`k`.
    c : int, optional
        The first consecutive power of :math:`\\alpha`. The default is 1.
    primitive_poly : galois.Poly, optional
        Optionally specify the primitive polynomial that defines the extension field :math:`\\mathrm{GF}(2^m)`. The default is
        `None` which uses the lexicographically-smallest primitive polynomial, i.e. `galois.primitive_poly(2, m, method="smallest")`.
        The use of the lexicographically-smallest primitive polynomial, as opposed to a Conway polynomial, is most common for the
        default in textbooks, Matlab, and Octave.
    primitive_element : int, galois.Poly, optional
        Optionally specify the primitive element :math:`\\alpha` whose powers are roots of the generator polynomial :math:`g(x)`.
        The default is `None` which uses the lexicographically-smallest primitive element in :math:`\\mathrm{GF}(2^m)`, i.e.
        `galois.primitive_element(2, m)`.
    systematic : bool, optional
        Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
        appended. The default is `True`.

    Returns
    -------
    galois.FieldArray
        The :math:`(n, k)` generator matrix :math:`G`, such that given a message :math:`m`, a codeword is defined by
        :math:`c = mG`.

    Examples
    --------
    .. ipython :: python

        galois.bch_generator_matrix(15, 7)
        galois.bch_generator_matrix(15, 7, systematic=False)
    """
    g = bch_generator_poly(n, k, c=c, primitive_poly=primitive_poly, primitive_element=primitive_element)
    return _convert_poly_to_matrix(n, k, g, systematic)


def _convert_poly_to_matrix(n, k, g, systematic):
    G = GF2.Zeros((k, n))
    for i in range(k):
        G[i, i:i + g.degree + 1] = g.coeffs

    if systematic:
        G = G.row_reduce()  # pylint: disable=no-member

    return G
