import numpy as np

from ..factor import prime_factors
from ..field import Field, Poly
from ..field import primitive_poly as primitive_poly_
from ..overrides import set_module

from .common import generator_poly_to_matrix, roots_to_parity_check_matrix

__all__ = ["rs_generator_poly", "rs_generator_matrix", "rs_parity_check_matrix"]


###############################################################################
# Reed-Solomon Functions
###############################################################################

def _check_and_compute_field(n, k, c, primitive_poly, primitive_element):
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"Argument `k` must be an integer, not {type(k)}.")
    if not isinstance(c, (int, np.integer)):
        raise TypeError(f"Argument `c` must be an integer, not {type(c)}.")
    if not isinstance(primitive_poly, (type(None), int, Poly)):
        raise TypeError(f"Argument `primitive_poly` must be None, an int, or galois.Poly, not {type(primitive_poly)}.")

    if not (n - k) % 2 == 0:
        raise ValueError("Arguments `n - k` must be even.")
    p, m = prime_factors(n + 1)
    if not (len(p) == 1 and len(m) == 1):
        raise ValueError(f"Argument `n` must have value `q - 1` for a prime power `q`, not {n}.")
    if not c >= 1:
        raise ValueError(f"Argument `c` must be at least 1, not {c}.")
    p, m = p[0], m[0]

    if primitive_poly is None:
        primitive_poly = primitive_poly_(p, m, method="smallest")

    GF = Field(p**m, irreducible_poly=primitive_poly, primitive_element=primitive_element)

    return GF


@set_module("galois")
def rs_generator_poly(n, k, c=1, primitive_poly=None, primitive_element=None, roots=False):
    """
    Returns the generator polynomial :math:`g(x)` for the :math:`\\textrm{RS}(n, k)` code.

    The Reed-Solomon generator polynomial :math:`g(x)` is defined as :math:`g(x) = (x - \\alpha^{c})(x - \\alpha^{c + 1}) \\dots (x - \\alpha^{c + 2t - 1})`,
    where :math:`\\alpha` is a primitive element of :math:`\\mathrm{GF}(q)` and :math:`q` is a prime power of the form :math:`q = n + 1`.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = q - 1`.
    k : int
        The message size :math:`k`. The error-correcting capability :math:`t` is defined by :math:`n - k = 2t`.
    c : int, optional
        The first consecutive power of :math:`\\alpha`. The default is 1.
    primitive_poly : galois.Poly, optional
        Optionally specify the primitive polynomial that defines the field :math:`\\mathrm{GF}(q)`. The default is
        `None` which uses the lexicographically-smallest primitive polynomial, i.e. `galois.primitive_poly(p, m, method="smallest")`
        where :math:`q = p^m`. The use of the lexicographically-smallest primitive polynomial, as opposed to a Conway polynomial,
        is most common for the default in textbooks, Matlab, and Octave.
    primitive_element : int, galois.Poly, optional
        Optionally specify the primitive element :math:`\\alpha` of :math:`\\mathrm{GF}(q)` whose powers are roots of the generator polynomial :math:`g(x)`.
        The default is `None` which uses the lexicographically-smallest primitive element in :math:`\\mathrm{GF}(q)`, i.e.
        `galois.primitive_element(p, m)`.
    roots : bool, optional
        Indicates to optionally return the :math:`2t` roots (in :math:`\\mathrm{GF}(q)`) of the generator polynomial. The default is `False`.

    Returns
    -------
    galois.Poly
        The generator polynomial :math:`g(x)` over :math:`\\mathrm{GF}(q)`.

    Examples
    --------
    .. ipython:: python

        galois.rs_generator_poly(63, 57)
        galois.rs_generator_poly(63, 57, roots=True)
    """
    GF = _check_and_compute_field(n, k, c, primitive_poly, primitive_element)
    alpha = GF.primitive_element
    t = (n - k) // 2
    roots_ = alpha**(c + np.arange(0, 2*t))
    g = Poly.Roots(roots_)

    if not roots:
        return g
    else:
        return g, roots_


@set_module("galois")
def rs_generator_matrix(n, k, c=1, primitive_poly=None, primitive_element=None, systematic=True):
    """
    Returns the generator matrix :math:`\\mathbf{G}` for the :math:`\\textrm{RS}(n, k)` code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = q - 1`.
    k : int
        The message size :math:`k`. The error-correcting capability :math:`t` is defined by :math:`n - k = 2t`.
    c : int, optional
        The first consecutive power of :math:`\\alpha`. The default is 1.
    primitive_poly : galois.Poly, optional
        Optionally specify the primitive polynomial that defines the field :math:`\\mathrm{GF}(q)`. The default is
        `None` which uses the lexicographically-smallest primitive polynomial, i.e. `galois.primitive_poly(p, m, method="smallest")`
        where :math:`q = p^m`. The use of the lexicographically-smallest primitive polynomial, as opposed to a Conway polynomial,
        is most common for the default in textbooks, Matlab, and Octave.
    primitive_element : int, galois.Poly, optional
        Optionally specify the primitive element :math:`\\alpha` of :math:`\\mathrm{GF}(q)` whose powers are roots of the generator polynomial :math:`g(x)`.
        The default is `None` which uses the lexicographically-smallest primitive element in :math:`\\mathrm{GF}(q)`, i.e.
        `galois.primitive_element(p, m)`.
    systematic : bool, optional
        Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
        appended. The default is `True`.

    Returns
    -------
    galois.FieldArray
        The :math:`(k, n)` generator matrix :math:`\\mathbf{G}`, such that given a message :math:`\\mathbf{m}`, a codeword is defined by
        :math:`\\mathbf{c} = \\mathbf{m}\\mathbf{G}`.

    Examples
    --------
    .. ipython :: python

        galois.rs_generator_poly(15, 9)
        galois.rs_generator_matrix(15, 9, systematic=False)
        galois.rs_generator_matrix(15, 9)
    """
    g = rs_generator_poly(n, k, c=c, primitive_poly=primitive_poly, primitive_element=primitive_element)
    G = generator_poly_to_matrix(n, g, systematic=systematic)
    return G


@set_module("galois")
def rs_parity_check_matrix(n, k, c=1, primitive_poly=None, primitive_element=None):
    """
    Returns the parity-check matrix :math:`\\mathbf{H}` for the :math:`\\textrm{RS}(n, k)` code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`, must be :math:`n = q - 1`.
    k : int
        The message size :math:`k`. The error-correcting capability :math:`t` is defined by :math:`n - k = 2t`.
    c : int, optional
        The first consecutive power of :math:`\\alpha`. The default is 1.
    primitive_poly : galois.Poly, optional
        Optionally specify the primitive polynomial that defines the field :math:`\\mathrm{GF}(q)`. The default is
        `None` which uses the lexicographically-smallest primitive polynomial, i.e. `galois.primitive_poly(p, m, method="smallest")`
        where :math:`q = p^m`. The use of the lexicographically-smallest primitive polynomial, as opposed to a Conway polynomial,
        is most common for the default in textbooks, Matlab, and Octave.
    primitive_element : int, galois.Poly, optional
        Optionally specify the primitive element :math:`\\alpha` of :math:`\\mathrm{GF}(q)` whose powers are roots of the generator polynomial :math:`g(x)`.
        The default is `None` which uses the lexicographically-smallest primitive element in :math:`\\mathrm{GF}(q)`, i.e.
        `galois.primitive_element(p, m)`.

    Returns
    -------
    galois.FieldArray
        The :math:`(n-k, n)` parity-check matrix :math:`\\mathbf{H}`, such that given a codeword :math:`\\mathbf{c}`, the syndrome is defined by
        :math:`\\mathbf{s} = \\mathbf{c}\\mathbf{H}^T`.

    Examples
    --------
    .. ipython :: python

        G = galois.rs_generator_matrix(15, 9); G
        H = galois.rs_parity_check_matrix(15, 9); H
        # The message
        m = type(G).Random(9); m
        # The codeword
        c = m @ G; c
        # Error pattern
        e = type(G).Zeros(15); e[0] = type(G).Random(low=1); e
        # c is a valid codeword, so the syndrome is 0
        s = c @ H.T; s
        # c + e is not a valid codeword, so the syndrome is not 0
        s = (c + e) @ H.T; s
    """
    _, roots = rs_generator_poly(n, k, c=c, primitive_poly=primitive_poly, primitive_element=primitive_element, roots=True)
    H = roots_to_parity_check_matrix(n, roots)
    return H
