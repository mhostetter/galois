import numpy as np

from ..field import FieldArray, Poly
from ..overrides import set_module

__all__ = ["generator_poly_to_matrix", "roots_to_parity_check_matrix"]


@set_module("galois")
def generator_poly_to_matrix(n, g, systematic=True):
    """
    Converts the generator polynomial :math:`g(x)` into the generator matrix :math:`\\mathbf{G}` for :math:`[n, k]` code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`.
    g : galois.Poly
        The generator polynomial :math:`g(x)`.
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

        g = galois.bch_generator_poly(15, 7); g
        galois.generator_poly_to_matrix(15, g, systematic=False)
        galois.generator_poly_to_matrix(15, g, systematic=True)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(g, Poly):
        raise TypeError(f"Argument `g` must be a galois.Poly, not {type(g)}.")
    if not isinstance(systematic, bool):
        raise TypeError(f"Argument `systematic` must be a bool, not {type(systematic)}.")

    # Assign the generator polynomial coefficients with highest degree starting along the diagonal
    k = n - g.degree
    G = g.field.Zeros((k, n))
    for i in range(k):
        G[i, i:i + g.degree + 1] = g.coeffs

    # Convert G to the form [I | P]
    if systematic:
        G = G.row_reduce()  # pylint: disable=no-member

    return G


@set_module("galois")
def roots_to_parity_check_matrix(n, roots):
    """
    Converts the generator polynomial roots into the parity-check matrix :math:`\\mathbf{H}` for :math:`[n, k]` code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`.
    roots : galois.FieldArray
        The generator polynomial roots.

    Returns
    -------
    galois.FieldArray
        The :math:`(n-k, n)` parity-check matrix :math:`\\mathbf{H}`, such that given a codeword :math:`\\mathbf{c}`, the syndrome is defined by
        :math:`\\mathbf{s} = \\mathbf{c}\\mathbf{H}^T`.

    Examples
    --------
    .. ipython :: python

        g, roots = galois.bch_generator_poly(15, 7, roots=True); g, roots
        galois.roots_to_parity_check_matrix(15, roots)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(roots, FieldArray):
        raise TypeError(f"Argument `roots` must be a galois.FieldArray, not {type(roots)}.")

    H = np.power.outer(roots, np.arange(n - 1, -1, -1))

    return H
