"""
A module containing common functions for cyclic codes.
"""
import numpy as np

from .._fields import FieldArray
from .._overrides import set_module
from .._polys import Poly

__all__ = ["poly_to_generator_matrix", "roots_to_parity_check_matrix"]


@set_module("galois")
def poly_to_generator_matrix(n, generator_poly, systematic=True):
    r"""
    Converts the generator polynomial :math:`g(x)` into the generator matrix :math:`\mathbf{G}` for an :math:`[n, k]` cyclic code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`.
    generator_poly : galois.Poly
        The generator polynomial :math:`g(x)`.
    systematic : bool, optional
        Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
        appended. The default is `True`.

    Returns
    -------
    galois.FieldArray
        The :math:`(k, n)` generator matrix :math:`\mathbf{G}`, such that given a message :math:`\mathbf{m}`, a codeword is defined by
        :math:`\mathbf{c} = \mathbf{m}\mathbf{G}`.

    Examples
    --------
    Compute the generator matrix for the :math:`\mathrm{Hamming}(7, 4)` code.

    .. ipython :: python

        g = galois.primitive_poly(2, 3); g
        galois.poly_to_generator_matrix(7, g, systematic=False)
        galois.poly_to_generator_matrix(7, g, systematic=True)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(generator_poly, Poly):
        raise TypeError(f"Argument `generator_poly` must be a galois.Poly, not {type(generator_poly)}.")
    if not isinstance(systematic, bool):
        raise TypeError(f"Argument `systematic` must be a bool, not {type(systematic)}.")

    GF = generator_poly.field
    k = n - generator_poly.degree

    if systematic:
        # This is a more efficient Gaussian elimination of the generator matrix G that is produced
        # in non-systematic form
        I = GF.Identity(k)
        P = GF.Zeros((k, n-k))
        P[0,:] = generator_poly.coeffs[0:-1] / generator_poly.coeffs[-1]
        for i in range(1, k):
            P[i,0] = 0
            P[i,1:] = P[i-1,0:-1]
            if P[i-1,-1] > 0:
                P[i,:] -= P[i-1,-1] * P[0,:]
        G = np.hstack((I, P))
    else:
        # Assign the generator polynomial coefficients with highest degree starting along
        # the diagonals
        G = GF.Zeros((k, n))
        for i in range(k):
            G[i, i:i + generator_poly.degree + 1] = generator_poly.coeffs

    return G


@set_module("galois")
def roots_to_parity_check_matrix(n, roots):
    r"""
    Converts the generator polynomial roots into the parity-check matrix :math:`\mathbf{H}` for an :math:`[n, k]` cyclic code.

    Parameters
    ----------
    n : int
        The codeword size :math:`n`.
    roots : galois.FieldArray
        The :math:`2t` roots of the generator polynomial :math:`g(x)`.

    Returns
    -------
    galois.FieldArray
        The :math:`(2t, n)` parity-check matrix :math:`\mathbf{H}`, such that given a codeword :math:`\mathbf{c}`, the syndrome is defined by
        :math:`\mathbf{s} = \mathbf{c}\mathbf{H}^T`.

    Examples
    --------
    Compute the parity-check matrix for the :math:`\mathrm{RS}(15, 9)` code.

    .. ipython :: python

        GF = galois.GF(2**4)
        alpha = GF.primitive_element
        t = 3
        roots = alpha**np.arange(1, 2*t + 1); roots
        g = galois.Poly.Roots(roots); g
        galois.roots_to_parity_check_matrix(15, roots)
    """
    if not isinstance(n, (int, np.integer)):
        raise TypeError(f"Argument `n` must be an integer, not {type(n)}.")
    if not isinstance(roots, FieldArray):
        raise TypeError(f"Argument `roots` must be a galois.FieldArray, not {type(roots)}.")

    H = np.power.outer(roots, np.arange(n - 1, -1, -1))

    return H
