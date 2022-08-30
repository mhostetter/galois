"""
A module containing common functions for cyclic codes.
"""
from __future__ import annotations

import numpy as np

from .._fields import FieldArray
from .._helper import export, verify_isinstance
from .._polys import Poly


@export
def poly_to_generator_matrix(n: int, generator_poly: Poly, systematic: bool = True) -> FieldArray:
    r"""
    Converts the generator polynomial :math:`g(x)` into the generator matrix :math:`\mathbf{G}` for an :math:`[n, k]` cyclic code.

    Parameters
    ----------
    n
        The codeword size :math:`n`.
    generator_poly
        The generator polynomial :math:`g(x)`.
    systematic
        Optionally specify if the encoding should be systematic, meaning the codeword is the message with parity
        appended. The default is `True`.

    Returns
    -------
    :
        The :math:`(k, n)` generator matrix :math:`\mathbf{G}`, such that given a message :math:`\mathbf{m}`, a codeword is defined by
        :math:`\mathbf{c} = \mathbf{m}\mathbf{G}`.

    Examples
    --------
    Compute the generator matrix for the :math:`\mathrm{Hamming}(7, 4)` code.

    .. ipython :: python

        g = galois.primitive_poly(2, 3); g
        galois.poly_to_generator_matrix(7, g, systematic=False)
        galois.poly_to_generator_matrix(7, g, systematic=True)

    :group: fec
    """
    verify_isinstance(n, int)
    verify_isinstance(generator_poly, Poly)
    verify_isinstance(systematic, bool)

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


@export
def roots_to_parity_check_matrix(n: int, roots: FieldArray) -> FieldArray:
    r"""
    Converts the generator polynomial roots into the parity-check matrix :math:`\mathbf{H}` for an :math:`[n, k]` cyclic code.

    Parameters
    ----------
    n
        The codeword size :math:`n`.
    roots
        The :math:`2t` roots of the generator polynomial :math:`g(x)`.

    Returns
    -------
    :
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

    :group: fec
    """
    verify_isinstance(n, int)
    verify_isinstance(roots, FieldArray)

    GF = type(roots)
    H = np.power.outer(roots, np.arange(n - 1, -1, -1, dtype=GF.dtypes[-1]))

    return H
