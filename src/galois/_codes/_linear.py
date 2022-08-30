"""
A module containing common functions for linear block codes.
"""
from __future__ import annotations

import numpy as np

from .._fields import FieldArray
from .._helper import export, verify_isinstance


@export
def generator_to_parity_check_matrix(G: FieldArray) -> FieldArray:
    r"""
    Converts the generator matrix :math:`\mathbf{G}` of a linear :math:`[n, k]` code into its parity-check matrix :math:`\mathbf{H}`.

    The generator and parity-check matrices satisfy the equations :math:`\mathbf{G}\mathbf{H}^T = \mathbf{0}`.

    Parameters
    ----------
    G
        The :math:`(k, n)` generator matrix :math:`\mathbf{G}` in systematic form
        :math:`\mathbf{G} = [\mathbf{I}_{k,k}\ |\ \mathbf{P}_{k,n-k}]`.

    Returns
    -------
    :
        The :math:`(n-k, n)` parity-check matrix :math:`\mathbf{H} = [-\mathbf{P}_{k,n-k}^T\ |\ \mathbf{I}_{n-k,n-k}]``.

    Examples
    --------
    .. ipython:: python

        g = galois.primitive_poly(2, 3); g
        G = galois.poly_to_generator_matrix(7, g); G
        H = galois.generator_to_parity_check_matrix(G); H
        G @ H.T

    :group: fec
    """
    verify_isinstance(G, FieldArray)

    field = type(G)
    k, n = G.shape
    if not np.array_equal(G[:,0:k], np.eye(k)):
        raise ValueError("Argument `G` must be in systematic form [I | P].")

    P = G[:, k:]
    I = field.Identity(n-k)
    H = np.hstack((-P.T, I))

    return H


@export
def parity_check_to_generator_matrix(H: FieldArray) -> FieldArray:
    r"""
    Converts the parity-check matrix :math:`\mathbf{H}` of a linear :math:`[n, k]` code into its generator matrix :math:`\mathbf{G}`.

    The generator and parity-check matrices satisfy the equations :math:`\mathbf{G}\mathbf{H}^T = \mathbf{0}`.

    Parameters
    ----------
    H
        The :math:`(n-k, n)` parity-check matrix :math:`\mathbf{G}` in systematic form
        :math:`\mathbf{H} = [-\mathbf{P}_{k,n-k}^T\ |\ \mathbf{I}_{n-k,n-k}]``.

    Returns
    -------
    :
        The :math:`(k, n)` generator matrix :math:`\mathbf{G} = [\mathbf{I}_{k,k}\ |\ \mathbf{P}_{k,n-k}]`.

    Examples
    --------
    .. ipython:: python

        g = galois.primitive_poly(2, 3); g
        G = galois.poly_to_generator_matrix(7, g); G
        H = galois.generator_to_parity_check_matrix(G); H
        G2 = galois.parity_check_to_generator_matrix(H); G2
        G2 @ H.T

    :group: fec
    """
    verify_isinstance(H, FieldArray)

    field = type(H)
    n_k, n = H.shape
    k = n - n_k
    if not np.array_equal(H[:,k:], np.eye(n - k)):
        raise ValueError("Argument `H` must be in systematic form [-P^T | I].")

    P = -H[:, 0:k].T
    I = field.Identity(k)
    G = np.hstack((I, P))

    return G
