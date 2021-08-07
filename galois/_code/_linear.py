"""
A module containing common functions for linear block codes.
"""
import numpy as np

from .._field import FieldArray
from .._overrides import set_module

__all__ = ["generator_to_parity_check_matrix", "parity_check_to_generator_matrix"]


@set_module("galois")
def generator_to_parity_check_matrix(G):
    r"""
    Converts the generator matrix :math:`\mathbf{G}` of a linear :math:`[n, k]` code into its parity-check matrix :math:`\mathbf{H}`.

    The generator and parity-check matrices satisfy the equations :math:`\mathbf{G}\mathbf{H}^T = \mathbf{0}`.

    Parameters
    ----------
    G : galois.FieldArray
        The :math:`(k, n)` generator matrix :math:`\mathbf{G}` in systematic form
        :math:`\mathbf{G} = [\mathbf{I}_{k,k}\ |\ \mathbf{P}_{k,n-k}]`.

    Returns
    -------
    galois.FieldArray
        The :math:`(n-k, n)` parity-check matrix :math:`\mathbf{H} = [-\mathbf{P}_{k,n-k}^T\ |\ \mathbf{I}_{n-k,n-k}]``.

    Examples
    --------
    .. ipython:: python

        g = galois.primitive_poly(2, 3); g
        G = galois.poly_to_generator_matrix(7, g); G
        H = galois.generator_to_parity_check_matrix(G); H
        G @ H.T
    """
    if not isinstance(G, (FieldArray)):
        raise TypeError(f"Argument `G` must be a galois.FieldArray, not {type(G)}.")

    field = type(G)
    k, n = G.shape
    if not np.array_equal(G[:,0:k], np.eye(k)):
        raise ValueError("Argument `G` must be in systematic form [I | P].")

    P = G[:, k:]
    I = field.Identity(n-k)
    H = np.hstack((-P.T, I))

    return H


@set_module("galois")
def parity_check_to_generator_matrix(H):
    r"""
    Converts the parity-check matrix :math:`\mathbf{H}` of a linear :math:`[n, k]` code into its generator matrix :math:`\mathbf{G}`.

    The generator and parity-check matrices satisfy the equations :math:`\mathbf{G}\mathbf{H}^T = \mathbf{0}`.

    Parameters
    ----------
    H : galois.FieldArray
        The :math:`(n-k, n)` parity-check matrix :math:`\mathbf{G}` in systematic form
        :math:`\mathbf{H} = [-\mathbf{P}_{k,n-k}^T\ |\ \mathbf{I}_{n-k,n-k}]``.

    Returns
    -------
    galois.FieldArray
        The :math:`(k, n)` generator matrix :math:`\mathbf{G} = [\mathbf{I}_{k,k}\ |\ \mathbf{P}_{k,n-k}]`.

    Examples
    --------
    .. ipython:: python

        g = galois.primitive_poly(2, 3); g
        G = galois.poly_to_generator_matrix(7, g); G
        H = galois.generator_to_parity_check_matrix(G); H
        G2 = galois.parity_check_to_generator_matrix(H); G2
        G2 @ H.T
    """
    if not isinstance(H, (FieldArray)):
        raise TypeError(f"Argument `H` must be a galois.FieldArray, not {type(H)}.")

    field = type(H)
    n_k, n = H.shape
    k = n - n_k
    if not np.array_equal(H[:,k:], np.eye(n - k)):
        raise ValueError("Argument `H` must be in systematic form [-P^T | I].")

    P = -H[:, 0:k].T
    I = field.Identity(k)
    G = np.hstack((I, P))

    return G
