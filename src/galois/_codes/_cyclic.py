"""
A module containing common functions for cyclic codes.
"""
from __future__ import annotations

from typing import Tuple, Union, overload
from typing_extensions import Literal

import numpy as np

from .._fields import FieldArray
from .._helper import verify_isinstance, extend_docstring
from .._polys import Poly
from .._polys._dense import divmod_jit
from ..typing import ArrayLike

from ._linear import _LinearCode


class _CyclicCode(_LinearCode):
    """
    A FEC base class for cyclic codes.
    """
    # pylint: disable=abstract-method

    def __init__(self, n: int, k: int, d: int, generator_poly: Poly, roots: FieldArray, systematic: bool):
        verify_isinstance(n, int)
        verify_isinstance(k, int)
        verify_isinstance(d, int)
        verify_isinstance(generator_poly, Poly)
        verify_isinstance(roots, FieldArray)
        verify_isinstance(systematic, bool)

        self._generator_poly = generator_poly
        self._roots = roots

        # Calculate the parity-check polynomial h(x) = (x^n - 1) / g(x)
        parity_check_poly, remainder_poly = divmod(Poly.Degrees([n, 0], [1, -1], field=generator_poly.field), generator_poly)
        assert remainder_poly == 0
        self._parity_check_poly = parity_check_poly

        G = _poly_to_generator_matrix(n, generator_poly, systematic)
        H = _poly_to_parity_check_matrix(n, parity_check_poly, False)

        super().__init__(n, k, d, G, H, systematic)

    @extend_docstring(_LinearCode.encode, {},
        r"""
        Notes
        -----
        The message vector :math:`\mathbf{m}` is a member of :math:`\mathrm{GF}(q)^k`. The corresponding
        message polynomial :math:`m(x)` is a degree-:math:`k` polynomial over :math:`\mathrm{GF}(q)`.

        .. math::
            \mathbf{m} = [m_{k-1},\ \dots,\ m_1,\ m_0] \in \mathrm{GF}(q)^k

        .. math::
            m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0 \in \mathrm{GF}(q)[x]

        The codeword vector :math:`\mathbf{c}` is a member of :math:`\mathrm{GF}(q)^n`. The corresponding
        codeword polynomial :math:`c(x)` is a degree-:math:`n` polynomial over :math:`\mathrm{GF}(q)`.

        .. math::
            \mathbf{c} = [c_{n-1},\ \dots,\ c_1,\ c_0] \in \mathrm{GF}(q)^n

        .. math::
            c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0 \in \mathrm{GF}(q)[x]

        The codeword vector is computed by matrix multiplication of the message vector with the generator matrix.
        The equivalent polynomial operation is multiplication of the message polynomial with the generator
        polynomial.

        .. math::
            \mathbf{c} = \mathbf{m} \mathbf{G}

        .. math::
            c(x) = m(x) g(x)
        """
    )
    def encode(self, message: ArrayLike, parity_only: bool = False) -> FieldArray:
        return super().encode(message, parity_only=parity_only)

    @overload
    def decode(self, codeword: ArrayLike, errors: Literal[False] = False) -> FieldArray:
        ...
    @overload
    def decode(self, codeword: ArrayLike, errors: Literal[True]) -> Tuple[FieldArray, Union[int, np.ndarray]]:  # pylint: disable=signature-differs
        ...
    @extend_docstring(_LinearCode.decode, {},
        r"""
        Notes
        -----
        The message vector :math:`\mathbf{m}` is a member of :math:`\mathrm{GF}(q)^k`. The corresponding
        message polynomial :math:`m(x)` is a degree-:math:`k` polynomial over :math:`\mathrm{GF}(q)`.

        .. math::
            \mathbf{m} = [m_{k-1},\ \dots,\ m_1,\ m_0] \in \mathrm{GF}(q)^k

        .. math::
            m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0 \in \mathrm{GF}(q)[x]

        The codeword vector :math:`\mathbf{c}` is a member of :math:`\mathrm{GF}(q)^n`. The corresponding
        codeword polynomial :math:`c(x)` is a degree-:math:`n` polynomial over :math:`\mathrm{GF}(q)`.
        Each codeword polynomial :math:`c(x)` is divisible by the generator polynomial :math:`g(x)`.

        .. math::
            \mathbf{c} = [c_{n-1},\ \dots,\ c_1,\ c_0] \in \mathrm{GF}(q)^n

        .. math::
            c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0 \in \mathrm{GF}(q)[x]
        """
    )
    def decode(self, codeword, errors=False):
        return super().decode(codeword, errors=errors)

    def _convert_codeword_to_message(self, codeword: FieldArray) -> FieldArray:
        if self.is_systematic:
            message = codeword[..., 0:self.k]
        else:
            message, _ = divmod_jit(self.field)(codeword, self.generator_poly.coeffs)
        return message

    def _convert_codeword_to_parity(self, codeword: FieldArray) -> FieldArray:
        if self.is_systematic:
            parity = codeword[..., -(self.n - self.k):]
        else:
            _, parity = divmod_jit(self.field)(codeword, self.generator_poly.coeffs)
        return parity

    @property
    def generator_poly(self) -> Poly:
        r"""
        The generator polynomial :math:`g(x)` over :math:`\mathrm{GF}(q)`.

        Every codeword :math:`\mathbf{c}` can be represented as a degree-:math:`n` polynomial :math:`c(x)`.
        Each codeword polynomial :math:`c(x)` is a multiple of :math:`g(x)`.
        """
        return self._generator_poly

    @property
    def parity_check_poly(self) -> Poly:
        r"""
        The parity-check polynomial :math:`h(x)`.

        The parity-check polynomial is the generator polynomial of the dual code.
        """
        return self._parity_check_poly

    @property
    def roots(self) -> FieldArray:
        r"""
        The :math:`d - 1` roots of the generator polynomial :math:`g(x)`.
        """
        return self._roots


def _poly_to_generator_matrix(n: int, generator_poly: Poly, systematic: bool) -> FieldArray:
    """
    Converts the generator polynomial g(x) into the generator matrix G over GF(q).
    """
    GF = generator_poly.field
    k = n - generator_poly.degree

    if systematic:
        # This is a more efficient Gaussian elimination of the generator matrix G that is produced
        # in non-systematic form
        I = GF.Identity(k)
        P = GF.Zeros((k, n - k))
        if n - k > 0:
            # Skip this section is n = k, which is the identity code (contains no parity)
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


def _poly_to_parity_check_matrix(n: int, parity_check_poly: Poly, systematic: bool) -> FieldArray:
    """
    Converts the generator polynomial g(x) into the generator matrix G over GF(q).
    """
    return _poly_to_generator_matrix(n, parity_check_poly.reverse(), systematic)



# def _roots_to_parity_check_matrix(n: int, roots: FieldArray, field: Type[FieldArray]) -> FieldArray:
#     r"""
#     Converts the generator polynomial g(x) roots in GF(q^m) into the parity-check matrix H over GF(q).
#     """
#     extension_field = type(roots)
#     m = ilog(extension_field.order, field.order)

#     # Create the parity-check matrix over GF(q^m) with shape (#roots, n)
#     H = np.power.outer(roots, np.arange(n - 1, -1, -1, dtype=extension_field.dtypes[-1]))
#     print(H)

#     if m > 1:
#         # Convert the parity-check matrix to GF(q) with shape (m * #roots, n)
#         H = H.T.vector().reshape((n, -1)).T
#         print(H)
#     assert isinstance(H, field)

#     return H


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
