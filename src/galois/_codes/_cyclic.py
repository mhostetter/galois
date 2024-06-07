"""
A module containing common functions for cyclic codes.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from typing_extensions import Literal

from .._fields import FieldArray
from .._helper import extend_docstring, verify_isinstance
from .._polys import Poly
from .._polys._dense import divmod_jit
from ..typing import ArrayLike
from ._linear import _LinearCode


class _CyclicCode(_LinearCode):
    """
    A FEC base class for cyclic codes.
    """

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        generator_poly: Poly,
        roots: FieldArray,
        systematic: bool,
    ):
        verify_isinstance(n, int)
        verify_isinstance(k, int)
        verify_isinstance(d, int)
        verify_isinstance(generator_poly, Poly)
        verify_isinstance(roots, FieldArray)
        verify_isinstance(systematic, bool)

        self._generator_poly = generator_poly
        self._roots = roots

        # Calculate the parity-check polynomial h(x) = (x^n - 1) / g(x)
        f = Poly.Degrees([n, 0], [1, -1], field=generator_poly.field)
        parity_check_poly, remainder_poly = divmod(f, generator_poly)
        assert remainder_poly == 0
        self._parity_check_poly = parity_check_poly

        G = _poly_to_generator_matrix(n, generator_poly, systematic)
        H = _poly_to_parity_check_matrix(n, parity_check_poly, False)

        super().__init__(n, k, d, G, H, systematic)

    @extend_docstring(
        _LinearCode.encode,
        {},
        r"""
        Notes:
            The message vector $\mathbf{m}$ is a member of $\mathrm{GF}(q)^k$. The corresponding
            message polynomial $m(x)$ is a degree-$k$ polynomial over $\mathrm{GF}(q)$.

            $$\mathbf{m} = [m_{k-1},\ \dots,\ m_1,\ m_0] \in \mathrm{GF}(q)^k$$

            $$m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0 \in \mathrm{GF}(q)[x]$$

            The codeword vector $\mathbf{c}$ is a member of $\mathrm{GF}(q)^n$. The corresponding
            codeword polynomial $c(x)$ is a degree-$n$ polynomial over $\mathrm{GF}(q)$.

            $$\mathbf{c} = [c_{n-1},\ \dots,\ c_1,\ c_0] \in \mathrm{GF}(q)^n$$

            $$c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0 \in \mathrm{GF}(q)[x]$$

            The codeword vector is computed by matrix multiplication of the message vector with the generator matrix.
            The equivalent polynomial operation is multiplication of the message polynomial with the generator
            polynomial.

            $$\mathbf{c} = \mathbf{m} \mathbf{G}$$

            $$c(x) = m(x) g(x)$$
        """,
    )
    def encode(self, message: ArrayLike, output: Literal["codeword", "parity"] = "codeword") -> FieldArray:
        return super().encode(message, output=output)

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[False] = False,
    ) -> FieldArray: ...

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[True] = True,
    ) -> tuple[FieldArray, int | np.ndarray]: ...

    @extend_docstring(
        _LinearCode.decode,
        {},
        r"""
        Notes:
            The message vector $\mathbf{m}$ is a member of $\mathrm{GF}(q)^k$. The corresponding
            message polynomial $m(x)$ is a degree-$k$ polynomial over $\mathrm{GF}(q)$.

            $$\mathbf{m} = [m_{k-1},\ \dots,\ m_1,\ m_0] \in \mathrm{GF}(q)^k$$

            $$m(x) = m_{k-1} x^{k-1} + \dots + m_1 x + m_0 \in \mathrm{GF}(q)[x]$$

            The codeword vector $\mathbf{c}$ is a member of $\mathrm{GF}(q)^n$. The corresponding
            codeword polynomial $c(x)$ is a degree-$n$ polynomial over $\mathrm{GF}(q)$.
            Each codeword polynomial $c(x)$ is divisible by the generator polynomial $g(x)$.

            $$\mathbf{c} = [c_{n-1},\ \dots,\ c_1,\ c_0] \in \mathrm{GF}(q)^n$$

            $$c(x) = c_{n-1} x^{n-1} + \dots + c_1 x + c_0 \in \mathrm{GF}(q)[x]$$
        """,
    )
    def decode(self, codeword, output="message", errors=False):
        return super().decode(codeword, output=output, errors=errors)

    def _convert_codeword_to_message(self, codeword: FieldArray) -> FieldArray:
        ns = codeword.shape[-1]  # The number of codeword symbols (could be less than self.n for shortened codes)
        ks = self.k - (self.n - ns)  # The number of message symbols (could be less than self.k for shortened codes)

        if self.is_systematic:
            message = codeword[..., 0:ks]
        else:
            message, _ = divmod_jit(self.field)(codeword, self.generator_poly.coeffs)

        return message

    def _convert_codeword_to_parity(self, codeword: FieldArray) -> FieldArray:
        if self.is_systematic:
            parity = codeword[..., -(self.n - self.k) :]
        else:
            _, parity = divmod_jit(self.field)(codeword, self.generator_poly.coeffs)

        return parity

    @property
    def generator_poly(self) -> Poly:
        r"""
        The generator polynomial $g(x)$ over $\mathrm{GF}(q)$.

        Notes:
            Every codeword $\mathbf{c}$ can be represented as a degree-$n$ polynomial $c(x)$.
            Each codeword polynomial $c(x)$ is a multiple of $g(x)$.

        Group:
            Polynomials

        Order:
            72
        """
        return self._generator_poly

    @property
    def parity_check_poly(self) -> Poly:
        r"""
        The parity-check polynomial $h(x)$.

        Notes:
            The parity-check polynomial is the generator polynomial of the dual code.

        Group:
            Polynomials

        Order:
            72
        """
        return self._parity_check_poly

    @property
    def roots(self) -> FieldArray:
        r"""
        The $d - 1$ roots of the generator polynomial $g(x)$.

        Group:
            Polynomials

        Order:
            72
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
            P[0, :] = generator_poly.coeffs[0:-1] / generator_poly.coeffs[-1]
            for i in range(1, k):
                P[i, 0] = 0
                P[i, 1:] = P[i - 1, 0:-1]
                if P[i - 1, -1] > 0:
                    P[i, :] -= P[i - 1, -1] * P[0, :]
        G = np.hstack((I, P))
    else:
        # Assign the generator polynomial coefficients with highest degree starting along
        # the diagonals
        G = GF.Zeros((k, n))
        for i in range(k):
            G[i, i : i + generator_poly.degree + 1] = generator_poly.coeffs

    return G


def _poly_to_parity_check_matrix(n: int, parity_check_poly: Poly, systematic: bool) -> FieldArray:
    """
    Converts the generator polynomial g(x) into the generator matrix G over GF(q).
    """
    return _poly_to_generator_matrix(n, parity_check_poly.reverse(), systematic)
