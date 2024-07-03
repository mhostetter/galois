"""
A module containing common functions for linear block codes.
"""

from __future__ import annotations

from typing import Type, overload

import numpy as np
from typing_extensions import Literal

from .._fields import FieldArray
from .._helper import verify_isinstance, verify_literal
from ..typing import ArrayLike


class _LinearCode:
    """
    A FEC base class for linear codes.
    """

    def __init__(
        self,
        n: int,
        k: int,
        d: int,
        G: FieldArray,
        H: FieldArray,
        systematic: bool,
    ):
        verify_isinstance(n, int)
        verify_isinstance(k, int)
        verify_isinstance(d, int)
        verify_isinstance(G, FieldArray)
        verify_isinstance(H, FieldArray)
        verify_isinstance(systematic, bool)

        if not n >= k:
            raise ValueError(f"Argument `n` must be greater than or equal to `k`, {n} is not greater than {k}.")
        if not d >= 1:
            raise ValueError(f"Argument `d` must be at least 1, not {d}.")
        if not type(G) is type(H):
            raise ValueError(f"Arguments `G` and `H` must be over the same field, not {type(G)} and {type(H)}.")
        if not G.shape == (k, n):
            raise ValueError(f"Argument `G` must be have shape (k, n), {G.shape} is not ({k}, {n}).")
        if not H.shape == (n - k, n):
            raise ValueError(f"Argument `H` must be have shape (n - k, n), {H.shape} is not ({n - k}, {n}).")

        self._n = n
        self._k = k
        self._d = d
        self._field = type(G)
        self._G = G
        self._H = H
        self._is_systematic = systematic

    def encode(self, message: ArrayLike, output: Literal["codeword", "parity"] = "codeword") -> FieldArray:
        r"""
        Encodes the message $\mathbf{m}$ into the codeword $\mathbf{c}$.

        Arguments:
            message: The message as either a $k$-length vector or $(N, k)$ matrix, where $N$ is the
                number of messages.

                .. info::
                    :title: Shortened codes

                    For the shortened $[n-s,\ k-s,\ d]$ code (only applicable for systematic codes),
                    pass $k-s$ symbols into :func:`encode` to return the $n-s$-symbol message.

            output: Specify whether to return the codeword or parity symbols only. The default is `"codeword"`.

        Returns:
            If `output="codeword"`, the codeword as either a $n$-length vector or $(N, n)$ matrix.
            If `output="parity"`, the parity symbols as either a $n-k$-length vector or $(N, n-k)$ matrix.
        """
        verify_literal(output, ["codeword", "parity"])

        if output == "parity" and not self.is_systematic:
            raise ValueError("Argument `output` may only be 'parity' for systematic codes.")

        message, is_message_1d = self._check_and_convert_message(message)
        codeword = self._encode_message(message)

        if is_message_1d:
            codeword = codeword[0, :]

        if output == "codeword":
            return codeword

        parity = self._convert_codeword_to_parity(codeword)
        return parity

    def detect(self, codeword: ArrayLike) -> bool | np.ndarray:
        r"""
        Detects if errors are present in the codeword $\mathbf{c}$.

        Arguments:
            codeword: The codeword as either a $n$-length vector or $(N, n)$ matrix, where $N$ is the
                number of codewords.

                .. info::
                    :title: Shortened codes

                    For the shortened $[n-s,\ k-s,\ d]$ code (only applicable for systematic codes),
                    pass $n-s$ symbols into :func:`detect`.

        Returns:
            A boolean scalar or $N$-length array indicating if errors were detected in the corresponding codeword.
        """
        codeword, is_codeword_1d = self._check_and_convert_codeword(codeword)
        detected = self._detect_errors(codeword)

        if is_codeword_1d:
            detected = bool(detected[0])

        return detected

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

    def decode(self, codeword, output="message", errors=False):
        r"""
        Decodes the codeword $\mathbf{c}$ into the message $\mathbf{m}$.

        Arguments:
            codeword: The codeword as either a $n$-length vector or $(N, n)$ matrix, where $N$ is the
                number of codewords.

                .. info::
                    :title: Shortened codes

                    For the shortened $[n-s,\ k-s,\ d]$ code (only applicable for systematic codes),
                    pass $n-s$ symbols into :func:`decode` to return the $k-s$-symbol message.

            output: Specify whether to return the error-corrected message or entire codeword. The default is
                `"message"`.
            errors: Optionally specify whether to return the number of corrected errors. The default is `False`.

        Returns:
            - If `output="message"`, the error-corrected message as either a $k$-length vector or
              $(N, k)$ matrix. If `output="codeword"`, the error-corrected codeword as either a $n$-length
              vector or $(N, n)$ matrix.
            - If `errors=True`, returns the number of corrected symbol errors as either a scalar or $N$-length
              array. Valid number of corrections are in $[0, t]$. If a codeword has too many errors and cannot
              be corrected, -1 will be returned.
        """
        verify_literal(output, ["message", "codeword"])

        codeword, is_codeword_1d = self._check_and_convert_codeword(codeword)
        dec_codeword, N_errors = self._decode_codeword(codeword)

        if output == "message":
            decoded = self._convert_codeword_to_message(dec_codeword)
        else:
            decoded = dec_codeword

        if is_codeword_1d:
            decoded, N_errors = decoded[0, :], int(N_errors[0])

        if errors:
            return decoded, N_errors

        return decoded

    # def dual_code(self) -> _LinearCode:
    #     n = self.n
    #     k = self.n - self.k
    #     d = 0
    #     field = self.field
    #     G = self.G.null_space()
    #     H = self.H.null_space()
    #     systematic = self.is_systematic
    #     return super(n, k, d, field, G, H, systematic)

    ###############################################################################
    # Helper functions
    ###############################################################################

    def _check_and_convert_message(self, message: ArrayLike) -> tuple[FieldArray, bool]:
        """
        Converts the array-like message into a 2-D FieldArray with shape (N, ks).
        """
        # Convert the array-like message into a FieldArray
        message = self.field(message)

        if message.ndim > 2:
            raise ValueError(f"Argument `message` can be either 1-D or 2-D, not {message.ndim}-D.")
        if self.is_systematic:
            if not message.shape[-1] <= self.k:
                raise ValueError(
                    f"For a systematic code, argument `message` must be a 1-D or 2-D array "
                    f"with last dimension less than or equal to {self.k}, not shape {message.shape}."
                )
        else:
            if not message.shape[-1] == self.k:
                raise ValueError(
                    f"For a non-systematic code, argument `message` must be a 1-D or 2-D array "
                    f"with last dimension equal to {self.k}, not shape {message.shape}."
                )

        # Record if the original message was 1-D and then convert to 2-D
        is_message_1d = message.ndim == 1
        message = np.atleast_2d(message)

        return message, is_message_1d

    def _check_and_convert_codeword(self, codeword: FieldArray) -> FieldArray:
        """
        Converts the array-like codeword into a 2-D FieldArray with shape (N, ns).
        """
        # Convert the array-like codeword into a FieldArray
        codeword = self.field(codeword)

        if self.is_systematic:
            if not codeword.shape[-1] <= self.n:
                raise ValueError(
                    f"For a systematic code, argument `codeword` must be a 1-D or 2-D array "
                    f"with last dimension less than or equal to {self.n}, not shape {codeword.shape}."
                )
        else:
            if not codeword.shape[-1] == self.n:
                raise ValueError(
                    f"For a non-systematic code, argument `codeword` must be a 1-D or 2-D array "
                    f"with last dimension equal to {self.n}, not shape {codeword.shape}."
                )

        # Record if the original codeword was 1-D and then convert to 2-D
        is_codeword_1d = codeword.ndim == 1
        codeword = np.atleast_2d(codeword)

        return codeword, is_codeword_1d

    def _convert_codeword_to_message(self, codeword: FieldArray) -> FieldArray:
        """
        Returns the message portion (N, k) of the codeword (N, ns).
        """
        raise NotImplementedError

    def _convert_codeword_to_parity(self, codeword: FieldArray) -> FieldArray:
        """
        Returns the parity portion (N, n - k) of the codeword (N, ns).
        """
        raise NotImplementedError

    ###############################################################################
    # Actual implementation functions that must be implemented in children
    # classes
    ###############################################################################

    def _encode_message(self, message: FieldArray) -> FieldArray:
        """
        Encodes the message with shape (N, ks) into the codeword with shape (N, ns).
        """
        ks = message.shape[-1]  # The number of input message symbols (could be less than self.k for shortened codes)

        if self.is_systematic:
            parity = message @ self.G[-ks:, self.k :]
            codeword = np.hstack((message, parity))
        else:
            codeword = message @ self.G

        return codeword

    def _detect_errors(self, codeword: FieldArray) -> np.ndarray:
        """
        Returns a boolean array (N,) indicating if errors are present in the codeword.
        """
        ns = codeword.shape[-1]  # The number of input codeword symbols (could be less than self.n for shortened codes)

        # Compute the syndrome with shape (N, n-k) over GF(q) by matrix multiplying with the parity-check matrix
        syndrome = codeword @ self.H[:, -ns:].T

        # Any non-zero syndrome row has errors
        detected = ~np.all(syndrome == 0, axis=1)

        return detected

    def _decode_codeword(self, codeword: FieldArray) -> tuple[FieldArray, np.ndarray]:
        """
        Decodes errors in the received codeword. Returns the corrected codeword (N, ns) and array of number of
        corrected errors (N,).
        """
        raise NotImplementedError

    ###############################################################################
    # Instance properties
    ###############################################################################

    @property
    def field(self) -> Type[FieldArray]:
        r"""
        The Galois field $\mathrm{GF}(q)$ that defines the codeword alphabet.
        """
        return self._field

    @property
    def n(self) -> int:
        """
        The codeword size $n$ of the $[n, k, d]_q$ code. This is also called the code *length*.
        """
        return self._n

    @property
    def k(self) -> int:
        """
        The message size $k$ of the $[n, k, d]_q$ code. This is also called the code *dimension*.
        """
        return self._k

    @property
    def d(self) -> int:
        r"""
        The minimum distance $d$ of the $[n, k, d]_q$ code.
        """
        return self._d

    @property
    def t(self) -> int:
        r"""
        The error-correcting capability $t$ of the code.

        Notes:
            The code can correct $t$ symbol errors in a codeword.

            $$t = \bigg\lfloor \frac{d - 1}{2} \bigg\rfloor$$
        """
        return (self.d - 1) // 2

    @property
    def G(self) -> FieldArray:
        r"""
        The generator matrix $\mathbf{G}$ with shape $(k, n)$.

        Group:
            Matrices

        Order:
            71
        """
        return self._G

    @property
    def H(self) -> FieldArray:
        r"""
        The parity-check matrix $\mathbf{H}$ with shape $(n - k, n)$.

        Group:
            Matrices

        Order:
            71
        """
        return self._H

    @property
    def is_systematic(self) -> bool:
        """
        Indicates if the code is *systematic*, meaning the codewords have parity appended to the message.
        """
        return self._is_systematic


def generator_to_parity_check_matrix(G: FieldArray) -> FieldArray:
    r"""
    Converts the generator matrix $\mathbf{G}$ of a linear $[n, k]$ code into its parity-check matrix
    $\mathbf{H}$.

    The generator and parity-check matrices satisfy the equations $\mathbf{G}\mathbf{H}^T = \mathbf{0}$.

    Arguments:
        G: The $(k, n)$ generator matrix $\mathbf{G}$ in systematic form
            $\mathbf{G} = [\mathbf{I}_{k,k} \mid \mathbf{P}_{k,n-k}]$.

    Returns:
        The $(n-k, n)$ parity-check matrix
        $\mathbf{H} = [-\mathbf{P}_{k,n-k}^T \mid \mathbf{I}_{n-k,n-k}]$`.

    Examples:
        .. ipython:: python

            g = galois.primitive_poly(2, 3); g
            G = galois.poly_to_generator_matrix(7, g); G
            H = galois.generator_to_parity_check_matrix(G); H
            G @ H.T

    Group:
        fec
    """
    verify_isinstance(G, FieldArray)

    field = type(G)
    k, n = G.shape
    if not np.array_equal(G[:, 0:k], np.eye(k)):
        raise ValueError("Argument 'G' must be in systematic form [I | P].")

    P = G[:, k:]
    I = field.Identity(n - k)
    H = np.hstack((-P.T, I))

    return H


def parity_check_to_generator_matrix(H: FieldArray) -> FieldArray:
    r"""
    Converts the parity-check matrix $\mathbf{H}$ of a linear $[n, k]$ code into its generator matrix
    $\mathbf{G}$.

    The generator and parity-check matrices satisfy the equations $\mathbf{G}\mathbf{H}^T = \mathbf{0}$.

    Arguments:
        H: The $(n-k, n)$ parity-check matrix $\mathbf{G}$ in systematic form
            $\mathbf{H} = [-\mathbf{P}_{k,n-k}^T \mid \mathbf{I}_{n-k,n-k}]$`.

    Returns:
        The $(k, n)$ generator matrix $\mathbf{G} = [\mathbf{I}_{k,k} \mid \mathbf{P}_{k,n-k}]$.

    Examples:
        .. ipython:: python

            g = galois.primitive_poly(2, 3); g
            G = galois.poly_to_generator_matrix(7, g); G
            H = galois.generator_to_parity_check_matrix(G); H
            G2 = galois.parity_check_to_generator_matrix(H); G2
            G2 @ H.T

    Group:
        fec
    """
    verify_isinstance(H, FieldArray)

    field = type(H)
    n_k, n = H.shape
    k = n - n_k
    if not np.array_equal(H[:, k:], np.eye(n - k)):
        raise ValueError("Argument 'H' must be in systematic form [-P^T | I].")

    P = -H[:, 0:k].T
    I = field.Identity(k)
    G = np.hstack((I, P))

    return G
