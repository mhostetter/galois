"""
G is kxn
H is (n-k)xn
m is 1xk
c is 1xn
"""
from __future__ import annotations

from typing import overload

import numpy as np
from typing_extensions import Literal

from .._fields import GF, FieldArray
from ..typing import ArrayLike
from ._linear import _LinearCode, parity_check_to_generator_matrix


class HammingCode(_LinearCode):

    def __init__(self, r: int, field: FieldArray):
        # Construct the Generator and Parity Check Matrix and then initialise the parent class
        self._r = r
        self._q = field.characteristic

        n = int((self._q ** self._r - 1)/(self._q - 1))
        k = n - r
        d = 3

        H = self._generate_systematic_parity_check_matrix(r, field)
        G = parity_check_to_generator_matrix(H)

        super().__init__(n, k, d, G, H, systematic=True)

    def encode(self, message: ArrayLike, output: Literal["codeword", "parity"] = "codeword") -> FieldArray:
        # Call the parent class's encode method
        return super().encode(message=message, output=output)

    def detect(self, codeword: FieldArray) -> np.ndarray:
        # Call the parent class's detect method
        return super().detect(codeword=codeword)

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[False] = False,
    ) -> FieldArray:
        ...

    @overload
    def decode(
        self,
        codeword: ArrayLike,
        output: Literal["message", "codeword"] = "message",
        errors: Literal[True] = True,
    ) -> tuple[FieldArray, int | np.ndarray]:
        ...

    def decode(self, codeword, output="message", errors=False):
        # Call the parent class's decode method, but implement the _decode_codeword method
        return super().decode(codeword=codeword, output=output, errors=errors)

    def _decode_codeword(self, codeword: FieldArray) -> tuple[FieldArray, np.ndarray]:

        # Using syndrome error correction we will find the position of the error and then fix it
        n_errors = np.zeros(codeword.shape[0])
        syndromes = codeword @ self.H.T
        errors_detected = ~np.all(syndromes == 0, axis=1)

        # We will return this variable finally after making corrections in it
        decoded_codeword = codeword.copy()
        # We iterate over every codeword's syndrome
        for i, (error_detected, syndrome) in enumerate(zip(errors_detected, syndromes)):
            if not error_detected:
                continue
            # If an error is detected, then we find the position of the error
            # Since Hamming codes are single error correcting codes
            # Hc = H*(v + e) = Hv + He = He
            # Here c is corrupted codeword to which an error vector e is added
            # Here e is a weight-1 vector
            # So He(syndrome) is a column of the parity check matrix
            # The location of this column is the position where the error occurred
            error_position = 0
            while not np.all(syndrome == self.H[:, error_position]):
                error_position += 1
            decoded_codeword[i, error_position] ^= 1
            n_errors[i] = 1

        return (decoded_codeword, n_errors)

    def _convert_codeword_to_message(self, codeword: FieldArray) -> FieldArray:
        message = None
        if self.is_systematic:
            message = codeword[..., :self.k]
        return message

    def _convert_codeword_to_parity(self, codeword: FieldArray) -> FieldArray:
        parity = None
        if self.is_systematic:
            parity = codeword[..., -(self.n - self.k):]
        return parity

    ##############################################
    # Helper Functions
    ##############################################

    def _generate_systematic_parity_check_matrix(self, r, field):

        q = field.characteristic
        n = int((q ** r - 1)/(q - 1))
        H = field(np.zeros((r,  n), dtype=int))

        for col in range(1, n + 1):
            for row in range(r):
                if col & 1 << row > 0:
                    H[row, col - 1] = 1

        H = H.row_reduce(eye="right")
        return H


if __name__ == "__main__":
    f = GF(2)
    hm = HammingCode(3, f)
    m = f(np.array([[0, 1, 1, 0], [1, 0, 1, 0]]))
    print(m)
    c = hm.encode(m)
    print(c)
    c[0][0] = 1
    print(c)
    dec_c, n_errors = hm.decode(c, output="message", errors=True)
    print(dec_c)
    print(n_errors)
