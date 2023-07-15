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
from .._helper import export, verify_isinstance, verify_issubclass
from ..typing import ArrayLike
from ._linear import _LinearCode, parity_check_to_generator_matrix


@export
class Hamming(_LinearCode):

    def __init__(self, n: int | None = None, k: int | None = None, field: FieldArray = GF(2), r: int | None = None, extended: bool = False):
        # Construct the Generator and Parity Check Matrix and then initialise the parent class

        if n is not None and k is not None:
            r = n - k
            q = field.characteristic
            expected_n = int((q ** r - 1)/(q - 1))
            if n != expected_n:
                r = n - k - 1
                expected_n_ext = int((q ** r - 1)/(q - 1)) + 1
                if n != expected_n_ext:
                    raise Exception(
                        f"Given n = {n}, k = {k} but expected n = {expected_n} or {expected_n_ext}")
                else:
                    extended = True
        elif r is not None:
            q = field.characteristic
            n = int((q ** r - 1)/(q - 1))
            k = n - r
            if extended:
                n += 1

        self._r = r
        self._q = field.characteristic
        self._extended = extended

        k = int((self._q ** self._r - 1)/(self._q - 1)) - r
        n = k + r
        if extended:
            n += 1
        d = 4 if extended else 3

        H = self._generate_systematic_parity_check_matrix(r, field, extended)
        G = parity_check_to_generator_matrix(H)

        super().__init__(n, k, d, G, H, systematic=True)

    @property
    def r(self) -> int:
        return self._r

    @property
    def q(self) -> int:
        return self._q

    @property
    def extended(self) -> bool:
        return self._extended

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
        n_errors = np.zeros(codeword.shape[0], dtype=int)
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
            # So He(syndrome) is a column of the parity check matrix scaled by a constant
            # The location of this column is the position where the error occurred
            error_position = 0
            constant_scale = 0
            while error_position < self.n:
                parity_column = self.H[:, error_position]
                for a in self._field.elements:
                    if np.all(syndrome == a * parity_column):
                        constant_scale = a
                        break
                if constant_scale != 0:
                    break
                error_position += 1
            if error_position < self.n:
                decoded_codeword[i, error_position] -= constant_scale
                n_errors[i] = 1
            else:
                n_errors[i] = -1

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

    @staticmethod
    def _generate_systematic_parity_check_matrix(r, field, extended):

        q = field.characteristic
        n = int((q ** r - 1)/(q - 1))
        H = field(np.zeros((r,  n), dtype=int))

        gf = GF(q**r)

        # Add the parity columns first
        col_idx = 0
        for num in gf.elements:
            if num == 0:
                continue
            vec = gf(num).vector()
            vec_weight = np.count_nonzero(vec)
            # If the vector is a weight-1 vector, then it will be added at the end
            # in the identity matrix
            vec_exists = True if vec_weight == 1 else False
            for factor in range(1, q):
                scaled_vec = vec * factor
                if gf.Vector(scaled_vec) < num:
                    vec_exists = True
            if not vec_exists:
                H[:, col_idx] = vec
                col_idx += 1

        # Add the identity matrix
        for pow in range(r-1, -1, -1):
            H[:, col_idx] = gf(q**pow).vector()
            col_idx += 1

        if extended:
            # Concatenate a zeros column to the right of the matrix
            H = np.concatenate((H, np.zeros((r, 1), dtype=int)), axis=1)
            # Concatenate a ones row to the bottom of the matrix
            H = np.concatenate((H, np.ones((1, n + 1), dtype=int)), axis=0)
            H = H.row_reduce(eye="right")

        return H
