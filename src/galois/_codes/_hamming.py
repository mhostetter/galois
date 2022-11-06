from __future__ import annotations

from typing import Type, Union, overload

import numpy as np
import itertools

from .._helper import export
from ._linear import _LinearCode, parity_check_to_generator_matrix
from .._fields import FieldArray, GF2
from ..typing import ArrayLike

@export
class Hamming(_LinearCode):

    def __init__(
        self, 
        n: int | None = None,
        k: int | None = None,
        field: Type[FieldArray] = GF2,
        r: int | None = None,
        extended: bool = False
    ):
        r"""    
        """
        # Logic to infer what is n and k and field and r based on (n, k, field, r and is_extended)
        self._q  = field.order
        self._extended = extended

        if(n is not None and k is not None):
            if (n, k) == self._get_hamming_code_params(self._q, n - k, False):
                self._extended = False
                self._r     = n - k
            elif (n, k) == self._get_hamming_code_params(self._q, n - k - 1, True):
                self._extended = True
                self._r     = n - k - 1
            else:
                print("Not a valid Hamming code parameters")
                return
        elif(r is not None):
            n, k = self._get_hamming_code_params(self._q, r, self._extended)
            print(n, k)
            self._r = r
        else:
            print("Specify the parameters correctly")
            exit()

        # Generate the parity check matrix - function call and set the parity check matrix to _H
        H = self._generate_parity_check_matrix(field)
        # Generate the generator matrix - function call and set the generator matrix to _G
        G = parity_check_to_generator_matrix(H)
        # _LinearCode.__init__(field, G, H)
        super().__init__(n = n, k = k, d = 4 if extended else 3, G=G, H=H,systematic=True)

    def __str__(self) -> str:
        """
        Returns the string representation of the code
        """
        string = ""
        string += f"Hamming Code [{self.n}, {self.k}, {self.d}] code over {self.field.order} "
        string += f"is extended = {self._extended}\n"
        return string

    def encode(self, message: ArrayLike,  parity_only: bool = False) -> FieldArray:
        return super().encode(message, parity_only)

    def detect(self, codeword: ArrayLike) -> Union[bool, np.ndarray]:
        return super().detect(codeword)

    # @overload
    # def decode():
    #     pass

    # Properties
    @property
    def n(self):
        return super().n

    @property
    def k(self):
        return super().k

    @property
    def d(self):
        return super().d

    @property
    def field(self):
        return super().field

    @property
    def G(self):
        return super().G

    @property
    def H(self):
        return super().H

    @property
    def q(self):
        return self._q

    @property
    def r(self):
        return self._r

    @property
    def is_extended(self):
        return self._is_extended
    
    # Private Functions
    @staticmethod
    def _get_hamming_code_params(q, r, is_extended):
        """
        Returns the tuple (n, k) of the Hamming code
        """
        n = int((np.power(q, r) - 1) / (q - 1))
        k = n - r
        if is_extended:
            n += 1
        return n, k

    def _generate_parity_check_matrix(self, field: Type[FieldArray]) -> FieldArray:
        """
        Constructs the parity check matrix of the code
        """
        
        H = np.array([list(i) for i in itertools.product([0, 1], repeat = self._r) if not all(item == 0 for item in i)], dtype = int)
        H = H.T
        n, k = self._get_hamming_code_params(self._q, self._r, self._extended)
        if self._extended:
            H = np.hstack((H, np.zeros((self._r, 1), dtype=int)))
            H = np.vstack((H, np.ones((1, n), dtype=int)))
        H = field(H)
        H = np.linalg.inv(H[:, k:]) @ H
        return H

    def _convert_codeword_to_message(self, codeword: FieldArray) -> FieldArray:
        """
        Returns the message part of the codeword
        """
        if self.is_systematic:
            return codeword[:self.k]

    def _convert_codeword_to_parity(self, codeword: FieldArray) -> FieldArray:
        """
        Returns the parity part of the codeword
        """
        if self.is_systematic:
            return codeword[self.k:]

    def _decode_codeword(self, codeword: FieldArray) -> FieldArray:
        """
        Decodes the given codeword 
        """
        pass

# Testing the init function

from .._fields import Field

print(Hamming(7, 4))
print(Hamming(8, 4))
# print(Hamming(10, 4))
# print(Hamming(r=4))
print(Hamming(r=3))
print(Hamming(r=3, extended=True))

# Testing the encode function

hm = Hamming(7, 4)
message = GF2([1, 0, 1, 1])
codeword = hm.encode(message, parity_only=False)
codeword[1] = 1
codeword[2] = GF2(1) - codeword[2]
codeword[3] = GF2(1) - codeword[3]
print(hm.detect(codeword))
hm = Hamming(8, 4)
codeword = hm.encode(message, parity_only=False)
codeword[1] = 1
codeword[2] = GF2(1) - codeword[2]
codeword[3] = GF2(1) - codeword[3]
print(hm.detect(codeword))



