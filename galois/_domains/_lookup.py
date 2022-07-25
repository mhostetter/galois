"""
A module containing various ufunc dispatchers with lookup table arithmetic added. These "lookup" implementations use
exponential, logarithm (base primitive element), and Zech logarithm (base primitive element) lookup tables to reduce
the complex finite field arithmetic to a few table lookups and an integer addition/subtraction.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import _ufunc

if TYPE_CHECKING:
    from ._array import Array


class add_ufunc(_ufunc.add_ufunc):
    """
    Addition ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global EXP, LOG, ZECH_LOG, ZECH_E
        EXP = self.field._EXP
        LOG = self.field._LOG
        ZECH_LOG = self.field._ZECH_LOG
        ZECH_E = self.field._ZECH_E

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a + b = α^m + α^n
              = α^m * (1 + α^(n - m))  # If n is larger, factor out α^m
              = α^m * α^ZECH_LOG(n - m)
              = α^(m + ZECH_LOG(n - m))
        """
        if a == 0:
            return b
        elif b == 0:
            return a

        m = LOG[a]
        n = LOG[b]

        if m > n:
            # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If m is
            # larger than n, switch a and b in the addition.
            m, n = n, m

        if n - m == ZECH_E:
            # zech_log(zech_e) = -Inf and α^(-Inf) = 0
            return 0
        else:
            return EXP[m + ZECH_LOG[n - m]]


class negative_ufunc(_ufunc.negative_ufunc):
    """
    Additive inverse ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global EXP, LOG, ZECH_E
        EXP = self.field._EXP
        LOG = self.field._LOG
        ZECH_E = self.field._ZECH_E

    @staticmethod
    def lookup(a: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        -a = -α^m
           = -1 * α^m
           = α^e * α^m
           = α^(e + m)
        """
        if a == 0:
            return 0
        else:
            m = LOG[a]
            return EXP[ZECH_E + m]


class subtract_ufunc(_ufunc.subtract_ufunc):
    """
    Subtraction ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, EXP, LOG, ZECH_LOG, ZECH_E
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG
        ZECH_LOG = self.field._ZECH_LOG
        ZECH_E = self.field._ZECH_E

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a - b = α^m - α^n
              = α^m + (-α^n)
              = α^m + (-1 * α^n)
              = α^m + (α^e * α^n)
              = α^m + α^(e + n)
        """
        # Same as addition if n = log(b) + e
        m = LOG[a]
        n = LOG[b] + ZECH_E

        if b == 0:
            return a
        elif a == 0:
            return EXP[n]

        if m > n:
            # We want to factor out α^m, where m is smaller than n, such that `n - m` is always positive. If m is
            # larger than n, switch a and b in the addition.
            m, n = n, m

        z = n - m
        if z == ZECH_E:
            # zech_log(zech_e) = -Inf and α^(-Inf) = 0
            return 0
        if z >= ORDER - 1:
            # Reduce index of ZECH_LOG by the multiplicative order of the field `ORDER - 1`
            z -= ORDER - 1

        return EXP[m + ZECH_LOG[z]]


class multiply_ufunc(_ufunc.multiply_ufunc):
    """
    Multiplication ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global EXP, LOG
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a * b = α^m * α^n
              = α^(m + n)
        """
        if a == 0 or b == 0:
            return 0
        else:
            m = LOG[a]
            n = LOG[b]
            return EXP[m + n]


class reciprocal_ufunc(_ufunc.reciprocal_ufunc):
    """
    Multiplicative inverse ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, EXP, LOG
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m

        1 / a = 1 / α^m
              = α^(-m)
              = 1 * α^(-m)
              = α^(ORDER - 1) * α^(-m)
              = α^(ORDER - 1 - m)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        m = LOG[a]
        return EXP[(ORDER - 1) - m]


class divide_ufunc(_ufunc.divide_ufunc):
    """
    Division ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, EXP, LOG
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b = α^n

        a / b = α^m / α^n
              = α^(m - n)
              = 1 * α^(m - n)
              = α^(ORDER - 1) * α^(m - n)
              = α^(ORDER - 1 + m - n)
        """
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            return 0
        else:
            m = LOG[a]
            n = LOG[b]
            return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative


class power_ufunc(_ufunc.power_ufunc):
    """
    Exponentiation ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global ORDER, EXP, LOG
        ORDER = self.field.order
        EXP = self.field._EXP
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        α is a primitive element of GF(p^m)
        a = α^m
        b in Z

        a ** b = α^m ** b
               = α^(m * b)
               = α^(m * ((b // (ORDER - 1))*(ORDER - 1) + b % (ORDER - 1)))
               = α^(m * ((b // (ORDER - 1))*(ORDER - 1)) * α^(m * (b % (ORDER - 1)))
               = 1 * α^(m * (b % (ORDER - 1)))
               = α^(m * (b % (ORDER - 1)))
        """
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif a == 0:
            return 0
        else:
            m = LOG[a]
            return EXP[(m * b) % (ORDER - 1)]  # TODO: Do b % (ORDER - 1) first? b could be very large and overflow int64


class log_ufunc(_ufunc.log_ufunc):
    """
    Logarithm ufunc dispatcher with lookup table arithmetic added.
    """
    def set_lookup_globals(self):
        # pylint: disable=global-variable-undefined
        global LOG
        LOG = self.field._LOG

    @staticmethod
    def lookup(a: int, b: int) -> int:  # pragma: no cover
        """
        b is a primitive element of GF(p^m)
        a = b^c

        log(a, b) = log(b^m, b)
                  = c
        """
        # pylint: disable=unused-argument
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        return LOG[a]


class sqrt_ufunc(_ufunc.sqrt_ufunc):
    """
    Square root ufunc dispatcher with lookup table arithmetic added.
    """
    def implementation(self, a: Array) -> Array:
        """
        Computes the square root of an element in a Galois field or Galois ring.
        """
        raise NotImplementedError
