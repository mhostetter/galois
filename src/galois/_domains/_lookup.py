"""
A module containing various ufunc dispatchers with lookup table arithmetic added. These "lookup" implementations use
exponential, logarithm (base primitive element), and Zech logarithm (base primitive element) lookup tables to reduce
the complex finite field arithmetic to a few table lookups and an integer addition/subtraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _ufunc

if TYPE_CHECKING:
    from ._array import Array


class add_ufunc(_ufunc.add_ufunc):
    """
    Addition ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
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
        if b == 0:
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

        return EXP[m + ZECH_LOG[n - m]]


class negative_ufunc(_ufunc.negative_ufunc):
    """
    Additive inverse ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
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

        m = LOG[a]
        return EXP[ZECH_E + m]


class subtract_ufunc(_ufunc.subtract_ufunc):
    """
    Subtraction ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
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
        if a == 0:
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

        m = LOG[a]
        n = LOG[b]
        return EXP[m + n]


class reciprocal_ufunc(_ufunc.reciprocal_ufunc):
    """
    Multiplicative inverse ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
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

        m = LOG[a]
        n = LOG[b]
        return EXP[(ORDER - 1) + m - n]  # We add `ORDER - 1` to guarantee the index is non-negative


class power_ufunc(_ufunc.power_ufunc):
    """
    Exponentiation ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
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
        if a == 0:
            return 0

        m = LOG[a]
        return EXP[(m * b) % (ORDER - 1)]  # TODO: Do b % (ORDER - 1) first? b could be very large and overflow int64


class log_ufunc(_ufunc.log_ufunc):
    """
    Logarithm ufunc dispatcher with lookup table arithmetic added.
    """

    def set_lookup_globals(self):
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


###############################################################################
# Array mixin class
###############################################################################


class UFuncMixin(_ufunc.UFuncMixin):
    """
    The UFuncMixin class with lookup table construction added.
    """

    @classmethod
    def _build_lookup_tables(cls):
        """
        Construct EXP, LOG, and ZECH_LOG lookup tables to be used in the "lookup" arithmetic functions
        """
        # TODO: Make this faster by using JIT-compiled ufuncs and vector arithmetic, when possible
        primitive_element = int(cls._primitive_element)
        add = cls._add.python_calculate
        multiply = cls._multiply.python_calculate

        cls._EXP = np.zeros(2 * cls.order, dtype=np.int64)
        cls._LOG = np.zeros(cls.order, dtype=np.int64)
        cls._ZECH_LOG = np.zeros(cls.order, dtype=np.int64)
        if cls.characteristic == 2:
            cls._ZECH_E = 0
        else:
            cls._ZECH_E = (cls.order - 1) // 2

        element = 1
        cls._EXP[0] = element
        cls._LOG[0] = 0  # Technically -Inf
        for i in range(1, cls.order):
            # Increment by multiplying by the primitive element, which is a multiplicative generator of the field
            element = multiply(element, primitive_element)
            cls._EXP[i] = element

            # Assign to the log lookup table but skip indices greater than or equal to `order - 1`
            # because `EXP[0] == EXP[order - 1]`
            if i < cls.order - 1:
                cls._LOG[cls._EXP[i]] = i

        # Compute Zech log lookup table
        for i in range(0, cls.order):
            one_plus_element = add(1, cls._EXP[i])
            cls._ZECH_LOG[i] = cls._LOG[one_plus_element]

        if not cls._EXP[cls.order - 1] == 1:
            raise RuntimeError(
                f"The anti-log lookup table for {cls.name} is not cyclic with size {cls.order - 1}, which means "
                f"the primitive element {cls._primitive_element} does not have multiplicative order {cls.order - 1} "
                f"and therefore isn't a multiplicative generator for {cls.name}."
            )
        if not len(set(cls._EXP[0 : cls.order - 1])) == cls.order - 1:
            raise RuntimeError(
                f"The anti-log lookup table for {cls.name} is not unique, "
                f"which means the primitive element {cls._primitive_element} has order less than {cls.order - 1} "
                f"and is not a multiplicative generator of {cls.name}."
            )
        if not len(set(cls._LOG[1 : cls.order])) == cls.order - 1:
            raise RuntimeError(f"The log lookup table for {cls.name} is not unique.")

        # Double the EXP table to prevent computing a `% (order - 1)` on every multiplication lookup
        cls._EXP[cls.order : 2 * cls.order] = cls._EXP[1 : 1 + cls.order]
