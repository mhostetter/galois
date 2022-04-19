"""
A module that contains a metaclass mixin that provides GF(p) arithmetic using explicit calculation.
"""
import numba
import numpy as np

from .._domains._array import DTYPES

from ._array import FieldArray

RECIPROCAL = lambda a, *args: 1 / a


class GFp(FieldArray):
    """
    An base class for all GF(p) classes.
    """
    # Need to have a unique cache of "calculate" functions for GF(p)
    _FUNC_CACHE_CALCULATE = {}

    @classmethod
    def _determine_dtypes(cls):
        """
        The only valid dtypes are ones that can hold x*x for x in [0, order).
        """
        max_dtype = DTYPES[-1]
        dtypes = [dtype for dtype in DTYPES if np.iinfo(dtype).max >= cls.order - 1 and np.iinfo(max_dtype).max >= (cls.order - 1)**2]
        if len(dtypes) == 0:
            dtypes = [np.object_]
        return dtypes

    @classmethod
    def _ufunc(cls, name):
        # Some explicit calculation functions are faster than using lookup tables. See https://github.com/mhostetter/galois/pull/92#issuecomment-835548405.
        if name not in cls._ufuncs and cls.ufunc_mode == "jit-lookup" and name in ["add", "negative", "subtract"]:
            cls._ufuncs[name] = cls._ufunc_calculate(name)
        return super()._ufunc(name)

    @classmethod
    def _set_globals(cls, name):
        global RECIPROCAL

        if name in ["divide", "power"]:
            RECIPROCAL = cls._func_calculate("reciprocal", reset=False)

    @classmethod
    def _reset_globals(cls):
        global RECIPROCAL

        RECIPROCAL = cls._func_python("reciprocal")

    ###############################################################################
    # Arithmetic functions using explicit calculation
    ###############################################################################

    @staticmethod
    @numba.extending.register_jitable
    def _add_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        c = a + b
        if c >= CHARACTERISTIC:
            c -= CHARACTERISTIC

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _negative_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a == 0:
            c = 0
        else:
            c = CHARACTERISTIC - a

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _subtract_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if a >= b:
            c = a - b
        else:
            c = CHARACTERISTIC + a - b

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _multiply_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        c = (a * b) % CHARACTERISTIC

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _reciprocal_calculate(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        s*x + t*y = gcd(x, y) = 1
        x = p
        y = a in GF(p)
        t = a**-1 in GF(p)
        """
        if a == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        r2, r1 = CHARACTERISTIC, a
        t2, t1 = 0, 1

        while r1 != 0:
            q = r2 // r1
            r2, r1 = r1, r2 - q*r1
            t2, t1 = t1, t2 - q*t1

        if t2 < 0:
            t2 += CHARACTERISTIC

        return t2

    @staticmethod
    @numba.extending.register_jitable
    def _divide_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        if b == 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if a == 0:
            c = 0
        else:
            b_inv = RECIPROCAL(b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            c = (a * b_inv) % CHARACTERISTIC

        return c

    @staticmethod
    @numba.extending.register_jitable
    def _power_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        Square and Multiply Algorithm

        a^13 = (1) * (a)^13
             = (a) * (a)^12
             = (a) * (a^2)^6
             = (a) * (a^4)^3
             = (a * a^4) * (a^4)^2
             = (a * a^4) * (a^8)
             = result_m * result_s
        """
        if a == 0 and b < 0:
            raise ZeroDivisionError("Cannot compute the multiplicative inverse of 0 in a Galois field.")

        if b == 0:
            return 1
        elif b < 0:
            a = RECIPROCAL(a, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY)
            b = abs(b)

        result_s = a  # The "squaring" part
        result_m = 1  # The "multiplicative" part

        while b > 1:
            if b % 2 == 0:
                result_s = (result_s * result_s) % CHARACTERISTIC
                b //= 2
            else:
                result_m = (result_m * result_s) % CHARACTERISTIC
                b -= 1

        result = (result_m * result_s) % CHARACTERISTIC

        return result

    @staticmethod
    @numba.extending.register_jitable
    def _log_calculate(a, b, CHARACTERISTIC, DEGREE, IRREDUCIBLE_POLY):
        """
        TODO: Replace this with more efficient algorithm
        a = α^m
        b is a primitive element of the field

        c = log(a, b)
        a = b^c
        """
        if a == 0:
            raise ArithmeticError("Cannot compute the discrete logarithm of 0 in a Galois field.")

        ORDER = CHARACTERISTIC**DEGREE

        # Naive algorithm
        result = 1
        for i in range(0, ORDER - 1):
            if result == a:
                break
            result = (result * b) % CHARACTERISTIC

        return i

    ###############################################################################
    # Ufuncs written in NumPy operations (not JIT compiled)
    ###############################################################################

    @staticmethod
    def _sqrt(a):
        """
        Algorithm 3.34 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        Algorithm 3.36 from https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
        """
        field = type(a)
        p = field.characteristic

        if p % 4 == 3:
            roots = a ** ((field.order + 1)//4)

        elif p % 8 == 5:
            d = a ** ((field.order - 1)//4)
            roots = field.Zeros(a.shape)

            idxs = np.where(d == 1)
            roots[idxs] = a[idxs] ** ((field.order + 3)//8)

            idxs = np.where(d == p - 1)
            roots[idxs] = 2*a[idxs] * (4*a[idxs]) ** ((field.order - 5)//8)

        else:
            # Find a quadratic non-residue element `b`
            while True:
                b = field.Random(low=1)
                if not b.is_quadratic_residue():
                    break

            # Write p - 1 = 2^s * t
            n = field.order - 1
            s = 0
            while n % 2 == 0:
                n >>= 1
                s += 1
            t = n
            assert field.order - 1 == 2**s * t

            roots = field.Zeros(a.shape)  # Empty array of roots

            # Compute a root `r` for the non-zero elements
            idxs = np.where(a > 0)  # Indices where a has a reciprocal
            a_inv = np.reciprocal(a[idxs])
            c = b ** t
            r = a[idxs] ** ((t + 1)//2)
            for i in range(1, s):
                d = (r**2 * a_inv) ** (2**(s - i - 1))
                r[np.where(d == p - 1)] *= c
                c = c**2
            roots[idxs] = r  # Assign non-zero roots to the original array

        roots = field._view(np.minimum(roots, -roots))  # Return only the smaller root

        return roots

GFp._reset_globals()
