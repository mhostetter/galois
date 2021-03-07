from .gf import GFBase, DTYPES


class GF2(GFBase):
    """
    asdf

    Examples
    --------

    GF2 class properties

    .. ipython:: python

        print(galois.GF2)
        galois.GF2.characteristic
        galois.GF2.power
        galois.GF2.order
        galois.GF2.prim_poly

    Construct arrays in GF2

    .. ipython:: python

        a = galois.GF2([1,0,1,1]); a
        b = galois.GF2([1,1,1,1]); b

    Arithmetic with GF2 arrays

    .. ipython:: python

        # Element-wise addition
        a + b
        # Element-wise subtraction
        a - b
        # Element-wise multiplication
        a * b
        # Element-wise division
        a / b
    """

    characteristic = 2
    power = 1
    order = 2
    prim_poly = None  # Will set this in __init__.py
    alpha = 1
    dtypes = DTYPES

    @staticmethod
    def _add(a, b):
        # Calculate a + b
        return a ^ b

    @staticmethod
    def _subtract(a, b):
        # Calculate a - b
        return a ^ b

    @staticmethod
    def _multiply(a, b):
        # Calculate a * b
        return a & b

    @staticmethod
    def _divide(a, b):
        # Calculate a / b
        return a & b

    @staticmethod
    def _negative(a):
        # Calculate -a
        return a

    @staticmethod
    def _power(a, b):
        # Calculate a**b
        if b == 0:
            return 1
        elif a == 0:
            return 0
        else:
            return a

    @staticmethod
    def _log(a):  # pylint: disable=unused-argument
        # Calculate np.log(a)
        return 0

    @staticmethod
    def _poly_eval(coeffs, values, results):
        # Calculate p(a)
        def _add(a, b):
            return a ^ b

        def _multiply(a, b):
            return a & b

        for i in range(values.size):
            results[i] = coeffs[0]
            for j in range(1, coeffs.size):
                results[i] = _add(coeffs[j], _multiply(results[i], values[i]))
