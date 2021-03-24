import numpy as np


@np.vectorize
def modular_exp(base, exponent, modulus):
    """
    Compute the modular exponentiation :math:`base^exponent \\textrm{mod}\\ modulus`.

    Parameters
    ----------
    base : array_like
        The base of exponential, an int or an array (follows numpy broadcasting rules).
    exponent : array_like
        The exponent, an int or an array (follows numpy broadcasting rules).
    modulus : array_like
        The modulus of the computation, an int or an array (follows numpy broadcasting rules).

    Returns
    -------
    array_like
        The results of :math:`base^exponent \\textrm{mod}\\ modulus`.
    """
    assert exponent >= 0
    if exponent == 0:
        return 1
    if modulus == 1:
        return 0

    result_s = base  # The "squaring" part
    result_m = 1  # The "multiplicative" part

    while exponent > 1:
        if exponent % 2 == 0:
            result_s = (result_s * result_s) % modulus
            exponent //= 2
        else:
            result_m = (result_m * result_s) % modulus
            exponent -= 1

    result = (result_m * result_s) % modulus

    return result
