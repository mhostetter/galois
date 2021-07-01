from ._field import FieldArray, Poly
from ._overrides import set_module

__all__ = ["berlekamp_massey"]


@set_module("galois")
def berlekamp_massey(sequence):
    r"""
    Finds the minimum-degree polynomial :math:`c(x)` that produces the sequence in :math:`\mathrm{GF}(p^m)`.

    This function implements the Berlekamp-Massey algorithm.

    Parameters
    ----------
    sequence : galois.FieldArray
        A sequence of Galois field elements in :math:`\mathrm{GF}(p^m)`.

    Returns
    -------
    galois.Poly
        The minimum-degree polynomial :math:`c(x) \in \mathrm{GF}(p^m)(x)` that produces
        the input sequence.

    Examples
    --------
    TODO: Add an LFSR example once they're added.
    """
    if not isinstance(sequence, FieldArray):
        raise TypeError(f"Argument `sequence` must be a Galois field array, not {type(sequence)}.")
    field = type(sequence)

    coeffs = field._berlekamp_massey(sequence)  # pylint: disable=protected-access

    return Poly(coeffs, field=field)
