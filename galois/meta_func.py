import numpy as np

# List of functions that are not valid on arrays over finite groups, rings, and fields
UNSUPPORTED_FUNCTIONS_UNARY = [
    np.packbits, np.unpackbits,
    np.unwrap,
    np.around, np.round_, np.fix,
    np.gradient, np.trapz,
    np.i0, np.sinc,
    np.angle, np.real, np.imag, np.conj, np.conjugate,
]

UNSUPPORTED_FUNCTIONS_BINARY = [
    np.lib.scimath.logn,
    np.cross,
]

FUNCTIONS_REQUIRING_VIEW = [
    np.copy, np.concatenate,
    np.broadcast_to,
    np.trace,
]


class Func(type):
    """
    A base class for :obj:`GroupFunc`, :obj:`RingFunc`, and :obj:`FieldFunc`.
    """
    # pylint: disable=no-value-for-parameter

    _unsupported_functions = UNSUPPORTED_FUNCTIONS_UNARY + UNSUPPORTED_FUNCTIONS_BINARY
    _functions_requiring_view = FUNCTIONS_REQUIRING_VIEW
    _overridden_functions = {}
    _overridden_linalg_functions = {}

    def _compile_funcs(cls, target):
        raise NotImplementedError
