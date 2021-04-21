import numpy as np

from .linalg import matrix_rank, solve, inv, det

UNSUPPORTED_ONE_ARG_FUNCTIONS = [
    np.packbits, np.unpackbits,
    np.unwrap,
    np.around, np.round_, np.fix,
    np.gradient, np.trapz,
    np.i0, np.sinc,
]

UNSUPPORTED_TWO_ARG_FUNCTIONS = [
    np.lib.scimath.logn,
    np.cross,
]

UNSUPPORTED_FUNCTIONS = UNSUPPORTED_ONE_ARG_FUNCTIONS + UNSUPPORTED_TWO_ARG_FUNCTIONS

OVERRIDDEN_FUNCTIONS = {
    # np.inner: "inner",
    # np.dot: "dot",
    # np.tensordot: "tensordot",
    np.linalg.matrix_rank: matrix_rank,
    np.linalg.inv: inv,
    np.linalg.det: det,
    np.linalg.solve: solve
}

FUNCTIONS_REQUIRING_VIEW = [
    np.copy, np.concatenate,
    np.broadcast_to,
    np.trace
]

class FunctionMixin(np.ndarray):
    """
    A mixin class to provide functionality to override native numpy functions.
    """

    def __array_function__(self, func, types, args, kwargs):
        if func in OVERRIDDEN_FUNCTIONS:
            output = OVERRIDDEN_FUNCTIONS[func](*args, **kwargs)

        elif func in UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(f"The numpy function '{func.__name__}' is not supported on Galois field arrays. If you believe this function should be supported, please submit a GitHub issue at https://github.com/mhostetter/galois/issues.\n\nIf you'd like to perform this operation on the data (but not necessarily a Galois field array), you should first call `array.view(np.ndarray)` and then call the function.")

        else:
            if func is np.insert:
                args = list(args)
                args[2] = self._check_array_like_object(args[2])
                args = tuple(args)

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in FUNCTIONS_REQUIRING_VIEW:
                if np.isscalar(output):
                    output = type(self)(output, dtype=self.dtype)
                else:
                    output = output.view(type(self))

        return output
