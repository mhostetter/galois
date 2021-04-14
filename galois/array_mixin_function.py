import numpy as np

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
    np.inner: "_inner",
    np.dot: "_dot",
    np.tensordot: "_tensordot",
}


class FunctionMixin(np.ndarray):
    """
    A mixin class to provide functionality to override native numpy functions.
    """

    def __array_function__(self, func, types, args, kwargs):
        if func in UNSUPPORTED_FUNCTIONS:
            raise NotImplementedError(f"Numpy function '{func.__name__}' is not supported on Galois field arrays.")

        if func in OVERRIDDEN_FUNCTIONS:
            output = getattr(self, OVERRIDDEN_FUNCTIONS[func])(func, types, args, kwargs)
        else:
            if func is np.insert:
                args[2] = self._check_array_like_object(args[2])

            output = super().__array_function__(func, types, args, kwargs)  # pylint: disable=no-member

            if func in [np.copy, np.concatenate, np.broadcast_to]:
                output = output.view(self.__class__)

        return output
