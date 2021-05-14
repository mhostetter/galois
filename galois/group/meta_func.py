from ..meta_func import Func


class GroupFunc(Func):
    """
    A mixin class that provides the basics for compiling ufuncs.
    """
    # pylint: disable=no-value-for-parameter

    _overridden_functions = {}
    _overridden_linalg_functions = {}

    def _compile_funcs(cls, target):
        return
