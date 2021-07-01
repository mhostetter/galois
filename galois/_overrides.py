def set_module(module):
    """
    A decorator to update the __module__ variable as is done in numpy.

    References
    ----------
    * https://numpy.org/devdocs/release/1.16.0-notes.html#module-attribute-now-points-to-public-modules
    * https://github.com/numpy/numpy/blob/544094aed5fdca536b300d0820fe41f22729ec66/numpy/core/overrides.py#L94-L109
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
