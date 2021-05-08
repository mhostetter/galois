def convolve(a, b):
    """
    Convolves the input arrays.

    See: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        a = GF.Random(10)
        b = GF.Random(10)
        np.convolve(a, b)

        # Equivalent implementation with native numpy
        np.convolve(a.view(np.ndarray).astype(int), b.view(np.ndarray).astype(int)) % 31

    .. ipython:: python

        GF = galois.GF(2**8)
        a = GF.Random(10)
        b = GF.Random(10)
        np.convolve(a, b)
    """
    return
