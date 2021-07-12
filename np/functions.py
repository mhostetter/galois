def copy(a):
    """
    Returns a copy of a given Galois field array.

    See: https://numpy.org/doc/stable/reference/generated/numpy.copy.html

    Warning
    -------
    This function returns an :obj:`numpy.ndarray`, not an instance of the subclass. To return a copy of the subclass, pass
    `subok=True` (for numpy version 1.19 and above) or use `a.copy()`.

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(2**3)
        a = GF.Random(5, low=1); a
        # NOTE: b is an ndarray
        b = np.copy(a); b
        type(b)
        a[0] = 0; a
        # b is unmodified
        b

    .. ipython:: python

        a.copy()
    """
    return


def concatenate(arrays, axis=0):
    """
    Concatenates the input arrays along the given axis.

    See: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(2**3)
        A = GF.Random((2,2)); A
        B = GF.Random((2,2)); B
        np.concatenate((A,B), axis=0)
        np.concatenate((A,B), axis=1)
    """
    return


def insert(array, object, values, axis=None):
    """
    Inserts values along the given axis.

    See: https://numpy.org/doc/stable/reference/generated/numpy.insert.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(2**3)
        x = GF.Random(5); x
        np.insert(x, 1, [0,1,2,3])
    """
    return
