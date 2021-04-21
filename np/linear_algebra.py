def trace(x):
    """
    Returns the sum along the diagonal of a Galois field array.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.trace.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Random((5,6)); A
        np.trace(A)
        A[0,0] + A[1,1] + A[2,2] + A[3,3] + A[4,4]

    .. ipython:: python

        np.trace(A, offset=1)
        A[0,1] + A[1,2] + A[2,3] + A[3,4] + A[4,5]
    """
    return
