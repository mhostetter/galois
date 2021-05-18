def dot(a, b):
    """
    Returns the dot product of two Galois field arrays.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.dot.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        a = GF.Random(3); a
        b = GF.Random(3); b
        np.dot(a, b)

    .. ipython:: python

        A = GF.Random((3,3)); A
        B = GF.Random((3,3)); B
        np.dot(A, B)
    """
    return


def vdot(a, b):
    """
    Returns the dot product of two Galois field vectors.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.vdot.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        a = GF.Random(3); a
        b = GF.Random(3); b
        np.vdot(a, b)

    .. ipython:: python

        A = GF.Random((3,3)); A
        B = GF.Random((3,3)); B
        np.vdot(A, B)
    """
    return


def inner(a, b):
    """
    Returns the inner product of two Galois field arrays.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.inner.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        a = GF.Random(3); a
        b = GF.Random(3); b
        np.inner(a, b)

    .. ipython:: python

        A = GF.Random((3,3)); A
        B = GF.Random((3,3)); B
        np.inner(A, B)
    """
    return


def outer(a, b):
    """
    Returns the outer product of two Galois field arrays.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.outer.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        a = GF.Random(3); a
        b = GF.Random(3); b
        np.outer(a, b)
    """
    return


def matmul(a, b):
    """
    Returns the matrix multiplication of two Galois field arrays.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Random((3,3)); A
        B = GF.Random((3,3)); B
        np.matmul(A, B)
        A @ B
    """
    return


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
