def matrix_power(x):
    """
    Raises a square Galois field matrix to an integer power.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Random((3,3)); A
        np.linalg.matrix_power(A, 3)
        A @ A @ A

    .. ipython:: python

        GF = galois.GF(31)
        # Ensure A is full rank and invertible
        while True:
            A = GF.Random((3,3))
            if np.linalg.matrix_rank(A) == 3:
                break
        A
        np.linalg.matrix_power(A, -3)
        A_inv = np.linalg.inv(A)
        A_inv @ A_inv @ A_inv
    """
    return


def det(A):
    """
    Computes the determinant of the matrix.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Random((2,2)); A
        np.linalg.det(A)
        A[0,0]*A[1,1] - A[0,1]*A[1,0]
    """
    return


def matrix_rank(x):
    """
    Returns the rank of a Galois field matrix.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Identity(4); A
        np.linalg.matrix_rank(A)

    One column is a linear combination of another.

    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Random((4,4)); A
        A[:,2] = A[:,1] * GF(17); A
        np.linalg.matrix_rank(A)

    One row is a linear combination of another.

    .. ipython:: python

        GF = galois.GF(31)
        A = GF.Random((4,4)); A
        A[3,:] = A[2,:] * GF(8); A
        np.linalg.matrix_rank(A)
    """
    return


def solve(x):
    """
    Solves the system of linear equations.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        # Ensure A is full rank and invertible
        while True:
            A = GF.Random((4,4))
            if np.linalg.matrix_rank(A) == 4:
                break
        A
        b = GF.Random(4); b
        x = np.linalg.solve(A, b); x
        A @ x

    .. ipython:: python

        GF = galois.GF(31)
        # Ensure A is full rank and invertible
        while True:
            A = GF.Random((4,4))
            if np.linalg.matrix_rank(A) == 4:
                break
        A
        B = GF.Random((4,2)); B
        X = np.linalg.solve(A, B); X
        A @ X
    """
    return


def inv(A):
    """
    Computes the inverse of the matrix.

    References
    ----------
    * https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html

    Examples
    --------
    .. ipython:: python

        GF = galois.GF(31)
        # Ensure A is full rank and invertible
        while True:
            A = GF.Random((3,3))
            if np.linalg.matrix_rank(A) == 3:
                break
        A
        A_inv = np.linalg.inv(A); A_inv
        A_inv @ A
    """
    return

