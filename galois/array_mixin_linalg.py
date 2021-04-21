import numpy as np

from .linalg import row_reduce, lu_decompose, lup_decompose


class LinearAlgebraMixin(np.ndarray):
    """
    A mixin class that provides matrix decomposition routines.
    """

    def row_reduce(self, ncols=None):
        """
        Performs Gaussian elimination on the matrix to achieve reduced row echelon form.

        **Row reduction operations**

        1. Swap the position of any two rows.
        2. Multiply a row by a non-zero scalar.
        3. Add one row to a scalar multiple of another row.

        Parameters
        ----------
        ncols : int, optional
            The number of columns to perform Gaussian elimination over. The default is `None` which represents
            the number of columns of the input array.

        Returns
        -------
        galois.GFArray
            The reduced row echelon form of the input array.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        One column is a linear combination of another.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A[:,2] = A[:,1] * GF(17); A
            A.row_reduce()
            np.linalg.matrix_rank(A)

        One row is a linear combination of another.

        .. ipython:: python

            GF = galois.GF(31)
            A = GF.Random((4,4)); A
            A[3,:] = A[2,:] * GF(8); A
            A.row_reduce()
            np.linalg.matrix_rank(A)
        """
        return row_reduce(self, ncols=ncols)

    def lu_decompose(self):
        """
        Decomposes the input array into the product of lower and upper triangular matrices.

        Returns
        -------
        galois.GFArray
            The lower triangular matrix.
        galois.GFArray
            The upper triangular matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(5)

            # Not every square matrix has an LU decomposition
            A = GF([[2, 4, 4, 1], [3, 3, 1, 4], [4, 3, 4, 2], [4, 4, 3, 1]])
            L, U = A.lu_decompose()
            L
            U

            # A = L U
            np.array_equal(A, L @ U)
        """
        return lu_decompose(self)

    def lup_decompose(self):
        """
        Decomposes the input array into the product of lower and upper triangular matrices using partial pivoting.

        Returns
        -------
        galois.GFArray
            The lower triangular matrix.
        galois.GFArray
            The upper triangular matrix.
        galois.GFArray
            The permutation matrix.

        Examples
        --------
        .. ipython:: python

            GF = galois.GF(5)
            A = GF([[1, 3, 2, 0], [3, 4, 2, 3], [0, 2, 1, 4], [4, 3, 3, 1]])
            L, U, P = A.lup_decompose()
            L
            U
            P

            # P A = L U
            np.array_equal(P @ A, L @ U)
        """
        return lup_decompose(self)
