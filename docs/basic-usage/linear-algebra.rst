Linear Algebra
==============

Linear algebra on *Galois field arrays* is supported through both native NumPy linear algebra function calls
and additional linear algebra routines not included in NumPy.

In the sections below, the prime field :math:`\mathrm{GF}(31)` is used.

.. ipython:: python

    GF = galois.GF(31)
    print(GF)

NumPy routines
--------------

Most NumPy linear algebra routines are supported on vectors and matrices over finite fields. These routines are accessed
using the standard :obj:`numpy.linalg` functions.

Expand any section for more details.

.. details:: Dot product: `np.dot(a, b)`

    .. ipython:: python

        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.dot(a, b)

.. details:: Vector dot product: `np.vdot(a, b)`

    .. ipython:: python

        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.vdot(a, b)

.. details:: Inner product: `np.inner(a, b)`

    .. ipython:: python

        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.inner(a, b)

.. details:: Outer product: `np.outer(a, b)`

    .. ipython:: python

        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.outer(a, b)

.. details:: Matrix multiplication: `A @ B == np.matmul(A, B)`

    .. ipython:: python

        A = GF([[17, 25, 18, 8], [7, 9, 21, 15], [6, 16, 6, 30]]); A
        B = GF([[8, 18], [22, 0], [7, 8], [20, 24]]); B
        A @ B
        np.matmul(A, B)

.. details:: Matrix exponentiation: `np.linalg.matrix_power(A, z)`

    .. ipython:: python

        A = GF([[14, 1, 5], [3, 23, 6], [24, 27, 4]]); A
        np.linalg.matrix_power(A, 3)
        A @ A @ A

.. details:: Matrix determinant: `np.linalg.det(A)`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.linalg.det(A)

.. details:: Matrix rank: `np.linalg.matrix_rank(A, z)`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.linalg.matrix_rank(A)
        A.row_reduce()

.. details:: Matrix trace: `np.trace(A)`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.trace(A)
        A[0,0] + A[1,1] + A[2,2] + A[3,3]

.. details:: Solve a system of equations: `np.linalg.solve(A, b)`

    .. ipython:: python

        A = GF([[14, 21, 14, 28], [24, 22, 23, 23], [16, 30, 26, 18], [4, 23, 18, 3]]); A
        b = GF([15, 11, 6, 29]); b
        x = np.linalg.solve(A, b)
        A @ x == b

.. details:: Matrix inverse: `np.linalg.inv(A)`

    .. ipython:: python

        A = GF([[14, 21, 14, 28], [24, 22, 23, 23], [16, 30, 26, 18], [4, 23, 18, 3]]); A
        A_inv = np.linalg.inv(A); A_inv
        A @ A_inv

Additional routines
-------------------

.. details:: Row space: `A.row_space()`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.row_space()

    See :func:`galois.FieldArray.row_space` for more details.

.. details:: Column space: `A.column_space()`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.column_space()

    See :func:`galois.FieldArray.column_space` for more details.

.. details:: Left null space: `A.left_null_space()`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.left_null_space()

    See :func:`galois.FieldArray.left_null_space` for more details.

.. details:: Null space: `A.null_space()`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.null_space()

    See :func:`galois.FieldArray.null_space` for more details.

.. details:: Gaussian elimination: `A.row_reduce()`

    .. ipython:: python

        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.row_reduce()

    See :func:`galois.FieldArray.row_reduce` for more details.

.. details:: LU decomposition: `A.lu_decompose()`

    .. ipython:: python

        A = GF([[4, 1, 24], [7, 6, 1], [11, 20, 2]]); A
        L, U = A.lu_decompose()
        L
        U
        np.array_equal(L @ U, A)

    See :func:`galois.FieldArray.lu_decompose` for more details.

.. details:: PLU decomposition: `A.plu_decompose()`

    .. ipython:: python

        A = GF([[15, 4, 11], [7, 6, 1], [11, 20, 2]]); A
        P, L, U = A.plu_decompose()
        P
        L
        U
        np.array_equal(P @ L @ U, A)

    See :func:`galois.FieldArray.plu_decompose` for more details.
