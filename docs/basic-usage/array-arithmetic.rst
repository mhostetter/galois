Array Arithmetic
================

After creating a :obj:`~galois.FieldArray` subclass and one or two arrays, nearly any arithmetic operation can be
performed using normal NumPy ufuncs or Python operators.

In the sections below, the finite field :math:`\mathrm{GF}(3^5)` and arrays :math:`x` and :math:`y` are used.

.. ipython:: python

    GF = galois.GF(3**5)
    x = GF([184, 25, 157, 31]); x
    y = GF([179, 9, 139, 27]); y

Ufuncs
------

`NumPy ufuncs <https://numpy.org/devdocs/reference/ufuncs.html>`_ are universal functions that operate on scalars. Unary ufuncs operate on
a single scalar and binary ufuncs operate on two scalars. NumPy extends the scalar operation of ufuncs to operate on arrays in various ways.
This extensibility enables `NumPy broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

Expand any section for more details.

.. details:: Addition: `x + y == np.add(x, y)`

    .. ipython:: python

        x + y
        np.add(x, y)

.. details:: Additive inverse: `-x == np.negative(x)`

    .. ipython:: python

        -x
        np.negative(x)

    Any array added to its additive inverse results in zero.

    .. ipython:: python

        x + np.negative(x)

.. details:: Subtraction: `x - y == np.subtract(x, y)`

    .. ipython:: python

        x - y
        np.subtract(x, y)

.. details:: Multiplication: `x * y == np.multiply(x, y)`

    .. ipython:: python

        x * y
        np.multiply(x, y)

.. details:: Scalar multiplication: `x * z == np.multiply(x, z)`

    Scalar multiplication is essentially *repeated addition*. It is the "multiplication" of finite field elements
    and integers. The integer value indicates how many additions of the field element to sum.

    .. ipython:: python

        x * 4
        np.multiply(x, 4)
        x + x + x + x

    In finite fields :math:`\mathrm{GF}(p^m)`, the characteristic :math:`p` is the smallest value when multiplied by
    any non-zero field element that results in 0.

    .. ipython:: python

        p = GF.characteristic; p
        x * p

.. details:: Multiplicative inverse: `y ** -1 == np.reciprocal(y)`

    .. ipython:: python

        y ** -1
        GF(1) / y
        np.reciprocal(y)

    Any array multiplied by its multiplicative inverse results in one.

    .. ipython:: python

        y * np.reciprocal(y)

.. details:: Division: `x / y == x // y == np.divide(x, y)`

    .. ipython:: python

        x / y
        x // y
        np.divide(x, y)

.. details:: Remainder: `x % y == np.remainder(x, y)`

    .. ipython:: python

        x % y
        np.remainder(x, y)

.. details:: Divmod: `divmod(x, y) == np.divmod(x, y)`

    .. ipython:: python

        x / y, x % y
        divmod(x, y)
        np.divmod(x, y)

    .. ipython:: python

        q, r = divmod(x, y)
        q*y + r == x

.. details:: Exponentiation: `x ** z == np.power(x, z)`

    .. ipython:: python

        x ** 3
        np.power(x, 3)
        x * x * x

.. details:: Square root: `np.sqrt(x)`

    .. ipython:: python

        x.is_square()
        z = np.sqrt(x); z
        z ** 2 == x

    See also :func:`~galois.FieldArray.is_square`, :func:`~galois.FieldArray.squares`, and :func:`~galois.FieldArray.non_squares`.

.. details:: Logarithm: `np.log(x)` or `x.log()`

    Compute the logarithm base :math:`\alpha`, the primitive element of the field.

    .. ipython:: python

        z = np.log(y); z
        alpha = GF.primitive_element; alpha
        alpha ** z == y

    Compute the logarithm base :math:`\beta`, a different primitive element of the field. See :func:`FieldArray.log` for more
    details.

    .. ipython:: python

        beta = GF.primitive_elements[-1]; beta
        z = y.log(beta); z
        beta ** z == y

Ufunc methods
-------------

:obj:`~galois.FieldArray` instances support `NumPy ufunc methods <https://numpy.org/devdocs/reference/ufuncs.html#methods>`_. Ufunc methods allow
a user to apply a NumPy ufunc in a unique way across the target array. All arithmetic ufuncs are supported.

Expand any section for more details.

.. details:: `reduce()`

    The :obj:`~numpy.ufunc.reduce` methods reduce the input array's dimension by one, applying the ufunc across one axis.

    .. ipython:: python

        np.add.reduce(x)
        x[0] + x[1] + x[2] + x[3]

    .. ipython:: python

        np.multiply.reduce(x)
        x[0] * x[1] * x[2] * x[3]

.. details:: `accumulate()`

    The :obj:`~numpy.ufunc.accumulate` methods accumulate the result of the ufunc across a specified axis.

    .. ipython:: python

        np.add.accumulate(x)
        GF([x[0], x[0] + x[1], x[0] + x[1] + x[2], x[0] + x[1] + x[2] + x[3]])

    .. ipython:: python

        np.multiply.accumulate(x)
        GF([x[0], x[0] * x[1], x[0] * x[1] * x[2], x[0] * x[1] * x[2] * x[3]])

.. details:: `reduceat()`

    The :obj:`~numpy.ufunc.reduceat` methods reduces the input array's dimension by one, applying the ufunc across one axis
    in-between certain indices.

    .. ipython:: python

        np.add.reduceat(x, [0, 3])
        GF([x[0] + x[1] + x[2], x[3]])

    .. ipython:: python

        np.multiply.reduceat(x, [0, 3])
        GF([x[0] * x[1] * x[2], x[3]])

.. details:: `outer()`

    The :obj:`~numpy.ufunc.outer` methods applies the ufunc to each pair of inputs.

    .. ipython:: python

        np.add.outer(x, y)

    .. ipython:: python

        np.multiply.outer(x, y)

.. details:: `at()`

    The :obj:`~numpy.ufunc.at` methods performs the ufunc in-place at the specified indices.

    .. ipython:: python

        z = x.copy()
        # Negate indices 0 and 1 in-place
        np.negative.at(x, [0, 1]); x
        z[0:1] *= -1; z

.. _advanced-arithmetic:

Advanced arithmetic
-------------------

.. details:: Convolution: `np.convolve(x, y)`

    .. ipython:: python

        np.convolve(x, y)

.. details:: FFT: `np.fft.fft(x)`

    The Discrete Fourier Transform (DFT) of size :math:`n` over the finite field :math:`\mathrm{GF}(p^m)` exists when there
    exists a primitive :math:`n`-th root of unity. This occurs when :math:`n\ |\ p^m - 1`.

    .. ipython:: python

        GF = galois.GF(7**5)
        n = 6
        # n divides p^m - 1
        (GF.order - 1) % n
        x = GF.Random(n); x
        X = np.fft.fft(x); X
        np.fft.ifft(X)

    See also :func:`~galois.ntt` and :obj:`~galois.FieldArray.primitive_root_of_unity`.

.. details:: Inverse FFT: `np.fft.ifft(X)`

    The inverse Discrete Fourier Transform (DFT) of size :math:`n` over the finite field :math:`\mathrm{GF}(p^m)` exists when there
    exists a primitive :math:`n`-th root of unity. This occurs when :math:`n\ |\ p^m - 1`.

    .. ipython:: python

        GF = galois.GF(7**5)
        n = 6
        # n divides p^m - 1
        (GF.order - 1) % n
        x = GF.Random(n); x
        X = np.fft.fft(x); X
        np.fft.ifft(X)

    See also :func:`~galois.ntt` and :obj:`~galois.FieldArray.primitive_root_of_unity`.

Linear algebra
--------------

Linear algebra on :obj:`~galois.FieldArray` arrays/matrices is supported through both native NumPy linear algebra functions
in :obj:`numpy.linalg` and additional linear algebra routines not included in NumPy.

Expand any section for more details.

.. details:: Dot product: `np.dot(a, b)`

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.dot(a, b)

.. details:: Vector dot product: `np.vdot(a, b)`

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.vdot(a, b)

.. details:: Inner product: `np.inner(a, b)`

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.inner(a, b)

.. details:: Outer product: `np.outer(a, b)`

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.outer(a, b)

.. details:: Matrix multiplication: `A @ B == np.matmul(A, B)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[17, 25, 18, 8], [7, 9, 21, 15], [6, 16, 6, 30]]); A
        B = GF([[8, 18], [22, 0], [7, 8], [20, 24]]); B
        A @ B
        np.matmul(A, B)

.. details:: Matrix exponentiation: `np.linalg.matrix_power(A, z)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[14, 1, 5], [3, 23, 6], [24, 27, 4]]); A
        np.linalg.matrix_power(A, 3)
        A @ A @ A

.. details:: Matrix determinant: `np.linalg.det(A)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.linalg.det(A)

.. details:: Matrix rank: `np.linalg.matrix_rank(A, z)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.linalg.matrix_rank(A)
        A.row_reduce()

.. details:: Matrix trace: `np.trace(A)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.trace(A)
        A[0,0] + A[1,1] + A[2,2] + A[3,3]

.. details:: Solve a system of equations: `np.linalg.solve(A, b)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[14, 21, 14, 28], [24, 22, 23, 23], [16, 30, 26, 18], [4, 23, 18, 3]]); A
        b = GF([15, 11, 6, 29]); b
        x = np.linalg.solve(A, b)
        np.array_equal(A @ x, b)

.. details:: Matrix inverse: `np.linalg.inv(A)`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[14, 21, 14, 28], [24, 22, 23, 23], [16, 30, 26, 18], [4, 23, 18, 3]]); A
        A_inv = np.linalg.inv(A); A_inv
        A @ A_inv

Additional linear algebra
-------------------------

Below are additional linear algebra routines provided for :obj:`~galois.FieldArray` arrays/matrices that are
not included in NumPy.

.. details:: Row space: `A.row_space()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.row_space()

    See :func:`~galois.FieldArray.row_space` for more details.

.. details:: Column space: `A.column_space()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.column_space()

    See :func:`~galois.FieldArray.column_space` for more details.

.. details:: Left null space: `A.left_null_space()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.left_null_space()

    See :func:`~galois.FieldArray.left_null_space` for more details.

.. details:: Null space: `A.null_space()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.null_space()

    See :func:`~galois.FieldArray.null_space` for more details.

.. details:: Gaussian elimination: `A.row_reduce()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        A.row_reduce()

    See :func:`~galois.FieldArray.row_reduce` for more details.

.. details:: LU decomposition: `A.lu_decompose()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[4, 1, 24], [7, 6, 1], [11, 20, 2]]); A
        L, U = A.lu_decompose()
        L
        U
        np.array_equal(L @ U, A)

    See :func:`~galois.FieldArray.lu_decompose` for more details.

.. details:: PLU decomposition: `A.plu_decompose()`

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[15, 4, 11], [7, 6, 1], [11, 20, 2]]); A
        P, L, U = A.plu_decompose()
        P
        L
        U
        np.array_equal(P @ L @ U, A)

    See :func:`~galois.FieldArray.plu_decompose` for more details.
