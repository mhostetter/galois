Array Arithmetic
================

After creating a :obj:`~galois.FieldArray` subclass and one or two arrays, nearly any arithmetic operation can be
performed using normal NumPy ufuncs or Python operators.

In the sections below, the finite field $\mathrm{GF}(3^5)$ and arrays $x$ and $y$ are used.

.. ipython:: python

    GF = galois.GF(3**5)
    x = GF([184, 25, 157, 31]); x
    y = GF([179, 9, 139, 27]); y

Standard arithmetic
-------------------

`NumPy ufuncs <https://numpy.org/devdocs/reference/ufuncs.html>`_ are universal functions that operate on scalars.
Unary ufuncs operate on a single scalar and binary ufuncs operate on two scalars. NumPy extends the scalar operation
of ufuncs to operate on arrays in various ways. This extensibility enables
`NumPy broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

Expand any section for more details.

.. example:: Addition: `x + y == np.add(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        x + y
        np.add(x, y)

.. example:: Additive inverse: `-x == np.negative(x)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        -x
        np.negative(x)

    Any array added to its additive inverse results in zero.

    .. ipython-with-reprs:: int,poly,power

        x
        x + np.negative(x)

.. example:: Subtraction: `x - y == np.subtract(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        x - y
        np.subtract(x, y)

.. example:: Multiplication: `x * y == np.multiply(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        x * y
        np.multiply(x, y)

.. example:: Scalar multiplication: `x * 4 == np.multiply(x, 4)`
    :collapsible:

    Scalar multiplication is essentially *repeated addition*. It is the "multiplication" of finite field elements
    and integers. The integer value indicates how many additions of the field element to sum.

    .. ipython-with-reprs:: int,poly,power

        x
        x * 4
        np.multiply(x, 4)
        x + x + x + x

    In finite fields $\mathrm{GF}(p^m)$, the characteristic $p$ is the smallest value when multiplied by
    any non-zero field element that results in 0.

    .. ipython-with-reprs:: int,poly,power

        p = GF.characteristic; p
        x * p

.. example:: Multiplicative inverse: `y ** -1 == np.reciprocal(y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        y
        y ** -1
        GF(1) / y
        np.reciprocal(y)

    Any array multiplied by its multiplicative inverse results in one.

    .. ipython-with-reprs:: int,poly,power

        y * np.reciprocal(y)

.. example:: Division: `x / y == x // y == np.divide(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        x / y
        x // y
        np.divide(x, y)

.. example:: Remainder: `x % y == np.remainder(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        x % y
        np.remainder(x, y)

.. example:: Divmod: `divmod(x, y) == np.divmod(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        x // y, x % y
        divmod(x, y)
        np.divmod(x, y)

    .. ipython-with-reprs:: int,poly,power

        q, r = divmod(x, y)
        q*y + r == x

.. example:: Exponentiation: `x ** 3 == np.power(x, 3)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        x ** 3
        np.power(x, 3)
        x * x * x

.. example:: Square root: `np.sqrt(x)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        x.is_square()
        z = np.sqrt(x); z
        z ** 2 == x

    See also :func:`~galois.FieldArray.is_square`, :func:`~galois.FieldArray.squares`, and
    :func:`~galois.FieldArray.non_squares`.

.. example:: Logarithm: `np.log(x)` or `x.log()`
    :collapsible:

    Compute the logarithm base $\alpha$, the primitive element of the field.

    .. ipython-with-reprs:: int,poly,power

        y
        z = np.log(y); z
        alpha = GF.primitive_element; alpha
        alpha ** z == y

    Compute the logarithm base $\beta$, a different primitive element of the field. See :func:`FieldArray.log`
    for more details.

    .. ipython-with-reprs:: int,poly,power

        y
        beta = GF.primitive_elements[-1]; beta
        z = y.log(beta); z
        beta ** z == y

Ufunc methods
-------------

:obj:`~galois.FieldArray` instances support `NumPy ufunc methods
<https://numpy.org/devdocs/reference/ufuncs.html#methods>`_. Ufunc methods allow a user to apply a NumPy ufunc in a
unique way across the target array. All arithmetic ufuncs are supported.

Expand any section for more details.

.. example:: `reduce()`
    :collapsible:

    The :obj:`~numpy.ufunc.reduce` methods reduce the input array's dimension by one, applying the ufunc across one
    axis.

    .. ipython-with-reprs:: int,poly,power

        x
        np.add.reduce(x)
        x[0] + x[1] + x[2] + x[3]

    .. ipython-with-reprs:: int,poly,power

        np.multiply.reduce(x)
        x[0] * x[1] * x[2] * x[3]

.. example:: `accumulate()`
    :collapsible:

    The :obj:`~numpy.ufunc.accumulate` methods accumulate the result of the ufunc across a specified axis.

    .. ipython-with-reprs:: int,poly,power

        x
        np.add.accumulate(x)
        GF([x[0], x[0] + x[1], x[0] + x[1] + x[2], x[0] + x[1] + x[2] + x[3]])

    .. ipython-with-reprs:: int,poly,power

        np.multiply.accumulate(x)
        GF([x[0], x[0] * x[1], x[0] * x[1] * x[2], x[0] * x[1] * x[2] * x[3]])

.. example:: `reduceat()`
    :collapsible:

    The :obj:`~numpy.ufunc.reduceat` methods reduces the input array's dimension by one, applying the ufunc across one
    axis in-between certain indices.

    .. ipython-with-reprs:: int,poly,power

        x
        np.add.reduceat(x, [0, 3])
        GF([x[0] + x[1] + x[2], x[3]])

    .. ipython-with-reprs:: int,poly,power

        np.multiply.reduceat(x, [0, 3])
        GF([x[0] * x[1] * x[2], x[3]])

.. example:: `outer()`
    :collapsible:

    The :obj:`~numpy.ufunc.outer` methods applies the ufunc to each pair of inputs.

    .. ipython-with-reprs:: int,poly,power

        x
        y
        np.add.outer(x, y)

    .. ipython-with-reprs:: int,poly,power

        np.multiply.outer(x, y)

.. example:: `at()`
    :collapsible:

    The :obj:`~numpy.ufunc.at` methods performs the ufunc in-place at the specified indices.

    .. ipython-with-reprs:: int,poly,power

        x
        z = x.copy()
        # Negate indices 0 and 1 in-place
        np.negative.at(x, [0, 1]); x
        z[0:1] *= -1; z

.. _advanced-arithmetic:

Advanced arithmetic
-------------------

.. example:: Convolution: `np.convolve(x, y)`
    :collapsible:

    .. ipython-with-reprs:: int,poly,power

        x
        y
        np.convolve(x, y)

.. example:: FFT: `np.fft.fft(x)`
    :collapsible:

    The Discrete Fourier Transform (DFT) of size $n$ over the finite field $\mathrm{GF}(p^m)$ exists when
    there exists a primitive $n$-th root of unity. This occurs when $n \mid p^m - 1$.

    .. ipython-with-reprs:: int,poly,power

        GF = galois.GF(7**5)
        n = 6
        # n divides p^m - 1
        (GF.order - 1) % n
        x = GF.Random(n, seed=1); x
        X = np.fft.fft(x); X
        np.fft.ifft(X)

    See also :func:`~galois.ntt` and :obj:`~galois.FieldArray.primitive_root_of_unity`.

.. example:: Inverse FFT: `np.fft.ifft(X)`
    :collapsible:

    The inverse Discrete Fourier Transform (DFT) of size $n$ over the finite field $\mathrm{GF}(p^m)$
    exists when there exists a primitive $n$-th root of unity. This occurs when $n \mid p^m - 1$.

    .. ipython-with-reprs:: int,poly,power

        GF = galois.GF(7**5)
        n = 6
        # n divides p^m - 1
        (GF.order - 1) % n
        x = GF.Random(n, seed=1); x
        X = np.fft.fft(x); X
        np.fft.ifft(X)

    See also :func:`~galois.ntt` and :obj:`~galois.FieldArray.primitive_root_of_unity`.

Linear algebra
--------------

Linear algebra on :obj:`~galois.FieldArray` arrays/matrices is supported through both native NumPy linear algebra
functions in :obj:`numpy.linalg` and additional `linear algebra methods
<https://mhostetter.github.io/galois/latest/api/galois.FieldArray/#linear-algebra>`_ not included in NumPy.

Expand any section for more details.

.. example:: Dot product: `np.dot(a, b)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.dot(a, b)

.. example:: Vector dot product: `np.vdot(a, b)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.vdot(a, b)

.. example:: Inner product: `np.inner(a, b)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.inner(a, b)

.. example:: Outer product: `np.outer(a, b)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        a = GF([29, 0, 2, 21]); a
        b = GF([23, 5, 15, 12]); b
        np.outer(a, b)

.. example:: Matrix multiplication: `A @ B == np.matmul(A, B)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[17, 25, 18, 8], [7, 9, 21, 15], [6, 16, 6, 30]]); A
        B = GF([[8, 18], [22, 0], [7, 8], [20, 24]]); B
        A @ B
        np.matmul(A, B)

.. example:: Matrix exponentiation: `np.linalg.matrix_power(A, 3)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[14, 1, 5], [3, 23, 6], [24, 27, 4]]); A
        np.linalg.matrix_power(A, 3)
        A @ A @ A

.. example:: Matrix determinant: `np.linalg.det(A)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.linalg.det(A)

.. example:: Matrix rank: `np.linalg.matrix_rank(A)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.linalg.matrix_rank(A)
        A.row_reduce()

.. example:: Matrix trace: `np.trace(A)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[23, 11, 3, 3], [13, 6, 16, 4], [12, 10, 5, 3], [17, 23, 15, 28]]); A
        np.trace(A)
        A[0,0] + A[1,1] + A[2,2] + A[3,3]

.. example:: Solve a system of equations: `np.linalg.solve(A, b)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[14, 21, 14, 28], [24, 22, 23, 23], [16, 30, 26, 18], [4, 23, 18, 3]]); A
        b = GF([15, 11, 6, 29]); b
        x = np.linalg.solve(A, b)
        np.array_equal(A @ x, b)

.. example:: Matrix inverse: `np.linalg.inv(A)`
    :collapsible:

    .. ipython:: python

        GF = galois.GF(31)
        A = GF([[14, 21, 14, 28], [24, 22, 23, 23], [16, 30, 26, 18], [4, 23, 18, 3]]); A
        A_inv = np.linalg.inv(A); A_inv
        A @ A_inv
