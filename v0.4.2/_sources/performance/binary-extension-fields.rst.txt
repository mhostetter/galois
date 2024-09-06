Binary Extension Fields
=======================

This page compares the performance of :obj:`galois` performing finite field multiplication in $\mathrm{GF}(2^m)$ with
native NumPy performing *only* modular multiplication.

Native NumPy cannot easily perform finite field multiplication in $\mathrm{GF}(2^m)$ because it involves polynomial multiplication
(convolution) followed by reducing modulo the irreducible polynomial. To make a *similar* comparison, NumPy will perform integer
multiplication followed by integer remainder division.

.. important::

    Native NumPy is not computing the correct result! This is not a fair fight!

These are *not* fair comparisons because NumPy is not computing the correct product. However, they are included here to
provide a performance reference point with native NumPy.

Lookup table performance
------------------------

This section tests :obj:`galois` when using the `"jit-lookup"` compilation mode. For finite fields with order less
than or equal to $2^{20}$, :obj:`galois` uses lookup tables by default for efficient arithmetic.

Below are examples computing 10 million multiplications in the binary extension field $\mathrm{GF}(2^8)$.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(2**8)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-lookup'

    In [4]: a = GF.Random(10_000_000, seed=1, dtype=int)

    In [5]: b = GF.Random(10_000_000, seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: a * b
    Out[6]: GF([181,  92, 148, ..., 255, 220, 153], order=2^8)

    In [7]: %timeit a * b
    33.9 ms ± 1.64 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

NumPy, even when computing the incorrect result, is ~1.9x slower than :obj:`galois`. This is because :obj:`galois` is using lookup
tables instead of explicitly performing the polynomial multiplication and division.

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: pp = int(GF.irreducible_poly)

    # This does not produce the correct result!
    In [11]: %timeit (aa * bb) % pp
    64 ms ± 747 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

Explicit calculation performance
--------------------------------

This section tests :obj:`galois` when using the `"jit-calculate"` compilation mode. For finite fields with order greater
than $2^{20}$, :obj:`galois` will use explicit arithmetic calculation by default rather than lookup tables.

Below are examples computing 10 million multiplications in the binary extension field $\mathrm{GF}(2^{32})$.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(2**32)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-calculate'

    In [4]: a = GF.Random(10_000_000, seed=1, dtype=int)

    In [5]: b = GF.Random(10_000_000, seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: a * b
    Out[6]:
    GF([1174047800, 3249326965, 3196014003, ..., 3195457330,  100242821,
        338589759], order=2^32)

    In [7]: %timeit a * b
    386 ms ± 14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

The :obj:`galois` library when using explicit calculation is only ~3.9x slower than native NumPy, which isn't even computing
the correct product.

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: pp = int(GF.irreducible_poly)

    # This does not produce the correct result!
    In [11]: %timeit (aa * bb) % pp
    100 ms ± 718 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

Linear algebra performance
--------------------------

Linear algebra performance in extension fields is definitely slower than native NumPy. This is because, unlike
with prime fields, it is not possible to use the BLAS/LAPACK implementations. Instead, entirely new JIT-compiled
ufuncs are generated, which are not as optimized for parallelism or hardware acceleration as BLAS/LAPACK.

Below are examples computing the matrix multiplication of two $100 \times 100$ matrices in the binary extension
field $\mathrm{GF}(2^{32})$.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(2**32)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-calculate'

    In [4]: A = GF.Random((100,100), seed=1, dtype=int)

    In [5]: B = GF.Random((100,100), seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: A @ B
    Out[6]:
    GF([[4203877556, 3977035749, 2623937858, ..., 3721257849, 4250999056,
        4026271867],
        [3120760606, 1017695431, 1111117124, ..., 1638387264, 2988805996,
        1734614583],
        [2508826906, 2800993411, 1720697782, ..., 3858180318, 2521070820,
        3906771227],
        ...,
        [ 624580545,  984724090, 3969931498, ..., 1692192269,  473079794,
        1029376699],
        [1232183301,  209395954, 2659712274, ..., 2967695343, 2747874320,
        1249453570],
        [3938433735,  828783569, 3286222384, ..., 3669775257,   33626526,
        4278384359]], order=2^32)

    In [7]: %timeit A @ B
    3.88 ms ± 102 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

The :obj:`galois` library is about ~5.5x slower than native NumPy (which isn't computing the correct product).

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: AA, BB = A.view(np.ndarray), B.view(np.ndarray)

    In [10]: pp = int(GF.irreducible_poly)

    # This does not produce the correct result!
    In [11]: %timeit (AA @ BB) % pp
    703 µs ± 1.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
