Prime Fields
============

This page compares the performance of :obj:`galois` to native NumPy when performing finite field
multiplication in $\mathrm{GF}(p)$. Native NumPy can perform finite field multiplication in $\mathrm{GF}(p)$
because prime fields are very simple. Multiplication is simply $xy\ \textrm{mod}\ p$.

Lookup table performance
------------------------

This section tests :obj:`galois` when using the `"jit-lookup"` compilation mode. For finite fields with order less
than or equal to $2^{20}$, :obj:`galois` uses lookup tables by default for efficient arithmetic.

Below are examples computing 10 million multiplications in the prime field $\mathrm{GF}(31)$.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(31)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-lookup'

    In [4]: a = GF.Random(10_000_000, seed=1, dtype=int)

    In [5]: b = GF.Random(10_000_000, seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: a * b
    Out[6]: GF([ 9, 27,  7, ..., 14, 21, 15], order=31)

    In [7]: %timeit a * b
    36 ms ± 1.07 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

The equivalent operation using native NumPy ufuncs is ~1.8x slower.

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: %timeit (aa * bb) % GF.order
    65.3 ms ± 1.26 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Explicit calculation performance
--------------------------------

This section tests :obj:`galois` when using the `"jit-calculate"` compilation mode. For finite fields with order greater
than $2^{20}$, :obj:`galois` will use explicit arithmetic calculation by default rather than lookup tables. *Even in these cases*,
:obj:`galois` is faster than NumPy!

Below are examples computing 10 million multiplications in the prime field $\mathrm{GF}(2097169)$.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(2097169)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-calculate'

    In [4]: a = GF.Random(10_000_000, seed=1, dtype=int)

    In [5]: b = GF.Random(10_000_000, seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: a * b
    Out[6]: GF([1879104, 1566761,  967164, ...,  744769,  975853, 1142138], order=2097169)

    In [7]: %timeit a * b
    32.7 ms ± 1.44 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

The equivalent operation using native NumPy ufuncs is ~2.5x slower.

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: %timeit (aa * bb) % GF.order
    78.8 ms ± 1.6 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Runtime floor
-------------

The :obj:`galois` ufunc runtime has a floor, however. This is due to a requirement of the ufuncs to `.view()`
the output array and convert its dtype with `.astype()`. Also the :obj:`galois` ufuncs must perform input
verification that NumPy ufuncs don't.

For example, for small array sizes, NumPy is faster than :obj:`galois`. This is true whether using lookup tables
or explicit calculation.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(2097169)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-calculate'

    In [4]: a = GF.Random(10, seed=1, dtype=int)

    In [5]: b = GF.Random(10, seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: a * b
    Out[6]:
    GF([1879104, 1566761,  967164, 1403108,  100593,  595358,  852783,
        1035698, 1207498,  989189], order=2097169)

    In [7]: %timeit a * b
    7.62 µs ± 390 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

The equivalent operation using native NumPy ufuncs is ~6x faster. However, in absolute terms, the
difference is only ~6 µs.

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: %timeit (aa * bb) % GF.order
    1.29 µs ± 12.6 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

Linear algebra performance
--------------------------

Linear algebra performance in prime fields is comparable to the native NumPy implementations, which use BLAS/LAPACK. This is
because :obj:`galois` uses the native NumPy ufuncs when possible.

If overflow is prevented, dot products in $\mathrm{GF}(p)$ can be computed by first computing the dot product in
$\mathbb{Z}$ and then reducing modulo $p$. In this way, the efficient BLAS/LAPACK implementations are used to
keep finite field linear algebra fast, whenever possible.

Below are examples computing the matrix multiplication of two $100 \times 100$ matrices in the prime field $\mathrm{GF}(2097169)$.

.. code-block:: ipython

    In [1]: import galois

    In [2]: GF = galois.GF(2097169)

    In [3]: GF.ufunc_mode
    Out[3]: 'jit-calculate'

    In [4]: A = GF.Random((100,100), seed=1, dtype=int)

    In [5]: B = GF.Random((100,100), seed=2, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [6]: A @ B
    Out[6]:
    GF([[1147163,   59466, 1841183, ...,  667877, 2084618,  799166],
        [ 306714, 1380503,  810935, ..., 1932687, 1690697,  329837],
        [ 325274,  575543, 1327001, ...,  167724,  422518,  696986],
        ...,
        [ 862992, 1143160,  588384, ...,  668891, 1285421, 1196448],
        [1026856, 1413416, 1844802, ...,   38844, 1643604,   10409],
        [ 401717,  329673,  860449, ..., 1551173, 1766877,  986430]],
    order=2097169)

    In [7]: %timeit A @ B
    708 µs ± 1.48 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

The equivalent operation using native NumPy ufuncs is slightly faster. This is because :obj:`galois` has some internal overhead
before invoking the same NumPy calculation.

.. code-block:: ipython

    In [8]: import numpy as np

    In [9]: AA, BB = A.view(np.ndarray), B.view(np.ndarray)

    In [10]: %timeit (AA @ BB) % GF.order
    682 µs ± 11.1 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
