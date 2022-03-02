Binary Extension Fields
=======================

This page compares the performance of :obj:`galois` performing finite field multiplication in :math:`\mathrm{GF}(2^m)` with
native NumPy performing *only* modular multiplication.

Native NumPy cannot easily perform finite field multiplication in :math:`\mathrm{GF}(2^m)` because it involves polynomial multiplication
(convolution) followed by reducing modulo the irreducible polynomial. To make a *similar* comparison, NumPy will perform integer
multiplication followed by integer remainder division.

.. important::

    Native NumPy is not computing the correct result! This is not a fair fight!

These are *not* fair comparisons because NumPy is not computing the correct product. However, they are included here to
provide a performance reference point with native NumPy.

Lookup table performance
------------------------

This section tests :obj:`galois` when using the `"jit-lookup"` compilation mode. For finite fields with order less
than or equal to :math:`2^{20}`, :obj:`galois` uses lookup tables by default for efficient arithmetic.

Below are examples computing 10 million multiplications in the binary extension field :math:`\mathrm{GF}(2^8)`.

.. code-block:: ipython

    In [1]: import numpy as np

    In [2]: import galois

    In [3]: GF = galois.GF(2**8)

    In [4]: GF.ufunc_mode
    Out[4]: 'jit-lookup'

    In [5]: a = GF.Random(10_000_000, dtype=int)

    In [6]: b = GF.Random(10_000_000, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [7]: a * b
    Out[7]: GF([ 48, 233,  52, ..., 153,  30,  54], order=2^8)

    In [8]: %timeit a * b
    47.6 ms ± 1.98 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

NumPy, even when computing the incorrect result, is slower than :obj:`galois`. This is because :obj:`galois` is using lookup
tables instead of explicitly performing the polynomial multiplication and division.

.. code-block:: ipython

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: pp = GF.irreducible_poly.integer

    # This does not produce the correct result!
    In [11]: %timeit (aa * bb) % pp
    64.5 ms ± 1.11 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Explicit calculation performance
--------------------------------

This section tests :obj:`galois` when using the `"jit-calculate"` compilation mode. For finite fields with order greater
than :math:`2^{20}`, :obj:`galois` will use explicit arithmetic calculation by default rather than lookup tables.

Below are examples computing 10 million multiplications in the binary extension field :math:`\mathrm{GF}(2^{32})`.

.. code-block:: ipython

    In [1]: import numpy as np

    In [2]: import galois

    In [3]: GF = galois.GF(2**32)

    In [4]: GF.ufunc_mode
    Out[4]: 'jit-calculate'

    In [5]: a = GF.Random(10_000_000, dtype=int)

    In [6]: b = GF.Random(10_000_000, dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [7]: a * b
    Out[7]:
    GF([3256691449, 3955002553, 3056152043, ..., 1113422699, 1048096312,
        603991153], order=2^32)

    In [8]: %timeit a * b
    407 ms ± 23.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

The :obj:`galois` library when using explicit calculation is only 4x slower than native NumPy, which isn't even computing
the correct product.

.. code-block:: ipython

    In [9]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

    In [10]: pp = GF.irreducible_poly.integer

    # This does not produce the correct result!
    In [11]: %timeit (aa * bb) % pp
    102 ms ± 2.23 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Linear algebra performance
--------------------------

Linear algebra performance in extension fields is definitely slower than native NumPy. This is because, unlike
with prime fields, it is not possible to use the BLAS/LAPACK implementations. Instead, entirely new JIT-compiled
ufuncs are generated, which are not as optimized for parallelism or hardware acceleration as BLAS/LAPACK.

Below are examples computing the matrix multiplication of two :math:`100 \times 100` matrices in the binary extension
field :math:`\mathrm{GF}(2^{32})`.

.. code-block:: ipython

    In [1]: import numpy as np

    In [2]: import galois

    In [3]: GF = galois.GF(2**32)

    In [4]: GF.ufunc_mode
    Out[4]: 'jit-calculate'

    In [5]: A = GF.Random((100,100), dtype=int)

    In [6]: B = GF.Random((100,100), dtype=int)

    # Invoke the ufunc once to JIT compile it, if necessary
    In [7]: A @ B
    Out[7]:
    GF([[ 695562520, 1842206254, 2844540327, ..., 3963691341, 1803659667,
        494558447],
        [4021484675,  698327780, 3411027960, ...,  281446711, 3543368975,
        3104392833],
        [1478167042, 2782017682, 3285476406, ..., 2314358464, 1480096862,
        3019599655],
        ...,
        [2289994312, 4161915469, 3260268436, ...,  273853796, 3467987921,
        2231560534],
        [2725361989, 2508085605, 1004990906, ..., 2607344348,  426951676,
        2256708701],
        [3601032548,  417715873,  364563230, ..., 2336992929, 4248844185,
        547379916]], order=2^32)

    In [8]: %timeit A @ B
    44.8 ms ± 72 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

The :obj:`galois` library is about 60x slower than native NumPy (which isn't computing the correct product).

.. code-block:: ipython

    In [9]: AA, BB = A.view(np.ndarray), B.view(np.ndarray)

    In [10]: pp = GF.irreducible_poly.integer

    # This does not produce the correct result!
    In [11]: %timeit (AA @ BB) % pp
    689 µs ± 2.87 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
