# Performance compared with native NumPy

To compare the performance of `galois` and native NumPy, we'll use a prime field `GF(p)`. This is because
it is the simplest field. Namely, addition, subtraction, and multiplication are modulo `p`, which can
be simply computed with NumPy arrays `(x + y) % p`. For extension fields `GF(p^m)`, the arithmetic is
computed using polynomials over `GF(p)` and can't be so tersely expressed in NumPy.

## Lookup performance

For fields with order less than or equal to `2^20`, `galois` uses lookup tables for efficiency.
Here is an example of multiplying two arrays in `GF(31)` using native NumPy and `galois`
with `ufunc_mode="jit-lookup"`.

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: GF = galois.GF(31)

In [4]: GF.ufunc_mode
Out[4]: 'jit-lookup'

In [5]: a = GF.Random(10_000, dtype=int)

In [6]: b = GF.Random(10_000, dtype=int)

In [7]: %timeit a * b
79.7 µs ± 1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [8]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

# Equivalent calculation of a * b using native numpy implementation
In [9]: %timeit (aa * bb) % GF.order
96.6 µs ± 2.4 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

The `galois` ufunc runtime has a floor, however. This is due to a requirement to `view` the output
array and convert its dtype with `astype()`. For example, for small array sizes NumPy is faster than
`galois` because it doesn't need to do these conversions.

```python
In [4]: a = GF.Random(10, dtype=int)

In [5]: b = GF.Random(10, dtype=int)

In [6]: %timeit a * b
45.1 µs ± 1.82 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [7]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

# Equivalent calculation of a * b using native numpy implementation
In [8]: %timeit (aa * bb) % GF.order
1.52 µs ± 34.8 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

However, for large N `galois` is strictly faster than NumPy.

```python
In [10]: a = GF.Random(10_000_000, dtype=int)

In [11]: b = GF.Random(10_000_000, dtype=int)

In [12]: %timeit a * b
59.8 ms ± 1.64 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [13]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

# Equivalent calculation of a * b using native numpy implementation
In [14]: %timeit (aa * bb) % GF.order
129 ms ± 8.01 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Calculation performance

For fields with order greater than `2^20`, `galois` will use explicit arithmetic calculation rather
than lookup tables. Even in these cases, `galois` is faster than NumPy!

Here is an example multiplying two arrays in `GF(2097169)` using NumPy and `galois` with
`ufunc_mode="jit-calculate"`.

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: GF = galois.GF(2097169)

In [4]: GF.ufunc_mode
Out[4]: 'jit-calculate'

In [5]: a = GF.Random(10_000, dtype=int)

In [6]: b = GF.Random(10_000, dtype=int)

In [7]: %timeit a * b
68.2 µs ± 2.09 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [8]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

# Equivalent calculation of a * b using native numpy implementation
In [9]: %timeit (aa * bb) % GF.order
93.4 µs ± 2.12 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

And again, the runtime comparison with NumPy improves with large N because the time of viewing
and type converting the output is small compared to the computation time. `galois` achieves better
performance than NumPy because the multiplication and modulo operations are compiled together into
one ufunc rather than two.

```python
In [10]: a = GF.Random(10_000_000, dtype=int)

In [11]: b = GF.Random(10_000_000, dtype=int)

In [12]: %timeit a * b
51.2 ms ± 1.08 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [13]: aa, bb = a.view(np.ndarray), b.view(np.ndarray)

# Equivalent calculation of a * b using native numpy implementation
In [14]: %timeit (aa * bb) % GF.order
111 ms ± 1.48 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Linear algebra performance

Linear algebra over Galois fields is highly optimized. For prime fields `GF(p)`, the performance is
comparable to the native NumPy implementation (using BLAS/LAPACK).

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: GF = galois.GF(31)

In [4]: A = GF.Random((100,100), dtype=int)

In [5]: B = GF.Random((100,100), dtype=int)

In [6]: %timeit A @ B
720 µs ± 5.36 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [7]: AA, BB = A.view(np.ndarray), B.view(np.ndarray)

# Equivalent calculation of A @ B using the native numpy implementation
In [8]: %timeit (AA @ BB) % GF.order
777 µs ± 4.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```

For extension fields `GF(p^m)`, the performance of `galois` is close to native NumPy linear algebra
(about 10x slower). However, for extension fields, each multiplication operation is equivalently
a convolution (polynomial multiplication) of two `m`-length arrays and polynomial remainder division with the
irreducible polynomial. So it's not an apples-to-apples comparison.

Below is a comparison of `galois` computing the correct matrix multiplication over `GF(2^8)` and NumPy
computing a normal integer matrix multiplication (which is not the correct result!). This
comparison is just for a performance reference.

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: GF = galois.GF(2**8)

In [4]: A = GF.Random((100,100), dtype=int)

In [5]: B = GF.Random((100,100), dtype=int)

In [6]: %timeit A @ B
7.13 ms ± 114 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

In [7]: AA, BB = A.view(np.ndarray), B.view(np.ndarray)

# Native numpy matrix multiplication, which doesn't produce the correct result!!
In [8]: %timeit AA @ BB
651 µs ± 12.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
