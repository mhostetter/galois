# Performance

To compare the performance of `galois` and native numpy, we'll use a prime field `GF(p)`. This is because
it is the simplest field. Namely, addition, subtraction, and multiplication are modulo `p`, which can
be simply computed with numpy arrays `(x + y) % p`. For extension fields `GF(p^m)`, the arithmetic is
computed using polynomials over `GF(p)` and can't be so tersely expressed in numpy.

## Lookup performance

For fields with order less than or equal to `2^20`, `galois` uses lookup tables for efficiency.
Here is an example of multiplying two arrays in `GF(31)` using native numpy and `galois`
with `ufunc_mode="jit-lookup"`.

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: GF = galois.GF(31); print(GF.properties)
GF(31):
  characteristic: 31
  degree: 1
  order: 31
  irreducible_poly: Poly(x + 28, GF(31))
  is_primitive_poly: True
  primitive_element: GF(3, order=31)
  dtypes: ['uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'int64']
  ufunc_mode: 'jit-lookup'
  ufunc_target: 'cpu'

In [4]: def construct_arrays(GF, N):
   ...:     a = np.random.randint(1, GF.order, N, dtype=int)
   ...:     b = np.random.randint(1, GF.order, N, dtype=int)
   ...:     ga = a.view(GF)
   ...:     gb = b.view(GF)
   ...:     return a, b, ga, gb
   ...:

In [5]: N = int(10e3)

In [6]: a, b, ga, gb = construct_arrays(GF, N)

In [7]: a
Out[7]: array([29, 20, 29, ..., 29, 22, 24])

In [8]: ga
Out[8]: GF([29, 20, 29, ..., 29, 22, 24], order=31)

In [9]: %timeit (a * b) % GF.order
88.2 µs ± 931 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [10]: %timeit ga * gb
67.9 µs ± 425 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

The `galois` ufunc runtime has a floor, however. This is due to a requirement to `view` the output
array and convert its dtype with `astype()`. For example, for small array sizes numpy is faster than
`galois` because it doesn't need to do these conversions.

```python
In [15]: N = 10

In [16]: a, b, ga, gb = construct_arrays(GF, N)

In [17]: a
Out[17]: array([17, 22,  9, 11,  7, 14, 27, 16, 21, 30])

In [18]: ga
Out[18]: GF([17, 22,  9, 11,  7, 14, 27, 16, 21, 30], order=31)

In [19]: %timeit (a * b) % GF.order
1.32 µs ± 22.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

In [20]: %timeit ga * gb
35.1 µs ± 879 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

This runtime discrepancy can be explained by the time numpy takes to perform the type conversion
and view.

```python
In [21]: %timeit a.astype(np.uint8).view(GF)
31.2 µs ± 5.53 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

However, for large N `galois` is strictly faster than numpy.

```python
In [22]: N = int(10e6)

In [23]: a, b, ga, gb = construct_arrays(GF, N)

In [24]: a
Out[24]: array([29,  9, 16, ..., 15, 24,  9])

In [25]: ga
Out[25]: GF([29,  9, 16, ..., 15, 24,  9], order=31)

In [26]: %timeit (a * b) % GF.order
109 ms ± 1.01 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [27]: %timeit ga * gb
55.2 ms ± 1.18 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Calculation performance

For fields with order greater than `2^20`, `galois` will use explicit arithmetic calculation rather
than lookup tables. Even in these cases, `galois` is faster than numpy!

Here is an example multiplying two arrays in `GF(2097169)` using numpy and `galois` with `ufunc_mode="jit-calculate"`.

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: prime = galois.next_prime(2**21); prime
Out[3]: 2097169

In [4]: GF = galois.GF(prime); print(GF.properties)
GF(2097169):
  characteristic: 2097169
  degree: 1
  order: 2097169
  irreducible_poly: Poly(x + 2097122, GF(2097169))
  is_primitive_poly: True
  primitive_element: GF(47, order=2097169)
  dtypes: ['uint32', 'int32', 'int64']
  ufunc_mode: 'jit-calculate'
  ufunc_target: 'cpu'

In [5]: def construct_arrays(GF, N):
   ...:     a = np.random.randint(1, GF.order, N, dtype=int)
   ...:     b = np.random.randint(1, GF.order, N, dtype=int)
   ...:     ga = a.view(GF)
   ...:     gb = b.view(GF)
   ...:     return a, b, ga, gb
   ...:

In [6]: N = int(10e3)

In [7]: a, b, ga, gb = construct_arrays(GF, N)

In [8]: a
Out[8]: array([331469, 337477, 453485, ..., 186502, 794636, 535201])

In [9]: ga
Out[9]: GF([331469, 337477, 453485, ..., 186502, 794636, 535201], order=2097169)

In [10]: %timeit (a * b) % GF.order
88.3 µs ± 557 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [11]: %timeit ga * gb
57.2 µs ± 749 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```

And again, the runtime comparison with numpy improves with large N because the time of viewing
and type converting the output is small compared to the computation time. `galois` achieves better
performance than numpy because the multiplication and modulo operations are compiled together into
one ufunc rather than two.

```python
In [12]: N = int(10e6)

In [13]: a, b, ga, gb = construct_arrays(GF, N)

In [14]: a
Out[14]: array([2090232, 2071169, 1463892, ..., 1382279, 1067677, 1901668])

In [15]: ga
Out[15]: GF([2090232, 2071169, 1463892, ..., 1382279, 1067677, 1901668], order=2097169)

In [16]: %timeit (a * b) % GF.order
109 ms ± 781 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

In [17]: %timeit ga * gb
50.3 ms ± 619 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```
