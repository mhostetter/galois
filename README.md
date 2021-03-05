# Galois: A performant numpy extension for Galois fields

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Documentation Status](https://readthedocs.org/projects/galois/badge/?version=stable)](https://galois.readthedocs.io/en/stable/?badge=stable)
![Lint](https://github.com/mhostetter/galois/workflows/Lint/badge.svg?branch=master)
![Test](https://github.com/mhostetter/galois/workflows/Test/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/mhostetter/galois/branch/master/graph/badge.svg)](https://codecov.io/gh/mhostetter/galois)

A Python 3 package for Galois field arithmetic.

## Installation

### Install with `pip`

```bash
pip3 install galois
```

## Performance

### GF(31) addition speed test

```python
In [1]: import numpy as np
   ...: import galois

In [2]: GFp = galois.GF_factory(31, 1)

In [3]: def construct_arrays(GF, N):
   ...:     order = GF.order
   ...:
   ...:     a = np.random.randint(0, order, N, dtype=int)
   ...:     b = np.random.randint(0, order, N, dtype=int)
   ...:
   ...:     ga = GF(a)
   ...:     gb = GF(b)
   ...:
   ...:     return a, b, ga, gb, order
   ...:

In [4]: def pure_python_add(a, b, modulus):
   ...:     c = np.zeros(a.size, dtype=a.dtype)
   ...:     for i in range(a.size):
   ...:         c[i] = (a[i] + b[i]) % modulus
   ...:     return c
   ...:

In [5]: N = int(10e3)
   ...: a, b, ga, gb, order = construct_arrays(GFp, N)
   ...:
   ...: print(f"Pure python addition in GF({order})")
   ...: %timeit pure_python_add(a, b, order)
   ...:
   ...: print(f"\nNative numpy addition in GF({order})")
   ...: %timeit (a + b) % order
   ...:
   ...: print(f"\n`galois` implementation of addition in GF({order})")
   ...: %timeit ga + gb
Pure python addition in GF(31)
5.84 ms ± 218 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

Native numpy addition in GF(31)
112 µs ± 14.7 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

`galois` implementation of addition in GF(31)
73.1 µs ± 746 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
