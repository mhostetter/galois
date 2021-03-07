# Galois: A performant numpy extension for Galois fields

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Documentation Status](https://readthedocs.org/projects/galois/badge/?version=stable)](https://galois.readthedocs.io/en/stable/?badge=stable)
![Lint](https://github.com/mhostetter/galois/workflows/Lint/badge.svg?branch=master)
![Test](https://github.com/mhostetter/galois/workflows/Test/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/mhostetter/galois/branch/master/graph/badge.svg)](https://codecov.io/gh/mhostetter/galois)

A Python 3 package for Galois field arithmetic.

- [Motivation](#motivation)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [Galois field array construction](#galois-field-array-construction)
  - [Galois field array arithmetic](#galois-field-array-arithmetic)
  - [Galois field polynomial construction](#galois-field-polynomial-construction)
  - [Galois field polynomial arithmetic](#galois-field-polynomial-arithmetic)
- [Performance](#performance)
  - [GF(31) addition speed test](#gf31-addition-speed-test)

## Motivation

The project goals are for `galois` to be:

- _**General:**_ Support all Galois field types: `GF(2)`, `GF(2^m)`, `GF(p)`, `GF(p^m)`
- _**Accurate:**_ Guarantee arithmetic accuracy -- tests against industry-standard mathematics software
- _**Compatible:**_ Seamlessley integrate with `numpy` arrays -- arithmetic operators (`x + y`), broadcasting, view casting, type casting, ufuncs, methods
- _**Performant:**_ Run as fast as `numpy` or C -- avoids the speed sinkhole of Python `for` loops
- _**Reconfigurable:**_ Dynamically optimize performance based on data size and processor (single-core CPU, multi-core CPU, or GPU)

## Installation

The latest version of `galois` can be installed from [PyPI](https://pypi.org/project/galois/) via `pip`.

```bash
pip3 install galois
```

For development, the lastest code on `master` can be checked out and installed locally in "editable" mode.

```bash
git clone https://github.com/mhostetter/galois.git
pip3 install -e galois
```

## Basic Usage

### Galois field array construction

Construct Galois field array classes using the `GF_factory()` class factory function.

```python
>>> import numpy as np
>>> import galois

>>> GF = galois.GF_factory(31, 1)
>>> print(GF)
<Galois Field: GF(31^1), prim_poly = x + 28 (None decimal)>

>>> print(GF.alpha)
3

>>> print(GF.prim_poly)
Poly(x + 28 , GF31)
```

Create arrays from existing `numpy` arrays.

```python
# Represents an existing numpy array
>>> np_x = np.random.randint(0, GF.order, 10, dtype=int); np_x
array([ 6, 28,  2, 23, 17,  6,  3,  0, 10,  4])

# Explicit Galois field construction
>>> GF(np_x)
GF31([ 6, 28,  2, 23, 17,  6,  3,  0, 10,  4])

# Numpy view casting to a Galois field, supported for integer dtypes
>>> np_x.view(GF)
GF31([ 6, 28,  2, 23, 17,  6,  3,  0, 10,  4])
```

Or, create Galois field arrays using alternate constructors.

```python
>>> x = GF.Random(10); x
GF31([20, 29, 27, 20, 27,  4, 29, 25, 11, 28])

# Construct a random array without zeros to prevent ZeroDivisonError later on
>>> y = GF.Random(10, low=1); y
GF31([28, 22, 22,  6,  7,  3,  3, 13, 29, 30])
```

### Galois field array arithmetic

Galois field arrays support traditional `numpy` array operations

```python
>>> x + y
GF31([17, 20, 18, 26,  3,  7,  1,  7,  9, 27])

>>> -x
GF31([11,  2,  4, 11,  4, 27,  2,  6, 20,  3])

# Multiply a Galois field array with any integer
>>> x * -3  # NOTE: -3 is outside the field
GF31([ 2,  6, 12,  2, 12, 19,  6, 18, 29,  9])

>>> 1 / y
GF31([10, 24, 24, 26,  9, 21, 21, 12, 15, 30])

# Exponentiate a Galois field array with any integer
>>> y ** -2  # NOTE: -2 is outside the field
GF31([ 7, 18, 18, 25, 19,  7,  7, 20,  8,  1])

# Log base alpha (the field's primitive element)
>>> np.log(y)
array([16, 17, 17, 25, 28,  1,  1, 11,  9, 15])
```

Galois field arrays support `numpy` array broadcasting.

```python
>>> a = GF.Random((2,5)); a
GF31([[ 0, 24,  3,  0,  1],
      [14,  5, 19, 22, 30]])

>>> b = GF.Random(5); b
GF31([23,  7, 28, 23, 19])

>>> a + b
GF31([[23,  0,  0, 23, 20],
      [ 6, 12, 16, 14, 18]])
```

Galois field arrays also support `numpy` ufunc methods.

```python
# Valid ufunc methods include "reduce", "accumulate", "reduceat", "outer", "at"
>>> np.add.reduce(a, axis=0)
GF31([14, 29, 22, 22,  0])

>>> np.multiply.outer(x, y)
GF31([[ 2,  6,  6, 27, 16, 29, 29, 12, 22, 11],
      [ 6, 18, 18, 19, 17, 25, 25,  5,  4,  2],
      [12,  5,  5,  7,  3, 19, 19, 10,  8,  4],
      [ 2,  6,  6, 27, 16, 29, 29, 12, 22, 11],
      [12,  5,  5,  7,  3, 19, 19, 10,  8,  4],
      [19, 26, 26, 24, 28, 12, 12, 21, 23, 27],
      [ 6, 18, 18, 19, 17, 25, 25,  5,  4,  2],
      [18, 23, 23, 26, 20, 13, 13, 15, 12,  6],
      [29, 25, 25,  4, 15,  2,  2, 19,  9, 20],
      [ 9, 27, 27, 13, 10, 22, 22, 23,  6,  3]])
```

### Galois field polynomial construction

Construct Galois field polynomials.

```python
# Construct a polynomial by specifying all the coefficients in descending-degree order
>>> p = galois.Poly([1, 22, 0, 17, 25], field=GF); p
Poly(x^4 + 22x^3 + 17x + 25 , GF31)

# Construct a polynomial by specifying only the non-zero coefficients
>>> q = galois.Poly.NonZero([4, 14],  [2, 0], field=GF); q
Poly(4x^2 + 14 , GF31)
```

### Galois field polynomial arithmetic

Galois field polynomial arithmetic is similar to `numpy` array operations.

```python
>>> p + q
Poly(x^4 + 22x^3 + 4x^2 + 17x + 8 , GF31)

>>> p // q, p % q
(Poly(8x^2 + 21x + 3 , GF31), Poly(2x + 14 , GF31))

>>> p ** 2
Poly(x^8 + 13x^7 + 19x^6 + 3x^5 + 23x^4 + 15x^3 + 10x^2 + 13x + 5 , GF31)
```

Galois field polynomials can also be evaluated at constants or arrays.

```python
>>> p(1)
GF31(3)

>>> p(a)
GF31([[ 1, 18, 17, 16,  5],
      [ 8, 21, 17, 23, 18]])
```

## Performance

### GF(31) addition speed test

```python
>>> import numpy as np
>>> import galois

>>> GFp = galois.GF_factory(31, 1)
>>> print(GFp)
<Galois Field: GF(31^1), prim_poly = x + 28 (None decimal)>

>>> def construct_arrays(GF, N):
...     order = GF.order
...
...     a = np.random.randint(0, order, N, dtype=int)
...     b = np.random.randint(0, order, N, dtype=int)
...
...     ga = GF(a)
...     gb = GF(b)
...
...     return a, b, ga, gb, order

>>> def pure_python_add(a, b, modulus):
...     c = np.zeros(a.size, dtype=a.dtype)
...     for i in range(a.size):
...         c[i] = (a[i] + b[i]) % modulus
...     return c

>>> N = int(10e3)
... a, b, ga, gb, order = construct_arrays(GFp, N)
...
... print(f"Pure python addition in GF({order})")
... %timeit pure_python_add(a, b, order)
...
... print(f"\nNative numpy addition in GF({order})")
... %timeit (a + b) % order
...
... print(f"\n`galois` implementation of addition in GF({order})")
... %timeit ga + gb
Pure python addition in GF(31)
5.84 ms ± 218 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

Native numpy addition in GF(31)
112 µs ± 14.7 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

`galois` implementation of addition in GF(31)
73.1 µs ± 746 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
