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

>>> GF31 = galois.GF_factory(31, 1)

>>> print(GF31)
<Galois Field: GF(31^1), prim_poly = x + 28 (59 decimal)>

>>> GF31.alpha
GF(3, order=31)

>>> GF31.prim_poly
Poly(x + 28, GF31)
```

Create any Galois field array class type: `GF(2^m)`, `GF(p)`, or `GF(p^m)`. Even arbitrarily-large fields!

```python
# Field used in AES
>>> GF256 = galois.GF_factory(2, 8); print(GF256)
<Galois Field: GF(2^8), prim_poly = x^8 + x^4 + x^3 + x^2 + 1 (285 decimal)>

# Large prime field
>>> prime = 36893488147419103183

>>> galois.is_prime(prime)
True

>>> GFp = galois.GF_factory(prime, 1); print(GFp)
<Galois Field: GF(36893488147419103183^1), prim_poly = x + 36893488147419103180 (73786976294838206363 decimal)>

# Large characteristic-2 field
>>> GF2_100 = galois.GF_factory(2, 100); print(GF2_100)
<Galois Field: GF(2^100), prim_poly = x^100 + x^57 + x^56 + x^55 + x^52 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^41 + x^37 + x^36 + x^35 + x^34 + x^31 + x^30 + x^27 + x^25 + x^24 + x^22 + x^20 + x^19 + x^16 + x^15 + x^11 + x^9 + x^8 + x^6 + x^5 + x^3 + 1 (1267650600228486663289456659305 decimal)>
```

Create arrays from existing `numpy` arrays.

```python
# Represents an existing numpy array
>>> array = np.random.randint(0, GF256.order, 10, dtype=int); array
array([ 71, 240, 210,  27, 124, 254,  13, 170, 221, 166])

# Explicit Galois field construction
>>> GF256(array)
GF([ 71, 240, 210,  27, 124, 254,  13, 170, 221, 166], order=2^8)

# Numpy view casting to a Galois field
>>> array.view(GF256)
GF([ 71, 240, 210,  27, 124, 254,  13, 170, 221, 166], order=2^8)
```

Or, create Galois field arrays using alternate constructors.

```python
>>> x = GF256.Random(10); x
GF([118,  49, 122, 166, 136, 118,  53,  19, 233, 119], order=2^8)

# Construct a random array without zeros to prevent ZeroDivisonError
>>> y = GF256.Random(10, low=1); y
GF([239,  63,  81, 225, 150,  12,  56,  24,  98, 245], order=2^8)
```

### Galois field array arithmetic

Galois field arrays support traditional `numpy` array operations

```python
>>> x + y
GF([153,  14,  43,  71,  30, 122,  13,  11, 139, 130], order=2^8)

>>> -x
GF([118,  49, 122, 166, 136, 118,  53,  19, 233, 119], order=2^8)

>>> x * y
GF([231,  91,  98, 212,  24,  82,  44, 181, 123,  90], order=2^8)

>>> x / y
GF([209,  38, 117, 199, 171, 113, 182,  88, 161, 194], order=2^8)

# Multiple addition of a Galois field array with any integer
>>> x * -3  # NOTE: -3 is outside the field
GF([118,  49, 122, 166, 136, 118,  53,  19, 233, 119], order=2^8)

# Exponentiate a Galois field array with any integer
>>> y ** -2  # NOTE: -2 is outside the field
GF([253, 171, 113,  60,  85,  56, 208,  14, 154,  70], order=2^8)

# Log base alpha (the field's primitive element)
>>> np.log(y)
array([215, 166, 208,  89, 180,  27, 201,  28, 182, 231])
```

Even field arithmetic on extremely large fields!

```python
>>> m = GFp.Random(3)

>>> n = GFp.Random(3)

>>> m + n
GF([13628406805756046913, 32523641596143053984, 6779710980842128868],
   order=36893488147419103183)

>>> m ** 123456
GF([25998377155280209892, 5155207233739881631, 35111337553358290170],
   order=36893488147419103183)

>>> r = GF2_100.Random(3); r
GF([1208438098225504529643982094397, 1043242720211594207819544907628,
    908206781068122096007332904544], order=2^100)

# With characteristic 2, this will always be zero
>>> r + r
GF([0, 0, 0], order=2^100)

# This is equivalent
>>> r * 2
GF([0, 0, 0], order=2^100)

# But this will result in `r`
>>> r * 3
GF([1208438098225504529643982094397, 1043242720211594207819544907628,
    908206781068122096007332904544], order=2^100)
```

Galois field arrays support `numpy` array broadcasting.

```python
>>> a = GF31.Random((2,5)); a
GF([[28, 30, 17, 21, 22],
    [23, 29, 23, 27, 17]], order=31)

>>> b = GF31.Random(5); b
Out[33]: GF([ 7, 10, 12, 20, 24], order=31)

>>> a + b
GF([[ 4,  9, 29, 10, 15],
    [30,  8,  4, 16, 10]], order=31)
```

Galois field arrays also support `numpy` ufunc methods.

```python
# Valid ufunc methods include "reduce", "accumulate", "reduceat", "outer", "at"
>>> np.multiply.reduce(a, axis=0)
GF([24,  2, 19,  9,  2], order=31)

>>> np.multiply.outer(x, y)
GF([[231, 157, 137,  89, 159,  82, 194, 164,  70, 175],
    [ 21,  91, 218,  38,  52,  81, 204, 162, 208, 213],
    [ 87, 132,  98, 161,  57,   2, 255,   4, 228, 167],
    [126, 199, 230, 212, 184, 251, 146, 235, 218, 196],
    [161,  19,  93, 130,  24,  46, 140,  92,  96, 240],
    [231, 157, 137,  89, 159,  82, 194, 164,  70, 175],
    [142, 167, 131, 133,  86,  97,  44, 194,  69,  38],
    [122, 150, 138, 136,  50, 212, 239, 181, 200, 233],
    [167, 228,   7, 240, 215, 152,  65,  45, 123,  69],
    [  8, 162, 216, 184,   9,  94, 250, 188,  36,  90]], order=2^8)
```

Display field elements as integers or polynomials.

```python
>>> print(x)
GF([118,  49, 122, 166, 136, 118,  53,  19, 233, 119], order=2^8)

# Temporarily set the display mode to represent GF(p^m) field elements as polynomials over GF(p)[x].
>>> with GF256.display("poly"):
...     print(x)
GF([x^6 + x^5 + x^4 + x^2 + x, x^5 + x^4 + 1, x^6 + x^5 + x^4 + x^3 + x,
    x^7 + x^5 + x^2 + x, x^7 + x^3, x^6 + x^5 + x^4 + x^2 + x,
    x^5 + x^4 + x^2 + 1, x^4 + x + 1, x^7 + x^6 + x^5 + x^3 + 1,
    x^6 + x^5 + x^4 + x^2 + x + 1], order=2^8)
```

### Galois field polynomial construction

Construct Galois field polynomials.

```python
# Construct a polynomial by specifying all the coefficients in descending-degree order
>>> p = galois.Poly([1, 22, 0, 17, 25], field=GF31); p
Poly(x^4 + 22x^3 + 17x + 25, GF31)

# Construct a polynomial by specifying only the non-zero coefficients
>>> q = galois.Poly.Degrees([2, 0], coeffs=[4, 14], field=GF31); q
Poly(4x^2 + 14, GF31)
```

### Galois field polynomial arithmetic

Galois field polynomial arithmetic is similar to `numpy` array operations.

```python
>>> p + q
Poly(x^4 + 22x^3 + 4x^2 + 17x + 8, GF31)

>>> p // q, p % q
(Poly(8x^2 + 21x + 3, GF31), Poly(2x + 14, GF31))

>>> p ** 2
Poly(x^8 + 13x^7 + 19x^6 + 3x^5 + 23x^4 + 15x^3 + 10x^2 + 13x + 5, GF31)
```

Galois field polynomials can also be evaluated at constants or arrays.

```python
>>> p
Poly(x^4 + 22x^3 + 17x + 25, GF31)

>>> a
GF([[28, 30, 17, 21, 22],
    [23, 29, 23, 27, 17]], order=31)

# Evaluate a polynomial at a single value
>>> p(1)
GF(3, order=31)

# Evaluate a polynomial at an array of values
>>> p(a)
GF([[19, 18,  0,  7,  5],
    [ 6, 17,  6, 14,  0]], order=31)
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
