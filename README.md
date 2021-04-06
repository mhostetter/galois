# Galois: A performant numpy extension for Galois fields

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Documentation Status](https://readthedocs.org/projects/galois/badge/?version=stable)](https://galois.readthedocs.io/en/stable/?badge=stable)
![Lint](https://github.com/mhostetter/galois/workflows/Lint/badge.svg?branch=master)
![Test](https://github.com/mhostetter/galois/workflows/Test/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/mhostetter/galois/branch/master/graph/badge.svg)](https://codecov.io/gh/mhostetter/galois)

- [Motivation](#motivation)
- [Documentation](#documentation)
- [Installation](#installation)
- [Versioning](#versioning)
- [Basic Usage](#basic-usage)
  - [Array construction](#array-construction)
  - [Field arithmetic](#field-arithmetic)
  - [Numpy functions](#numpy-functions)
  - [Numpy ufunc methods](#numpy-ufunc-methods)
  - [Field element display modes](#field-element-display-modes)
  - [Polynomial construction](#polynomial-construction)
  - [Polynomial arithmetic](#polynomial-arithmetic)
- [Performance](#performance)
  - [GF(31) addition speed test](#gf31-addition-speed-test)
- [Acknowledgements](#acknowledgements)

## Motivation

The project goals are for `galois` to be:

- _**General:**_ Support all Galois fields `GF(p^m)`, even arbitrarily large fields!
- _**Accurate:**_ Guarantee arithmetic accuracy -- tests against industry-standard mathematics software.
- _**Compatible:**_ Seamlessly integrate with `numpy` arrays -- arithmetic operators (`x + y`), broadcasting, view casting, type casting, numpy functions, ufuncs, ufunc methods.
- _**Performant:**_ Run as fast as `numpy` or C -- avoids the speed sinkhole of Python `for` loops.
- _**Reconfigurable:**_ Dynamically optimize JIT-compiled code for performance based on data size and processor (single-core CPU, multi-core CPU, or GPU).

<!-- ## Features

- asdf
- asdfasdf -->

## Documentation

Our documentation can be found at https://galois.readthedocs.io/en/stable/. The documentation includes [installation instructions](https://galois.readthedocs.io/en/stable/pages/installation.html), [development guide](https://galois.readthedocs.io/en/stable/pages/development.html), [basic usage](https://galois.readthedocs.io/en/stable/pages/basic_usage.html), [tutorials](https://galois.readthedocs.io/en/stable/pages/tutorials.html), and an [API reference](https://galois.readthedocs.io/en/stable/pages/build/_autosummary/galois.html#module-galois).

## Installation

The latest version of `galois` can be installed from [PyPI](https://pypi.org/project/galois/) via `pip`.

```bash
python3 -m pip install galois
```

## Versioning

This project uses [semantic versioning](https://semver.org/). Releases are versioned `major.minor.patch`. Major releases introduce API-changing features. Minor releases add features and are backwards compatible with other releases in `major.x.x`. Patch releases fix bugs in a minor release and are backwards compatible with other releases in `major.minor.x`.

Releases before `1.0.0` are alpha and beta releases. Alpha releases are `0.0.alpha`. There is no API compatibility guarantee for them. They can be thought of as `0.0.alpha-major`. Beta releases are `0.beta.x` and are API compatible. They can be thought of as `0.beta-major.beta-minor`.

## Basic Usage

### Array construction

Construct Galois field array classes using the `galois.GF()` class factory function.

```python
>>> import numpy as np

>>> import galois

>>> GF31 = galois.GF(31)

>>> print(GF31)
<class 'numpy.ndarray' over GF(31)>

>>> issubclass(GF31, np.ndarray)
True
```

Galois field array classes contain extra class attributes related to the finite field.

```python
# The size of the finite field
>>> GF31.order
31

# A primitive element of the finite field
>>> GF31.alpha
GF(3, order=31)

# The primitive polynomial of the finite field
>>> GF31.prim_poly
Poly(x + 28, GF(31))
```

Create any Galois field array class type: `GF(2^m)`, `GF(p)`, or `GF(p^m)`. Even arbitrarily-large fields!

```python
# Field used in AES
>>> GF256 = galois.GF(2**8); print(GF256)
<class 'numpy.ndarray' over GF(2^8)>

>>> prime = 36893488147419103183; galois.is_prime(prime)
True

# Large prime field
>>> GFp = galois.GF(prime); print(GFp)
<class 'numpy.ndarray' over GF(36893488147419103183)>

# Large characteristic-2 field
>>> GF2_100 = galois.GF(2**100); print(GF2_100)
<class 'numpy.ndarray' over GF(2^100)>
```

Create arrays from existing `numpy` arrays, either explicitly or by view casting.

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

### Field arithmetic

Here, `GF` is any Galois field array class created from `galois.GF`, `x` and `y` are `GF` arrays, and `z` is an integer `numpy.ndarray`. All arithmetic operations follow normal numpy [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) rules.

- Addition: `x + y == np.add(x, y)`
- Subtraction: `x - y == np.subtract(x, y)`
- Multiplication: `x * y == np.multiply(x, y)`
- Division: `x / y == x // y == np.divide(x, y)`
- Scalar multiplication: `x * z == z * x`, e.g. `x * 3 == x + x + x`
- Reciprocal: `1 / x == np.reciprocal(x)`
- Exponentiation: `x ** z == np.power(x, z)`
- Logarithm base `alpha`: `np.log(x)`, where `alpha == GF.alpha`
- **COMING SOON**: Logarithm base `b`: `GF.log(x, b)`, where `b` is any field element
- **COMING SOON**: Matrix multiplication: `x @ y = np.matmul(x, y)`

**Note**

Generally, we don't allow Galois field array operations with scalars, i.e. `x + 5` or `x + z`, even if `5` is a valid element in `x`'s Galois field, or `z`'s integers are elements too. We prefer *explicit over implicit*. Instead, the correct notation would be `x + GF(5)` and `y = GF(y); x + y`.

There are a couple exceptions: scalar multiplication and exponentiation.

For multiplication, `x * y` is interpreted as field multiplication. Whereas, `x * 3`, which is valid syntax, is interpreted as `x + x + x`. In prime fields, `x * GF(3) == x * 3`. In extension fields, `x * GF(3) != x * 3` so **be careful!**

For exponentiation, `x ** 3` or `x ** -2` are valid. The exponent can be any integer, not just a field element.

### Numpy functions

The `galois` package also supports linear algebra routines. They can be accessed using the
natural numpy syntax.

- **COMING SOON**: [`np.inner`](https://numpy.org/doc/stable/reference/generated/numpy.inner.html)
- **COMING SOON**: [`np.dot`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot)
- **COMING SOON**: [`np.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html#numpy.tensordot)

### Numpy ufunc methods

Galois field arrays also support numpy ufunc methods. This allows you to apply a ufunc in a unique was across the target
array.

The ufunc method signature is `<ufunc>.<method>(*args, **kwargs)`. Below are the supported ufuncs and their methods.

- `<ufunc>`: [`np.add`](https://numpy.org/doc/stable/reference/generated/numpy.add.html), [`np.subtract`](https://numpy.org/doc/stable/reference/generated/numpy.subtract.html), [`np.multiply`](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html), [`np.divide`](https://numpy.org/doc/stable/reference/generated/numpy.divide.html), [`np.true_divide`](https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html), [`np.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html), [`np.negative`](https://numpy.org/doc/stable/reference/generated/numpy.negative.html), [`np.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html), [`np.square`](https://numpy.org/doc/stable/reference/generated/numpy.square.html), [`np.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html)

- `<method>`: [`reduce`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html), [`accumulate`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.accumulate.html), [`reduceat`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html), [`outer`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html), [`at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html)

Below is are examples of how to use the `reduce` and `outer` methods with the `np.multiply` ufunc.

```python
>>> a = GF31.Random((2,5)); a
GF([[28, 30, 17, 21, 22],
    [23, 29, 23, 27, 17]], order=31)

>>> np.multiply.reduce(a, axis=0)
GF([24,  2, 19,  9,  2], order=31)
```

```python
>>> x = GF256.Random(10); x
GF([118,  49, 122, 166, 136, 118,  53,  19, 233, 119], order=2^8)

>>> y = GF256.Random(10, low=1); y
GF([239,  63,  81, 225, 150,  12,  56,  24,  98, 245], order=2^8)

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

### Field element display modes

The user may display the finite field elements as either integers or polynomials.

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

### Polynomial construction

Construct Galois field polynomials.

```python
# Construct a polynomial by specifying all the coefficients in descending-degree order
>>> p = galois.Poly([1, 22, 0, 17, 25], field=GF31); p
Poly(x^4 + 22x^3 + 17x + 25, GF(31))

# Construct a polynomial by specifying only the non-zero coefficients
>>> q = galois.Poly.Degrees([2, 0], coeffs=[4, 14], field=GF31); q
Poly(4x^2 + 14, GF(31))
```

### Polynomial arithmetic

Galois field polynomial arithmetic is similar to `numpy` array operations.

```python
>>> p + q
Poly(x^4 + 22x^3 + 4x^2 + 17x + 8, GF(31))

>>> p // q, p % q
(Poly(8x^2 + 21x + 3, GF(31)), Poly(2x + 14, GF(31)))

>>> p ** 2
Poly(x^8 + 13x^7 + 19x^6 + 3x^5 + 23x^4 + 15x^3 + 10x^2 + 13x + 5, GF(31))
```

Galois field polynomials can also be evaluated at constants or arrays.

```python
>>> p
Poly(x^4 + 22x^3 + 17x + 25, GF(31))

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

>>> GFp = galois.GF(31)
>>> print(GFp)>
<class 'numpy.ndarray' over GF(31)>

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

## Acknowledgements

This package heavily relies on [Numba](https://numba.pydata.org/) and its just-in-time compiler for performance.
We use Frank Luebeck's [compilation of Conway polynomials](http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html)
for computing primitive polynomials for extension fields. We utilize [SageMath](https://www.sagemath.org/) for generating test vectors.
