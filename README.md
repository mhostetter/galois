# Galois: A performant numpy extension for Galois fields

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Documentation Status](https://readthedocs.org/projects/galois/badge/?version=stable)](https://galois.readthedocs.io/en/stable/?badge=stable)
![Lint](https://github.com/mhostetter/galois/workflows/Lint/badge.svg?branch=master)
![Test](https://github.com/mhostetter/galois/workflows/Test/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/mhostetter/galois/branch/master/graph/badge.svg)](https://codecov.io/gh/mhostetter/galois)

The main idea of the `galois` package can be summarized as follows. The user creates a "Galois field array class" using `GF = galois.GF(p**m)`.
A Galois field array class `GF` is a subclass of `np.ndarray` and its constructor `x = GF(array_like)` mimics
the call signature of `np.array`. A Galois field array `x` is operated on like any other numpy array, but all
arithmetic is performed in `GF(p^m)` not **Z** or **R**.

Internally, the Galois field arithmetic is implemented by replacing [numpy ufuncs](https://np.org/doc/stable/reference/ufuncs.html).
The new ufuncs are written in python and then [just-in-time compiled](https://numba.pydata.org/numba-doc/dev/user/vectorize.html) with
[numba](https://numba.pydata.org/). The ufuncs can be configured to use either lookup tables (for speed) or explicit
calculation (for memory savings). Numba also provides the ability to ["target"](https://numba.readthedocs.io/en/stable/user/vectorize.html?highlight=target)
the JIT-compiled ufuncs for CPUs or GPUs.

In addition to normal array arithmetic, `galois` also supports linear algebra (with `np.linalg` functions) and polynomials over Galois fields
(with the `galois.Poly` class).

- [Features](#features)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Installation](#installation)
- [Versioning](#versioning)
- [Basic Usage](#basic-usage)
  - [Class construction](#class-construction)
  - [Array creation](#array-creation)
  - [Field arithmetic](#field-arithmetic)
  - [Linear algebra](#linear-algebra)
  - [Numpy ufunc methods](#numpy-ufunc-methods)
  - [Numpy functions](#numpy-functions)
  - [Polynomial construction](#polynomial-construction)
  - [Polynomial arithmetic](#polynomial-arithmetic)
- [Performance](#performance)
  - [Lookup performance](#lookup-performance)
  - [Calculation performance](#calculation-performance)
- [Acknowledgements](#acknowledgements)

## Features

- Supports all Galois fields `GF(p^m)`, even arbitrarily-large fields!
- **Faster** than native numpy! `GF(x) * GF(y)` is faster than `(x * y) % p` for `GF(p)`
- Seamless integration with numpy -- normal numpy functions work on Galois field arrays
- Linear algebra over Galois fields using native `np.linalg` functions
- Polynomials over Galois fields with `galois.Poly`, both dense and sparse polynomials
- Compile ufuncs to target GPUs for massive data processing

## Roadmap

Planned future functionality of `galois` includes:

- Ring support
- Linear feedback shift registers over arbitrary Galois fields
- Number-theoretic transform, DFT over Galois fields
- Elliptic curves over Galois fields
- Cryptographic ciphers using Galois fields (RSA, AES, ECC, etc)

## Documentation

Our documentation can be found at https://galois.readthedocs.io/en/stable/. The documentation includes
[installation instructions](https://galois.readthedocs.io/en/stable/pages/installation.html), [basic usage](https://galois.readthedocs.io/en/stable/pages/basic_usage.html),
[tutorials](https://galois.readthedocs.io/en/stable/pages/tutorials.html), [development guide](https://galois.readthedocs.io/en/stable/pages/development.html),
and an [API reference](https://galois.readthedocs.io/en/stable/pages/build/_autosummary/galois.html#module-galois).

## Installation

The latest version of `galois` can be installed from [PyPI](https://pypi.org/project/galois/) using `pip`.

```bash
python3 -m pip install galois
```

## Versioning

This project uses [semantic versioning](https://semver.org/). Releases are versioned `major.minor.patch`. Major releases introduce API-changing
features. Minor releases add features and are backwards compatible with other releases in `major.x.x`. Patch releases fix bugs in a minor release
and are backwards compatible with other releases in `major.minor.x`.

Releases before `1.0.0` are alpha and beta releases. Alpha releases are `0.0.alpha`. There is no API compatibility guarantee for them. They can
be thought of as `0.0.alpha-major`. Beta releases are `0.beta.x` and are API compatible. They can be thought of as `0.beta-major.beta-minor`.

## Basic Usage

### Class construction

Galois field array classes are created using the `galois.GF()` class factory function.

```python
>>> import numpy as np

>>> import galois

>>> GF256 = galois.GF(2**8)

>>> print(GF256)
<class 'np.ndarray over GF(2^8)'>
```

These classes are subclasses of `galois.GFArray` (which itself subclasses `np.ndarray`) and have `galois.GFMeta` as their metaclass.

```python
>>> issubclass(GF256, np.ndarray)
True

>>> issubclass(GF256, galois.GFArray)
True

>>> issubclass(type(GF256), galois.GFMeta)
True
```

A Galois field array class contains attributes relating to its Galois field and methods to modify how the field
is calculated or displayed. See the attributes and methods in `galois.GFMeta`.

```python
# Summarizes some properties of the Galois field
>>> print(GF256.properties)
GF(2^8):
  characteristic: 2
  degree: 8
  order: 256
  irreducible_poly: Poly(x^8 + x^4 + x^3 + x^2 + 1, GF(2))
  is_primitive_poly: True
  primitive_element: GF(2, order=2^8)

# Access each attribute individually
>>> GF256.irreducible_poly
Poly(x^8 + x^4 + x^3 + x^2 + 1, GF(2))
```

The `galois` package even supports arbitrarily-large fields! This is accomplished by using numpy arrays
with `dtype=object` and pure-python ufuncs. This comes at a performance penalty compared to smaller fields
which use numpy integer dtypes (e.g., `np.uint32`) and have compiled ufuncs.

```python
>>> GF = galois.GF(36893488147419103183); print(GF.properties)
GF(36893488147419103183):
  characteristic: 36893488147419103183
  degree: 1
  order: 36893488147419103183
  irreducible_poly: Poly(x + 36893488147419103180, GF(36893488147419103183))
  is_primitive_poly: True
  primitive_element: GF(3, order=36893488147419103183)

>>> GF = galois.GF(2**100); print(GF.properties)
GF(2^100):
  characteristic: 2
  degree: 100
  order: 1267650600228229401496703205376
  irreducible_poly: Poly(x^100 + x^57 + x^56 + x^55 + x^52 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^41 + x^37 + x^36 + x^35 + x^34 + x^31 + x^30 + x^27 + x^25 + x^24 + x^22 + x^20 + x^19 + x^16 + x^15 + x^11 + x^9 + x^8 + x^6 + x^5 + x^3 + 1, GF(2))
  is_primitive_poly: True
  primitive_element: GF(2, order=2^100)
```

### Array creation

Galois field arrays can be created from existing numpy arrays.

```python
# Represents an existing numpy array
>>> array = np.random.randint(0, GF256.order, 10, dtype=int); array
array([ 31, 254, 155, 154, 121, 185,  16, 246, 216, 244])

# Explicit Galois field array creation (a copy is performed)
>>> GF256(array)
GF([ 31, 254, 155, 154, 121, 185,  16, 246, 216, 244], order=2^8)

# Or view an existing numpy array as a Galois field array (no copy is performed)
>>> array.view(GF256)
GF([ 31, 254, 155, 154, 121, 185,  16, 246, 216, 244], order=2^8)
```

Or they can be created from "array-like" objects. These include strings representing a Galois field element
as a polynomial over its prime subfield.

```python
# Arrays can be specified as iterables of iterables
>>> GF256([[217, 130, 42], [74, 208, 113]])
GF([[217, 130,  42],
    [ 74, 208, 113]], order=2^8)

# You can mix-and-match polynomial strings and integers
>>> GF256(["x^6 + 1", 2, "1", "x^5 + x^4 + x"])
GF([65,  2,  1, 50], order=2^8)

# Single field elements are 0-dimensional arrays
>>> GF256("x^6 + x^4 + 1")
GF(81, order=2^8)
```

Galois field arrays also have constructor class methods for convenience. They include:

- `GFArray.Zeros`, `GFArray.Ones`, `GFArray.Identity`, `GFArray.Range`, `GFArray.Random`, `GFArray.Elements`

Galois field elements can either be displayed using their integer representation, polynomial representation, or
power representation. Calling `GFMeta.display` will change the element representation. If called as a context
manager, the display mode will only be temporarily changed.

```python
>>> x = GF256(["y**6 + 1", 0, 2, "1", "y**5 + y**4 + y"]); x
GF([65,  0,  2,  1, 50], order=2^8)

# Set the display mode to represent GF(2^8) field elements as polynomials over GF(2) with degree less than 8
>>> GF256.display("poly"):

>>> x
GF([α^6 + 1, 0, α, 1, α^5 + α^4 + α], order=2^8)

# Temporarily set the display mode to represent GF(2^8) field elements as powers of the primitive element
>>> with GF256.display("power"):
...     print(x)

GF([α^191, -∞, α, 1, α^194], order=2^8)

# Resets the display mode to the integer representation
>>> GF256.display():
```

### Field arithmetic

Galois field arrays are treated like any other numpy array. Array arithmetic is performed using python operators or numpy
functions.

In the list below, `GF` is a Galois field array class created by `GF = galois.GF(p**m)`, `x` and `y` are `GF` arrays, and `z` is an
integer `np.ndarray`. All arithmetic operations follow normal numpy [broadcasting](https://np.org/doc/stable/user/basics.broadcasting.html) rules.

- Addition: `x + y == np.add(x, y)`
- Subtraction: `x - y == np.subtract(x, y)`
- Multiplication: `x * y == np.multiply(x, y)`
- Division: `x / y == x // y == np.divide(x, y)`
- Scalar multiplication: `x * z == np.multiply(x, z)`, e.g. `x * 3 == x + x + x`
- Additive inverse: `-x == np.negative(x)`
- Multiplicative inverse: `GF(1) / x == np.reciprocal(x)`
- Exponentiation: `x ** z == np.power(x, z)`, e.g. `x ** 3 == x * x * x`
- Logarithm: `np.log(x)`, e.g. `GF.primitive_element ** np.log(x) == x`
- **COMING SOON:** Logarithm base `b`: `GF.log(x, b)`, where `b` is any field element
- Matrix multiplication: `A @ B == np.matmul(A, B)`

```python
>>> x = GF256.Random((2,5)); x
GF([[166,  71, 214, 164, 228],
    [168, 202,  73,  54, 180]], order=2^8)

>>> y = GF256.Random(5); y
GF([ 25, 102, 131, 233, 188], order=2^8)

# y is broadcast over the last dimension of x
>>> x + y
GF([[191,  33,  85,  77,  88],
    [177, 172, 202, 223,   8]], order=2^8)
```

### Linear algebra

The `galois` package intercepts relevant calls to numpy's linear algebra functions and performs the specified
operation in `GF(p^m)` not in **R**. Some of these functions include:

- `np.trace`
- `np.dot`, `np.inner`, `np.outer`
- `np.linalg.matrix_rank`, `np.linalg.matrix_power`
- `np.linalg.det`, `np.linalg.inv`, `np.linalg.solve`

```python
>>> A = GF256.Random((3,3)); A
GF([[151, 147, 229],
    [162, 192,  59],
    [ 52, 213,  37]], order=2^8)

>>> b = GF256.Random(3); b
GF([154, 193, 235], order=2^8)

>>> x = np.linalg.solve(A, b); x
GF([114, 170, 178], order=2^8)

>>> np.array_equal(A @ x, b)
True
```

Galois field arrays also contain matrix decomposition routines not included in numpy. These include:

- `GFArray.row_reduce`, `GFArray.lu_decompose`, `GFArray.lup_decompose`

### Numpy ufunc methods

Galois field arrays support [numpy ufunc methods](https://np.org/devdocs/reference/ufuncs.html#methods). This allows the user to apply a ufunc in a unique way across the target
array. The ufunc method signature is `<ufunc>.<method>(*args, **kwargs)`. All arithmetic ufuncs are supported. Below
is a list of their ufunc methods.

- `<method>`: `reduce`, `accumulate`, `reduceat`, `outer`, `at`

```python
>>> X = GF256.Random((2,5)); X
GF([[210,  67, 167, 137,  95],
    [104,  74, 178,  13, 142]], order=2^8)

>>> np.multiply.reduce(X, axis=0)
GF([ 63, 169, 209, 171, 161], order=2^8)
```

```python
>>> x = GF256.Random(5); x
GF([210,  49,  66, 251, 148], order=2^8)

>>> y = GF256.Random(5); y
GF([  3, 123, 247, 144, 197], order=2^8)

>>> np.multiply.outer(x, y)
GF([[107, 245,  37, 192,  98],
    [ 83,  67, 183, 146, 140],
    [198,  93, 248, 206, 128],
    [ 16, 170, 178,  83,  68],
    [161,  89,  38, 116, 209]], order=2^8)
```

### Numpy functions

Many other relevant numpy functions are supported on Galois field arrays. These include:

- `np.copy`, `np.concatenate`, `np.insert`, `np.reshape`

### Polynomial construction

The `galois` package supports polynomials over Galois fields with the `galois.Poly` class. `galois.Poly`
does not subclass `np.ndarray` but instead contains a `GFArray` of coefficients as an attribute
(implements the "has-a", not "is-a", architecture).

Polynomials can be created by specifying the polynomial coefficients as either a `GFArray` or an "array-like"
object with the `field` keyword argument.

```python
>>> p = galois.Poly([172, 22, 0, 0, 225], field=GF256); p
Poly(172x^4 + 22x^3 + 225, GF(2^8))

>>> coeffs = GF256([33, 17, 0, 225]); coeffs
GF([ 33,  17,   0, 225], order=2^8)

>>> p = galois.Poly(coeffs, order="asc"); p
Poly(225x^3 + 17x + 33, GF(2^8))
```

Polynomials over Galois fields can also display the field elements as polynomials over their prime subfields.
This can be quite confusing to read, so be warned!

```python
>>> print(p)
Poly(225x^3 + 17x + 33, GF(2^8))

>>> with GF256.display("poly"):
...     print(p)

Poly((α^7 + α^6 + α^5 + 1)x^3 + (α^4 + 1)x + (α^5 + 1), GF(2^8))
```

Polynomials can also be created using a number of constructor class methods. They include:

- `Poly.Zero`, `Poly.One`, `Poly.Identity`, `Poly.Random`, `Poly.Integer`, `Poly.Degrees`, `Poly.Roots`

```python
# Construct a polynomial by specifying its roots
>>> q = galois.Poly.Roots([155, 37], field=GF256); q
Poly(x^2 + 190x + 71, GF(2^8))

>>> q.roots()
GF([ 37, 155], order=2^8)
```

### Polynomial arithmetic

Polynomial arithmetic is performed using python operators.

```python
>>> p
Poly(225x^3 + 17x + 33, GF(2^8))

>>> q
Poly(x^2 + 190x + 71, GF(2^8))

>>> p + q
Poly(225x^3 + x^2 + 175x + 102, GF(2^8))

>>> divmod(p, q)
(Poly(225x + 57, GF(2^8)), Poly(56x + 104, GF(2^8)))

>>> p ** 2
Poly(171x^6 + 28x^2 + 117, GF(2^8))
```

Polynomials over Galois fields can be evaluated at scalars or arrays of field elements.

```python
>>> p = galois.Poly.Degrees([4, 3, 0], [172, 22, 225], field=GF256); p
Poly(172x^4 + 22x^3 + 225, GF(2^8))

# Evaluate the polynomial at a single value
>>> p(1)
GF(91, order=2^8)

>>> x = GF256.Random((2,5)); x
GF([[212, 211, 244, 125,  75],
    [113, 139, 247, 223, 106]], order=2^8)

# Evaluate the polynomial at an array of values
>>> p(x)
GF([[158, 129,  28, 122, 186],
    [184, 132, 179,  49, 223]], order=2^8)
```

Polynomials can also be evaluated in superfields. For example, evaluating a Galois field’s irreducible polynomial at one of its elements.

```python
# Notice the irreducible polynomial is over GF(2), not GF(2^8)
>>> p = GF256.irreducible_poly; p
Poly(x^8 + x^4 + x^3 + x^2 + 1, GF(2))

>>> GF256.is_primitive_poly
True

# Notice the primitive element is in GF(2^8)
>>> alpha = GF256.primitive_element; alpha
GF(2, order=2^8)

# Since p(x) is a primitive polynomial, alpha is one of its roots
>>> p(alpha, field=GF256)
GF(0, order=2^8)
```

## Performance

To compare the performance of `galois` and native numpy, we'll use a prime field `GF(p)`. This is because
it is the simplest field. Namely, addition, subtraction, and multiplication are modulo `p`, which can
be simply computed with numpy arrays `(x + y) % p`. For extension fields `GF(p^m)`, the arithmetic is
computed using polynomials over `GF(p)` and can't be so tersely expressed in numpy.

### Lookup performance

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

### Calculation performance

For fields with order greater than `2^20`, `galois` will use explicit arithmetic calculation rather
than lookup tables. Even in these cases, `galois` is faster than numpy!

Here is an example multiplying two arrays in `GF(2097169)` using numpy and `galois` with
`ufunc_mode="jit-calculate"`.

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

## Acknowledgements

This package heavily relies on [Numba](https://numba.pydata.org/) and its just-in-time compiler for performance.
We use Frank Luebeck's [compilation of Conway polynomials](http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html)
for computing primitive polynomials for extension fields. We utilize [SageMath](https://www.sagemath.org/) for generating test vectors.
