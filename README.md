# <img alt="Galois: A performant NumPy extension for Galois fields and their applications" align="middle" src="/logo/galois-heading.png">

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![Downloads](https://pepy.tech/badge/galois/month)](https://pepy.tech/project/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Documentation Status](https://readthedocs.org/projects/galois/badge/?version=stable)](https://galois.readthedocs.io/en/latest/?badge=latest)
![Lint](https://github.com/mhostetter/galois/workflows/Lint/badge.svg?branch=master)
![Test](https://github.com/mhostetter/galois/workflows/Test/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/mhostetter/galois/branch/master/graph/badge.svg)](https://codecov.io/gh/mhostetter/galois)

The main idea of the `galois` package can be summarized as follows. The user creates a "Galois field array class" using `GF = galois.GF(p**m)`.
A Galois field array class `GF` is a subclass of `np.ndarray` and its constructor `x = GF(array_like)` mimics
the call signature of `np.array()`. A Galois field array `x` is operated on like any other NumPy array, but all
arithmetic is performed in `GF(p^m)` not **Z** or **R**.

Internally, the Galois field arithmetic is implemented by replacing [NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html).
The new ufuncs are written in Python and then [just-in-time compiled](https://numba.pydata.org/numba-doc/dev/user/vectorize.html) with
[Numba](https://numba.pydata.org/). The ufuncs can be configured to use either lookup tables (for speed) or explicit
calculation (for memory savings).

- [Features](#features)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Installation](#installation)
- [Versioning](#versioning)
- [Basic Usage](#basic-usage)
- [Performance](#performance)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Features

- Supports all [Galois fields](https://galois.readthedocs.io/en/stable/api/galois-fields.html#) `GF(p^m)`, even arbitrarily-large fields!
- **Faster** than native NumPy! `GF(x) * GF(y)` is faster than `(x * y) % p` for `GF(p)`
- Seamless integration with NumPy -- normal NumPy functions work on Galois field arrays
- [Linear algebra](https://galois.readthedocs.io/en/stable/api/numpy-examples.html#linear-algebra) on Galois field matrices using normal `np.linalg` functions
- [Functions](https://galois.readthedocs.io/en/stable/api/polys.html#special-polynomial-creation) to generate irreducible, primitive, and Conway polynomials
- [Polynomials](https://galois.readthedocs.io/en/stable/api/polys.html) over Galois fields with `galois.Poly`
- [Forward error correction codes](https://galois.readthedocs.io/en/stable/api/fec.html) with `galois.BCH` and `galois.ReedSolomon`
- Fibonacci and Galois [linear feedback shift registers](https://galois.readthedocs.io/en/stable/api/linear-sequences.html) with `galois.LFSR`, both binary and p-ary
- Various [number theoretic functions](https://galois.readthedocs.io/en/stable/api/number-theory.html)
- [Integer factorization](https://galois.readthedocs.io/en/stable/api/integer-factorization.html) and accompanying algorithms
- [Prime number generation](https://galois.readthedocs.io/en/stable/api/primes.html#prime-number-generation) and [primality testing](https://galois.readthedocs.io/en/stable/api/primes.html#primality-tests)

## Roadmap

- Elliptic curves over Galois fields
- Number-theoretic transform, DFT over Galois fields
- Group and ring arrays
- GPU support

## Documentation

The documentation for `galois` can be found at https://galois.readthedocs.io/en/stable/. It includes
[installation instructions](https://galois.readthedocs.io/en/stable/installation.html), [basic usage](https://galois.readthedocs.io/en/stable/basic-usage.html),
[tutorials](https://galois.readthedocs.io/en/stable/tutorials.html), a [development guide](https://galois.readthedocs.io/en/stable/development.html), an [API reference](https://galois.readthedocs.io/en/stable/api/galois.html), and [release notes](https://galois.readthedocs.io/en/stable/release-notes.html).

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

### Galois field arrays

#### Class construction

Galois field array classes are created using the `galois.GF()` class factory function.

```python
>>> import numpy as np

>>> import galois

>>> GF256 = galois.GF(2**8)

>>> print(GF256)
<class 'numpy.ndarray over GF(2^8)'>
```

These classes are subclasses of `galois.FieldArray` (which itself subclasses `np.ndarray`) and have `galois.FieldClass` as their metaclass.

```python
>>> isinstance(GF256, galois.FieldClass)
True

>>> issubclass(GF256, galois.FieldArray)
True

>>> issubclass(GF256, np.ndarray)
True
```

A Galois field array class contains attributes relating to its Galois field and methods to modify how the field
is calculated or displayed. See the attributes and methods in `galois.FieldClass`.

```python
# Summarizes some properties of the Galois field
>>> print(GF256.properties)
GF(2^8):
  characteristic: 2
  degree: 8
  order: 256
  irreducible_poly: x^8 + x^4 + x^3 + x^2 + 1
  is_primitive_poly: True
  primitive_element: x

# Access each attribute individually
>>> GF256.irreducible_poly
Poly(x^8 + x^4 + x^3 + x^2 + 1, GF(2))
```

The `galois` package even supports arbitrarily-large fields! This is accomplished by using NumPy arrays
with `dtype=object` and pure-Python ufuncs. This comes at a performance penalty compared to smaller fields
which use NumPy integer dtypes (e.g., `np.uint32`) and have compiled ufuncs.

```python
>>> GF = galois.GF(36893488147419103183); print(GF.properties)
GF(36893488147419103183):
  characteristic: 36893488147419103183
  degree: 1
  order: 36893488147419103183
  irreducible_poly: x + 36893488147419103180
  is_primitive_poly: True
  primitive_element: 3

>>> GF = galois.GF(2**100); print(GF.properties)
GF(2^100):
  characteristic: 2
  degree: 100
  order: 1267650600228229401496703205376
  irreducible_poly: x^100 + x^57 + x^56 + x^55 + x^52 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^41 + x^37 + x^36 + x^35 + x^34 + x^31 + x^30 + x^27 + x^25 + x^24 + x^22 + x^20 + x^19 + x^16 + x^15 + x^11 + x^9 + x^8 + x^6 + x^5 + x^3 + 1
  is_primitive_poly: True
  primitive_element: x
```

#### Array creation

Galois field arrays can be created from existing NumPy arrays.

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

- `FieldArray.Zeros`, `FieldArray.Ones`, `FieldArray.Identity`, `FieldArray.Range`, `FieldArray.Random`, `FieldArray.Elements`

Galois field elements can either be displayed using their integer representation, polynomial representation, or
power representation. Calling `FieldClass.display` will change the element representation. If called as a context
manager, the display mode will only be temporarily changed.

```python
>>> a = GF256(["x^6 + 1", 0, 2, "1", "x^5 + x^4 + x"]); a
GF([65,  0,  2,  1, 50], order=2^8)

# Set the display mode to represent GF(2^8) field elements as polynomials over GF(2) with degree less than 8
>>> GF256.display("poly");

>>> a
GF([α^6 + 1, 0, α, 1, α^5 + α^4 + α], order=2^8)

# Temporarily set the display mode to represent GF(2^8) field elements as powers of the primitive element
>>> with GF256.display("power"):
...     print(a)

GF([α^191, 0, α, 1, α^194], order=2^8)

# Resets the display mode to the integer representation
>>> GF256.display();
```

#### Field arithmetic

Galois field arrays are treated like any other NumPy array. Array arithmetic is performed using Python operators or NumPy
functions.

In the list below, `GF` is a Galois field array class created by `GF = galois.GF(p**m)`, `x` and `y` are `GF` arrays, and `z` is an
integer `np.ndarray`. All arithmetic operations follow normal NumPy [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) rules.

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

#### Linear algebra

The `galois` package intercepts relevant calls to NumPy's linear algebra functions and performs the specified
operation in `GF(p^m)` not in **R**. Some of these functions include:

- `np.dot`, `np.vdot`, `np.inner`, `np.outer`, `np.matmul`, `np.linalg.matrix_power`
- `np.linalg.det`, `np.linalg.matrix_rank`, `np.trace`
- `np.linalg.solve`, `np.linalg.inv`

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

Galois field arrays also contain matrix decomposition routines not included in NumPy. These include:

- `FieldArray.row_reduce`, `FieldArray.lu_decompose`, `FieldArray.lup_decompose`

#### NumPy ufunc methods

Galois field arrays support [NumPy ufunc methods](https://numpy.org/devdocs/reference/ufuncs.html#methods). This allows the user to apply a ufunc in a unique way across the target
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

### Polynomials over Galois fields

The `galois` package supports polynomials over Galois fields with the `galois.Poly` class. `galois.Poly`
does not subclass `np.ndarray` but instead contains a `FieldArray` of coefficients as an attribute
(implements the "has-a", not "is-a", architecture).

Polynomials can be created by specifying the polynomial coefficients as either a `FieldArray` or an "array-like"
object with the `field` keyword argument.

```python
>>> p = galois.Poly([172, 22, 0, 0, 225], field=GF256); p
Poly(172x^4 + 22x^3 + 225, GF(2^8))

>>> coeffs = GF256([33, 17, 0, 225]); coeffs
GF([ 33,  17,   0, 225], order=2^8)

>>> p = galois.Poly(coeffs, order="asc"); p
Poly(225x^3 + 17x + 33, GF(2^8))
```

Polynomials over Galois fields can also display their coefficients as polynomials over their prime subfields.
This can be quite confusing to read, so be warned!

```python
>>> print(p)
Poly(225x^3 + 17x + 33, GF(2^8))

>>> with GF256.display("poly"):
...     print(p)

Poly((α^7 + α^6 + α^5 + 1)x^3 + (α^4 + 1)x + (α^5 + 1), GF(2^8))
```

Polynomials can also be created using a number of constructor class methods. They include:

- `Poly.Zero`, `Poly.One`, `Poly.Identity`, `Poly.Random`, `Poly.Integer`, `Poly.String`, `Poly.Degrees`, `Poly.Roots`

```python
# Construct a polynomial by specifying its roots
>>> q = galois.Poly.Roots([155, 37], field=GF256); q
Poly(x^2 + 190x + 71, GF(2^8))

>>> q.roots()
GF([ 37, 155], order=2^8)
```

Polynomial arithmetic is performed using Python operators.

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

### Forward error correction codes

To demonstrate the FEC code API, here is an example using BCH codes. Other FEC codes have a similar API.

```python
>>> import numpy as np

>>> import galois

>>> bch = galois.BCH(15, 7); bch
<BCH Code: [15, 7, 5] over GF(2)>

>>> bch.generator_poly
Poly(x^8 + x^7 + x^6 + x^4 + 1, GF(2))

# The error-correcting capability
>>> bch.t
2
```

A message can be either a 1-D vector or a 2-D matrix of messages. Shortened codes are also supported. See the docs for more details.

```python
# Create a matrix of two messages
>>> M = galois.GF2.Random((2, bch.k)); M
GF([[1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 0]], order=2)
```

Encoding the message(s) is performed with `encode()`.

```python
>>> C = bch.encode(M); C
GF([[1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]], order=2)
```

Decoding the codeword(s) is performed with `decode()`.

```python
# Corrupt the first bit in each codeword
C[:,0] ^= 1; C
GF([[0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]], order=2)

bch.decode(C)
GF([[1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 0]], order=2)
```

## Performance

To compare the performance of `galois` and native NumPy, we'll use a prime field `GF(p)`. This is because
it is the simplest field. Namely, addition, subtraction, and multiplication are modulo `p`, which can
be simply computed with NumPy arrays `(x + y) % p`. For extension fields `GF(p^m)`, the arithmetic is
computed using polynomials over `GF(p)` and can't be so tersely expressed in NumPy.

### Lookup performance

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

### Calculation performance

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

### Linear algebra performance

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

## Acknowledgements

- This library is an extension of, and completely dependent on, [NumPy](https://numpy.org/).
- We heavily rely on [Numba](https://numba.pydata.org/) and its just-in-time compiler for optimizing performance of Galois field arithmetic.
- We use Frank Luebeck's compilation of [Conway polynomials](http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html).
- We also use Wolfram's compilation of [primitive polynomials](https://datarepository.wolframcloud.com/resources/Primitive-Polynomials/).
- We extensively use [SageMath](https://www.sagemath.org/) for generating test vectors.
- We also use [Octave](https://www.gnu.org/software/octave/index) for generating test vectors.

Many thanks!

## Citation

If this library was useful to you in your research, feel free to cite us. Following other Python package
[citation standards](https://github.com/leonoverweel/bibtex-python-package-citations), here is the recommended citation.

```bibtex
@misc{galois,
    title={Galois: A performant NumPy extension for Galois fields},
    author={Matt Hostetter},
    year={2020},
    howpublished={\url{https://galois.readthedocs.io/en/stable/}},
}
```
