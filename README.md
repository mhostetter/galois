# Galois: A performant numpy extension for Galois fields and their applications

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Documentation Status](https://readthedocs.org/projects/galois/badge/?version=stable)](https://galois.readthedocs.io/en/stable/?badge=stable)
![Lint](https://github.com/mhostetter/galois/workflows/Lint/badge.svg?branch=master)
![Test](https://github.com/mhostetter/galois/workflows/Test/badge.svg?branch=master)
[![Codecov](https://codecov.io/gh/mhostetter/galois/branch/master/graph/badge.svg)](https://codecov.io/gh/mhostetter/galois)

The main idea of the `galois` package can be summarized as follows. The user creates a "Galois field array class" using `GF = galois.GF(p**m)`.
A Galois field array class `GF` is a subclass of `np.ndarray` and its constructor `x = GF(array_like)` mimics
the call signature of `np.array()`. A Galois field array `x` is operated on like any other numpy array, but all
arithmetic is performed in `GF(p^m)` not **Z** or **R**.

Internally, the Galois field arithmetic is implemented by replacing [numpy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html).
The new ufuncs are written in python and then [just-in-time compiled](https://numba.pydata.org/numba-doc/dev/user/vectorize.html) with
[numba](https://numba.pydata.org/). The ufuncs can be configured to use either lookup tables (for speed) or explicit
calculation (for memory savings).

In addition to normal array arithmetic, `galois` also supports linear algebra (with `np.linalg` functions), polynomials
over Galois fields (with `galois.Poly`), and forward error correction codes (with `galois.BCH` and `galois.ReedSolomon`).

- [Features](#features)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Installation](#installation)
- [Versioning](#versioning)
- [Basic Usage](#basic-usage)
  - [Galois field arrays](#galois-field-arrays)
    - [Class construction](#class-construction)
    - [Array creation](#array-creation)
    - [Field arithmetic](#field-arithmetic)
    - [Linear algebra](#linear-algebra)
    - [Numpy ufunc methods](#numpy-ufunc-methods)
  - [Polynomials over Galois fields](#polynomials-over-galois-fields)
  - [BCH codes](#bch-codes)
  - [Reed-Solomon codes](#reed-solomon-codes)
- [Performance](#performance)
  - [Lookup performance](#lookup-performance)
  - [Calculation performance](#calculation-performance)
  - [Linear algebra performance](#linear-algebra-performance)
- [Acknowledgements](#acknowledgements)

## Features

- Supports all Galois fields `GF(p^m)`, even arbitrarily-large fields!
- **Faster** than native numpy! `GF(x) * GF(y)` is faster than `(x * y) % p` for `GF(p)`
- Seamless integration with numpy -- normal numpy functions work on Galois field arrays
- Linear algebra on Galois field matrices using normal `np.linalg` functions
- Polynomials over Galois fields with `galois.Poly`
- Forward error correction codes with `galois.BCH` and `galois.ReedSolomon`

## Roadmap

- Linear feedback shift registers over arbitrary Galois fields
- Number-theoretic transform, DFT over Galois fields
- Elliptic curves over Galois fields
- Cryptographic ciphers using Galois fields (RSA, AES, ECC, etc)
- Group and ring arrays
- GPU support

## Documentation

The documentation for `galois` can be found at https://galois.readthedocs.io/en/stable/. It includes
[installation instructions](https://galois.readthedocs.io/en/stable/pages/installation.html), [basic usage](https://galois.readthedocs.io/en/stable/pages/basic_usage.html),
[tutorials](https://galois.readthedocs.io/en/stable/pages/tutorials.html), a [development guide](https://galois.readthedocs.io/en/stable/pages/development.html), an [API reference](https://galois.readthedocs.io/en/stable/pages/api.html), and [release notes](https://galois.readthedocs.io/en/stable/pages/release_notes.html).

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
<class 'np.ndarray over GF(2^8)'>
```

These classes are subclasses of `galois.FieldArray` (which itself subclasses `np.ndarray`) and are instances of `galois.FieldClass`.

```python
>>> isinstance(GF256, galois.FieldClass)
True

>>> issubclass(GF256, galois.FieldArray)
True

>>> issubclass(GF256, np.ndarray)
True
```

A Galois field array class contains attributes relating to its Galois field and has methods to modify how the field
is calculated or displayed. See the attributes and methods in `galois.FieldClass`.

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

>>> GF = galois.GF(2**100); print(GF.properties)
GF(2^100):
  characteristic: 2
  degree: 100
  order: 1267650600228229401496703205376
  irreducible_poly: Poly(x^100 + x^57 + x^56 + x^55 + x^52 + x^48 + x^47 + x^46 + x^45 + x^44 + x^43 + x^41 + x^37 + x^36 + x^35 + x^34 + x^31 + x^30 + x^27 + x^25 + x^24 + x^22 + x^20 + x^19 + x^16 + x^15 + x^11 + x^9 + x^8 + x^6 + x^5 + x^3 + 1, GF(2))
  is_primitive_poly: True
  primitive_element: GF(2, order=2^100)
```

#### Array creation

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

- `FieldArray.Zeros`, `FieldArray.Ones`, `FieldArray.Identity`, `FieldArray.Range`, `FieldArray.Random`, `FieldArray.Elements`

Galois field elements can either be displayed using their integer representation, polynomial representation, or
power representation. Calling `FieldClass.display` will change the element representation. If called as a context
manager, the display mode will only be temporarily changed.

```python
>>> x = GF256(["y**6 + 1", 0, 2, "1", "y**5 + y**4 + y"]); x
GF([65,  0,  2,  1, 50], order=2^8)

# Set the display mode to represent GF(2^8) field elements as polynomials over GF(2) with degree less than 8
>>> GF256.display("poly");

>>> x
GF([α^6 + 1, 0, α, 1, α^5 + α^4 + α], order=2^8)

# Temporarily set the display mode to represent GF(2^8) field elements as powers of the primitive element
>>> with GF256.display("power"):
...     print(x)

GF([α^191, 0, α, 1, α^194], order=2^8)

# Resets the display mode to the integer representation
>>> GF256.display();
```

#### Field arithmetic

Galois field arrays are treated like any other numpy array. Array arithmetic is performed using python operators or numpy
functions.

In the list below, `GF` is a Galois field array class created by `GF = galois.GF(p**m)`, `x` and `y` are `GF` arrays, and `z` is an
integer `np.ndarray`. All arithmetic operations follow normal numpy [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) rules.

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

The `galois` package intercepts relevant calls to numpy's linear algebra functions and performs the specified
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

Galois field arrays also contain matrix decomposition routines not included in numpy. These include:

- `FieldArray.row_reduce`, `FieldArray.lu_decompose`, `FieldArray.lup_decompose`

#### Numpy ufunc methods

Galois field arrays support [numpy ufunc methods](https://numpy.org/devdocs/reference/ufuncs.html#methods). This allows the user to apply a ufunc in a unique way across the target
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

- `Poly.Zero`, `Poly.One`, `Poly.Identity`, `Poly.Random`, `Poly.Integer`, `Poly.String`, `Poly.Degrees`, `Poly.Roots`

```python
# Construct a polynomial by specifying its roots
>>> q = galois.Poly.Roots([155, 37], field=GF256); q
Poly(x^2 + 190x + 71, GF(2^8))

>>> q.roots()
GF([ 37, 155], order=2^8)
```

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

### BCH codes

See full [documentation](https://galois.readthedocs.io/en/stable/pages/galois.html#forward-error-correcting-codes).

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: bch = galois.BCH(15, 7)

# Messages can be either vectors or matrices of np.ndarray or galois.FieldArray (galois.GF2 in this case)
In [4]: M = np.random.randint(0, 2, (5,bch.k)); M
Out[4]:
array([[1, 0, 0, 0, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0, 1, 0],
       [1, 1, 0, 0, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1]])

In [5]: C = bch.encode(M); C
Out[5]:
array([[1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
       [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
       [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
       [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
       [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]])

# Corrupt the first bit in each codeword
In [6]: C[:,0] ^= 1; C
Out[6]:
array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
       [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
       [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]])

In [7]: bch.decode(C)
Out[7]:
array([[1, 0, 0, 0, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1],
       [1, 0, 0, 0, 0, 1, 0],
       [1, 1, 0, 0, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 1]])
```

### Reed-Solomon codes

See full [documentation](https://galois.readthedocs.io/en/stable/pages/galois.html#forward-error-correcting-codes).

```python
In [1]: import numpy as np

In [2]: import galois

In [3]: rs = galois.ReedSolomon(15, 9)

In [4]: (rs.n, rs.k, rs.t)
Out[4]: (15, 9, 3)

In [5]: GF = rs.field; GF
Out[5]: <class 'numpy.ndarray over GF(2^4)'>

# Messages can be either vectors or matrices of np.ndarray or galois.FieldArray
In [6]: M = GF.Random((5,rs.k)); M
Out[6]:
GF([[ 0, 11, 13,  7,  9,  9,  3,  2, 12],
    [ 0,  8, 15, 10, 13,  2,  6,  2,  6],
    [ 1,  9, 13,  1, 13,  2,  6,  4, 12],
    [ 5, 14, 11, 10,  9, 15,  5,  0,  0],
    [ 6,  1,  4,  9,  9,  3, 14, 11, 13]], order=2^4)

In [7]: C = rs.encode(M); C
Out[7]:
GF([[ 0, 11, 13,  7,  9,  9,  3,  2, 12,  6,  3, 13,  0,  8,  4],
    [ 0,  8, 15, 10, 13,  2,  6,  2,  6,  1, 15,  8, 14,  0, 15],
    [ 1,  9, 13,  1, 13,  2,  6,  4, 12,  8, 11,  7,  1,  5, 13],
    [ 5, 14, 11, 10,  9, 15,  5,  0,  0,  1,  8, 13, 12, 13,  3],
    [ 6,  1,  4,  9,  9,  3, 14, 11, 13, 10,  0, 12,  3,  0,  1]],
   order=2^4)

# Corrupt the first symbol in each codeword
In [8]: C[:,0] += GF(13); C
Out[8]:
GF([[13, 11, 13,  7,  9,  9,  3,  2, 12,  6,  3, 13,  0,  8,  4],
    [13,  8, 15, 10, 13,  2,  6,  2,  6,  1, 15,  8, 14,  0, 15],
    [12,  9, 13,  1, 13,  2,  6,  4, 12,  8, 11,  7,  1,  5, 13],
    [ 8, 14, 11, 10,  9, 15,  5,  0,  0,  1,  8, 13, 12, 13,  3],
    [11,  1,  4,  9,  9,  3, 14, 11, 13, 10,  0, 12,  3,  0,  1]],
   order=2^4)

In [9]: rs.decode(C)
Out[9]:
GF([[ 0, 11, 13,  7,  9,  9,  3,  2, 12],
    [ 0,  8, 15, 10, 13,  2,  6,  2,  6],
    [ 1,  9, 13,  1, 13,  2,  6,  4, 12],
    [ 5, 14, 11, 10,  9, 15,  5,  0,  0],
    [ 6,  1,  4,  9,  9,  3, 14, 11, 13]], order=2^4)
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
array and convert its dtype with `astype()`. For example, for small array sizes numpy is faster than
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

However, for large N `galois` is strictly faster than numpy.

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
than lookup tables. Even in these cases, `galois` is faster than numpy!

Here is an example multiplying two arrays in `GF(2097169)` using numpy and `galois` with
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

And again, the runtime comparison with numpy improves with large N because the time of viewing
and type converting the output is small compared to the computation time. `galois` achieves better
performance than numpy because the multiplication and modulo operations are compiled together into
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
comparable to the native numpy implementation (using BLAS/LAPACK).

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

For extension fields `GF(p^m)`, the performance of `galois` is close to native numpy linear algebra
(about 10x slower). However, for extension fields, each multiplication operation is equivalently
a convolution (polynomial multiplication) of two `m`-length arrays and polynomial remainder division with the
irreducible polynomial. So it's not an apples-to-apples comparison.

Below is a comparison of `galois` computing the correct matrix multiplication over `GF(2^8)` and numpy
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

This package heavily relies on [Numba](https://numba.pydata.org/) and its just-in-time compiler for performance.
We use Frank Luebeck's [compilation of Conway polynomials](http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html)
for computing primitive polynomials for extension fields. We utilize [SageMath](https://www.sagemath.org/) for generating test vectors.
