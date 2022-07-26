# ![Galois: A performant NumPy extension for Galois fields and their applications](https://raw.githubusercontent.com/mhostetter/galois/master/logo/galois-heading.png)

[![PyPI version](https://badge.fury.io/py/galois.svg)](https://badge.fury.io/py/galois)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/galois)](https://pypistats.org/packages/galois)
[![Supported Versions](https://img.shields.io/pypi/pyversions/galois.svg)](https://pypi.org/project/galois)
[![Read the Docs](https://img.shields.io/readthedocs/galois)](https://galois.readthedocs.io/en/latest/)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/mhostetter/galois/Test)](https://github.com/mhostetter/galois/actions)
[![Codecov](https://img.shields.io/codecov/c/github/mhostetter/galois)](https://codecov.io/gh/mhostetter/galois)
[![Twitter](https://img.shields.io/twitter/follow/galois_py?label=galois_py&style=flat&logo=twitter)](https://twitter.com/galois_py)

The `galois` library is a Python 3 package that extends NumPy arrays to operate over finite fields.

The user creates a [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) subclass using `GF = galois.GF(p**m)`.
`GF` is a subclass of `np.ndarray` and its constructor `x = GF(array_like)` mimics the signature of `np.array()`. The
[`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) `x` is operated on like any other NumPy array except
all arithmetic is performed in $\mathrm{GF}(p^m)$, not $\mathbb{R}$.

Internally, the finite field arithmetic is implemented by replacing [NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html).
The new ufuncs are written in pure Python and [just-in-time compiled](https://numba.pydata.org/numba-doc/dev/user/vectorize.html) with
[Numba](https://numba.pydata.org/). The ufuncs can be configured to use either lookup tables (for speed) or explicit calculation (for memory savings).

| :warning: Disclaimer    |
|:------------------------|
| The algorithms implemented in the NumPy ufuncs are not constant-time, but were instead designed for performance. As such, the library could be vulnerable to a [side-channel timing attack](https://en.wikipedia.org/wiki/Timing_attack). This library is not intended for production security, but instead for research & development, reverse engineering, cryptanalysis, experimentation, and general education. |

## Features

- Supports all [Galois fields](https://galois.readthedocs.io/en/latest/api/galois.GF/) $\mathrm{GF}(p^m)$, even arbitrarily-large fields!
- [**Faster**](https://galois.readthedocs.io/en/latest/performance/prime-fields/) than native NumPy! `GF(x) * GF(y)` is faster than `(x * y) % p` for $\mathrm{GF}(p)$.
- Seamless integration with NumPy -- normal NumPy functions work on [`FieldArray`s](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/).
- Linear algebra over finite fields using normal `np.linalg` functions.
- Linear transforms over finite fields, such as the FFT with `np.fft.fft()` and the NTT with [`ntt()`](https://galois.readthedocs.io/en/latest/api/galois.ntt/).
- Functions to generate [irreducible](https://galois.readthedocs.io/en/latest/api/#irreducible-polynomials), [primitive](https://galois.readthedocs.io/en/latest/api/#primitive-polynomials), and [Conway](https://galois.readthedocs.io/en/latest/api/galois.conway_poly/) polynomials.
- Univariate polynomials over finite fields with [`Poly`](https://galois.readthedocs.io/en/latest/api/galois.Poly/).
- Forward error correction codes with [`BCH`](https://galois.readthedocs.io/en/latest/api/galois.BCH/) and [`ReedSolomon`](https://galois.readthedocs.io/en/latest/api/galois.ReedSolomon/).
- Fibonacci and Galois linear-feedback shift registers over any finite field with [`FLFSR`](https://galois.readthedocs.io/en/latest/api/galois.FLFSR/) and [`GLFSR`](https://galois.readthedocs.io/en/latest/api/galois.GLFSR/).
- Various [number theoretic functions](https://galois.readthedocs.io/en/latest/api/#number-theory).
- [Integer factorization](https://galois.readthedocs.io/en/latest/api/#factorization) and accompanying algorithms.
- [Prime number generation](https://galois.readthedocs.io/en/latest/api/#prime-number-generation) and [primality testing](https://galois.readthedocs.io/en/latest/api/#primality-tests).

## Roadmap

- Elliptic curves over finite fields
- Galois ring arrays
- GPU support

## Documentation

The documentation for `galois` is located at https://galois.readthedocs.io/en/latest/.

## Getting Started

The [Getting Started](https://galois.readthedocs.io/en/latest/getting-started/) guide is intended to assist the user with installing the
library, creating two example arrays, and performing basic array arithmetic. See [Basic Usage](https://galois.readthedocs.io/en/latest/basic-usage/array-classes/)
for more detailed discussions and examples.

### Install the package

The latest version of `galois` can be installed from [PyPI](https://pypi.org/project/galois/) using `pip`.

```sh
$ python3 -m pip install galois
```

Import the `galois` package in Python.

```python
In [1]: import galois

In [2]: galois.__version__
Out[2]: '0.0.31'
```

### Create a [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) subclass

Next, create a [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) subclass
for the specific finite field you'd like to work in. This is created using the `galois.GF()` class factory. In this example, we are
working in $\mathrm{GF}(3^5)$.

```python
In [3]: GF = galois.GF(3**5)

In [4]: print(GF.properties)
Galois Field:
  name: GF(3^5)
  characteristic: 3
  degree: 5
  order: 243
  irreducible_poly: x^5 + 2x + 1
  is_primitive_poly: True
  primitive_element: x
```

The [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) subclass `GF` is a subclass of
`np.ndarray` that performs all arithmetic in the Galois field $\mathrm{GF}(3^5)$, not in $\mathbb{R}$.

```python
In [5]: issubclass(GF, galois.FieldArray)
Out[5]: True

In [6]: issubclass(GF, np.ndarray)
Out[6]: True
```

See [Array Classes](https://galois.readthedocs.io/en/latest/basic-usage/array-classes/) for more details.

### Create two [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) instances

Next, create a new [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) `x` by passing an
[`ArrayLike`](https://galois.readthedocs.io/en/latest/api/galois.typing.ArrayLike/) object to `GF`'s constructor.

```python
In [7]: x = GF([236, 87, 38, 112]); x
Out[7]: GF([236,  87,  38, 112], order=3^5)
```

The array `x` is an instance of [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) and also
an instance of `np.ndarray`.

```python
In [8]: isinstance(x, galois.FieldArray)
Out[8]: True

In [9]: isinstance(x, np.ndarray)
Out[9]: True
```

Create a second [`FieldArray`](https://galois.readthedocs.io/en/latest/api/galois.FieldArray/) `y` by converting an existing
NumPy array (without copying it) by invoking `.view()`. When finished working in the finite field, view it back as a NumPy array
with `.view(np.ndarray)`.

```python
# y represents an array created elsewhere in the code
In [10]: y = np.array([109, 17, 108, 224]); y
Out[10]: array([109,  17, 108, 224])

In [11]: y = y.view(GF); y
Out[11]: GF([109,  17, 108, 224], order=3^5)
```

See [Array Creation](https://galois.readthedocs.io/en/latest/basic-usage/array-creation/) for more details.

### Change the element representation

The display representation of finite field elements can be set to either the integer (`"int"`), polynomial (`"poly"`),
or power (`"power"`) representation. The default representation is the integer representation since that is natural when
working with integer NumPy arrays.

Set the display mode by passing the `display` keyword argument to `galois.GF()` or by calling the `display()` classmethod.
Choose whichever element representation is most convenient for you.

```python
# The default representation is the integer representation
In [12]: x
Out[12]: GF([236,  87,  38, 112], order=3^5)

In [13]: GF.display("poly"); x
Out[13]: 
GF([2α^4 + 2α^3 + 2α^2 + 2,               α^4 + 2α,
             α^3 + α^2 + 2,      α^4 + α^3 + α + 1], order=3^5)

In [14]: GF.display("power"); x
Out[14]: GF([α^204,  α^16, α^230,  α^34], order=3^5)

# Reset to the integer representation
In [15]: GF.display("int");
```

See [Element Representation](https://galois.readthedocs.io/en/latest/basic-usage/element-representation/) for more details.

### Perform array arithmetic

Once you have two Galois field arrays, nearly any arithmetic operation can be performed using normal NumPy arithmetic.
The traditional [NumPy broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html) apply.

Standard element-wise array arithmetic -- addition, subtraction, multiplication, and division -- are easily preformed.

```python
In [16]: x + y
Out[16]: GF([ 18,  95, 146,   0], order=3^5)

In [17]: x - y
Out[17]: GF([127, 100, 173, 224], order=3^5)

In [18]: x * y
Out[18]: GF([ 21, 241, 179,  82], order=3^5)

In [19]: x / y
Out[19]: GF([ 67,  47, 192,   2], order=3^5)
```

More complicated arithmetic, like square root and logarithm base $\alpha$, are also supported.

```python
In [20]: np.sqrt(x)
Out[20]: GF([ 51, 135,  40,  16], order=3^5)

In [21]: np.log(x)
Out[21]: array([204,  16, 230,  34])
```

See [Array Arithmetic](https://galois.readthedocs.io/en/latest/basic-usage/array-arithmetic/) for more details.

## Acknowledgements

The `galois` library is an extension of, and completely dependent on, [NumPy](https://numpy.org/). It also heavily
relies on [Numba](https://numba.pydata.org/) and the [LLVM just-in-time compiler](https://llvm.org/) for optimizing performance
of the finite field arithmetic.

[Frank Luebeck's compilation](http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html) of Conway polynomials and
[Wolfram's compilation](https://datarepository.wolframcloud.com/resources/Primitive-Polynomials/) of primitive polynomials are used
for efficient polynomial lookup, when possible.

[Sage](https://www.sagemath.org/) is used extensively for generating test vectors for finite field arithmetic and polynomial arithmetic.
[SymPy](https://www.sympy.org/en/index.html) is used to generate some test vectors. [Octave](https://www.gnu.org/software/octave/index)
is used to generate test vectors for forward error correction codes.

This library would not be possible without all of the other libraries mentioned. Thank you to all their developers!

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@software{Hostetter_Galois_2020,
    title = {{Galois: A performant NumPy extension for Galois fields}},
    author = {Hostetter, Matt},
    month = {11},
    year = {2020},
    url = {https://github.com/mhostetter/galois},
}
```

### APA

```
Hostetter, M. (2020). Galois: A performant NumPy extension for Galois fields [Computer software]. https://github.com/mhostetter/galois
```
