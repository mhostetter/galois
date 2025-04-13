# ![Galois: A performant NumPy extension for Galois fields and their applications](https://raw.githubusercontent.com/mhostetter/galois/main/logo/galois-heading.png)

<div align=center>
  <a href="https://pypi.org/project/galois"><img src="https://img.shields.io/pypi/v/galois"></a>
  <a href="https://pypi.org/project/galois"><img src="https://img.shields.io/pypi/pyversions/galois"></a>
  <a href="https://pypi.org/project/galois"><img src="https://img.shields.io/pypi/wheel/galois"></a>
  <a href="https://pypistats.org/packages/galois"><img src="https://img.shields.io/pypi/dm/galois"></a>
  <a href="https://pypi.org/project/galois"><img src="https://img.shields.io/pypi/l/galois"></a>
</div>
<div align=center>
  <a href="https://github.com/mhostetter/galois/actions/workflows/lint.yaml"><img src="https://github.com/mhostetter/galois/actions/workflows/lint.yaml/badge.svg?branch=main"></a>
  <a href="https://github.com/mhostetter/galois/actions/workflows/build.yaml"><img src="https://github.com/mhostetter/galois/actions/workflows/build.yaml/badge.svg?branch=main"></a>
  <a href="https://github.com/mhostetter/galois/actions/workflows/test.yaml"><img src="https://github.com/mhostetter/galois/actions/workflows/test.yaml/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/mhostetter/galois"><img src="https://codecov.io/gh/mhostetter/galois/branch/main/graph/badge.svg?token=3FJML79ZUK"></a>
  <a href="https://twitter.com/galois_py"><img src="https://img.shields.io/static/v1?label=follow&message=@galois_py&color=blue&logo=twitter"></a>
</div>

The `galois` library is a Python 3 package that extends NumPy arrays to operate over finite fields.

> Enjoying the library? Give us a :star: on [GitHub](https://github.com/mhostetter/galois)!

The user creates a [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) subclass using `GF = galois.GF(p**m)`.
`GF` is a subclass of `np.ndarray` and its constructor `x = GF(array_like)` mimics the signature of `np.array()`. The
[`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) `x` is operated on like any other NumPy array except
all arithmetic is performed in $\mathrm{GF}(p^m)$, not $\mathbb{R}$.

Internally, the finite field arithmetic is implemented by replacing [NumPy ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html).
The new ufuncs are written in pure Python and [just-in-time compiled](https://numba.pydata.org/numba-doc/dev/user/vectorize.html) with
[Numba](https://numba.pydata.org/). The ufuncs can be configured to use either lookup tables (for speed) or explicit calculation (for memory savings).

> **Warning**
> The algorithms implemented in the NumPy ufuncs are not constant-time, but were instead designed for performance. As such, the library could be vulnerable to a [side-channel timing attack](https://en.wikipedia.org/wiki/Timing_attack). This library is not intended for production security, but instead for research & development, reverse engineering, cryptanalysis, experimentation, and general education.

## Features

- Supports all [Galois fields](https://mhostetter.github.io/galois/latest/api/galois.GF/) $\mathrm{GF}(p^m)$, even arbitrarily large fields!
- [**Faster**](https://mhostetter.github.io/galois/latest/performance/prime-fields/) than native NumPy! `GF(x) * GF(y)` is faster than `(x * y) % p` for $\mathrm{GF}(p)$.
- Seamless integration with NumPy -- normal NumPy functions work on [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/)s.
- Linear algebra over finite fields using normal [`np.linalg`](https://mhostetter.github.io/galois/latest/basic-usage/array-arithmetic/#linear-algebra) functions.
- Linear transforms over finite fields, such as the FFT with [`np.fft.fft()`](https://mhostetter.github.io/galois/latest/basic-usage/array-arithmetic/#advanced-arithmetic) and the NTT with [`ntt()`](https://mhostetter.github.io/galois/latest/api/galois.ntt/).
- Functions to generate [irreducible](https://mhostetter.github.io/galois/latest/api/#irreducible-polynomials), [primitive](https://mhostetter.github.io/galois/latest/api/#primitive-polynomials), and [Conway](https://mhostetter.github.io/galois/latest/api/galois.conway_poly/) polynomials.
- Univariate polynomials over finite fields with [`Poly`](https://mhostetter.github.io/galois/latest/api/galois.Poly/).
- Forward error correction codes with [`BCH`](https://mhostetter.github.io/galois/latest/api/galois.BCH/) and [`ReedSolomon`](https://mhostetter.github.io/galois/latest/api/galois.ReedSolomon/).
- Fibonacci and Galois linear-feedback shift registers over any finite field with [`FLFSR`](https://mhostetter.github.io/galois/latest/api/galois.FLFSR/) and [`GLFSR`](https://mhostetter.github.io/galois/latest/api/galois.GLFSR/).
- Various [number theoretic functions](https://mhostetter.github.io/galois/latest/api/#number-theory).
- [Integer factorization](https://mhostetter.github.io/galois/latest/api/#factorization) and accompanying algorithms.
- [Prime number generation](https://mhostetter.github.io/galois/latest/api/#prime-number-generation) and [primality testing](https://mhostetter.github.io/galois/latest/api/#primality-tests).

## Roadmap

- Elliptic curves over finite fields
- Galois ring arrays
- GPU support

## Documentation

The documentation for `galois` is located at https://mhostetter.github.io/galois/latest/.

## Getting Started

The [Getting Started](https://mhostetter.github.io/galois/latest/getting-started/) guide is intended to assist the user with installing the
library, creating two example arrays, and performing basic array arithmetic. See [Basic Usage](https://mhostetter.github.io/galois/latest/basic-usage/array-classes/)
for more detailed discussions and examples.

### Install the package

The latest version of `galois` can be installed from [PyPI](https://pypi.org/project/galois/) using `pip`.

```console
$ python3 -m pip install galois
```

Import the `galois` package in Python.

```python
In [1]: import galois

In [2]: galois.__version__
Out[2]: '0.4.5'
```

### Create a [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) subclass

Next, create a [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) subclass
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

The [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) subclass `GF` is a subclass of
`np.ndarray` that performs all arithmetic in the Galois field $\mathrm{GF}(3^5)$, not in $\mathbb{R}$.

```python
In [5]: issubclass(GF, galois.FieldArray)
Out[5]: True

In [6]: issubclass(GF, np.ndarray)
Out[6]: True
```

See [Array Classes](https://mhostetter.github.io/galois/latest/basic-usage/array-classes/) for more details.

### Create two [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) instances

Next, create a new [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) `x` by passing an
[`ArrayLike`](https://mhostetter.github.io/galois/latest/api/galois.typing.ArrayLike/) object to `GF`'s constructor.

```python
In [7]: x = GF([236, 87, 38, 112]); x
Out[7]: GF([236,  87,  38, 112], order=3^5)
```

The array `x` is an instance of [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) and also
an instance of `np.ndarray`.

```python
In [8]: isinstance(x, galois.FieldArray)
Out[8]: True

In [9]: isinstance(x, np.ndarray)
Out[9]: True
```

Create a second [`FieldArray`](https://mhostetter.github.io/galois/latest/api/galois.FieldArray/) `y` by converting an existing
NumPy array (without copying it) by invoking `.view()`. When finished working in the finite field, view it back as a NumPy array
with `.view(np.ndarray)`.

```python
# y represents an array created elsewhere in the code
In [10]: y = np.array([109, 17, 108, 224]); y
Out[10]: array([109,  17, 108, 224])

In [11]: y = y.view(GF); y
Out[11]: GF([109,  17, 108, 224], order=3^5)
```

See [Array Creation](https://mhostetter.github.io/galois/latest/basic-usage/array-creation/) for more details.

### Change the element representation

The representation of finite field elements can be set to either the integer (`"int"`), polynomial (`"poly"`),
or power (`"power"`) representation. The default representation is the integer representation since integers are natural when
working with integer NumPy arrays.

Set the element representation by passing the `repr` keyword argument to `galois.GF()` or by calling the `repr()`
classmethod. Choose whichever element representation is most convenient.

```python
# The default is the integer representation
In [12]: x
Out[12]: GF([236,  87,  38, 112], order=3^5)

In [13]: GF.repr("poly"); x
Out[13]:
GF([2α^4 + 2α^3 + 2α^2 + 2,               α^4 + 2α,
             α^3 + α^2 + 2,      α^4 + α^3 + α + 1], order=3^5)

In [14]: GF.repr("power"); x
Out[14]: GF([α^204,  α^16, α^230,  α^34], order=3^5)

# Reset to the integer representation
In [15]: GF.repr("int");
```

See [Element Representation](https://mhostetter.github.io/galois/latest/basic-usage/element-representation/) for more details.

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

See [Array Arithmetic](https://mhostetter.github.io/galois/latest/basic-usage/array-arithmetic/) for more details.

## Acknowledgements

The `galois` library is an extension of, and completely dependent on, [NumPy](https://numpy.org/). It also heavily
relies on [Numba](https://numba.pydata.org/) and the [LLVM just-in-time compiler](https://llvm.org/) for optimizing performance
of the finite field arithmetic.

[Frank Luebeck's compilation](http://www.math.rwth-aachen.de/~Frank.Luebeck/data/ConwayPol/index.html) of Conway polynomials and
[Wolfram's compilation](https://datarepository.wolframcloud.com/resources/Primitive-Polynomials/) of primitive polynomials are used
for efficient polynomial lookup, when possible.

[The Cunningham Book's tables](https://homes.cerias.purdue.edu/~ssw/cun/third/index.html) of prime factorizations, $b^n \pm 1$
for $b \in \{2, 3, 5, 6, 7, 10, 11, 12\}$, are used to generate factorization lookup tables. These lookup tables speed-up the
creation of large finite fields by avoiding the need to factor large integers.

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
