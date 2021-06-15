Basic Usage
===========

The main idea of the :obj:`galois` package can be summarized as follows. The user creates a "Galois field array class" using `GF = galois.GF(p**m)`.
A Galois field array class `GF` is a subclass of :obj:`numpy.ndarray` and its constructor `x = GF(array_like)` mimics
the call signature of :func:`numpy.array`. A Galois field array `x` is operated on like any other numpy array, but all
arithmetic is performed in :math:`\mathrm{GF}(p^m)` not :math:`\mathbb{Z}` or :math:`\mathbb{R}`.

Internally, the Galois field arithmetic is implemented by replacing `numpy ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html>`_.
The new ufuncs are written in python and then `just-in-time compiled <https://numba.pydata.org/numba-doc/dev/user/vectorize.html>`_ with
`numba <https://numba.pydata.org/>`_. The ufuncs can be configured to use either lookup tables (for speed) or explicit
calculation (for memory savings). Numba also provides the ability to `"target" <https://numba.readthedocs.io/en/stable/user/vectorize.html?highlight=target>`_
the JIT-compiled ufuncs for CPUs or GPUs.

In addition to normal array arithmetic, :obj:`galois` also supports linear algebra (with :obj:`numpy.linalg` functions) and polynomials over Galois fields
(with the :obj:`galois.Poly` class).

Class construction
------------------

Galois field array classes are created using the :func:`galois.GF` class factory function.

.. ipython:: python

   import numpy as np
   import galois

   GF256 = galois.GF(2**8)
   print(GF256)

These classes are subclasses of :obj:`galois.FieldArray` (which itself subclasses :obj:`numpy.ndarray`) and have :obj:`galois.FieldClass` as their metaclass.

.. ipython:: python

   issubclass(GF256, np.ndarray)
   issubclass(GF256, galois.FieldArray)
   isinstance(GF256, galois.FieldClass)

A Galois field array class contains attributes relating to its Galois field and methods to modify how the field
is calculated or displayed. See the attributes and methods in :obj:`galois.FieldClass`.

.. ipython:: python

   # Summarizes some properties of the Galois field
   print(GF256.properties)

   # Access each attribute individually
   GF256.irreducible_poly

The :obj:`galois` package even supports arbitrarily-large fields! This is accomplished by using numpy arrays
with `dtype=object` and pure-python ufuncs. This comes at a performance penalty compared to smaller fields
which use numpy integer dtypes (e.g., :obj:`numpy.uint32`) and have compiled ufuncs.

.. ipython:: python

   GF = galois.GF(36893488147419103183); print(GF.properties)
   GF = galois.GF(2**100); print(GF.properties)

Array creation
--------------

Galois field arrays can be created from existing numpy arrays.

.. ipython:: python

   # Represents an existing numpy array
   array = np.random.randint(0, GF256.order, 10, dtype=int); array

   # Explicit Galois field array creation (a copy is performed)
   GF256(array)

   # Or view an existing numpy array as a Galois field array (no copy is performed)
   array.view(GF256)

Or they can be created from "array-like" objects. These include strings representing a Galois field element
as a polynomial over its prime subfield.

.. ipython:: python

   # Arrays can be specified as iterables of iterables
   GF256([[217, 130, 42], [74, 208, 113]])

   # You can mix-and-match polynomial strings and integers
   GF256(["x^6 + 1", 2, "1", "x^5 + x^4 + x"])

   # Single field elements are 0-dimensional arrays
   GF256("x^6 + x^4 + 1")

Galois field arrays also have constructor class methods for convenience. They include:

- :func:`galois.FieldArray.Zeros`, :func:`galois.FieldArray.Ones`, :func:`galois.FieldArray.Identity`, :func:`galois.FieldArray.Range`, :func:`galois.FieldArray.Random`, :func:`galois.FieldArray.Elements`

Galois field elements can either be displayed using their integer representation, polynomial representation, or
power representation. Calling :func:`galois.FieldClass.display` will change the element representation. If called as a context
manager, the display mode will only be temporarily changed.

.. ipython:: python

   x = GF256(["y**6 + 1", 0, 2, "1", "y**5 + y**4 + y"]); x

   # Set the display mode to represent GF(2^8) field elements as polynomials over GF(2) with degree less than 8
   GF256.display("poly");
   x

   # Temporarily set the display mode to represent GF(2^8) field elements as powers of the primitive element
   with GF256.display("power"):
      print(x)

   # Resets the display mode to the integer representation
   GF256.display();

Field arithmetic
----------------

Galois field arrays are treated like any other numpy array. Array arithmetic is performed using python operators or numpy
functions.

In the list below, `GF` is a Galois field array class created by `GF = galois.GF(p**m)`, `x` and `y` are `GF` arrays, and `z` is an
integer `numpy.ndarray`. All arithmetic operations follow normal numpy `broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ rules.

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

.. ipython:: python

   x = GF256.Random((2,5)); x
   y = GF256.Random(5); y
   # y is broadcast over the last dimension of x
   x + y

Linear algebra
--------------

The :obj:`galois` package intercepts relevant calls to numpy's linear algebra functions and performs the specified
operation in :math:`\mathrm{GF}(p^m)` not in :math:`\mathbb{R}`. Some of these functions include:

- :func:`np.dot`, :func:`np.vdot`, :func:`np.inner`, :func:`np.outer`, :func:`np.matmul`, :func:`np.linalg.matrix_power`
- :func:`np.linalg.det`, :func:`np.linalg.matrix_rank`, :func:`np.trace`
- :func:`np.linalg.solve`, :func:`np.linalg.inv`

.. ipython:: python

   A = GF256.Random((3,3)); A
   # Ensure A is invertible
   while np.linalg.matrix_rank(A) < 3:
      A = GF256.Random((3,3)); A
   b = GF256.Random(3); b
   x = np.linalg.solve(A, b); x
   np.array_equal(A @ x, b)

Galois field arrays also contain matrix decomposition routines not included in numpy. These include:

- :func:`galois.FieldArray.row_reduce`, :func:`galois.FieldArray.lu_decompose`, :func:`galois.FieldArray.lup_decompose`

Numpy ufunc methods
-------------------

Galois field arrays support `numpy ufunc methods <https://numpy.org/devdocs/reference/ufuncs.html#methods>`_. This allows the user to apply a ufunc in a unique way across the target
array. The ufunc method signature is `<ufunc>.<method>(*args, **kwargs)`. All arithmetic ufuncs are supported. Below
is a list of their ufunc methods.

- `<method>`: `reduce`, `accumulate`, `reduceat`, `outer`, `at`

.. ipython:: python

   X = GF256.Random((2,5)); X
   np.multiply.reduce(X, axis=0)

.. ipython:: python

   x = GF256.Random(5); x
   y = GF256.Random(5); y
   np.multiply.outer(x, y)

Numpy functions
---------------

Many other relevant numpy functions are supported on Galois field arrays. These include:

- :func:`np.copy`, :func:`np.concatenate`, :func:`np.insert`, :func:`np.reshape`

Polynomial construction
-----------------------

The :obj:`galois` package supports polynomials over Galois fields with the :obj:`galois.Poly` class. :obj:`galois.Poly`
does not subclass :obj:`numpy.ndarray` but instead contains a :obj:`galois.FieldArray` of coefficients as an attribute
(implements the "has-a", not "is-a", architecture).

Polynomials can be created by specifying the polynomial coefficients as either a :obj:`galois.FieldArray` or an "array-like"
object with the `field` keyword argument.

.. ipython:: python

   p = galois.Poly([172, 22, 0, 0, 225], field=GF256); p

   coeffs = GF256([33, 17, 0, 225]); coeffs
   p = galois.Poly(coeffs, order="asc"); p

Polynomials over Galois fields can also display the field elements as polynomials over their prime subfields.
This can be quite confusing to read, so be warned!

.. ipython:: python

   print(p)
   with GF256.display("poly"):
      print(p)

Polynomials can also be created using a number of constructor class methods. They include:

- :func:`galois.Poly.Zero`, :func:`galois.Poly.One`, :func:`galois.Poly.Identity`, :func:`galois.Poly.Random`, :func:`galois.Poly.Integer`, :func:`galois.Poly.String`, :func:`galois.Poly.Degrees`, :func:`galois.Poly.Roots`

.. ipython:: python

   # Construct a polynomial by specifying its roots
   q = galois.Poly.Roots([155, 37], field=GF256); q
   q.roots()

Polynomial arithmetic
---------------------

Polynomial arithmetic is performed using python operators.

.. ipython:: python

   p
   q
   p + q
   divmod(p, q)
   p ** 2

Polynomials over Galois fields can be evaluated at scalars or arrays of field elements.

.. ipython:: python

   p = galois.Poly.Degrees([4, 3, 0], [172, 22, 225], field=GF256); p

   # Evaluate the polynomial at a single value
   p(1)

   x = GF256.Random((2,5)); x

   # Evaluate the polynomial at an array of values
   p(x)

Polynomials can also be evaluated in superfields. For example, evaluating a Galois field's irreducible
polynomial at one of its elements.

.. ipython:: python

   # Notice the irreducible polynomial is over GF(2), not GF(2^8)
   p = GF256.irreducible_poly; p
   GF256.is_primitive_poly

   # Notice the primitive element is in GF(2^8)
   alpha = GF256.primitive_element; alpha

   # Since p(x) is a primitive polynomial, alpha is one of its roots
   p(alpha, field=GF256)
