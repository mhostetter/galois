Galois: A performant numpy extension for Galois fields
======================================================

Installation
------------

The latest version of `galois` can be installed from PyPI via `pip`.

.. code-block::

   pip3 install galois

The the lastest code from `master` can be checked out and installed locally in an "editable" fashion.

.. code-block::

   git clone https://github.com/mhostetter/galois.git
   pip3 install -e galois

API v\ |version|
----------------

.. autosummary::
   :template: module.rst
   :toctree: build/_autosummary

   galois

Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials/constructing_fields.rst
   tutorials/field_arithmetic.rst

Basic Usage
-----------

Construct Galois field array classes using the `GF_factory()` class factory function.

.. ipython:: python

   import numpy as np
   import galois

   GF = galois.GF_factory(31, 1)
   print(GF)
   print(GF.alpha)
   print(GF.prim_poly)

Create arrays from existing `numpy` arrays.

.. ipython:: python

   # Represents an existing numpy array
   array = np.random.randint(0, GF.order, 10, dtype=int); array

   # Explicit Galois field construction
   GF(array)

   # Numpy view casting to a Galois field
   array.view(GF)

Or, create Galois field arrays using alternate constructors.

.. ipython:: python

   x = GF.Random(10); x

   # Construct a random array without zeros to prevent ZeroDivisonError
   y = GF.Random(10, low=1); y

Galois field arrays support traditional numpy array operations

.. ipython:: python

   x + y

   -x

   x * y

   x / y

   # Multiple addition of a Galois field array with any integer
   x * -3  # NOTE: -3 is outside the field

   # Exponentiate a Galois field array with any integer
   y ** -2  # NOTE: 87 is outside the field

   # Log base alpha (the field's primitive element)
   np.log(y)

Galois field arrays support numpy array broadcasting.

.. ipython:: python

   a = GF.Random((2,5)); a
   b = GF.Random(5); b

   a + b

Galois field arrays also support numpy ufunc methods.

.. ipython:: python

   # Valid ufunc methods include "reduce", "accumulate", "reduceat", "outer", "at"
   np.multiply.reduce(a, axis=0)

   np.multiply.outer(x, y)

Construct Galois field polynomials.

.. ipython:: python

   # Construct a polynomial by specifying all the coefficients in descending-degree order
   p = galois.Poly([1, 22, 0, 17, 25], field=GF); p

   # Construct a polynomial by specifying only the non-zero coefficients
   q = galois.Poly.NonZero([4, 14],  [2, 0], field=GF); q

Galois field polynomial arithmetic is similar to numpy array operations.

.. ipython:: python

   p + q
   p // q, p % q
   p ** 2

Galois field polynomials can also be evaluated at constants or arrays.

.. ipython:: python

   p(1)
   p(a)

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
