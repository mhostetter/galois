Galois field array arithmetic
=============================


Addition, subtraction, multiplication, division
-----------------------------------------------

A finite field is a set that defines the operations addition, subtraction, multiplication, and division. The field
is closed under these operations.

.. ipython:: python

   GF7 = galois.GF_factory(7)
   print(GF7)

   # Create a random GF(7) array with 10 elements
   x = GF7.Random(10); x

   # Create a random GF(7) array with 10 elements, with the lowest element being 1 (used to prevent ZeroDivisionError later on)
   y = GF7.Random(10, low=1); y

   # Addition in the finite field
   x + y

   # Subtraction in the finite field
   x - y

   # Multiplication in the finite field
   x * y

   # Divison in the finite field
   x / y
   x // y


One can easily create the addition, subtraction, multiplication, and divison tables for any field. Here is an
example using :math:`\mathrm{GF}(7)`.

.. ipython:: python

   X, Y = np.meshgrid(GF7.Elements(), GF7.Elements(), indexing="ij")
   X + Y
   X - Y
   X * Y

   X, Y = np.meshgrid(GF7.Elements(), GF7.Elements()[1:], indexing="ij")
   X / Y


Multiple addition
-----------------

A finite field :math:`\mathrm{GF}(p^m)` is a set that is closed under four operations: addition, subtraction, multiplication,
and division. For multiplication, :math:`x * y = z` for :math:`x, y, z \in \mathrm{GF}(p^m)`.

Let's define another notation :math:`x \cdot r = z` for :math:`x, z \in \mathrm{GF}(p^m)` and :math:`r \in \mathbb{Z}`,
which represents :math:`r` additions of :math:`x`, i.e. :math:`x + \dotsb + x = z`. In prime fields :math:`\mathrm{GF}(p)`,
multiplication :math:`*` and "multiple addition" :math:`\cdot` are equivalent. However, in extension fields :math:`\mathrm{GF}(p^m)`
they are not.

.. warning::

   There is a difference between `GF8(6) * GF8(2)` and `GF8(6) * 2`. The former represents the field element "6"
   multiplied by the field element "2" using field multiplication. The latter represents adding the field element
   "6" two times.

   .. ipython:: python

      GF8 = galois.GF_factory(2**3)
      print(GF8)

      a = GF8.Random(10)

      # Calculates a x "2" in the finite field
      a * GF8(2)

      # Calculates a + a
      a * 2


Exponentiation
--------------

.. ipython:: python

   GF7 = galois.GF_factory(7)
   print(GF7)

   x = GF7.Random(10); x

   # Calculates "x" * "x", note 2 is not a field element
   x ** 2


Logarithm
---------

.. ipython:: python

   GF7 = galois.GF_factory(7)
   print(GF7)

   # The primitive element of the field
   GF7.alpha

   x = GF7.Random(10, low=1); x

   # Notice the outputs of log(x) are not field elements, but integers
   e = np.log(x); e

   GF7.alpha**e

   np.all(GF7.alpha**e == x)
