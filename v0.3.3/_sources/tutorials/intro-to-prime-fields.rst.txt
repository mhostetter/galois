Intro to Prime Fields
=====================

A Galois field is a finite field named in honor of `Évariste Galois <https://en.wikipedia.org/wiki/%C3%89variste_Galois>`_,
one of the fathers of group theory. A *field* is a set that is closed under addition, subtraction, multiplication, and division.
To be *closed* under an operation means that performing the operation on any two elements of the set will result in another
element from the set. A *finite field* is a field with a finite set of elements.

.. quote:: Évariste Galois

   Ask Jacobi or Gauss publicly to give their opinion, not as to the truth, but as to the importance of these theorems.
   Later there will be, I hope, some people who will find it to their advantage to decipher all this mess.

   *- May 29, 1832 (two days before his death)*

Galois proved that finite fields exist only when their *order* (or size of the set) is a prime power :math:`p^m`. Accordingly,
finite fields can be broken into two categories: prime fields :math:`\mathrm{GF}(p)` and extension fields :math:`\mathrm{GF}(p^m)`.
This tutorial will focus on prime fields.

Prime field
-----------

In this tutorial, we will consider the prime field :math:`\mathrm{GF}(7)`. Using the :obj:`galois` library, the :obj:`~galois.FieldArray`
subclass `GF7` is created using the class factory :func:`~galois.GF`.

.. ipython-with-reprs:: int,power
   :name: GF7

   GF7 = galois.GF(7)
   print(GF7.properties)

.. info::

   In this tutorial, we suggest using the integer representation to display the elements. However, sometimes it is useful to view elements
   in their power representation :math:`\{0, 1, \alpha, \alpha^2, \dots, \alpha^{p^m - 2}\}`. Switch the display between these two
   representations using the tabbed sections. Note, the polynomial representation is not shown because it is identical to the integer
   representation for prime fields.

   See :doc:`/basic-usage/element-representation` for more details.

Elements
--------

The elements of the finite field :math:`\mathrm{GF}(p)` are naturally represented as the integers
:math:`\{0, 1, \dots, p - 1\}`.

The elements of the finite field are retrieved in a 1-D array using the :func:`~galois.FieldArray.Elements` classmethod.

.. ipython-with-reprs:: int,power
   :name: GF7

   GF7.elements

Arithmetic
----------

Addition, subtraction, and multiplication in :math:`\mathrm{GF}(p)` is equivalent to integer addition, subtraction,
and multiplication reduced modulo :math:`p`. Mathematically speaking, this is the integer ring :math:`\mathbb{Z} / p\mathbb{Z}`.

In this tutorial, consider two field elements :math:`a = 3` and :math:`b = 5`. We will use :obj:`galois` to perform explicit modular
integer arithmetic and then prime field arithmetic.

Here are :math:`a` and :math:`b` represented as Python integers.

.. ipython:: python

   a_int = 3
   b_int = 5
   p = GF7.characteristic; p

Here are :math:`a` and :math:`b` represented as prime field elements. See :doc:`/basic-usage/array-creation` for more details.

.. ipython-with-reprs:: int,power
   :name: GF7

   a = GF7(3); a
   b = GF7(5); b


Addition
........

We can see that :math:`3 + 5 \equiv 1\ (\textrm{mod}\ 7)`. So accordingly, :math:`3 + 5 = 1` in :math:`\mathrm{GF}(7)`.

.. ipython-with-reprs:: int,power
   :name: GF7

   (a_int + b_int) % p
   a + b


The :obj:`galois` library includes the ability to display the arithmetic tables for any finite field. The table is only readable
for small fields, but nonetheless the capability is provided. Select a few computations at random and convince yourself the
answers are correct.

.. ipython-with-reprs:: int,power
   :name: GF7

   print(GF7.arithmetic_table("+"))

Subtraction
...........

As with addition, we can see that :math:`3 - 5 \equiv 5\ (\textrm{mod}\ 7)`. So accordingly, :math:`3 - 5 = 5` in :math:`\mathrm{GF}(7)`.

.. ipython-with-reprs:: int,power
   :name: GF7

   (a_int - b_int) % p
   a - b

Here is the subtraction table for completeness.

.. ipython-with-reprs:: int,power
   :name: GF7

   print(GF7.arithmetic_table("-"))

Multiplication
..............

Similarly, we can see that :math:`3 \cdot 5 \equiv 1\ (\textrm{mod}\ 7)`. So accordingly, :math:`3 \cdot 5 = 1`
in :math:`\mathrm{GF}(7)`.

.. ipython-with-reprs:: int,power
   :name: GF7

   (a_int * b_int) % p
   a * b

Here is the multiplication table for completeness.

.. ipython-with-reprs:: int,power
   :name: GF7

   print(GF7.arithmetic_table("*"))

Multiplicative inverse
......................

Division in :math:`\mathrm{GF}(p)` is a little more difficult. Division can't be as simple as taking :math:`a / b\ (\textrm{mod}\ p)` because
many integer divisions do not result in integers! The division :math:`a / b` can be reformulated into :math:`a b^{-1}`, where :math:`b^{-1}`
is the multiplicative inverse of :math:`b`. Let's first learn the multiplicative inverse before returning to division.

`Euclid <https://en.wikipedia.org/wiki/Euclid>`_ discovered an efficient algorithm to solve the `Bézout Identity <https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity>`_,
which is used to find the multiplicative inverse. It is now called the `Extended Euclidean Algorithm <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#:~:text=Extended%20Euclidean%20algorithm%20also%20refers,a%20and%20b%20are%20coprime.>`_.
Given two integers :math:`x` and :math:`y`, the Extended Euclidean Algorithm finds the integers :math:`s` and :math:`t` such that
:math:`xs + yt = \textrm{gcd}(x, y)`. This algorithm is implemented in :func:`~galois.egcd`.

If :math:`x = 5` is a field element of :math:`\mathrm{GF}(7)` and :math:`y = 7` is the prime characteristic, then :math:`s = x^{-1}`
in :math:`\mathrm{GF}(7)`. Note, the GCD will always be 1 because :math:`y` is prime.

.. ipython:: python

   # Returns (gcd, s, t)
   galois.egcd(b_int, p)

The :obj:`galois` library uses the Extended Euclidean Algorithm to compute multiplicative inverses (and division) in prime fields.
The inverse of 5 in :math:`\mathrm{GF}(7)` can be easily computed in the following way.

.. ipython-with-reprs:: int,power
   :name: GF7

   b ** -1
   np.reciprocal(b)

Division
........

Now let's return to division in finite fields. As mentioned earlier, :math:`a / b` is equivalent to :math:`a b^{-1}`, and we have
already learned multiplication and multiplicative inversion in finite fields.

To compute :math:`3 / 5` in :math:`\mathrm{GF}(7)`, we can equivalently compute :math:`3 \cdot 5^{-1}` in :math:`\mathrm{GF}(7)`.

.. ipython-with-reprs:: int,power
   :name: GF7

   _, b_inv_int, _ = galois.egcd(b_int, p)
   (a_int * b_inv_int) % p
   a * b**-1
   a / b

Here is the division table for completeness. Notice that division is not defined for :math:`y = 0`.

.. ipython-with-reprs:: int,power
   :name: GF7

   print(GF7.arithmetic_table("/"))

Primitive elements
------------------

A property of finite fields is that some elements produce the non-zero elements of the field by their powers.

A *primitive element* :math:`g` of :math:`\mathrm{GF}(p)` is an element such that :math:`\mathrm{GF}(p) = \{0, 1, g, g^2, \dots, g^{p - 2}\}`.
The non-zero elements :math:`\{1, g, g^2, \dots, g^{p - 2}\}` form the cyclic multiplicative group :math:`\mathrm{GF}(p)^{\times}`.
A primitive element has multiplicative order :math:`\textrm{ord}(g) = p - 1`.

In prime fields :math:`\mathrm{GF}(p)`, the generators or primitive elements of :math:`\mathrm{GF}(p)` are *primitive roots mod p*.

Primitive roots mod :math:`p`
.............................

An integer :math:`g` is a *primitive root mod p* if every number coprime to :math:`p` can be represented as a power of :math:`g`
mod :math:`p`. Namely, every :math:`a` coprime to :math:`p` can be represented as :math:`g^k \equiv a\ (\textrm{mod}\ p)` for some :math:`k`.
In prime fields, since :math:`p` is prime, every integer :math:`1 \le a < p` is coprime to :math:`p`.

Finding primitive roots mod :math:`p` is implemented in :func:`~galois.primitive_root` and :func:`~galois.primitive_roots`.

.. ipython:: python

   galois.primitive_root(7)

A primitive element
...................

In :obj:`galois`, a primitive element of a finite field is provided by the :obj:`~galois.FieldArray.primitive_element`
class property.

.. ipython-with-reprs:: int,power
   :name: GF7

   print(GF7.properties)
   g = GF7.primitive_element; g

The :obj:`galois` package allows you to easily display all powers of an element and their equivalent polynomial, vector, and integer
representations using :func:`~galois.FieldArray.repr_table`. Let's ignore the polynomial and vector representations for now.
They will become useful for extension fields.

Here is the representation table using the default generator :math:`g = 3`. Notice its multiplicative order is :math:`p - 1`.

.. ipython:: python

   g.multiplicative_order()
   print(GF7.repr_table())

Other primitive elements
........................

There are multiple primitive elements of any finite field. All primitive elements are provided in the
:obj:`~galois.FieldArray.primitive_elements` class property.

.. ipython-with-reprs:: int,power
   :name: GF7

   list(galois.primitive_roots(7))
   GF7.primitive_elements
   g = GF7(5); g

This means that 3 and 5 generate the multiplicative group :math:`\mathrm{GF}(7)^\times`.
We can examine this by viewing the representation table using different generators.

Here is the representation table using a different generator :math:`g = 5`. Notice it also has
multiplicative order :math:`p- 1`.

.. ipython:: python

   g.multiplicative_order()
   print(GF7.repr_table(g))

Non-primitive elements
......................

All other elements of the field cannot generate the multiplicative group. They have multiplicative
orders less than :math:`p - 1`.

For example, the element :math:`e = 2` is not a primitive element.

.. ipython-with-reprs:: int,power
   :name: GF7

   e = GF7(2); e

It has :math:`\textrm{ord}(e) = 3`. Notice elements 3, 5, and 6 are not represented by the powers of :math:`e`.

.. ipython:: python

   e.multiplicative_order()
   print(GF7.repr_table(e))
