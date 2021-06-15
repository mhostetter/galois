Intro to Galois Fields: Prime Fields
====================================

A Galois field is a finite field named in honor of Évariste Galois, one of the fathers of group theory. A *field*
is a set that is closed under addition, subtraction, multiplication, and division. To be *closed* under an operation
means that performing the operation on any two elements of the set will result in a third element from the set. A *finite
field* is a field with a finite set.

Galois proved that finite fields exist only when their *order* (or size of the set) is a prime power :math:`p^m`. Accordingly,
finite fields can be broken into two categories: prime fields :math:`\mathrm{GF}(p)` and extension fields :math:`\mathrm{GF}(p^m)`.
This tutorial will focus on prime fields.

Elements
--------

The elements of the Galois field :math:`\mathrm{GF}(p)` are naturally represented as the integers
:math:`\{0, 1, \dots, p - 1\}`.

Using the :obj:`galois` package, a Galois field array class is created using the class factory :func:`galois.GF`.

.. ipython:: python

   GF7 = galois.GF(7); GF7
   print(GF7.properties)

The elements of the Galois field can be represented as a 1-dimensional array using the :func:`galois.FieldArray.Elements` method.

.. ipython:: python

   GF7.Elements()

This array should be read as "a Galois field array [0, 1, 2, 3, 4, 5, 6] over the finite field with order 7".

Arithmetic mod p
----------------

Addition, subtraction, and multiplication in :math:`\mathrm{GF}(p)` is equivalent to integer addition, subtraction,
and multiplication reduced modulo :math:`p`. Mathematically speaking, this is the ring of integers mod :math:`p`, :math:`\mathbb{Z}/p\mathbb{Z}`.

With :obj:`galois`, we can represent a single Galois field element using `GF7(int)`. For example, `GF7(3)` to represent the field element :math:`3`.
We can see that :math:`3 + 5 \equiv 1\ (\textrm{mod}\ 7)`, so accordingly :math:`3 + 5 = 1` in :math:`\mathrm{GF}(7)`. The same
can be shown for subtraction and multiplication.

.. ipython:: python

   GF7(3) + GF7(5)
   GF7(3) - GF7(5)
   GF7(3) * GF7(5)

The power of :obj:`galois`, however, is array arithmetic not scalar arithmetic. Random arrays over :math:`\mathrm{GF}(7)`
can be created using :func:`galois.FieldArray.Random`. Normal binary operators work on Galois field arrays just like
numpy arrays.

.. ipython:: python

   x = GF7.Random(10); x
   y = GF7.Random(10); y
   x + y
   x - y
   x * y

The :obj:`galois` package includes the ability to display the arithmetic tables for a given finite field. The table is only readable
for small fields, but nonetheless the capability is provided. Select a few computations at random and convince yourself the
answers are correct.

.. ipython:: python

   print(GF7.arithmetic_table("+"))
   print(GF7.arithmetic_table("-"))
   print(GF7.arithmetic_table("*"))

Division in :math:`\mathrm{GF}(p)` is a little more difficult. Division can't be as simple as taking :math:`x / y\ (\textrm{mod}\ p)` because
many integer divisions do not result in integers. The division of :math:`x / y = z` can be reformulated as the question "what :math:`z` multiplied by :math:`y`
results in :math:`x`?". This is an equivalent problem to "what :math:`z` multiplied by :math:`y` results in :math:`1`?", where :math:`z` is the
multiplicative inverse of :math:`y`.

To find the multiplicative inverse of :math:`y`, one can simply perform trial multiplication until the result of :math:`1` is found.
For instance, suppose :math:`y = 4` in :math:`\mathrm{GF}(7)`. We can multiply :math:`4` by every element in the field until the product
is :math:`1` and we'll find that :math:`4^{-1} = 2` in :math:`\mathrm{GF}(7)`, namely :math:`2 * 4 = 1` in :math:`\mathrm{GF}(7)`.

.. ipython:: python

   y = GF7(4); y
   # Hypothesize each element from GF(7)
   guesses = GF7.Elements(); guesses
   results = y * guesses; results
   y_inv = guesses[np.where(results == 1)[0][0]]; y_inv

This algorithm is terribly inefficient for large fields, however. Fortunately, Euclid came up with an efficient algorithm, now called the Extended Eulcidean
Algorithm. Given two integers :math:`a` and :math:`b`, the Extended Euclidean Algorithm finds the integers :math:`x` and :math:`y` such that
:math:`xa + yb = \textrm{gcd}(a, b)`. This algorithm is implemented in :func:`galois.gcd`.

If :math:`a` is a field element of :math:`\mathrm{GF}(7)` and :math:`b = 7`, then :math:`x = a^{-1}` in :math:`\mathrm{GF}(7)`.
Note, the GCD will always be :math:`1` because :math:`p` is prime.

.. ipython:: python

   galois.gcd(4, 7)

The :obj:`galois` package uses the Extended Euclidean Algorithm to compute multiplicative inverses (and division) in prime fields.
The inverse of :math:`4` in :math:`\mathrm{GF}(7)` can be easily computed in the following way.

.. ipython:: python

   y = GF7(4); y
   np.reciprocal(y)
   y ** -1

With this in mind, the division table for :math:`\mathrm{GF}(7)` can be calculated. Note that division is not defined for :math:`y = 0`.

.. ipython:: python

   print(GF7.arithmetic_table("/"))

Primitive elements
------------------

A property of finite fields is that some elements can produce the entire field by their powers. Namely, a *primitive element*
:math:`g` of :math:`\mathrm{GF}(p)` is an element such that :math:`\mathrm{GF}(p) = \{0, g^0, g^1, \dots, g^{p - 1}\}`. In prime fields
:math:`\mathrm{GF}(p)`, the generators or primitive elements of :math:`\mathrm{GF}(p)` are *primitive roots mod p*.

The integer :math:`g` is a *primitive root mod p* if every number coprime to :math:`p` can be represented as a power of :math:`g`
mod :math:`p`. Namely, every :math:`a` coprime to :math:`p` can be represented as :math:`g^k \equiv a\ (\textrm{mod}\ p)` for some :math:`k`.
In prime fields, since :math:`p` is prime, every integer :math:`1 \le a < p` is coprime to :math:`p`. Finding primitive roots mod :math:`p` is implemented in
:func:`galois.primitive_root` and :func:`galois.primitive_roots`.

.. ipython:: python

   galois.primitive_root(7)

Since :math:`3` is a primitive root mod :math:`7`, the claim is that the elements of :math:`\mathrm{GF}(7)`
can be written as :math:`\mathrm{GF}(7) = \{0, 3^0, 3^1, \dots, 3^6\}`. :math:`0` is a special element. It can
technically be represented as :math:`g^{-\infty}`, however that can't be computed on a computer. For the non-zero elements,
they can easily be calculated as powers of :math:`g`. The set :math:`\{3^0, 3^1, \dots, 3^6\}` forms a cyclic multiplicative group,
namely :math:`\mathrm{GF}(7)^{\times}`.

.. ipython:: python

   g = GF7(3); g
   g ** np.arange(0, GF7.order - 1)

A primitive element of :math:`\mathrm{GF}(p)` can be accessed through :obj:`galois.FieldClass.primitive_element`.

.. ipython:: python

   GF7.primitive_element

The :obj:`galois` package allows you to easily display all powers of an element and their equivalent polynomial, vector, and integer
representations. Let's ignore the polynomial and vector representations for now; they will become useful for extension fields.

.. ipython:: python

   print(GF7.repr_table())

There are multiple primitive elements of a given field. In the case of :math:`\mathrm{GF}(7)`, :math:`3` and :math:`5`
are primitive elements.

.. ipython:: python

   GF7.primitive_elements

.. ipython:: python

   print(GF7.repr_table(GF7(5)))

And it can be seen that every other element of :math:`\mathrm{GF}(7)` is not a generator of the multiplicative group. For instance,
:math:`2` does not generate the multiplicative group :math:`\mathrm{GF}(7)^\times`.

.. ipython:: python

   print(GF7.repr_table(GF7(2)))
