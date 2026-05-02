Intro to Extension Fields
=========================

As discussed in the :doc:`intro-to-prime-fields` tutorial, a finite field is a finite set that is closed under addition, subtraction, multiplication,
and division. Galois proved that finite fields exist only when their *order* (or size of the set) is a prime power $p^m$.

When the order is prime, the arithmetic is *mostly* computed using integer arithmetic modulo $p$. When the order is a prime power, namely
extension fields $\mathrm{GF}(p^m)$, the arithmetic is *mostly* computed using polynomial arithmetic modulo the irreducible
polynomial $f(x)$.

Extension field
---------------

In this tutorial, we will consider the extension field $\mathrm{GF}(3^2)$. Using the :obj:`galois` library, the :obj:`~galois.FieldArray` subclass
`GF9` is created using the class factory :func:`~galois.GF`.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   GF9 = galois.GF(3**2)
   print(GF9.properties)

.. info::

   In this tutorial, we suggest using the polynomial representation to display the elements. Although, it is common to use the default
   integer representation $\{0, 1, \dots, p^m - 1\}$ to display the arrays more compactly. Switch the display between the three
   representations using the tabbed sections.

   See :doc:`/basic-usage/element-representation` for more details.

Elements
--------

The elements of $\mathrm{GF}(p^m)$ are polynomials over $\mathrm{GF}(p)$ with degree less than $m$.
Formally, they are all polynomials $a_{m-1}x^{m-1} + \dots + a_1x^1 +  a_0 \in \mathrm{GF}(p)[x]$. There are
exactly $p^m$ elements.

The elements of the finite field are retrieved in a 1-D array using the :func:`~galois.FieldArray.Elements` classmethod.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   GF9.elements

Irreducible polynomial
----------------------

Every extension field must be defined with respect to an irreducible polynomial $f(x)$. This polynomial defines the
arithmetic of the field.

When creating a :obj:`~galois.FieldArray` subclass in :obj:`galois`, if an irreducible polynomial is not explicitly specified, a default
is chosen. The default is the Conway polynomial $C_{p,m}(x)$, which is irreducible *and* primitive. See :func:`~galois.conway_poly`
for more information.

Notice $f(x)$ is over $\mathrm{GF}(3)$ with degree 2.

.. ipython:: python

   f = GF9.irreducible_poly; f

Also note, when factored, $f(x)$ has no irreducible factors other than itself -- an analogue of a prime number.

.. ipython:: python

   f.is_irreducible()
   f.factors()

Arithmetic
----------

Addition, subtraction, and multiplication in $\mathrm{GF}(p^m)$ with irreducible polynomial $f(x)$ is equivalent to polynomial
addition, subtraction, and multiplication over $\mathrm{GF}(p)$ reduced modulo $f(x)$. Mathematically speaking, this is
the polynomial ring $\mathrm{GF}(p)[x] / f(x)$.

In this tutorial, consider two field elements $a = x + 2$ and $b = x + 1$. We will use :obj:`galois` to perform explicit polynomial
calculations and then extension field arithmetic.

Here are $a$ and $b$ represented using :obj:`~galois.Poly` objects.

.. ipython:: python

   GF3 = galois.GF(3)
   a_poly = galois.Poly([1, 2], field=GF3); a_poly
   b_poly = galois.Poly([1, 1], field=GF3); b_poly

Here are $a$ and $b$ represented as extension field elements. Extension field elements can be specified as integers
or polynomial strings. See :doc:`/basic-usage/array-creation` for more details.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   a = GF9("x + 2"); a
   b = GF9("x + 1"); b

Addition
........

In polynomial addition, the polynomial coefficients add degree-wise in $\mathrm{GF}(p)$. Addition of polynomials with degree
less than $m$ will never result in a polynomial of degree $m$ or greater. Therefore, it is unnecessary to reduce modulo
the degree-$m$ polynomial $f(x)$, since the quotient will always be zero.

We can see that $a + b = (1 + 1)x + (2 + 1) = 2x$.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   a_poly + b_poly
   a + b

The :obj:`galois` library includes the ability to display the arithmetic tables for any finite field. The table is only readable
for small fields, but nonetheless the capability is provided. Select a few computations at random and convince yourself the
answers are correct.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   print(GF9.arithmetic_table("+"))

Subtraction
...........

Subtraction, like addition, is performed on coefficients degree-wise and will never result in a polynomial with greater degree.

We can see that $a - b = (1 - 1)x + (2 - 1) = 1$.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   a_poly - b_poly
   a - b

Here is the entire subtraction table for completeness.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   print(GF9.arithmetic_table("-"))


Multiplication
..............

Multiplication of polynomials with degree less than $m$, however, will often result in a polynomial of degree $m$
or greater. Therefore, it is necessary to reduce the result modulo $f(x)$.

First compute $ab = (x + 2)(x + 1) = x^2 + 2$. Notice that $x^2 + 2$ has degree 2, but the elements of
$\mathrm{GF}(3^2)$ can have degree at most 1. Therefore, reduction modulo $f(x)$ is required. After remainder
division, we see that $ab\ \equiv x\ \textrm{mod}\ f(x)$.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   # Note the degree is greater than 1
   a_poly * b_poly
   (a_poly * b_poly) % f
   a * b

Here is the entire multiplication table for completeness.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   print(GF9.arithmetic_table("*"))

Multiplicative inverse
......................

As with prime fields, the division $a(x) / b(x)$ is reformulated into $a(x) b(x)^{-1}$. So, first we must compute the multiplicative
inverse $b^{-1}$ before continuing onto division.

The `Extended Euclidean Algorithm <https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm#:~:text=Extended%20Euclidean%20algorithm%20also%20refers,a%20and%20b%20are%20coprime.>`_,
which was used in prime fields on integers, can be used for extension fields on polynomials. Given two polynomials $a(x)$ and
$b(x)$, the Extended Euclidean Algorithm finds the polynomials $s(x)$ and $t(x)$ such that
$a(x)s(x) + b(x)t(x) = \textrm{gcd}(a(x), b(x))$. This algorithm is implemented in :func:`~galois.egcd`.

If $a(x) = x + 1$ is a field element of $\mathrm{GF}(3^2)$ and $b(x) = f(x)$ is the irreducible polynomial, then
$s(x) = a^{-1}$ in $\mathrm{GF}(3^2)$. Note, the GCD will always be 1 because $f(x)$ is irreducible.

.. ipython:: python

   # Returns (gcd, s, t)
   galois.egcd(b_poly, f)

The :obj:`galois` library uses the Extended Euclidean Algorithm to compute multiplicative inverses (and division) in extension fields.
The inverse of $x + 1$ in $\mathrm{GF}(3^2)$ can be easily computed in the following way.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   b ** -1
   np.reciprocal(b)

Division
........

Now let's return to division in finite fields. As mentioned earlier, $a(x) / b(x)$ is equivalent to $a(x) b(x)^{-1}$, and we have
already learned multiplication and multiplicative inversion in finite fields.

Let's compute $a / b = (x + 2)(x + 1)^{-1}$ in $\mathrm{GF}(3^2)$.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   _, b_inv_poly, _ = galois.egcd(b_poly, f)
   (a_poly * b_inv_poly) % f
   a * b**-1
   a / b

Here is the division table for completeness. Notice that division is not defined for $y = 0$.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   print(GF9.arithmetic_table("/"))

Primitive elements
------------------

A property of finite fields is that some elements produce the non-zero elements of the field by their powers.

A *primitive element* $g$ of $\mathrm{GF}(p^m)$ is an element such that $\mathrm{GF}(p^m) = \{0, 1, g, g^2, \dots, g^{p^m - 2}\}$.
The non-zero elements $\{1, g, g^2, \dots, g^{p^m - 2}\}$ form the cyclic multiplicative group $\mathrm{GF}(p^m)^{\times}$.
A primitive element has multiplicative order $\textrm{ord}(g) = p^m - 1$.

A primitive element
...................

In :obj:`galois`, a primitive element of a finite field is provided by the :obj:`~galois.FieldArray.primitive_element`
class property.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   print(GF9.properties)
   g = GF9.primitive_element; g

The :obj:`galois` package allows you to easily display all powers of an element and their equivalent polynomial, vector, and integer
representations using :func:`~galois.FieldArray.repr_table`.

Here is the representation table using the default generator $g = x$. Notice its multiplicative order is $p^m - 1$.

.. ipython:: python

   g.multiplicative_order()
   print(GF9.repr_table())

Other primitive elements
........................

There are multiple primitive elements of any finite field. All primitive elements are provided in the
:obj:`~galois.FieldArray.primitive_elements` class property.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   GF9.primitive_elements
   g = GF9("2x + 1"); g

This means that $x$, $x + 2$, $2x$, and $2x + 1$ all generate the multiplicative
group $\mathrm{GF}(3^2)^\times$. We can examine this by viewing the representation table using
different generators.

Here is the representation table using a different generator $g = 2x + 1$. Notice it also has
multiplicative order $p^m - 1$.

.. ipython:: python

   g.multiplicative_order()
   print(GF9.repr_table(g))

Non-primitive elements
......................

All other elements of the field cannot generate the multiplicative group. They have multiplicative
orders less than $p^m - 1$.

For example, the element $e = x + 1$ is not a primitive element. It has $\textrm{ord}(e) = 4$.
Notice elements $x$, $x + 2$, $2x$, and $2x + 1$ are not represented by the powers of $e$.

.. ipython-with-reprs:: int,poly,power
   :name: GF9

   e = GF9("x + 1"); e

.. ipython:: python

   e.multiplicative_order()
   print(GF9.repr_table(e))
