Intro to Galois Fields: Extension Fields
========================================

As discussed in the previous tutorial, a finite field is a finite set that is closed under addition, subtraction, multiplication,
and division. Galois proved that finite fields exist only when their *order* (or size of the set) is a prime power :math:`p^m`.
When the order is prime, the arithmetic can be *mostly* computed using integer arithmetic mod :math:`p`. In the case of prime power
order, namely extension fields :math:`\mathrm{GF}(p^m)`, the finite field arithmetic is computed using polynomials over :math:`\mathrm{GF}(p)`
with degree less than :math:`m`.

Elements
--------

The elements of the Galois field :math:`\mathrm{GF}(p^m)` can be thought of as the integers
:math:`\{0, 1, \dots, p^m - 1\}`, although their arithmetic doesn't obey integer arithmetic. A more common interpretation
is to view the elements of :math:`\mathrm{GF}(p^m)` as polynomials over :math:`\mathrm{GF}(p)` with degree less than :math:`m`,
for instance :math:`a_{m-1}x^{m-1} + a_{m-2}x^{m-2} + \dots + a_1x^1 +  a_0 \in \mathrm{GF}(p)[x]`.

For example, consider the finite field :math:`\mathrm{GF}(3^2)`. The order of the field is 9, so we know there are 9 elements.
The only question is what to call each element and how to represent them.

.. ipython:: python

   GF9 = galois.GF(3**2); GF9
   print(GF9.properties)

In :obj:`galois`, the default element display mode is the integer representation. This is natural when storing and working with
integer numpy arrays. However, there are other representations and at times it may be useful to view the elements
in one of those representations.

.. ipython:: python

   GF9.Elements()

Below, we will view the representation table again to compare and contrast the different equivalent representations.

.. ipython:: python

   print(GF9.repr_table())

As before, there are some elements whose powers generate the field; we'll skip them for now. The main takeaway from this table
is the equivalence of the integer representation and the polynomial (or vector) representation. In :math:`\mathrm{GF}(3^2)`, the
element :math:`2\alpha + 1` is a polynomial that can be thought of as :math:`2x + 1` (we'll explain why :math:`\alpha` is used later). The conversion between the
polynomial and integer representation is performed simply by substituting :math:`x = 3` into the polynomial :math:`2*3 + 1 = 7`, using normal
integer arithmetic.

With :obj:`galois`, we can represent a single Galois field element using `GF9(int)` or `GF9(string)`.

.. ipython:: python

   # Create a single field element from its integer representation
   GF9(7)
   # Create a single field element from its polynomial representation
   GF9("2x + 1")
   # Create a single field element from its vector representation
   GF9.Vector([2,1])

In addition to scalars, these conversions work for arrays.

.. ipython:: python

   GF9([4, 8, 7])
   GF9(["x + 1", "2x + 2", "2x + 1"])
   GF9.Vector([[1,1], [2,2], [2,1]])

Anytime you have a large array, you can easily view its elements in whichever mode is most illustrative.

.. ipython:: python

   x = GF9.Elements(); x
   # Temporarily print x using the power representation
   with GF9.display("power"):
      print(x)
   # Permanently set the display mode to the polynomial representation
   GF9.display("poly"); x
   # Reset the display mode to the integer representation
   GF9.display(); x

   # Or convert the (10,) array of GF(p^m) elements to a (10,2) array of vectors over GF(p)
   x.vector()

Arithmetic mod p(x)
-------------------

In prime fields :math:`\mathrm{GF}(p)`, integer arithmetic (addition, subtraction, and multiplication) was performed and then reduced modulo :math:`p`.
In extension fields :math:`\mathrm{GF}(p^m)`, polynomial arithmetic (addition, subtraction, and multiplication) is performed over :math:`\mathrm{GF}(p)`
and then reduced by a polynomial :math:`p(x)`. This polynomial is called an irreducible polynomial because it cannot be factored over :math:`\mathrm{GF}(p)` --
an analogue of a prime number.

When constructing an extension field, if an explicit irreducible polynomial is not specified, a default is chosen. The default
polynomial is a Conway polynomial which is irreducible and *primitive*, see :func:`galois.conway_poly` for more information.

.. ipython:: python

   p = GF9.irreducible_poly; p
   galois.is_irreducible(p)
   # Explicit polynomial factorization returns itself as a multiplicity-1 factor
   galois.poly_factors(p)

Polynomial addition and subtract never result in polynomials of larger degree, so it is unnecessary to reduce them
modulo :math:`p(x)`. Let's try an example of addition. Suppose two field elements :math:`a = x + 2` and :math:`b = x + 1`.
These polynomials add degree-wise in :math:`\mathrm{GF}(p)`. Relatively easily we can see that :math:`a + b = (1 + 1)x + (2 + 1) = 2x`. But we can use
:obj:`galois` and :obj:`galois.Poly` to confirm this.

.. ipython:: python

   GF3 = galois.GF(3)
   # Explicitly create a polynomial over GF(3) to represent a
   a = galois.Poly([1, 2], field=GF3); a
   a.integer
   # Explicitly create a polynomial over GF(3) to represent b
   b = galois.Poly([1, 1], field=GF3); b
   b.integer
   c = a + b; c
   c.integer

We can do the equivalent calculation directly in the field :math:`\mathrm{GF}(3^2)`.

.. ipython:: python

   a = GF9("x + 2"); a
   b = GF9("x + 1"); b
   c = a + b; c

   # Or view the answer in polynomial form
   with GF9.display("poly"):
      print(c)

From here, we can view the entire addition arithmetic table. And we can choose to view the elements in
the integer representation or polynomial representation.

.. ipython:: python

   print(GF9.arithmetic_table("+"))
   with GF9.display("poly"):
      print(GF9.arithmetic_table("+"))

Polynomial multiplication, however, often results in products of larger degree than the multiplicands.
In this case, the result must be reduced modulo :math:`p(x)`.

Let's use the same example from before with :math:`a = x + 2` and :math:`b = x + 1`. To compute :math:`c = ab`, we
need to multiply the polynomials :math:`c = (x + 2)(x + 1) = x^2 + 2` in :math:`\mathrm{GF}(3)`. The issue is that
:math:`x^2 + 2` has degree-:math:`2` and the elements of :math:`\mathrm{GF}(3^2)` can have degree at most :math:`1`,
hence the need to reduce modulo :math:`p(x)`. After remainder division, we see that :math:`c = ab\ \equiv x\ \textrm{mod}\ p`.

As before, let's compute this polynomial product explicitly first.

.. ipython:: python

   # The irreducible polynomial for GF(3^2)
   p = GF9.irreducible_poly; p
   # Explicitly create a polynomial over GF(3) to represent a
   a = galois.Poly([1, 2], field=GF3); a
   a.integer
   # Explicitly create a polynomial over GF(3) to represent b
   b = galois.Poly([1, 1], field=GF3); b
   b.integer
   c = (a * b) % p; c
   c.integer

And now we'll compare that direct computation of this finite field multiplication is equivalent.

.. ipython:: python

   a = GF9("x + 2"); a
   b = GF9("x + 1"); b
   c = a * b; c

   # Or view the answer in polynomial form
   with GF9.display("poly"):
      print(c)

Now the entire multiplication table can be shown for completeness.

.. ipython:: python

   with GF9.display("poly"):
      print(GF9.arithmetic_table("*"))

Division, as in :math:`\mathrm{GF}(p)`, is a little more difficult. Fortunately the Extended Euclidean Algorithm, which
was used in prime fields on integers, can be used for extension fields on polynomials. Given two polynomials :math:`a`
and :math:`b`, the Extended Euclidean Algorithm finds the polynomials :math:`x` and :math:`y` such that
:math:`xa + yb = \textrm{gcd}(a, b)`. This algorithm is implemented in :func:`galois.poly_egcd`.

If :math:`a = x + 2` is a field element of :math:`\mathrm{GF}(3^2)` and :math:`b = p(x)`, the field's irreducible polynomial, then :math:`x = a^{-1}` in :math:`\mathrm{GF}(3^2)`.
Note, the GCD will always be :math:`1` because :math:`p(x)` is irreducible.

.. ipython:: python

   p = GF9.irreducible_poly; p
   a = galois.Poly([1, 2], field=GF3); a
   gcd, x, y = galois.poly_egcd(a, p); gcd, x, y

The claim is that :math:`(x + 2)^{-1} = x` in :math:`\mathrm{GF}(3^2)` or, equivalently, :math:`(x + 2)(x)\ \equiv 1\ \textrm{mod}\ p(x)`. This
can be easily verified with :obj:`galois`.

.. ipython:: python

   (a * x) % p

:obj:`galois` performs all this arithmetic under the hood. With :obj:`galois`, performing finite field arithmetic
is as simple as invoking the appropriate numpy function or binary operator.

.. ipython:: python

   a = GF9("x + 2"); a
   np.reciprocal(a)
   a ** -1

   # Or view the answer in polynomial form
   with GF9.display("poly"):
      print(a ** -1)

And finally, for completeness, we'll include the division table for :math:`\mathrm{GF}(3^2)`. Note, division
is not defined for :math:`y = 0`.

.. ipython:: python

   with GF9.display("poly"):
      print(GF9.arithmetic_table("/"))

Primitive elements
------------------

A property of finite fields is that some elements can produce the entire field by their powers. Namely, a *primitive element*
:math:`g` of :math:`\mathrm{GF}(p^m)` is an element such that :math:`\mathrm{GF}(p^m) = \{0, g^0, g^1, \dots, g^{p^m - 1}\}`.

In :obj:`galois`, the primitive elements of an extension field can be found by the class attribute :obj:`galois.FieldClass.primitive_element`
and :obj:`galois.FieldClass.primitive_elements`.

.. ipython:: python

   # Switch to polynomial display mode
   GF9.display("poly");
   p = GF9.irreducible_poly; p
   GF9.primitive_element
   GF9.primitive_elements

This means that :math:`x`, :math:`x + 2`, :math:`2x`, and :math:`2x + 1` can all generate the nonzero multiplicative
group :math:`\mathrm{GF}(3^2)^\times`. We can examine this by viewing the representation table using
different generators.

Here is the representation table using the default generator :math:`g = x`.

.. ipython:: python

   print(GF9.repr_table())

And here is the representation table using a different generator :math:`g = 2x + 1`.

.. ipython:: python

   print(GF9.repr_table(GF9("2x + 1")))

All other elements cannot generate the multiplicative subgroup. Another way of putting that is that their multiplicative
order is less than :math:`p^m - 1`. For example, the element :math:`e = x + 1` has :math:`\textrm{ord}(e) = 4`. This can
be seen because :math:`e^4 = 1`.

.. ipython:: python

   print(GF9.repr_table(GF9("x + 1")))

Primitive polynomials
---------------------

Some irreducible polynomials have special properties, these are primitive polynomial. A degree-:math:`m`
polynomial is *primitive* over :math:`\mathrm{GF}(p)` if it has as a root that is a generator of :math:`\mathrm{GF}(p^m)`.

In :obj:`galois`, the default choice of irreducible polynomial is a Conway polynomial, which is also a
primitive polynomial. Consider the finite field :math:`\mathrm{GF}(2^4)`. The Conway polynomial for :math:`\mathrm{GF}(2^4)`
is :math:`C_{2,4} = x^4 + x + 1`, which is irreducible and primitive.

.. ipython:: python

   GF16 = galois.GF(2**4)
   print(GF16.properties)

Since :math:`p(x) = C_{2,4}` is primitive, it has the primitive element of :math:`\mathrm{GF}(2^4)` as a root.

.. ipython:: python

   p = GF16.irreducible_poly; p
   galois.is_irreducible(p)
   galois.is_primitive(p)
   # Evaluate the irreducible polynomial over GF(2^4) at the primitive element
   p(GF16.primitive_element, field=GF16)

Since the irreducible polynomial is primitive, we write the field elements in polynomial basis with
indeterminate :math:`\alpha` instead of :math:`x`, where :math:`\alpha` represents the primitive element
of :math:`\mathrm{GF}(p^m)`. For powers of :math:`\alpha` less than 4, it can be seen that :math:`\alpha = x`, :math:`\alpha^2 = x^2`,
and :math:`\alpha^3 = x^3`.

.. ipython:: python

   print(GF16.repr_table())

Extension fields do not need to be constructed from primitive polynomials, however. The polynomial :math:`p(x) = x^4 + x^3 + x^2 + x + 1` is
irreducible, but not primitive. This polynomial can define arithmetic in :math:`\mathrm{GF}(2^4)`. The two fields (the first defined by a primitive
polynomial and the second defined by a non-primitive polynomial) are *isomorphic* to one another.

.. ipython:: python

   p = galois.Poly.Degrees([4,3,2,1,0]); p
   galois.is_irreducible(p)
   galois.is_primitive(p)

.. ipython:: python

   GF16_v2 = galois.GF(2**4, irreducible_poly=p)
   print(GF16_v2.properties)

   with GF16_v2.display("poly"):
      print(GF16_v2.primitive_element)

Notice the primitive element of :math:`\mathrm{GF}(2^4)` with irreducible polynomial :math:`p(x) = x^4 + x^3 + x^2 + x + 1` does not have
:math:`x + 1` as root in :math:`\mathrm{GF}(2^4)`.

.. ipython:: python

   # Evaluate the irreducible polynomial over GF(2^4) at the primitive element
   p(GF16_v2.primitive_element, field=GF16_v2)

As can be seen in the representation table, for powers of :math:`\alpha` less than 4, :math:`\alpha \neq x`,
:math:`\alpha^2 \neq x^2`, and :math:`\alpha^3 \neq x^3`. Therefore the polynomial indeterminate used is :math:`x` to distinguish it from :math:`\alpha`, the primitive
element.

.. ipython:: python

   print(GF16_v2.repr_table())
