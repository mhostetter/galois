# Release Notes

## v0.0.22

### Breaking Changes

- Random integer generation is handled using [new style](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start) random generators. Now each `.Random()` call will generate a new seed rather than using the NumPy "global" seed used with `np.random.randint()`.
- Add a `seed=None` keyword argument to `FieldArray.Random()` and `Poly.Random()`. A reproducible script can be constructed like this:

  ```python
  rng = np.random.default_rng(123456789)
  x = GF.Random(10, seed=rng)
  y = GF.Random(10, seed=rng)
  poly = galois.Poly.Random(5, seed=rng, field=GF)
  ```

### Changes

- Official support for Python 3.9.
- Major performance improvements to "large" finite fields (those with `dtype=np.object_`).
- Minor performance improvements to all finite fields.
- Add the Number Theoretic Transform (NTT) in `ntt()` and `intt()`.
- Add the trace of finite field elements in `FieldArray.field_trace()`.
- Add the norm of finite field elements in `FieldArray.field_norm()`.
- Support `len()` on `Poly` objects, which returns the length of the coefficient array (polynomial order + 1).
- Support `x.dot(y)` syntax for the expression `np.dot(x, y)`.
- Minimum NumPy version bumped to 1.18.4 for [new style](https://numpy.org/doc/stable/reference/random/index.html#random-quick-start) random usage.
- Various bug fixes.

### Contributors

- Iyán Méndez Veiga (@iyanmv)
- Matt Hostetter (@mhostetter)

## v0.0.21

### Changes

- Fix docstrings and code completion for Python classes that weren't rendering correctly in an IDE.
- Various documentation improvements.

### Contributors

- Matt Hostetter (@mhostetter)

## v0.0.20

### Breaking Changes

- Move `poly_gcd()` functionality into `gcd()`.
- Move `poly_egcd()` functionality into `egcd()`.
- Move `poly_factors()` functionality into `factors()`.

### Changes

- Fix polynomial factorization algorithms. Previously only parital factorization was implemented.
- Support generating and testing irreducible and primitive polynomials over extension fields.
- Support polynomial input to `is_square_free()`.
- Minor documentation improvements.
- Pin Numba dependency to <0.54

### Contributors

- Matt Hostetter (@mhostetter)

## v0.0.19

### Breaking Changes

- Remove unnecessary `is_field()` function. Use `isinstance(x, galois.FieldClass)` or `isinstance(x, galois.FieldArray)` instead.
- Remove `log_naive()` function. Might be re-added later through `np.log()` on a multiplicative group array.
- Rename `mode` kwarg in `galois.GF()` to `compile`.
- Revert `np.copy()` override that always returns a subclass. Now, by default it does not return a sublcass. To return a Galois field array, use `x.copy()` or `np.copy(x, subok=True)` instead.

### Changes

- Improve documentation.
- Improve unit test coverage.
- Add benchmarking tests.
- Add initial LFSR implementation.
- Add `display` kwarg to `galois.GF()` class factory to set the display mode at class-creation time.
- Add `Poly.reverse()` method.
- Allow polynomial strings as input to `galois.GF()`. For example, `galois.GF(2**4, irreducible_poly="x^4 + x + 1")`.
- Enable `np.divmod()` and `np.remainder()` on Galois field arrays. The remainder is always zero, though.
- Fix bug in `bch_valid_codes()` where repetition codes weren't included.
- Various minor bug fixes.

### Contributors

- Matt Hostetter (@mhostetter)

## v0.0.18

### Breaking Changes

- Make API more consistent with software like Matlab and Wolfram:
  - Rename `galois.prime_factors()` to `galois.factors()`.
  - Rename `galois.gcd()` to `galois.egcd()` and add `galois.gcd()` for conventional GCD.
  - Rename `galois.poly_gcd()` to `galois.poly_egcd()` and add `galois.poly_gcd()` for conventional GCD.
  - Rename `galois.euler_totient()` to `galois.euler_phi()`.
  - Rename `galois.carmichael()` to `galois.carmichael_lambda()`.
  - Rename `galois.is_prime_fermat()` to `galois.fermat_primality_test()`.
  - Rename `galois.is_prime_miller_rabin()` to `galois.miller_rabin_primality_test()`.
- Rename polynomial search `method` keyword argument values from `["smallest", "largest", "random"]` to `["min", "max", "random"]`.

### Changes

- Clean up `galois` API and `dir()` so only public classes and functions are displayed.
- Speed-up `galois.is_primitive()` test and search for primitive polynomials in `galois.primitive_poly()`.
- Speed-up `galois.is_smooth()`.
- Add Reed-Solomon codes in `galois.ReedSolomon`.
- Add shortened BCH and Reed-Solomon codes.
- Add error detection for BCH and Reed-Solomon with the `detect()` method.
- Add general cyclic linear block code functions.
- Add Matlab default primitive polynomial with `galois.matlab_primitive_poly()`.
- Add number theoretic functions:
  - Add `galois.legendre_symbol()`, `galois.jacobi_symbol()`, `galois.kronecker_symbol()`.
  - Add `galois.divisors()`, `galois.divisor_sigma()`.
  - Add `galois.is_composite()`, `galois.is_prime_power()`, `galois.is_perfect_power()`, `galois.is_square_free()`, `galois.is_powersmooth()`.
  - Add `galois.are_coprime()`.
- Clean up integer factorization algorithms and add some to public API:
  - Add `galois.perfect_power()`, `galois.trial_division()`, `galois.pollard_p1()`, `galois.pollard_rho()`.
- Clean up API reference structure and hierarchy.
- Fix minor bugs in BCH codes.

### Contributors

- Matt Hostetter (@mhostetter)

## v0.0.17

### Breaking Changes

- Rename `FieldMeta` to `FieldClass`.
- Remove `target` keyword from `FieldClass.compile()` until there is better support for GPUs.
- Consolidate `verify_irreducible` and `verify_primitive` keyword arguments into `verify` for the `galois.GF()` class factory function.
- Remove group arrays until there is more complete support.

### Changes

- Speed-up Galois field class creation time.
- Speed-up JIT compilation time by caching functions.
- Speed-up `Poly.roots()` by JIT compiling it.
- Add BCH codes with `galois.BCH`.
- Add ability to generate irreducible polynomials with `irreducible_poly()` and `irreducible_polys()`.
- Add ability to generate primitive polynomials with `primitive_poly()` and `primitive_polys()`.
- Add computation of the minimal polynomial of an element of an extension field with `minimal_poly()`.
- Add display of arithmetic tables with `FieldClass.arithmetic_table()`.
- Add display of field element representation table with `FieldClass.repr_table()`.
- Add Berlekamp-Massey algorithm in `berlekamp_massey()`.
- Enable ipython tab-completion of Galois field classes.
- Cleanup API reference page.
- Add introduction to Galois fields tutorials.
- Fix bug in `is_primitive()` where some reducible polynomials were marked irreducible.
- Fix bug in integer<-->polynomial conversions for large binary polynomials.
- Fix bug in "power" display mode of 0.
- Other minor bug fixes.

### Contributors

- Dominik Wernberger (@Werni2A)
- Matt Hostetter (@mhostetter)

## v0.0.16

### Changes

- Add `Field()` alias of `GF()` class factory.
- Add finite groups modulo `n` with `Group()` class factory.
- Add `is_group()`, `is_field()`, `is_prime_field()`, `is_extension_field()`.
- Add polynomial constructor `Poly.String()`.
- Add polynomial factorization in `poly_factors()`.
- Add `np.vdot()` support.
- Fix PyPI packaging issue from v0.0.15.
- Fix bug in creation of 0-degree polynomials.
- Fix bug in `poly_gcd()` not returning monic GCD polynomials.

### Contributors

- Matt Hostetter (@mhostetter)

## v0.0.15

### Breaking Changes

- Rename `poly_exp_mod()` to `poly_pow()` to mimic the native `pow()` function.
- Rename `fermat_primality_test()` to `is_prime_fermat()`.
- Rename `miller_rabin_primality_test()` to `is_prime_miller_rabin()`.

### Changes

- Massive linear algebra speed-ups. (See #88)
- Massive polynomial speed-ups. (See #88)
- Various Galois field performance enhancements. (See #92)
- Support  `np.convolve()` for two Galois field arrays.
- Allow polynomial arithmetic with Galois field scalars (of the same field). (See #99), e.g.

```python
>>> GF = galois.GF(3)

>>> p = galois.Poly([1,2,0], field=GF)
Poly(x^2 + 2x, GF(3))

>>> p * GF(2)
Poly(2x^2 + x, GF(3))
```

- Allow creation of 0-degree polynomials from integers. (See #99), e.g.

```python
>>> p = galois.Poly(1)
Poly(1, GF(2))
```

- Add the four Oakley fields from RFC 2409.
- Speed-up unit tests.
- Restructure API reference.

### Contributors

- Matt Hostetter (@mhostetter)

## v0.0.14

### Breaking Changes

- Rename `GFArray.Eye()` to `GFArray.Identity()`.
- Rename `chinese_remainder_theorem()` to `crt()`.

### Changes

- Lots of performance improvements.
- Additional linear algebra support.
- Various bug fixes.

### Contributors

- Baalateja Kataru (@BK-Modding)
- Matt Hostetter (@mhostetter)
