Versioning
==========

The :obj:`galois` library follows `semantic versioning <https://semver.org/>`_.
Releases are versioned as `major.minor.patch`.

For versions `>= 1.0.0`:

- **Major** versions may introduce breaking API changes.
- **Minor** versions add backwards-compatible functionality.
- **Patch** versions contain backwards-compatible bug fixes.

Pre-1.0 releases
----------------

All releases with version `< 1.0.0` should be considered **unstable** in the
sense of Semantic Versioning. However, :obj:`galois` adopts the following
stability policy to provide stronger guarantees during development.

Early development (`0.0.x`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Versions `0.0.x` represent early development releases. No API stability
guarantees are provided, and breaking changes may occur between any releases.

Ongoing development (`0.y.z`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Versions `0.y.z` follow a relaxed form of semantic versioning:

- Patch releases (`0.y.z` → `0.y.(z+1)`) are backwards-compatible.
- Minor releases (`0.y.z` → `0.(y+1).0`) may introduce breaking API changes,
  but such changes are documented in the release notes.

This policy allows iterative development prior to `1.0.0` while maintaining
reasonable API stability for users.
