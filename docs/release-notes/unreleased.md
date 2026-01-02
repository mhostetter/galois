---
tocdepth: 2
---

# Unreleased

## Changes currently on `develop`

### Breaking changes

- Deprecated `galois.Field()` for removal in v0.5.0. ([#633](https://github.com/mhostetter/galois/issues/633))

### Features

- Make `FieldArray` subclasses hashable based on their properties. ([#639](https://github.com/mhostetter/galois/pull/639))

### Fixes

- Fixed bug where `FieldArray` instances couldn't be unpickled if the `FieldArray` class had not yet be created. ([#639](https://github.com/mhostetter/galois/pull/639))

### Performance

- …

### Documentation

- …

### Contributors

- Frank Yellin ([@fyellin](https://github.com/fyellin))
- Matt Hostetter ([@mhostetter](https://github.com/mhostetter))
