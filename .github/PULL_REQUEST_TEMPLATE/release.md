## Release PR (develop → main)

### Version

- Target tag: `vX.Y.Z`
- Release type:
  - [ ] Patch (0.4.x)
  - [ ] Minor/Major (0.5.0)

### Release notes

- [ ] Promoted `docs/release-notes/unreleased.md` → versioned notes (e.g., `docs/release-notes/v0.4.md`)
- [ ] Reset `docs/release-notes/unreleased.md` for the next cycle
- [ ] Docs build looks good (dev + versioned)

### Checks

- [ ] CI is green on `develop`
- [ ] CI is green on this PR

### Post-merge steps

- [ ] Tag `vX.Y.Z` on `main` and push tag
- [ ] Verify GitHub Pre-Release created (automation)
- [ ] Copy release notes into GitHub Release and publish
- [ ] Verify docs published for `vX.Y.Z` and `latest` points to it
- [ ] Verify new version published to PyPI
