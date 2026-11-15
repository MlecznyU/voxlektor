---
name: voxlektor-release
description: Use when starting work on a new feature or fix after a voxlektor release, when preparing a release, or when the owner mentions a version, tag, upgrade or "outdated daemon" — the package version must be bumped so running daemons are detected as outdated.
---

# voxlektor versioning

The CLI compares its `voxlektor.__version__` with the daemon's `status.version`; on a mismatch
`say`, `warm` and `watch` replace an idle daemon (`shutdown if_idle`) or report
`"daemon_outdated": true` (docs/protocol.md). That only works if **every released state has its
own version**. A forgotten bump means an old daemon keeps running after `uv tool upgrade` and
clients (dialogue-box) see features missing without any hint why.

## When to bump

| Moment | Action |
|---|---|
| First change on a branch after the last release (tag or published version) | bump **before** or with that first change, so dev daemons differ from the released one |
| New feature, protocol addition (`status.caps`, new fields), new CLI command/option | minor: `0.1.x` → `0.2.0` (pre-1.0: minor = features, also additive protocol changes) |
| Bug fix only | patch: `0.2.0` → `0.2.1` |
| Breaking change of a frozen surface (protocol v2, removed CLI option, exit code) | owner decides; never without an API change request |
| Release | check the version is not already used by an earlier tag/release; the owner tags and publishes |

If unsure whether the current version was already released, ask the owner: do not guess.

## Where

Exactly two places, kept equal by `tests/test_package.py::test_version_is_exposed_without_side_effects`:

- `pyproject.toml` → `[project] version`
- `src/voxlektor/__init__.py` → `__version__`

Examples in `docs/protocol.md` (`"version":"0.1.0"`) are illustrative; leave them unless the
example would mislead.

## After a bump

- `uv run poe check` (the version test runs in the default suite).
- Tell the owner to reinstall the tool when entry points changed:
  `uv tool install --reinstall --editable ~/development_local/voxlektor`.
- The running daemon is replaced at the next `say`/`warm`/`watch` when idle; `vxl stop` forces it.
- The owner commits; suggest a commit message, e.g. `chore: bump version to 0.2.0`.
