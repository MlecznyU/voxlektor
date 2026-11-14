# Contributing to voxlektor

Thanks for your interest. voxlektor is a personal project that is shared openly: you are welcome
to use it, fork it and build on it under the [Apache License 2.0](LICENSE).

## Issues

Issues are welcome, in particular:

- bugs (include `vxl status`, your macOS version and what you ran),
- requests for another speech engine or another platform,
- ideas that would make everyday reading better.

Open one at <https://github.com/MlecznyU/voxlektor/issues>.

## Pull requests

You may propose pull requests at <https://github.com/MlecznyU/voxlektor/pulls>. They are
reviewed and merged at the author's discretion; a pull request may be closed without being
merged, also when it is good, if it does not fit where the project is going. For anything larger
than a small fix, open an issue first so we can agree on the direction.

A pull request is considered only when:

- `uv run poe check` passes (lint, format, import layers, types, API snapshot, tests with
  coverage); see `CLAUDE.md` and `docs/architecture.md` for the rules it enforces,
- it keeps the layer rules and the frozen interfaces (protocol v1 changes are additive only),
- code, comments, docs and commit messages are in English.

By submitting a contribution you agree that it is licensed under the Apache License 2.0
(section 5 of the license).
