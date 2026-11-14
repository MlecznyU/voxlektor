# voxlektor

Local text-to-speech service plus CLI (`voxlektor` / `vxl`) for Apple Silicon Macs. A warm daemon
keeps an MLX engine loaded and speaks over a Unix socket (protocol v1); the CLI is a thin, fast
client. Python 3.12, `uv`, English only in every file (Polish only as test/sample data).
Map of the code: `docs/architecture.md` (binding). Specs and plans are local working files in
`docs/superpowers/` (git-ignored, not part of the repository).

## Layers (imports point downward only)

```
entrypoints   CLI, daemon, one-shot
    ↓
bootstrap     wiring: the only place that picks concrete adapters
    ↓
adapters | runtime   technology (MLX, sounddevice, socket, disk) | threads + asyncio loop
    ↓
app           use cases; Session = ALL mutable state; synchronous; ports only
    ↓
domain | contracts   pure logic, no I/O, no numpy | public versioned contracts
```
`|` = siblings that must not import each other. `testing/` is never imported by production code.

## Never

1. `domain` does no I/O and never imports numpy (or os, threading, asyncio, ...).
2. `app` has no threads and no asyncio; it talks to the world through ports only.
3. Only `runtime` owns threads and the event loop.
4. Engines are reached only through the registry (`voxlektor.engines` entry points), never imported.
5. The CLI control path imports no heavy libraries (numpy, MLX, sounddevice, asyncio).

## Before you change structure

Before adding or moving a module, load skill `voxlektor-architecture`. A new module must be added
to the `docs/architecture.md` section 2 map in the same change (`test_tree_documented` enforces it).

## Gate

`uv run poe check` must be green before you say a task is done (`poe check-fast` while iterating).
It runs ruff, ruff format, `lint-imports`, pyright strict, the API snapshot, pytest with branch
coverage and the core coverage check. Hooks run lint/format/types after each edit and the fast
gate when you stop. Enable the pre-commit hook once: `git config core.hooksPath .githooks`.

## Coverage policy

100% branch coverage for `domain`, `app` and `contracts`; at least 95% overall. Only
`adapters/engines/*_mlx.py` and `adapters/audio/sounddevice_out.py` are excluded (they are covered
by `-m model` / `-m audio`). Do not add `# pragma: no cover` to dodge the gate.

## Contracts

`src/voxlektor/contracts/`: `protocol.py` + `codec.py` (socket protocol v1), `engine.py`
(Engine SPI), `voicefmt.py` (voice format). In v1 protocol changes are strictly additive; new
commands are advertised in `status.caps`. Generated artefacts (`docs/protocol/v1.schema.json`)
are never edited by hand.

## API change requests

Frozen surface = everything in the plan's Interfaces blocks plus `contracts/*`, `app/ports.py`,
`app/commands.py`, `app/effects.py`, CLI commands/options/exit codes, config keys and `VOXLEKTOR_*`
env names. Python symbols and the CLI command tree are recorded in
`tests/architecture/api_snapshot.txt`; config keys and env names are pinned by
`tests/adapters/storage/test_settings_toml.py`, exit codes by `tests/entrypoints/cli/test_output.py`
(130: `test_app.py`). Tasks may only add the snapshot entries their own Interfaces block names
(`uv run poe api-add`). If a frozen interface turns out to be
wrong, stop that part, do not work around it, and send an API change request:

- **What**: current vs proposed, exact signatures; **Why**: what in the code forced it
- **Alternatives**: including "keep as is"; **Impact**: tasks, files, tests, protocol compatibility

Wait for an explicit owner "yes". Never edit the snapshot or the schema by hand; `api-approve` is
the owner's command (`! uv run poe api-approve`).

The owner commits: never run `git commit`, `push`, `reset`, `checkout --`, `stash` or `rebase`;
finish with a report (what was built, files, gate results, deviations) instead.

## Tests

- `src/voxlektor/testing/` holds the fakes (FakeEngine, MemoryOutput, FakeClock, in-memory
  repositories), the deterministic simulator and the engine contract suite.
- Test `Session` through the simulator (no threads, no sleeps).
- Markers: `model` (real model, Apple Silicon), `audio` (real device), `e2e`. The default run
  excludes `model` and `audio` (`addopts`). Test names and messages are English.
- `VOXLEKTOR_OUTPUT=memory` (device-less output and cues in the daemon) is for tests only.
- `tests/architecture/` guards the architecture itself; keep it green and never weaken it.
