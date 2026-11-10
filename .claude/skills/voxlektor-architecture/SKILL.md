---
name: voxlektor-architecture
description: Use before adding, moving or splitting any module in voxlektor, adding an engine, platform, CLI command, protocol command or Effect, or when lint-imports/pyright/coverage fail on structure.
---

# voxlektor architecture recipes

Binding map: `docs/architecture.md`. Rules are enforced by `lint-imports`, pyright strict, ruff and
`tests/architecture/`; a violation is a red `uv run poe check`, never a judgement call.

## Which layer?

| The code is... | Layer |
|---|---|
| pure logic, no I/O, no numpy, synchronous | `domain` |
| a public, versioned contract (wire, SPI, file format) | `contracts` |
| a use case that needs ports or mutable state | `app` (state only in `Session`) |
| threads, timers, the asyncio loop (no technology) | `runtime` |
| a library or the outside world (MLX, audio, socket, disk) | `adapters/<family>` |
| choosing and connecting concrete adapters | `bootstrap` |
| a user surface (CLI, daemon, one-shot) | `entrypoints` |
| fakes, simulator, contract suites | `testing` |

## Before any new or moved module

1. Pick the layer from the table; check the "Never" rules in `CLAUDE.md`.
2. Add the file to the `docs/architecture.md` section 2 map under the right directory line, in the same change.
3. Write the failing test first (mirror the path under `tests/`); domain/app/contracts need 100% branch coverage.
4. Run `uv run poe check-fast`, then `uv run poe check`.
5. If the module adds frozen symbols named by your task's Interfaces block, run `uv run poe api-add`.

## Add an engine

1. `adapters/engines/<id>.py` implementing `contracts.engine.Engine`; no file I/O, returns numpy copies.
2. Entry point in `pyproject.toml` under group `voxlektor.engines`; never import it from other layers.
3. Pass `testing/engine_contract.py` (`-m model`); add the file to the section 2 map.
4. Mark MLX-bound files `*_mlx.py` (coverage-omitted) and keep logic out of them.
5. Contract change? That is an API change request (see below), not an edit.

## Add a platform

1. `adapters/platform/<os>.py` implementing `app.ports.PlatformInfo`.
2. Register in `bootstrap/platform_select.py` (`sys.platform` / `machine`).
3. Tests for the selector and the adapter; map entry in section 2.

## Add a protocol command (additive only in v1)

1. `contracts/protocol.py`: add the Request/Reply types; never rename or remove (v1 is additive).
2. Advertise it in `status.caps`; add golden fixtures in `tests/contracts/protocol_v1/*.jsonl`.
3. `adapters/transport/mapping.py` (the only place that knows both wire and application types).
4. `app/commands.py` + `app/session.py` handling, then `entrypoints/cli/...`.
5. Regenerate `docs/protocol.md` and the schema with the generator (never by hand); `api-add`.

## Add an Effect

1. `app/effects.py`: new frozen dataclass; `Session` returns it in `Outcome.effects`.
2. Handle it in `runtime/loop.py` (the executor); `Session` never performs it itself.
3. Cover it in the simulator (`testing/simulator.py`) and the threaded runtime test.
4. Frozen surface: `api-add` if your Interfaces block names it, otherwise file an API change request.

## Add a CLI command

1. `entrypoints/cli/<name>.py`; control path imports no numpy/asyncio/MLX/runtime.
2. Exit codes: 0 ok, 1 user error, 2 service unavailable, 3 unsupported platform.
3. Test with the typer runner; run `test_cli_import_is_light`; update the README table and `api-add`.

## Add logic

`domain/` first (pure, 100% coverage), then `app/`. Wiring goes to `bootstrap/`, never into the use case.

## When a gate fails on structure

- `lint-imports`: read which contract broke; move the code to the right layer or add a port, never edit the contract.
- pyright: fix the types; no blanket `# type: ignore`.
- coverage: add tests; `exclude_also` is for Protocol stubs and typing helpers only.

## API change requests

The frozen surface (plan Interfaces blocks, `contracts/*`, `app/ports.py`, `app/commands.py`,
`app/effects.py`, CLI commands/options/exit codes, config keys, `VOXLEKTOR_*` names) is recorded in
`tests/architecture/api_snapshot.txt`. Tasks may only add entries their own Interfaces block names
(`uv run poe api-add`). If the interface is wrong, stop that part, do not work around it, and send:

- **What**: current vs proposed, exact signatures
- **Why**: what in the code forced it
- **Alternatives**: including "keep as is"
- **Impact**: tasks, files, tests, protocol compatibility

Wait for an explicit owner "yes". Never edit the snapshot or `docs/protocol/v1.schema.json` by hand;
`api-approve` is the owner's command. The owner commits: never run git state-changing commands.
