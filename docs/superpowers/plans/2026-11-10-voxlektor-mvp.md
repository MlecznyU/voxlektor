# voxlektor MVP (Etap 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local service + `vxl` CLI that reads any PL/EN text aloud with OmniVoice-16 speaking a cloned voice, with keep-warm, playback control over a versioned Unix-socket protocol, and a measurement of start latency and gaps at playback rates 1.0 / 1.25 / 1.5.

**Architecture:** Layered hexagon with a pure core — see `docs/architecture.md` (the module map, layer rules, thread ownership and public contracts; every task places files exactly where that map says). All mutable state lives in a synchronous `app.session.Session` called only from the asyncio loop thread; one engine worker thread and the PortAudio callback hand results back via `call_soon_threadsafe`. Engines load lazily via entry points so the CLI never imports MLX.

**Tech Stack:** Python 3.12 (pinned: `pytsmod` requires <3.13), `uv`, `mlx-audio==0.6.4`, `mlx`, `msgspec`, `sounddevice`, `numpy`, `soundfile`, `pytsmod`, `typer`, `tomli-w`; dev: `pytest`, `pytest-cov`, `hypothesis`, `pyright`, `ruff`, `import-linter>=2.1`, `poethepoet`.

**Spec:** `docs/superpowers/specs/2026-11-10-voxlektor-design.md` (Etap 1 scope in §11). **Architecture:** `docs/architecture.md` (binding; where it and the spec differ — Engine SPI returns an `Iterator`, cache owned by `app`, `.bin` cache name, LOADING/UNLOADING states — the architecture wins).

## Global Constraints

- Platform: macOS 14+ on Apple Silicon only; otherwise error `platform not supported yet`, exit code 3.
- Default engine id `omnivoice`, repo `mlx-community/OmniVoice-bf16`, `num_steps=16`, sample rate 24000; revision pinned in `OMNIVOICE_REVISION` (resolved in Task 4).
- `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` is set by `bootstrap.environment.prepare()` before any engine loads.
- Defaults: `idle_timeout="10m"`, `service.autostart=true`, history 20, voice `v0-default`, lang `auto`, base_lang `pl`, rate `1.0`, generate-ahead 2.
- Precedence: CLI flag > `VOXLEKTOR_*` env > `~/.config/voxlektor/config.toml` > defaults. `VOXLEKTOR_HOME` (tests) relocates config to `$VOXLEKTOR_HOME/config` and runtime to `$VOXLEKTOR_HOME/run`.
- Runtime dir `~/Library/Application Support/voxlektor/` mode `0700`; socket `voxlektor.sock` mode `0600`.
- No temp audio files. Text never reaches a shell; the daemon is spawned with an argv list.
- Protocol v1: one JSON object per line, ≤ 1 MiB, `"v":1`, request `id` echoed, errors `{"code","message"}` from closed `ErrorCode`.
- Exit codes: 0 ok, 1 user error, 2 service unavailable, 3 unsupported platform, 130 interrupted (Ctrl+C).
- Commands `voxlektor` and `vxl` (same entry point).
- Quality gate `uv run poe check` must be green at the end of every task: ruff, `lint-imports`, pyright strict, pytest with branch coverage **100% for `domain`, `app`, `contracts`; ≥ 95% overall**, excluding only `adapters/engines/*_mlx.py` and `adapters/audio/sounddevice_out.py`.
- **English only in the repository:** code, identifiers, comments, docstrings, log and CLI messages, docs, plans, commit messages. Polish appears **only as data**: PL sample/test texts in `tests/fixtures/samples.py` (single source, imported by tests and `scripts/bench_gaps.py`; the Polish strings quoted in this plan go there) and the bundled voices' `ref.txt` transcripts. Test names and assertion messages stay English.
- **The owner commits.** Agents never run `git commit`, `git push`, `git reset`, `git checkout --`, `git stash` or `git rebase` (blocked by `.claude/hooks/pre_bash_guard.sh`). Every task's final "Commit" step means: the controller stops and gives the owner a **task report**: what was built, files created/changed, `poe check` result (lint, imports, types, tests, coverage numbers), API snapshot additions, any deviation or open question, plus the suggested commit message. The next task starts only after the owner says so.
- Every task from Task 1 on ends (before its commit) with `uv run poe api-add` for the new symbols its Interfaces block names; if `api-add` refuses, follow **API Change Control** below.

## API Change Control (owner's rule: agents never change the API on their own)

- **Frozen surface** = everything named in this plan's **Interfaces** blocks plus: `contracts/*` (protocol v1, Engine SPI, voice format), `app/ports.py`, `app/commands.py`, `app/effects.py`, CLI commands/options/exit codes, config keys and `VOXLEKTOR_*` env names.
- **Approved by the owner** with this plan; recorded mechanically in `tests/architecture/api_snapshot.txt` (Task 0 tooling). Tasks may only **add** entries that their own Interfaces block names.
- **If implementation shows a frozen interface is wrong** (needs a different name, parameter, type, semantics, error code or file location): the implementer **stops that part**, does not work around it, and returns an **API change request** to the controller: *what* (current vs proposed, exact signatures), *why* (what in the code forced it), *alternatives* incl. "keep as is", *impact* (tasks/files/tests affected, protocol compatibility). The controller **asks the owner** and waits; only an explicit owner "yes" allows the change, after which this plan's Interfaces block is updated in the same commit.
- Enforcement: `scripts/api_snapshot.py --check` runs in `poe check` (fails on any diff); `--add` accepts **only pure additions**; `--approve` (accepts changes/removals) is blocked for agents by a PreToolUse hook and is run by the owner as `! uv run poe api-approve`.
- **Checkpoint:** after Task 8 the controller pauses and asks the owner to review `docs/protocol.md` + `api_snapshot.txt` before Tasks 9–11 (they and dialogue-box build on it).

## Review Focus

1. **Empty or whitespace-only input** (`vxl ""`, `echo | vxl`) → `nothing to read`, exit 0, no socket contact. Test in Task 10.
2. **Stale socket after a crash** → client removes it and autostarts a fresh daemon. Test in Task 10.
3. **Very long input** (100 000 chars) → never more than 2 generated-but-unplayed chunks. Test in Task 7 (simulator) and Task 9 (threaded).
4. **Text equal to a subcommand** (`vxl pause`) → command; `vxl say pause` / `vxl -- pause` reads it. Test in Task 10.
5. **Engine/audio failure mid-reading, or a request during LOADING** → reading stops with `last_error`, next `speak` works; `speak` during LOADING is queued, `unload` during LOADING unloads right after load. Tests in Tasks 6 and 7.

---

## File Structure

Exactly the tree in `docs/architecture.md` §2, minus the modules marked Etap 3. Tests mirror it: `tests/<layer>/test_<module>.py`, plus `tests/architecture/`, `tests/contracts/protocol_v1/*.jsonl`, `tests/e2e/`. Repo extras: `CLAUDE.md`, `.claude/skills/voxlektor-architecture/SKILL.md`, `.claude/settings.json`, `.claude/hooks/*.sh`, `.githooks/pre-commit`, `typings/`.

---

### Task 0: Architecture skeleton, quality gate, Claude guidance and hooks

**Files:**
- Create: `pyproject.toml`, `.gitignore`, every package directory from `docs/architecture.md` §2 with an empty `__init__.py` (`src/voxlektor/__init__.py` holds only `__version__ = "0.1.0"`), `src/voxlektor/__main__.py`, `typings/{mlx_audio,sounddevice,pytsmod}/__init__.pyi` (empty stubs for now), `tests/architecture/test_canaries.py`, `tests/architecture/test_tree_documented.py`, `CLAUDE.md`, `.claude/skills/voxlektor-architecture/SKILL.md`, `.claude/settings.json`, `.claude/hooks/post_edit.sh`, `.claude/hooks/stop_check.sh`, `.claude/hooks/pre_new_module.sh`, `.githooks/pre-commit`

**Interfaces:**
- Produces: `uv run poe check` (and `poe check-fast` = same without coverage thresholds), entry point group `voxlektor.engines` (filled in Task 4), pytest markers `model`, `audio`, `e2e` (`addopts = "-m 'not model and not audio'"`).

- [ ] **Step 1: Project.** `uv init --lib --python 3.12 --name voxlektor`; deps and dev deps from Tech Stack; scripts `voxlektor = "voxlektor.entrypoints.cli.app:main"`, `vxl = "voxlektor.entrypoints.cli.app:main"`. `.gitignore`: `.venv/ __pycache__/ .coverage htmlcov/ *.wav` with an exception for `src/voxlektor/bundled_voices/**`.
- [ ] **Step 2: Static config in `pyproject.toml`:**
  - `[tool.pyright]`: `typeCheckingMode = "strict"`, `pythonVersion = "3.12"`, `include = ["src", "tests"]`, `stubPath = "typings"`, `reportImportCycles = true`, `reportMissingTypeStubs = "error"`.
  - `[tool.ruff]`: `target-version = "py312"`, `line-length = 100`; lint `select = ["E","F","W","I","B","UP","SIM","RUF","PT","TID","ANN","S","PL"]`; `[tool.ruff.lint.flake8-tidy-imports.banned-api]` bans `threading`, `asyncio`, `subprocess`, `socket` with `per-file-ignores` allowing them only in `src/voxlektor/runtime/**`, `src/voxlektor/adapters/**`, `src/voxlektor/entrypoints/**`, `tests/**`; `banned-module-level-imports = ["numpy", "mlx", "mlx_audio", "sounddevice", "soundfile", "pytsmod"]` with per-file-ignores for `adapters/**`, `runtime/buffers.py`, `contracts/engine.py`, `testing/**`, `tests/**`.
  - `[tool.coverage.run] branch = true, source = ["voxlektor"], omit = ["*/adapters/engines/*_mlx.py", "*/adapters/audio/sounddevice_out.py"]`; `[tool.coverage.report] fail_under = 95, show_missing = true, exclude_also = ["if TYPE_CHECKING:", "@overload", "raise NotImplementedError", "assert_never"]`.
  - `[tool.poe.tasks]`: `lint = "ruff check ."`, `fmt = "ruff format --check ."`, `imports = "lint-imports"`, `types = "pyright"`, `test = "pytest --cov --cov-report=term-missing"`, `core-cov = "coverage report --include='src/voxlektor/domain/*,src/voxlektor/app/*,src/voxlektor/contracts/*' --fail-under=100"`, `check = ["lint","fmt","imports","types","test","core-cov"]`, `check-fast = ["lint","imports","types","pytest -x -q"]`.
  - `[tool.importlinter]`: copy **verbatim** the contracts block below.

```toml
[tool.importlinter]
root_packages = ["voxlektor"]
include_external_packages = true

[[tool.importlinter.contracts]]
name = "Layered architecture"
type = "layers"
containers = ["voxlektor"]
layers = ["entrypoints", "bootstrap", "adapters | runtime", "app", "domain | contracts"]
exhaustive = true
exhaustive_ignores = ["testing", "bundled_voices", "__main__"]

[[tool.importlinter.contracts]]
name = "Domain is pure and synchronous"
type = "forbidden"
source_modules = ["voxlektor.domain"]
forbidden_modules = ["asyncio", "threading", "concurrent", "multiprocessing", "socket", "subprocess", "selectors", "os", "shutil", "tempfile", "logging.handlers", "importlib.metadata", "numpy", "mlx", "mlx_audio", "torch", "sounddevice", "soundfile", "pytsmod", "msgspec", "tomllib", "tomli_w", "typer", "huggingface_hub"]

[[tool.importlinter.contracts]]
name = "App has no concurrency or technology"
type = "forbidden"
source_modules = ["voxlektor.app"]
forbidden_modules = ["asyncio", "threading", "concurrent", "multiprocessing", "socket", "subprocess", "mlx", "mlx_audio", "torch", "sounddevice", "soundfile", "pytsmod", "tomllib", "tomli_w", "typer", "huggingface_hub", "importlib.metadata", "voxlektor.contracts.protocol", "voxlektor.contracts.codec"]

[[tool.importlinter.contracts]]
name = "Protocol contract is self-contained"
type = "forbidden"
source_modules = ["voxlektor.contracts.protocol", "voxlektor.contracts.codec"]
forbidden_modules = ["numpy", "voxlektor.domain", "voxlektor.contracts.engine", "voxlektor.contracts.voicefmt"]

[[tool.importlinter.contracts]]
name = "Contracts are mutually independent"
type = "independence"
modules = ["voxlektor.contracts.protocol", "voxlektor.contracts.engine", "voxlektor.contracts.voicefmt"]
ignore_imports = ["voxlektor.contracts.codec -> voxlektor.contracts.protocol"]

[[tool.importlinter.contracts]]
name = "Runtime is technology-agnostic"
type = "forbidden"
source_modules = ["voxlektor.runtime"]
forbidden_modules = ["mlx", "mlx_audio", "torch", "sounddevice", "soundfile", "pytsmod", "typer", "socket", "subprocess", "huggingface_hub"]

[[tool.importlinter.contracts]]
name = "ML frameworks only in engine adapters"
type = "forbidden"
source_modules = ["voxlektor.contracts", "voxlektor.domain", "voxlektor.app", "voxlektor.runtime", "voxlektor.adapters.audio", "voxlektor.adapters.storage", "voxlektor.adapters.platform", "voxlektor.adapters.transport", "voxlektor.adapters.procs", "voxlektor.bootstrap", "voxlektor.entrypoints"]
forbidden_modules = ["mlx", "mlx_audio", "torch"]

[[tool.importlinter.contracts]]
name = "Audio libs only in audio adapters"
type = "forbidden"
source_modules = ["voxlektor.contracts", "voxlektor.domain", "voxlektor.app", "voxlektor.runtime", "voxlektor.adapters.engines", "voxlektor.adapters.storage", "voxlektor.adapters.platform", "voxlektor.adapters.transport", "voxlektor.adapters.procs", "voxlektor.bootstrap", "voxlektor.entrypoints"]
forbidden_modules = ["sounddevice", "soundfile", "pytsmod"]

[[tool.importlinter.contracts]]
name = "Adapter families are independent"
type = "independence"
modules = ["voxlektor.adapters.engines", "voxlektor.adapters.audio", "voxlektor.adapters.storage", "voxlektor.adapters.platform", "voxlektor.adapters.transport", "voxlektor.adapters.procs"]

[[tool.importlinter.contracts]]
name = "Engines are discovered, never imported"
type = "forbidden"
source_modules = ["voxlektor.bootstrap", "voxlektor.entrypoints", "voxlektor.runtime", "voxlektor.app"]
forbidden_modules = ["voxlektor.adapters.engines"]

[[tool.importlinter.contracts]]
name = "CLI control path is lightweight"
type = "forbidden"
source_modules = ["voxlektor.entrypoints.cli"]
forbidden_modules = ["numpy", "asyncio", "voxlektor.runtime", "voxlektor.adapters.audio", "voxlektor.bootstrap.daemon"]
ignore_imports = ["voxlektor.entrypoints.cli.speak -> voxlektor.entrypoints.oneshot"]

[[tool.importlinter.contracts]]
name = "Production does not import the test kit"
type = "forbidden"
source_modules = ["voxlektor.contracts", "voxlektor.domain", "voxlektor.app", "voxlektor.runtime", "voxlektor.adapters", "voxlektor.bootstrap", "voxlektor.entrypoints"]
forbidden_modules = ["voxlektor.testing"]
```

- [ ] **Step 3: Write failing architecture tests** in `tests/architecture/`:
  - `test_canaries.py::test_import_linter_catches_stdlib_violation`: copy `src/voxlektor` into `tmp_path`, add `domain/_bad.py` with `import threading`, run `lint-imports` with `cwd=tmp_path` (argv list) → non-zero exit. If import-linter cannot see stdlib modules, this fails; then Ruff TID251 from Step 2 is the stdlib guard, and the test is changed to assert `ruff check` fails on the same file (record which one applies in `docs/architecture.md` §6).
  - `test_canaries.py::test_cli_import_is_light`: subprocess `python -c "import voxlektor.entrypoints.cli.app, sys; bad={'mlx','numpy','sounddevice','mlx_audio'} & set(sys.modules); assert not bad, bad"` → exit 0. (Stays red until Task 10 creates `cli/app.py`; mark `xfail(strict=True)` until then.)
  - `test_tree_documented.py::test_every_module_is_in_architecture_map`: every `src/voxlektor/**/*.py` (except `__init__.py`) must have its file name present in `docs/architecture.md` §2 under the right directory line. A new module without a map entry fails `check`.
- [ ] **Step 4: Run** `uv run poe check-fast` → architecture tests fail/xfail as expected, all other gates green on the empty skeleton.
- [ ] **Step 5: `CLAUDE.md`** (≤ 80 lines): what the project is (one paragraph); the layer diagram from `docs/architecture.md` §1; the five "never" rules (domain has no I/O/numpy; app has no threads/asyncio; only `runtime` owns threads; engines only via the registry; CLI control path imports no heavy libs); "before adding or moving a module, load skill `voxlektor-architecture`"; the gate `uv run poe check` must be green before saying a task is done; coverage policy; where the contracts live and that protocol changes in v1 are additive only; test conventions (`testing/` fakes, simulator for Session, markers).
- [ ] **Step 6: Skill `.claude/skills/voxlektor-architecture/SKILL.md`.** Frontmatter `name: voxlektor-architecture`, description: "Use before adding, moving or splitting any module in voxlektor, adding an engine, platform, CLI command, protocol command or Effect, or when lint-imports/pyright/coverage fail on structure." Body: the recipes from `docs/architecture.md` §5 expanded into ordered checklists (files to touch, tests to add, contract/version rules, update the §2 map), plus a "which layer?" decision table (pure logic → domain; needs ports/state → app; threads/timers → runtime; a library → adapters; wiring → bootstrap; user surface → entrypoints).
- [ ] **Step 7: Hooks.** `.claude/settings.json`:
  - `PostToolUse`, matcher `Edit|Write|MultiEdit` → `.claude/hooks/post_edit.sh`: reads hook JSON from stdin, takes `tool_input.file_path`; if it is a `.py` under `src/` or `tests/`: `uv run ruff check --fix "$f" && uv run ruff format "$f" && uv run pyright "$f"`; on failure prints output to stderr and exits 2 (fed back to Claude).
  - `PreToolUse`, matcher `Write` → `.claude/hooks/pre_new_module.sh`: if the target is a **new** `.py` under `src/voxlektor/`, prints JSON `{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"New module: follow skill voxlektor-architecture and add it to docs/architecture.md §2 in this change."}}`, exit 0 (never blocks).
  - `Stop` → `.claude/hooks/stop_check.sh`: if stdin JSON has `stop_hook_active: true` → exit 0 (no loop); if `git status --porcelain -- src tests` is empty → exit 0; else `uv run poe check-fast`; on failure prints the tail of the output to stderr, exit 2.
  - `.githooks/pre-commit`: `uv run poe check`; `git config core.hooksPath .githooks`.
  - Hook scripts use `set -euo pipefail`, quote every variable, parse JSON with `python3 -c` reading stdin (no `eval`).
- [ ] **Step 7b: API snapshot tooling.** `scripts/api_snapshot.py`: walks the frozen modules (list in the script, extended by each task as modules appear) and writes one sorted line per public symbol: `module:Name(signature) -> return` via `inspect.signature`, dataclass fields with types, enum members, msgspec struct fields, and for the CLI the typer command tree (command, options, defaults). Modes: `--check` (exit 1 on any diff, printing it), `--add` (writes only if every old line is still present unchanged; otherwise exit 1 with `API change requires owner approval — file an API change request`), `--approve` (writes unconditionally). Poe tasks: `api-check`, `api-add`, `api-approve`; `api-check` joins `check` and `check-fast`. Hook `.claude/hooks/pre_bash_guard.sh` (PreToolUse, matcher `Bash`): blocks (exit 2, stderr `the owner commits — report instead`) any `git commit|push|reset|checkout --|stash|rebase`; and if the command contains `api_snapshot.py --approve`, `poe api-approve`, or writes to `tests/architecture/api_snapshot.txt` (e.g. `>`, `tee`, `cp`, `mv` targeting it) → stderr `API changes need the owner's approval: file an API change request instead`, exit 2. Same guard in `pre_new_module.sh` logic for `Edit|Write` on `tests/architecture/api_snapshot.txt` and `docs/protocol/v1.schema.json` (exit 2). Test `tests/architecture/test_api_snapshot_tool.py`: `--add` accepts a new line, rejects a changed line; `--check` fails on diff.
- [ ] **Step 7c: `CLAUDE.md` + skill** get a section "API change requests" copying the rule from this plan's *API Change Control* section, with the request template (what / why / alternatives / impact).
- [ ] **Step 8: Verify hooks manually:** `echo '{"tool_input":{"command":"uv run poe api-approve"}}' | .claude/hooks/pre_bash_guard.sh; echo $?` → `2`; `echo '{"tool_input":{"file_path":"src/voxlektor/domain/x.py"}}' | .claude/hooks/pre_new_module.sh` prints the JSON; `echo '{"stop_hook_active":true}' | .claude/hooks/stop_check.sh; echo $?` → `0`.
- [ ] **Step 9: Report to owner (owner commits)** — suggested message: `chore: architecture skeleton, quality gate, CLAUDE.md, skill and hooks`

---

### Task 1: Settings, platform adapters

**Files:**
- Create: `domain/types.py`, `domain/errors.py`, `domain/settings.py`, `app/ports.py` (only `PlatformInfo`, `SettingsStore` now; later tasks add ports), `adapters/storage/settings_toml.py`, `adapters/platform/macos.py`, `adapters/platform/unsupported.py`, `bootstrap/platform_select.py`, tests mirrored.

**Interfaces:**
- Produces:
  - `domain.types`: `VoiceName = NewType("VoiceName", str)`, `EngineId`, `ChunkId = NewType(int)`, `Epoch = NewType(int)`, `Lang = Literal["pl","en"]`, `LangPref = Literal["auto","pl","en"]`, `Seconds = float`.
  - `domain.errors.DomainError(code: str, message: str)`; subclasses `InvalidDuration`, `InvalidSetting`, `UnknownVoice(name, available: tuple[str, ...])` with codes `invalid_duration`, `invalid_setting`, `unknown_voice`.
  - `domain.settings`: `@dataclass(frozen=True, slots=True) Settings(engine: EngineId = "omnivoice", voice: VoiceName = "v0-default", lang: LangPref = "auto", base_lang: Lang = "pl", rate: float = 1.0, idle_timeout_s: int = 600, autostart: bool = True, history: int = 20, ahead: int = 2)`; `parse_duration(s: str) -> int` (`"0"`, `"90s"`, `"10m"`, `"1h"`; else `InvalidDuration`); `merge_layers(*layers: Mapping[str, object]) -> Settings` (later layers win; validates `rate` in `[0.5, 3.0]`, lang, base_lang (`pl`/`en` only), positive history/ahead; raises `InvalidSetting`). Layer keys are the TOML/flag names: `engine, voice, lang, base_lang, rate, history, idle_timeout, autostart`.
  - `app.ports.SettingsStore(Protocol)`: `read_file() -> dict[str, object]`, `read_env() -> dict[str, object]`, `write_value(key: str, value: str) -> None`.
  - `app.ports.PlatformInfo(Protocol)`: `config_dir() -> Path`, `runtime_dir() -> Path`, `socket_path() -> Path`, `log_path() -> Path`, `pid_path() -> Path`.
  - `adapters.storage.settings_toml.TomlSettingsStore(path: Path, environ: Mapping[str, str])` — TOML layout: top-level `config_version = 1`, `engine`, `voice`, `lang`, `base_lang`, `rate`, `history`; `[service] idle_timeout, autostart`; env `VOXLEKTOR_{ENGINE,VOICE,LANG,BASE_LANG,RATE,IDLE_TIMEOUT,AUTOSTART}`; `write_value` keeps unknown keys.
  - `adapters.platform.macos.MacOSPlatform(home: Path | None)` (honours `VOXLEKTOR_HOME`), dirs created `0700`; `unsupported.UnsupportedPlatform` raises `PlatformError("platform not supported yet")` on every call.
  - `bootstrap.platform_select.select_platform(sys_platform: str, machine: str, macos_version: str) -> PlatformInfo`.

- [ ] **Step 1: Failing tests:** `merge_layers` precedence (defaults < file < env < flags, each layer overriding `voice`); `parse_duration` table incl. `"abc"`, `"-5m"`, `"10x"` raising; `merge_layers({"rate": 5})` raises `InvalidSetting`; Hypothesis: `parse_duration(f"{n}m") == 60*n` for `n in 0..10_000`; TOML store round-trip keeps an unknown key `foo = 1`; env parsing (`VOXLEKTOR_AUTOSTART=false` → `False`); macOS dirs mode `0o700`; `select_platform("linux","x86_64")` → calls raise `PlatformError`.
- [ ] **Step 2: Run** `uv run pytest tests/domain tests/adapters/storage tests/adapters/platform tests/bootstrap -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** same → PASS; `uv run poe check` → green.
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `feat: layered settings and platform adapters`

---

### Task 2: Text chunking and language detection

**Files:** Create `domain/text/__init__.py`, `domain/text/chunking.py`, `domain/text/lang.py`, `tests/domain/text/test_chunking.py`, `test_lang.py`.

**Interfaces:**
- Produces: `split_text(text: str, max_chars: int = 300, merge_below: int = 60) -> list[str]`; `detect_lang(text: str, fallback: Lang) -> Lang` (fallback required; `Settings.base_lang` is passed by callers).
- Rules: paragraphs on blank lines; long paragraphs split on sentence ends (`.`, `!`, `?`, `…` + whitespace) and packed greedily ≤ `max_chars`; an overlong sentence split at the last space before the limit (never inside a word unless a single token exceeds `max_chars`); consecutive chunks shorter than `merge_below` merged with the next when the result fits; whitespace normalised, empties dropped. `detect_lang` → `"pl"` if any of `ąćęłńóśźż` (any case); else tokenize (lowercase, split on non-letters) and count DISTINCT markers: PL `{się, nie, jest, że, w, z, na, jak, ale, czy, tak, mamy}`, EN `{the, and, is, are, was, of, you, with, this, that, what, it, for, not, be, have}` (ambiguous `to, do, no, a, i, on` are in neither set); more PL → `"pl"`, more EN → `"en"`, tie (including 0:0) → `fallback`.

- [ ] **Step 1: Failing tests:** `"A.\n\nB."` → `["A. B."]`; two 100-char paragraphs → 2 chunks; `split_text("  \n\n ") == []`; 700-char unpunctuated sentence → all ≤ 300, no word cut; **Hypothesis** over text from an alphabet of letters/space/`.`/`\n`: every chunk ≤ `max_chars` (unless it is a single over-long token), `"".join(text.split()) == "".join("".join(chunks).split())`; lang table: `"Zmergowałem brancha z mainem"→pl`, `"I merged the branch"→en`, `"to nie jest ok"→pl`, `"I need to go to the store"` with fallback `pl` → en, `"to be or not to be"→en`, `"npm run dev"` → the fallback (both `pl` and `en`).
- [ ] **Step 2: Run** `uv run pytest tests/domain/text -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS; `poe check` green.
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `feat: text chunking and PL/EN detection`

---

### Task 3: Voice format, catalog and filesystem repository

**Files:** Create `contracts/voicefmt.py`, `app/voices.py`, `adapters/storage/voices_fs.py`, `src/voxlektor/bundled_voices/<7>/{ref.wav,ref.txt,voice.toml}`, tests mirrored. Modify `app/ports.py` (add `VoiceRepository`).

**Interfaces:**
- Produces:
  - `contracts.voicefmt`: `VOICE_FORMAT = 1`; `@dataclass(frozen=True) VoiceMeta(format: int, name: str, source: Literal["bundled","clone","design"], lang: str, created: str)`; `REF_WAV = "ref.wav"`, `REF_TXT = "ref.txt"`, `META = "voice.toml"`, `CACHE_DIR = "cache"`; `MAX_REF_SECONDS = 30`; `prompt_cache_name(engine_id: str, revision: str, prompt_format: int) -> str` → `f"{engine_id}-{revision[:8]}-p{prompt_format}.bin"`.
  - `app.ports.VoiceRepository(Protocol)`: `names() -> list[str]`; `ref(name: str) -> VoiceRef` (Task 4 type; until then return a `(ref_wav: Path, ref_text: str)` pair is NOT allowed — implement Task 3 after Task 4's `contracts/engine.py` or define `VoiceRef` here in `contracts/engine.py` first: **create `contracts/engine.py` with only `VoiceRef` in this task**); `read_cache(name: str, key: str) -> bytes | None`; `write_cache(name: str, key: str, data: bytes) -> None`; `install_bundled() -> None` (copies missing bundled voices, never overwrites).
  - `contracts.engine.VoiceRef`: `@dataclass(frozen=True) VoiceRef(name: str, ref_wav: Path, ref_text: str)`.
  - `app.voices.VoiceCatalog(repo: VoiceRepository)`: `list() -> list[str]` (sorted), `resolve(name: str) -> VoiceRef` (raises `UnknownVoice` with `available`), `ensure_bundled() -> None`.
  - `adapters.storage.voices_fs.FsVoiceRepository(root: Path)` — layout from `voicefmt`; cache writes are atomic (`write to .tmp` + `os.replace` inside `cache/`), and only files inside `cache/` are ever deleted.

- [ ] **Step 1: Copy assets:** for `v0-default f1-warm f2-narrator f3-bright m1-calm m2-narrator m3-casual`, copy `~/development_local/claude-mods/tts-lab/out/voices/<name>/ref.wav`; `ref.txt` = the exact `REF_TEXT` string from `~/development_local/claude-mods/tts-lab/voices.py`; `voice.toml` = `format=1, name, source="bundled", lang="pl", created="2026-11-10"`. Package data in `pyproject.toml`.
- [ ] **Step 2: Failing tests:** `test_prompt_cache_name`; catalog `list()` returns the 7 names sorted; `resolve("nope")` raises `UnknownVoice` with all 7 in `available`; `install_bundled` does not overwrite an edited `ref.txt`; cache round-trip `write_cache`/`read_cache`; `read_cache` of a missing key → `None`; an unknown extra file in a voice dir survives `install_bundled`.
- [ ] **Step 3: Run** `uv run pytest tests/contracts tests/app/test_voices.py tests/adapters/storage -v` → FAIL.
- [ ] **Step 4: Implement** (bundled files via `importlib.resources`).
- [ ] **Step 5: Run** → PASS; `poe check` green.
- [ ] **Step 6: Report to owner (owner commits)** — suggested message: `feat: voice format, catalog and filesystem repository with 7 bundled voices`

---

### Task 4: Engine SPI, fakes, contract suite, OmniVoice adapter, registry

**Files:** Modify `contracts/engine.py`. Create `testing/fakes.py` (FakeEngine part), `testing/engine_contract.py`, `adapters/engines/omnivoice_mlx.py`, `app/voice_prompts.py`, `bootstrap/environment.py`, `bootstrap/registry.py`, `typings/mlx_audio/...` (only symbols used), tests: `tests/contracts/test_engine_fake.py`, `tests/adapters/engines/test_omnivoice_mlx.py` (`-m model`), `tests/app/test_voice_prompts.py`, `tests/bootstrap/test_registry.py`. Modify `pyproject.toml` (entry points), `app/ports.py` (add `EngineProvider`).

**Interfaces:**
- Produces:
  - `contracts.engine`: `ENGINE_API = 1`; `class Capability(StrEnum): LIVE, RENDER, CLONE, DESIGN`; `@dataclass(frozen=True) EngineSpec(id: str, revision: str, prompt_format: int, sample_rate: int, langs: frozenset[str], capabilities: frozenset[Capability])`; `@dataclass(frozen=True) VoicePrompt(voice: str, prompt_format: int, data: object)`; `@dataclass(frozen=True) AudioChunk(samples: npt.NDArray[np.float32], sample_rate: int)` (numpy only under `TYPE_CHECKING`: `contracts.engine` and `app.ports` never import numpy at runtime);
    ```python
    class Engine(Protocol):
        spec: EngineSpec

        def load(self) -> None: ...
        def unload(self) -> None: ...
        def prepare_voice(self, ref: VoiceRef) -> VoicePrompt: ...
        def serialize_prompt(self, p: VoicePrompt) -> bytes: ...
        def deserialize_prompt(self, voice: str, data: bytes) -> VoicePrompt: ...
        def synthesize(self, text: str, lang: str, prompt: VoicePrompt) -> Iterator[AudioChunk]: ...
    ```
  - `testing.fakes.FakeEngine(sample_rate=24000, seconds_per_char=0.05, delay_s=0.0, fail_on: str | None = None)` — sine chunks, counters `load_calls`, `unload_calls`, `prepare_calls`, `synth_calls: list[tuple[str, str]]`; `spec.id = "fake"`, capabilities `{LIVE, CLONE}`.
  - `testing.engine_contract.run_engine_contract(engine: Engine, ref: VoiceRef) -> None` — asserts: `load` idempotent; prompt serialize/deserialize round-trip yields same audio length ±10%; `synthesize` yields ≥ 1 float32 mono chunk at `spec.sample_rate`; `unload` then `load` works.
  - `app.voice_prompts.get_or_build_prompt(engine: Engine, repo: VoiceRepository, ref: VoiceRef) -> VoicePrompt` — key = `prompt_cache_name(spec.id, spec.revision, spec.prompt_format)` (the contract suite checks that a prepared prompt's `prompt_format` equals `spec.prompt_format`); read cache → `deserialize_prompt`; else `prepare_voice` → `write_cache(serialize_prompt)`. A corrupt cache (`deserialize_prompt` raises `ValueError`) is rebuilt and overwritten.
  - `adapters.engines.omnivoice_mlx.OmniVoiceEngine`: `REPO`, `OMNIVOICE_REVISION`, `NUM_STEPS = 16`, `LANGS = {"pl": "polish", "en": "english"}`, `PROMPT_FORMAT = 1`; `load` via `snapshot_download(REPO, revision=OMNIVOICE_REVISION)` then `mlx_audio.tts.utils.load_model(local_path)`; `prepare_voice` via `mlx_audio.tts.models.omnivoice.utils.create_voice_clone_prompt(str(ref.ref_wav), tokenizer=model.audio_tokenizer, max_duration_s=10.0)`; serialize = `np.save` of `np.array(tokens)` into bytes; `synthesize` = `model.generate(text=, language=LANGS[lang], num_steps=NUM_STEPS, ref_tokens=mx.array(...), ref_text=prompt ref_text)` yielding `AudioChunk(np.array(r.audio, np.float32).copy(), 24000)`; `unload` drops the model, `gc.collect()`, `mx.clear_cache()`. **No file I/O.** (Store `ref_text` inside `VoicePrompt.data` together with tokens.)
  - `bootstrap.environment.prepare() -> None` — `os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")`.
  - `bootstrap.registry.load_engine(engine_id: str) -> Engine` — `importlib.metadata.entry_points(group="voxlektor.engines")`; calls `environment.prepare()` first; unknown id → `DomainError("unknown_engine", ...)` listing available ids. Entry points: `omnivoice = "voxlektor.adapters.engines.omnivoice_mlx:OmniVoiceEngine"`, `fake = "voxlektor.testing.fakes:FakeEngine"`.
  - `app.ports.EngineProvider(Protocol)`: `__call__(engine_id: str) -> Engine`.

- [ ] **Step 1: Resolve revision:** `curl -s https://huggingface.co/api/models/mlx-community/OmniVoice-bf16 | python3 -c "import json,sys;print(json.load(sys.stdin)['sha'])"` → `OMNIVOICE_REVISION`.
- [ ] **Step 2: Failing tests:** `run_engine_contract(FakeEngine(), ref)` passes; `FakeEngine(fail_on="BOOM")` raises on that text; `get_or_build_prompt` builds once then hits cache (`prepare_calls == 1` across two calls, with `InMemoryVoiceRepository` from `testing/fakes.py`); corrupt cache bytes → rebuilt; `load_engine("fake")` returns `FakeEngine`; `load_engine("nope")` raises with `available` listing `fake` and `omnivoice`; `load_engine` sets the HF env var. `-m model`: `run_engine_contract(OmniVoiceEngine(), bundled v0-default ref)` and a printed latency of the 110-char PL paragraph `"Sprawdziłem testy i wszystko przechodzi. Jedyny problem to ostrzeżenie przy buildzie, ale nie blokuje niczego."` with a warm, cached prompt (record only).
- [ ] **Step 3: Run** `uv run pytest tests/contracts tests/app/test_voice_prompts.py tests/bootstrap -v` → FAIL; `uv run pytest -m model tests/adapters/engines -v -s` → FAIL.
- [ ] **Step 4: Implement;** add `typings/mlx_audio` stubs for exactly the used symbols so pyright strict passes without ignores.
- [ ] **Step 5: Run** both → PASS; `poe check` green; put the printed latency in the commit body.
- [ ] **Step 6: Report to owner (owner commits)** — suggested message: `feat: engine SPI, contract suite, OmniVoice adapter and entry-point registry`

---

### Task 5: Playback queue, audio output port and adapters

**Files:** Create `domain/playback.py`, `adapters/audio/dsp.py`, `adapters/audio/sounddevice_out.py`, `testing/fakes.py` (add `MemoryOutput`, `FakeClock`), `typings/sounddevice`, `typings/pytsmod`, tests `tests/domain/test_playback.py`, `tests/adapters/audio/test_dsp.py`, `tests/adapters/audio/test_sounddevice_out.py` (`-m audio`). Modify `app/ports.py` (add `AudioOutput`, `Clock`).

**Interfaces:**
- Produces:
  - `domain.playback.PlaybackQueue[A]` (generic, numpy-free; `A` is `ChunkId` in production): `push(item: A)`, `start_next() -> A | None` (moves next to `current`), `finish_current() -> None` (current → history, history capped at `history`), `skip() -> A | None` (drops current without history, returns next), `back() -> list[A]` (returns `[previous, current]` re-queued at the front; no-op without history), `clear() -> list[A]` (returns dropped items), properties `current`, `queued: int`, `history: tuple[A, ...]`, `paused: bool` with `pause()`/`resume()`.
  - `app.ports.AudioOutput(Protocol)`: `start(chunk: ChunkId, samples: object, sample_rate: int) -> None`, `pause() -> None`, `resume() -> None`, `abort() -> None`; completion is reported through callbacks set once: `set_callbacks(on_done: Callable[[ChunkId], None], on_failed: Callable[[ChunkId, str], None]) -> None`.
  - `app.ports.Clock(Protocol)`: `now() -> float`.
  - `adapters.audio.dsp.time_stretch(samples: npt.NDArray[np.float32], rate: float) -> npt.NDArray[np.float32]` — identity at 1.0; else `pytsmod.wsola(samples, 1.0 / rate)` cast to float32.
  - `adapters.audio.sounddevice_out.SoundDeviceOutput()` — `sounddevice.OutputStream(samplerate, channels=1, dtype="float32", latency="high", callback=…)`; callback copies frames from the current buffer, writes zeros while paused, on end calls `on_done` and on `CallbackAbort`/device error `on_failed`. Callbacks are invoked as given (the runtime wraps them with `call_soon_threadsafe`).
  - `testing.fakes.MemoryOutput(auto_complete: bool = True)` — records `started: list[ChunkId]`, `aborted`, `paused`; `complete(chunk)` / `fail(chunk, msg)` to drive callbacks manually; `auto_complete` calls `on_done` synchronously on `start`.
  - `testing.fakes.FakeClock(t: float = 0.0)` with `advance(s)`.

- [ ] **Step 1: Failing tests:** queue order; `skip` drops current and returns next; `back` after A finished and B current → front of queue is `[A, B]`; `back` with empty history → `[current]` only/no-op; history capped at 20 after 30 finishes; `clear` returns all queued + current; **Hypothesis stateful test** (`RuleBasedStateMachine`) over push/start_next/finish/skip/back/clear asserting: no item is both current and queued; `len(history) <= cap`; no item is lost except via `skip`/`clear`. `time_stretch` of 1 s at rate 1.5 → length within 5% of 16000 samples; rate 1.0 is identity (same object or equal array).
- [ ] **Step 2: Run** `uv run pytest tests/domain/test_playback.py tests/adapters/audio/test_dsp.py -v` → FAIL.
- [ ] **Step 3: Implement;** stubs for `sounddevice`/`pytsmod` symbols used.
- [ ] **Step 4: Run** → PASS; `-m audio` smoke: `SoundDeviceOutput` plays 0.5 s sine and fires `on_done` within 2 s. `poe check` green.
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `feat: playback queue, audio output port, sounddevice and time-stretch adapters`

---

### Task 6: Keep-warm lifecycle and reading policy (domain)

**Files:** Create `domain/keepwarm.py`, `domain/reading.py`, tests mirrored.

**Interfaces:**
- Produces:
  - `domain.keepwarm`: `class Life(StrEnum): COLD, LOADING, WARM, UNLOADING`; events `NeedEngine`, `Loaded`, `LoadFailed(msg)`, `Idle(now)`, `Busy`, `UnloadRequested`, `Unloaded`, `Wake(now)`, `SetTimeout(seconds)`; actions `StartLoad`, `StartUnload`, `ScheduleWake(at: float)`, `CancelWake`, `FlushPending`; `@dataclass(frozen=True) KeepWarm(life: Life, timeout_s: int, deadline: float | None, unload_after_load: bool, reload_after_unload: bool = False, idle_since: float | None = None)` (API change F: `unload_after_load` = "unload right after the current/next load completes" in every state; `reload_after_unload` = "start a load when the current unload completes"; `idle_since` is set on WARM + `Idle(now)`, also when `timeout_s == 0`, and cleared on `Busy` and when leaving WARM); `transition(s: KeepWarm, e: Event) -> tuple[KeepWarm, list[Action]]`.
  - Rules: `COLD + NeedEngine → LOADING, [StartLoad]`; `LOADING + NeedEngine → LOADING, []` (caller queues the work); `LOADING + Loaded → WARM, [FlushPending]` then if `unload_after_load` → `UNLOADING, [StartUnload]`; `LOADING + UnloadRequested → LOADING (unload_after_load=True)`; `LOADING + LoadFailed → COLD`; `WARM + Idle(now)` → deadline `now+timeout_s`, `[ScheduleWake(deadline)]` unless `timeout_s == 0`; `WARM + Busy → deadline None, [CancelWake]`; `WARM + Wake(now ≥ deadline) → UNLOADING, [StartUnload]`; stale `Wake` (now < deadline or deadline None) ignored; `WARM + UnloadRequested → UNLOADING, [StartUnload]`; `UNLOADING + Unloaded → COLD`; `UNLOADING + NeedEngine` → stays, `reload_after_unload=True`, so `Unloaded` leads to `LOADING, [StartLoad]`; `UNLOADING + UnloadRequested` while a reload is pending → `unload_after_load=True` (kept through `Unloaded` → `LOADING` → `Loaded` → `UNLOADING, [FlushPending, StartUnload]`); `SetTimeout` in WARM-idle reschedules from `idle_since` (`deadline = idle_since + new timeout`, `ScheduleWake`; new timeout 0 → `CancelWake`, deadline None; a deadline already in the past is still emitted, `Wake` handles it).
  - `domain.reading`: `@dataclass(frozen=True) Chunk(id: ChunkId, text: str, lang: Lang, rate: float, epoch: Epoch)`; `class AheadWindow(ahead: int)`: `next_to_generate(pending: Sequence[Chunk], generated_unplayed: int) -> Chunk | None` (returns `pending[0]` only if `generated_unplayed < ahead`); `UnderrunTracker`: `on_queue_empty(more_expected: bool)` counts underruns; `plan_reading(text: str, lang: LangPref, base_lang: Lang, rate: float, epoch: Epoch, first_id: int) -> list[Chunk]` (ruling R19/R20: when `auto`, `split_text(text, merge_below=0)` + `detect_lang(chunk, base_lang)` per chunk, then adjacent same-language chunks are merged by the split_text rule; the base language comes from `Settings.base_lang`).

- [ ] **Step 1: Failing tests:** one test per rule above (table-driven); **Hypothesis stateful** over random event sequences asserting: never two `StartLoad` without a `Loaded`/`LoadFailed` between; `StartUnload` never emitted in LOADING; a `Wake` with stale deadline never unloads; with `timeout_s == 0` no `ScheduleWake` ever. `AheadWindow(2)` returns `None` at 2 generated-unplayed; `plan_reading` gives `["pl","en"]` langs for `"Zażółć gęślą jaźń.\n\nHello world, this is English."`; underrun counted only when `more_expected`.
- [ ] **Step 2: Run** `uv run pytest tests/domain/test_keepwarm.py tests/domain/test_reading.py -v` → FAIL.
- [ ] **Step 3: Implement** (pure; `match` with `assert_never`).
- [ ] **Step 4: Run** → PASS; `poe check` green (100% branch on domain).
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `feat: keep-warm lifecycle and reading policy as pure domain`

---

### Task 7: Session (single-writer core) and simulator

**Files:** Create `app/commands.py`, `app/effects.py`, `app/session.py`, `testing/simulator.py`, `tests/app/test_session.py`, `tests/app/test_session_stateful.py`.

**Interfaces:**
- Consumes: `KeepWarm`/`transition`, `PlaybackQueue`, `AheadWindow`, `UnderrunTracker`, `plan_reading` (Task 6), `Settings` (Task 1), `VoiceCatalog` (Task 3), ports.
- Produces:
  - `app.commands`: frozen dataclasses `Speak(text: str, voice: str | None, lang: LangPref | None, rate: float | None, now: bool)`, `Pause`, `Resume`, `Skip`, `Back`, `StopReading`, `Status`, `SetIdle(timeout: str)`, `Unload`, `Shutdown`; `Command = Union[...]`.
  - `app.effects`: `LoadEngine(engine_id)`, `UnloadEngine`, `PrepareVoice(ref: VoiceRef, fresh: bool = False)` (`fresh` = bypass the prompt cache and rebuild from the reference audio), `Generate(chunk: Chunk, voice: str)`, `StartPlayback(chunk: ChunkId)`, `PausePlayback`, `ResumePlayback`, `AbortPlayback`, `DropAudio(chunks: tuple[ChunkId, ...])`, `ScheduleWake(at)`, `CancelWake`, `Exit`, `PlayCue(cue: Cue)` with `class Cue(StrEnum): RETRY = "retry"; STOPPED = "stopped"`; `Effect = Union[...]`. (Owner-approved additions, Task 7 fix round 1.)
  - `@dataclass(frozen=True) Reply(ok: bool, data: Mapping[str, object], error: DomainError | None)`; `@dataclass Outcome(reply: Reply | None, effects: list[Effect])`.
  - `app.session.Session(settings: Settings, catalog: VoiceCatalog, clock: Clock)`:
    - `handle(cmd: Command) -> Outcome`
    - callbacks (each returns `list[Effect]`): `on_loaded()`, `on_load_failed(msg)`, `on_prompt_ready(voice: str)`, `on_generated(epoch: Epoch, chunk: ChunkId, audio_seconds: float, gen_seconds: float)`, `on_gen_failed(epoch, chunk, msg)`, `on_play_done(chunk)`, `on_play_failed(chunk, msg)`, `on_wake()`, `on_unloaded()`, `on_prompt_failed(voice: str, msg: str)` (owner-approved addition).
    - Retry ladder (owner-approved): prompt failure for a voice that a chunk still to be generated uses (otherwise it is dropped silently: voice un-requested, no cue, no `last_error`) → 1st per voice per reading and per engine load (R22): `PlayCue(RETRY)` + `PrepareVoice(ref, fresh=True)`; 2nd: stop reading, `last_error`, `PlayCue(STOPPED)`, voice no longer requested. Generation failure (current epoch, live chunk; per-chunk counter reset on success, everything reset on a new reading) → attempt 1: `PlayCue(RETRY)` + regenerate; attempt 2: `PlayCue(RETRY)` + engine reload through keep-warm, then regenerate; attempt 3: stop reading, `last_error`, `PlayCue(STOPPED)`. Playback or load failure → stop reading, `last_error`, `PlayCue(STOPPED)`. Stale callbacks produce no cues. Back re-queues chunks under fresh ChunkIds (R21). `Speak.rate` outside [0.5, 3.0] → `InvalidSetting`.
    - Behaviour: `Speak` with empty plan → reply `ok, chunks=0`, no effects. Otherwise plan chunks (new epoch if `now`, plus `AbortPlayback` + `DropAudio` of the old reading), emit `NeedEngine` into keep-warm; once WARM and the voice prompt is ready, emit `Generate` per `AheadWindow`; `on_generated` with stale epoch → `DropAudio`; first generated chunk starts playback; `on_play_done` → `finish_current`, start next generated or record underrun; queue empty and nothing pending → keep-warm `Idle(now)`; `on_gen_failed` → the retry ladder below (the reading stops, with `last_error = msg`, only on the third failure of a chunk); `on_play_failed`/`on_load_failed` → stop reading, `last_error = msg`, `PlayCue(STOPPED)`; the session stays usable. `Unload` while reading → `StopReading` semantics + `UnloadRequested`. `Shutdown` → stop + unload + `Exit`. `Status` reply data: `state`, `engine`, `voice`, `idle_timeout_s`, `idle_remaining_s` (None while busy or timeout 0), `queued`, `playing`, `paused`, `underruns`, `first_audio_s`, `chunks`, `gen_seconds`, `audio_seconds`, `last_error`, `caps` (list of supported command names). Unknown voice → `Reply(ok=False, error=UnknownVoice)`.
    - Voice prompts are prepared once per voice per engine load (`PrepareVoice` effect, then `on_prompt_ready`).
  - `testing.simulator.Simulator(session, engine: FakeEngine, output: MemoryOutput, clock: FakeClock)` — executes effects synchronously (`LoadEngine` → `on_loaded`, `Generate` → runs fake synth → `on_generated`, `StartPlayback` → records; `MemoryOutput.complete` drives `on_play_done`), `run_until_idle()`, `advance(seconds)` fires due wakes; records played cues.
  - `testing.fakes.FakeEngine` gains transient-failure knobs `fail_times: int | None = None` (only the first N matching syntheses fail; None = every one) and `fail_prepare: int = 0` (the first N `prepare_voice` calls fail) — owner-approved addition.

- [ ] **Step 1: Failing tests** (`test_session.py`, via `Simulator`): cold `Speak` → `LoadEngine` once, then WARM; same voice twice → one `PrepareVoice`; idle unload at `timeout_s` (599 s still WARM, 601 s COLD); paused reading never unloads; `timeout 0` never unloads; `SetIdle("1m")` applies live; `Unload` during reading → reading stopped, COLD; **Review Focus 5**: `fail_on="BOOM"` → after the retry ladder `last_error == "fake engine failure on 'BOOM'"` (FakeEngine's message), later chunk not played, next `Speak("fine")` plays; `Speak` during LOADING is queued and plays after load; `Unload` during LOADING ends COLD after load; **Review Focus 3**: 100 000-char text → at every step `generated_unplayed <= 2`; `now=True` → old reading aborted, new chunk plays first, late results of the old epoch dropped; `Back`/`Skip` map to the queue; `Status` contains every listed key; unknown voice → error code `unknown_voice`.
- [ ] **Step 2: `test_session_stateful.py`:** Hypothesis `RuleBasedStateMachine` interleaving speak/now-speak/pause/resume/skip/back/stop/unload/set-idle/advance-time/complete-play/fail-gen; invariants: at most one chunk playing; never `Generate` while not WARM; no stale-epoch chunk ever started; COLD only when no reading is active or after an explicit unload; `idle_remaining_s` is None whenever busy.
- [ ] **Step 3: Run** `uv run pytest tests/app -v` → FAIL.
- [ ] **Step 4: Implement** `commands.py`, `effects.py`, `session.py`, `testing/simulator.py`.
- [ ] **Step 5: Run** → PASS; `poe check` green (100% branch on app).
- [ ] **Step 6: Report to owner (owner commits)** — suggested message: `feat: single-writer session core with deterministic simulator`

---

### Task 8: Protocol v1 contract and wire mapping

**Files:** Create `contracts/protocol.py`, `contracts/codec.py`, `adapters/transport/mapping.py`, `docs/protocol.md`, `docs/protocol/v1.schema.json` (generated), `tests/contracts/protocol_v1/{requests,replies}.jsonl`, `tests/contracts/test_protocol.py`, `tests/adapters/transport/test_mapping.py`, `scripts/gen_schema.py`.

**Interfaces:**
- Produces:
  - `contracts.protocol`: `PROTOCOL_VERSION = 1`, `MAX_FRAME = 1 << 20`; `class ErrorCode(StrEnum)`: `bad_frame, bad_version, unknown_command, invalid_argument, unknown_voice, unknown_engine, invalid_duration, invalid_setting, engine_failed, audio_failed, unsupported, internal`; msgspec tagged-union request structs (tag field `cmd`): `speak{text, voice?, lang?, rate?, now=false}`, `pause`, `resume`, `skip`, `back`, `stop_reading`, `status`, `idle{timeout}`, `unload`, `shutdown`, `subscribe` (returns `unsupported` in MVP); every request has `v: int` and `id: str`. Reply `{v, id, ok, data?, error?{code, message}}`. Event envelope `{v, event, data}` defined but unused in MVP.
  - `contracts.codec`: `encode_line(msg: Struct) -> bytes` (adds `\n`), `decode_request(line: bytes) -> Request`, `decode_reply(line: bytes) -> Reply`; `ProtocolError(code: ErrorCode, message: str, request_id: str | None = None)` for oversize, invalid JSON, `v != 1`, unknown `cmd`, wrong types.
  - `adapters.transport.mapping`: `to_command(req: Request) -> Command`, `to_wire_reply(req_id: str, reply: app Reply) -> wire Reply` (maps `DomainError.code` → `ErrorCode`, unknown codes → `internal`), `error_reply(err: ProtocolError) -> wire Reply` (the reply `id` is `err.request_id`, null when unreadable).
  - `scripts/gen_schema.py` writes `msgspec.json.schema(...)` to `docs/protocol/v1.schema.json`.

- [ ] **Step 1: Failing tests:** every line of `requests.jsonl` decodes and re-encodes to the same JSON (golden); `decode_request` failures: `b"{bad"` → `bad_frame`, `MAX_FRAME + 1` bytes → `bad_frame`, `{"v":2,...}` → `bad_version`, `{"v":1,"id":"1","cmd":"rm"}` → `unknown_command`, `speak` with `rate:"x"` → `invalid_argument`; unknown extra fields are ignored (forward compatibility); `test_schema_is_current`: generated schema equals the committed file; mapping round-trip for each command; `UnknownVoice` → `unknown_voice` with message listing voices.
- [ ] **Step 2: Run** `uv run pytest tests/contracts/test_protocol.py tests/adapters/transport/test_mapping.py -v` → FAIL.
- [ ] **Step 3: Implement;** write `docs/protocol.md` (framing, every command with an example request/reply, error codes, additive-only rule, `caps`).
- [ ] **Step 4: Run** → PASS; `poe check` green.
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `feat: versioned socket protocol v1 with golden fixtures and schema`
- [ ] **Step 6: ⛔ OWNER CHECKPOINT.** Controller stops and asks the owner to review `docs/protocol.md` and `tests/architecture/api_snapshot.txt` (whole frozen surface so far). Do not start Task 9 without an explicit owner approval; requested changes go through the API change request flow.

---

### Task 9: Runtime loop, engine worker, Unix server, daemon wiring

**Files:** Create `runtime/loop.py`, `runtime/engine_worker.py`, `runtime/buffers.py`, `runtime/events.py` (minimal), `adapters/transport/unix_server.py`, `bootstrap/daemon.py`, `entrypoints/daemon.py`, tests `tests/runtime/test_loop.py`, `tests/runtime/test_engine_worker.py`, `tests/adapters/transport/test_unix_server.py`, `tests/e2e/test_daemon_fake.py`.

**Interfaces:**
- Consumes: `Session`, effects, callbacks (Task 7); `Engine`, `get_or_build_prompt` (Task 4); `AudioOutput` (Task 5); codec + mapping (Task 8); `PlatformInfo` (Task 1).
- Produces:
  - `runtime.engine_worker.EngineWorker(provider: EngineProvider, repo: VoiceRepository, stretch: Callable[[NDArray, float], NDArray])` — one daemon thread, FIFO `submit(job, on_result, on_error)`; jobs: load, unload, prepare prompt, generate (synthesize + join chunks + stretch by `chunk.rate`); `close()` joins the thread.
  - `runtime.buffers.AudioBuffers(max_items: int = 8)` — `put(chunk_id, AudioChunk)`, `take(chunk_id)`, `drop(ids)`; raises if exceeding `max_items` (guards Review Focus 3 at runtime).
  - `runtime.loop.CoreLoop(session: Session, worker: EngineWorker, output: AudioOutput, buffers: AudioBuffers, clock: Clock)` — `execute(effects)`, marshals worker/output callbacks with `loop.call_soon_threadsafe`, wake timers via `loop.call_at` (no polling); `async handle(cmd) -> Reply`; `async run_until_stopped()`.
  - `adapters.transport.unix_server.serve_unix(path: Path, handler: Callable[[bytes], Awaitable[bytes]], *, on_ready: Callable[[], None] | None) -> asyncio.Server` — `start_unix_server`, `chmod 0600`, reads lines with `limit=MAX_FRAME + 1`, oversize → error reply and connection close; multiple requests per connection.
  - `bootstrap.daemon.build_daemon(settings: Settings, *, transport: bool, platform: PlatformInfo | None = None, output: AudioOutput | None = None) -> Daemon` (`Daemon` exposes `loop: CoreLoop`, `async run()`); picks `registry.load_engine` as provider, `FsVoiceRepository`, `SoundDeviceOutput` unless `output` is injected, `time_stretch`.
  - `entrypoints.daemon.serve(settings: Settings) -> int` — `ensure_supported`; if a live daemon answers `status` → print `already running`, return 1; remove stale socket; write pid file; rotating log `log_path()` (1 MB × 2); run; on exit remove socket + pid.

- [ ] **Step 1: Failing tests:** `test_engine_worker`: jobs run in order on a single thread (record `threading.get_ident()`), errors go to `on_error`, `close()` joins; `test_loop`: with `FakeEngine` + `MemoryOutput(auto_complete=True)` a 3-chunk `Speak` plays 3 chunks and `Status` ends idle; idle unload fires via `call_at` with a tiny timeout (0.05 s) without polling; `AudioBuffers` never holds > 2 + 1 items during a 100 000-char reading (**Review Focus 3**, threaded); `test_unix_server`: socket mode `0o600`, `status` round-trip, `{"v":9,...}` → `bad_version` reply and the next request on a new connection still works, oversize frame → `bad_frame`; `test_daemon_fake` (`-m e2e` not required — runs by default with `VOXLEKTOR_ENGINE=fake`, `VOXLEKTOR_HOME=tmp`, `MemoryOutput` injected via `build_daemon(output=...)`): start in a thread, send `speak`, `status` shows `WARM`, `shutdown` removes the socket and pid file.
- [ ] **Step 2: Run** `uv run pytest tests/runtime tests/adapters/transport tests/e2e -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS; `poe check` green.
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `feat: runtime loop, engine worker, unix server and daemon wiring`

---

### Task 10: Client, spawn, CLI and one-shot mode

**Files:** Create `adapters/transport/unix_client.py`, `adapters/procs/spawn.py`, `bootstrap/client.py`, `entrypoints/cli/{app,speak,control,voices,config,output}.py`, `entrypoints/oneshot.py`, tests `tests/adapters/transport/test_unix_client.py`, `tests/adapters/procs/test_spawn.py`, `tests/entrypoints/cli/test_*.py`, `tests/e2e/test_cli_fake.py`. Modify `tests/architecture/test_canaries.py` (remove the `xfail` on `test_cli_import_is_light`).

**Interfaces:**
- Produces:
  - `unix_client.request(path: Path, req: dict[str, object], timeout: float = 5.0) -> dict[str, object]` (blocking `socket`, generates `id`, adds `v`); raises `NotRunning(ServiceUnavailable)` when nobody listens (missing socket / `ENOENT` / `ECONNREFUSED`) and plain `ServiceUnavailable` for a daemon that is there but does not answer properly (timeout over the whole exchange, EOF, undecodable or foreign reply) — owner decision L1, Task 10 review.
  - `spawn.ensure_daemon(platform: PlatformInfo, ping: Callable[[], bool], wait_s: float = 15.0) -> None` — if `ping()` false: remove a stale REGULAR file at the socket path (ruling R34: a refused socket is left to the daemon's flock-protected cleanup); `subprocess.Popen([sys.executable, "-m", "voxlektor", "serve"], start_new_session=True, stdin=DEVNULL, stdout/stderr → log file)`; poll `ping` every 0.1 s until true or raise `ServiceUnavailable`.
  - `bootstrap.client.build_client(flags: Mapping[str, object]) -> ClientContext` (settings merged from store + env + flags, platform, `send(cmd: str, **args) -> dict`, `ensure()`), importing nothing heavy.
  - CLI commands (typer, split by file as named): `say TEXT…`, `serve [--idle DUR]`, `pause`, `resume`, `skip`, `back`, `stop-reading`, `status`, `idle DUR`, `unload`, `stop`, `voices`, `config voice NAME`, `config show`. Speak options `--voice/-v`, `--lang/-l`, `--rate/-r`, `--now`, `-f FILE`.
  - `entrypoints.cli.app.main(argv: list[str] | None = None) -> int` — dispatch: first token is a command name → that command; `--` → rest is text; otherwise all non-option tokens are text for `say`; no args + non-TTY stdin → stdin is text; no args + TTY → help.
  - Speak path (`speak.py`): resolve text (arg / `-f` / stdin); `split_text` empty → `nothing to read`, exit 0, no socket; `autostart` true → `ensure()` then `send("speak")`; false → `oneshot.run(text, settings)`.
  - `entrypoints.oneshot.run(text: str, settings: Settings) -> int` — `build_daemon(settings, transport=False)`, speak, wait until idle, Ctrl+C → `StopReading`, unload, exit 0.
  - `output.py`: `status` one-liners (state, engine, voice, idle remaining, last metrics; no RAM line — ruling R32 withdrawn by owner decision M1); `stop` / `status` when not running (`NotRunning`) → `not running` / `OFF`, exit 0; any other `ServiceUnavailable` (including for `stop` / `status`) → message, exit 2; `DomainError` → message, exit 1; Ctrl+C → exit 130 (interrupted).

- [ ] **Step 1: Failing tests:** **Review Focus 1** `main([""])` and stdin `"\n"` → `nothing to read`, exit 0, no `request` calls (monkeypatched); **Review Focus 4** `["pause"]` → `pause`; `["say","pause"]`, `["--","pause"]` → speak `"pause"`; `["hello","world"]` → speak `"hello world"`; flags `-v m1-calm -r 1.25 hej` → request `voice`, `rate`; `-f notes.txt` → file text; `stop` with `ServiceUnavailable` → `not running`, 0; `config voice f2-narrator` persists; unknown voice → exit 1 listing names; **Review Focus 2** `test_spawn.py`: regular file at socket path + fake `Popen` that starts the fake daemon in a thread → `ensure_daemon` returns and the file was replaced by a live socket; `test_cli_fake.py` (e2e, real subprocess with `VOXLEKTOR_ENGINE=fake` and `VOXLEKTOR_OUTPUT=memory` honoured by `bootstrap.daemon` for tests only): `vxl "test"` → exit 0, `vxl status` → `WARM`, `vxl stop`, `vxl status` → `OFF`. Canary `test_cli_import_is_light` now passes.
- [ ] **Step 2: Run** `uv run pytest tests/adapters/transport tests/adapters/procs tests/entrypoints tests/e2e tests/architecture -v` → FAIL.
- [ ] **Step 3: Implement** (add `VOXLEKTOR_OUTPUT=memory` handling in `bootstrap/daemon.py`, documented in `CLAUDE.md` as test-only).
- [ ] **Step 4: Run** full `uv run poe check` → green.
- [ ] **Step 5: Manual smoke:** `uv run vxl "Cześć, to jest test voxlektora."` → audible; `uv run vxl status` → `WARM`; `uv run vxl stop`; `uv run vxl status` → `OFF`.
- [ ] **Step 6: Report to owner (owner commits)** — suggested message: `feat: CLI with daemon autostart, stale socket recovery and one-shot mode`

---

### Task 11: Gap benchmark, install, README

**Files:** Create `scripts/bench_gaps.py`, `README.md`.

- [ ] **Step 1: `scripts/bench_gaps.py`:** `build_daemon(settings, transport=False, output=<realtime MemoryOutput that sleeps for each chunk's duration>)`; voices `v0-default, f2-narrator, m1-calm` × rates `1.0, 1.25, 1.5` × langs `pl, en`, using the 3-paragraph texts from `~/development_local/claude-mods/tts-lab/paras.py` (copied into the script); print a table `voice | rate | lang | first_audio_s | underruns | gen/audio ratio` from `Status`. Run `uv run python scripts/bench_gaps.py` → table, no exceptions.
- [ ] **Step 2: Install:** `uv tool install --editable .` → `which vxl voxlektor` resolve to `~/.local/bin`; `vxl "Instalacja działa."` is audible.
- [ ] **Step 3: README.md:** requirements verbatim from the spec ("macOS 14+ on Apple Silicon, 8 GB RAM (16 GB recommended), ~4 GB disk. Other platforms: not yet."); install via `uv tool install`; command cheat-sheet; config + env table; keep-warm lifecycle diagram; links to `docs/architecture.md`, `docs/protocol.md`; the bench table under "Measured performance".
- [ ] **Step 4:** `uv run poe check` green.
- [ ] **Step 5: Report to owner (owner commits)** — suggested message: `docs: README and gap benchmark results`

---

## After the plan

- Three independent reviews of the whole branch: **security** (socket perms, spawn argv, protocol limits, file input, model revision pinning, hook scripts), **bugs/correctness** (worker/loop/callback marshalling, epochs, keep-warm vs pause, error recovery), **architecture contracts** (ignore lists, `typings/` stubs, coverage exclusions, map completeness).
- Then Etap 2 (dialogue-box engine speaking protocol v1) gets its own spec/plan on a separate branch of `claude-mods`.