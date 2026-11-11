# voxlektor: architecture

**Status:** adopted 2026-11-10 (variant A, "layers with a clean core", with two borrowings). Source: an architect's review comparing 3 structures (A: ports and adapters, B: per-feature modules, C: microkernel with plugins). B rejected, because after a few months the god module and threads in every module come back. C rejected, because it freezes the API prematurely and carries package overhead.

This document is a **map of "what lives where"**. Dependency rules are enforced automatically (`import-linter`, pyright strict, ruff): a violation means a red `check`, not a matter of discipline.

## 1. Layers and dependency direction

```
entrypoints   CLI, daemon, one-shot, (render)
    ↓
bootstrap     wiring: the only place that picks concrete adapters
    ↓
adapters | runtime   technology (MLX, sounddevice, socket, disk) | threads + asyncio loop, no technology
    ↓
app           use cases; Session = ALL mutable state; synchronous; ports only
    ↓
domain | contracts   pure logic with no I/O and no numpy | public, versioned contracts
```
`|` denotes siblings that must not import each other. `testing/` (fakes, simulator, engine contract suite) is never imported by production code.

## 2. Package tree

```
src/voxlektor/
  __init__.py                 only __version__ (no imports, so that `vxl` starts fast)
  __main__.py                 -> entrypoints.cli.app:main
  bundled_voices/<7>/{ref.wav,ref.txt,voice.toml}

  contracts/                  PUBLIC, VERSIONED
    protocol.py               PROTOCOL_VERSION=1, MAX_FRAME, Request/Reply/Event (msgspec tagged unions), ErrorCode, Capability
    codec.py                  encode_line / decode_request / decode_reply; ProtocolError(code)
    engine.py                 ENGINE_API=1, Engine SPI, EngineSpec, VoiceRef, VoicePrompt, AudioChunk
    voicefmt.py               VOICE_FORMAT=1, VoiceMeta (voice.toml), file names, prompt_cache_name(), ref limits

  domain/                     PURE: sync, no I/O, no numpy
    types.py                  NewType VoiceName/EngineId/ChunkId/Epoch; Lang; LangPref; Rate; Seconds
    errors.py                 DomainError with a stable `code`
    settings.py               Settings, merge_layers(), parse_duration()
    keepwarm.py               COLD/LOADING/WARM/UNLOADING; transition(state, event) -> (state, actions); time arrives in events
    playback.py               PlaybackQueue[A]: queued/current/history(20); skip/back/stop/pause
    reading.py                Chunk; AheadWindow (max. 2 ahead); UnderrunTracker; plan_reading() (per-chunk language, same-language merge)
    text/chunking.py          split_text()
    text/lang.py              detect_lang()
    text/limits.py            MAX_CHARS / MERGE_BELOW shared by split_text and plan_reading (not frozen)

  app/                        SYNC, no threads and no asyncio; ports only
    ports.py                  Clock, AudioOutput, VoiceRepository, SettingsStore, PlatformInfo, EngineProvider
    commands.py               application commands (independent of the protocol version)
    effects.py                Effect: LoadEngine, UnloadEngine, PrepareVoice, Generate, StartPlayback, Pause/Resume/AbortPlayback, ScheduleWake, Emit, Exit
    session.py                Session.handle(cmd, now) -> Outcome(reply, effects); on_loaded/on_generated/on_gen_failed/on_play_done/on_play_failed/on_wake
    voice_prompts.py          get_or_build_prompt(): cache via VoiceRepository + Engine.serialize_prompt
    voices.py                 VoiceCatalog: list/resolve/ensure_bundled

  runtime/                    imperative shell: threads + loop; knows ports, not technologies
    loop.py                   CoreLoop: asyncio, calls Session, executes Effects, call_soon_threadsafe, wake timers
    engine_worker.py          EngineWorker: ONE thread, FIFO, epochs, cooperative cancellation
    buffers.py                AudioBuffers: ChunkId -> AudioChunk (numpy outside Session), bounded
    events.py                 EventHub (groundwork for subscribe; unused by the CLI in the MVP)

  adapters/                   technology; families independent of each other
    engines/omnivoice_mlx.py  OmniVoice via mlx-audio; pinned revision; returns numpy copies; NO file I/O
    audio/sounddevice_out.py  AudioOutput on a PortAudio callback; pause = silence
    audio/dsp.py              time_stretch() (pytsmod WSOLA)
    audio/memory_out.py       MemoryOutput: device-less AudioOutput (VOXLEKTOR_OUTPUT=memory; also used by tests)
    storage/voices_fs.py      on-disk VoiceRepository per contracts.voicefmt; bundled voices via importlib.resources; atomic cache writes
    storage/settings_toml.py  SettingsStore: TOML + VOXLEKTOR_*
    platform/macos.py         paths (0700), hardware
    platform/unsupported.py   "platform not supported yet"
    transport/mapping.py      wire v1 <-> application commands (the ONLY place that knows both)
    transport/unix_server.py  asyncio server: frames, MAX_FRAME, 0600
    transport/unix_client.py  blocking client (no asyncio)
    procs/spawn.py            daemon spawn (argv, start_new_session), pidfile, stale socket

  bootstrap/
    environment.py            process settings (HF_HUB_DISABLE_IMPLICIT_TOKEN) before engines are loaded
    registry.py               engines via `voxlektor.engines` entry points (lazily)
    platform_select.py        sys.platform/machine -> platform adapter
    client.py                 lightweight graph for the CLI (transport + spawn + settings)
    daemon.py                 build_daemon(settings, *, transport: bool) -> CoreLoop

  entrypoints/
    cli/{app,speak,control,voices,config,output}.py
    daemon.py                 `serve`
    oneshot.py                in-process reading (CoreLoop without transport)

  testing/
    fakes.py                  FakeEngine, MemoryOutput, FakeClock, InMemoryVoiceRepository, InMemorySettings
    simulator.py              deterministic driver: Session + immediate worker + fake output
    engine_contract.py        suite that every engine passes
    protocol_fixtures.py      golden JSONL fixtures
```
Phase 3 adds (without changing the structure): `domain/text/markdown.py`, `app/{engines,doctor,render}.py`, `adapters/engines/voxcpm2_mlx.py`, `adapters/audio/wavfile.py`, `adapters/models/hf_hub.py`, `adapters/platform/linux.py`, `entrypoints/{render.py,cli/setup.py}`.

## 3. Threads: who owns what

| Thread / process | Owner | Does | Never |
|---|---|---|---|
| Daemon main (asyncio) | `runtime.loop.CoreLoop` | socket server, **all Session calls**, Effects, timers | does not block > a few ms; does not call the engine |
| Engine thread (exactly 1) | `runtime.engine_worker` | every Engine SPI call, time-stretch, copy to numpy | does not change Session state; hands results back via `call_soon_threadsafe` |
| PortAudio callback | `adapters.audio.sounddevice_out` | copies frames, silence while paused, end/error signal via `call_soon_threadsafe` | does not allocate, log or call app |
| CLI process | `entrypoints.cli` + `unix_client` | blocking socket I/O | does not import asyncio, numpy or MLX on the control path |
| One-shot | `entrypoints.oneshot` | the same CoreLoop without transport | has no reading logic of its own |

**Cancellation via epochs:** `--now`, `stop_reading` and `unload` bump the epoch; Session discards results with a stale epoch. A single OmniVoice `generate` cannot be interrupted, so in the worst case one chunk is wasted.

**MLX / audio rules:** MLX arrays never leave the engine thread. `unload` on the same thread as `load`. The daemon starts via `subprocess` (argv), never via `os.fork` after importing MLX/PortAudio. Audio output with `latency="high"`; playback starts once the first chunk is buffered.

## 4. Public contracts

| Contract | Source | Derived | Rules |
|---|---|---|---|
| **Socket protocol v1** | `contracts/protocol.py` + `codec.py` | `docs/protocol.md`, `docs/protocol/v1.schema.json` (generated; CI fails when out of date), `tests/contracts/protocol_v1/*.jsonl` | 1 JSON object per line, ≤ 1 MiB, `"v":1`; every request has an `id` echoed in the reply; errors `{"code","message"}` from a closed `ErrorCode`; events `{"v","event",…}` only after `subscribe`; in v1 changes are strictly additive, new commands advertised in `status.caps` |
| **Engine SPI** | `contracts/engine.py` (`ENGINE_API=1`) + `voxlektor.engines` entry points | `testing/engine_contract.py` | `EngineSpec` (id, revision, sample_rate, languages, capabilities ⊆ {LIVE, RENDER, CLONE, DESIGN}); `load`, `unload`, `prepare_voice(VoiceRef) -> VoicePrompt`, `serialize_prompt`/`deserialize_prompt` (bytes + `prompt_format`), `synthesize(text, lang, prompt) -> Iterator[AudioChunk]`; engines do not touch the file system |
| **Voice format** | `contracts/voicefmt.py` (`VOICE_FORMAT=1`) | `docs/voice-format.md` | `voices/<name>/ref.wav` (mono, ≤ 30 s), `ref.txt`, `voice.toml` = source of truth; `cache/<engine>-<rev8>-p<prompt_format>.bin` disposable; only `cache/` is ever deleted |
| Config (semi-public) | `domain/settings.py` + `adapters/storage/settings_toml.py` | table in README | `config_version = 1`; unknown keys preserved on write |
| CLI (semi-public) | `entrypoints/cli/app.py` | README | exit codes: 0 ok, 1 user error, 2 service unavailable, 3 unsupported platform |

## 5. How to add…

| What | Where |
|---|---|
| An engine | `adapters/engines/<id>.py` implementing `contracts.engine.Engine` + entry point in `pyproject.toml` + passing `testing/engine_contract.py` (`-m model`) |
| A platform | `adapters/platform/<os>.py` + an entry in `bootstrap/platform_select.py` |
| A client (e.g. dialogue-box) | speaks protocol v1 over the socket; zero dependencies on the Python code |
| A command | `contracts/protocol.py` (additively) → `adapters/transport/mapping.py` → `app/commands.py` + `app/session.py` → `entrypoints/cli/…` |
| Logic | `domain/` first (pure, 100% coverage), then `app/` |

Detailed step-by-step guide: the `.claude/skills/voxlektor-architecture/SKILL.md` skill.

## 6. Quality gate (`uv run poe check`)

`ruff check` + `ruff format --check` → `lint-imports` (contracts in `pyproject.toml`) → `pyright` (strict, `typings/` for `mlx_audio`, `sounddevice`, `pytsmod`) → `pytest --cov` (branch; 100% for `domain`/`app`/`contracts`, ≥ 95% overall; excluded only `adapters/engines/*_mlx.py` and `adapters/audio/sounddevice_out.py`, covered by `-m model` / `-m audio`).

**Stdlib guard (canary result, Task 0):** `import-linter` 2.16 sees stdlib modules, so `lint-imports` is the primary guard (`test_import_linter_catches_stdlib_violation`). Two limits: it rejects stdlib *submodules* (`logging.handlers`, `importlib.metadata`), so `importlib` is forbidden as a whole in `domain` and `app`, and `logging.handlers` is banned by Ruff TID251 (also asserted by `test_ruff_bans_threading_in_domain`). Contracts that name modules must find them, so empty placeholder modules exist for `contracts/{protocol,codec,engine,voicefmt}.py`, `bootstrap/daemon.py`, `entrypoints/oneshot.py` and `entrypoints/cli/speak.py` until their tasks fill them; the two contracts with `ignore_imports` set `unmatched_ignore_imports_alerting = "none"` for the same reason.
