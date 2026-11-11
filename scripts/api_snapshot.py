# pyright: basic
"""Snapshot of the frozen public API surface.

Writes one sorted line per public symbol of the frozen modules to
``tests/architecture/api_snapshot.txt``:

* ``module:Name(signature) -> return`` for functions and classes
* ``module:Name.field: type`` for dataclass and msgspec struct fields
* ``module:Name.MEMBER = value`` for enum members
* ``cli:command --option (default)`` for the typer command tree

Modes:
  --check    exit 1 on any difference and print it
  --add      write only if every old line is still present unchanged
  --approve  write unconditionally (owner only)
"""

import argparse
import dataclasses
import enum
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "tests" / "architecture" / "api_snapshot.txt"

# Extended by each task as frozen modules appear.
FROZEN_MODULES: tuple[str, ...] = (
    "voxlektor.domain.types",
    "voxlektor.domain.errors",
    "voxlektor.domain.settings",
    "voxlektor.domain.text.chunking",
    "voxlektor.domain.text.lang",
    "voxlektor.contracts.engine",
    "voxlektor.contracts.voicefmt",
    "voxlektor.app.ports",
    "voxlektor.app.voices",
    "voxlektor.adapters.storage.voices_fs",
    "voxlektor.adapters.storage.settings_toml",
    "voxlektor.adapters.platform.macos",
    "voxlektor.adapters.platform.unsupported",
    "voxlektor.bootstrap.platform_select",
    "voxlektor.app.voice_prompts",
    "voxlektor.adapters.engines.omnivoice_mlx",
    "voxlektor.bootstrap.environment",
    "voxlektor.bootstrap.registry",
    "voxlektor.domain.playback",
    "voxlektor.domain.keepwarm",
    "voxlektor.domain.reading",
    "voxlektor.adapters.audio.dsp",
    "voxlektor.adapters.audio.sounddevice_out",
    "voxlektor.adapters.audio.memory_out",
    "voxlektor.testing.fakes",
    "voxlektor.testing.engine_contract",
    "voxlektor.app.commands",
    "voxlektor.app.effects",
    "voxlektor.app.session",
    "voxlektor.testing.simulator",
)
# Typer applications as "module:attribute", extended by the CLI task.
FROZEN_CLI: tuple[str, ...] = ()

APPROVAL_MESSAGE = "API change requires owner approval — file an API change request"


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj, eval_str=True))
    except (TypeError, ValueError, NameError):
        try:
            return str(inspect.signature(obj))
        except (TypeError, ValueError):
            return "(...)"


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _public_names(module: Any) -> list[str]:
    declared: list[str] | None = getattr(module, "__all__", None)
    if declared is not None:
        return sorted(declared)
    return sorted(
        name
        for name, value in vars(module).items()
        if _is_public(name)
        and not inspect.ismodule(value)
        and getattr(value, "__module__", module.__name__) == module.__name__
    )


def _annotation(tp: Any) -> str:
    return tp if isinstance(tp, str) else inspect.formatannotation(tp)


def _class_lines(prefix: str, cls: type) -> list[str]:
    lines = [f"{prefix}{_signature(cls)}"]
    if issubclass(cls, enum.Enum):
        lines += [f"{prefix}.{member.name} = {member.value!r}" for member in cls]
        return lines
    if dataclasses.is_dataclass(cls):
        lines += [f"{prefix}.{f.name}: {_annotation(f.type)}" for f in dataclasses.fields(cls)]
    struct_fields: tuple[str, ...] | None = getattr(cls, "__struct_fields__", None)
    if struct_fields is not None:
        hints = inspect.get_annotations(cls)
        lines += [f"{prefix}.{name}: {_annotation(hints.get(name, '?'))}" for name in struct_fields]
    for name, member in sorted(vars(cls).items()):
        if _is_public(name) and inspect.isfunction(member):
            lines.append(f"{prefix}.{name}{_signature(member)}")
    return lines


def module_lines(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    lines: list[str] = []
    for name in _public_names(module):
        value = getattr(module, name)
        prefix = f"{module_name}:{name}"
        if inspect.isclass(value):
            lines += _class_lines(prefix, value)
        elif callable(value):
            lines.append(f"{prefix}{_signature(value)}")
        else:
            lines.append(f"{prefix} = {value!r}")
    return lines


def cli_lines(spec: str) -> list[str]:
    import typer.main  # noqa: PLC0415

    module_name, attribute = spec.split(":")
    app = getattr(importlib.import_module(module_name), attribute)
    root = typer.main.get_command(app)
    lines: list[str] = []

    def walk(command: Any, path: str) -> None:
        for param in command.params:
            options = " ".join(getattr(param, "opts", [param.name]))
            lines.append(f"cli:{path} {options} (default={param.default!r})")
        for name, sub in sorted(getattr(command, "commands", {}).items()):
            lines.append(f"cli:{path} {name}")
            walk(sub, f"{path} {name}")

    walk(root, root.name or "vxl")
    return lines


def collect(modules: tuple[str, ...], cli: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for module_name in modules:
        lines += module_lines(module_name)
    for spec in cli:
        lines += cli_lines(spec)
    return sorted(set(lines))


def _read(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line]


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines))


def _diff(old: list[str], new: list[str]) -> list[str]:
    removed = [f"- {line}" for line in old if line not in new]
    added = [f"+ {line}" for line in new if line not in old]
    return removed + added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, description=__doc__.splitlines()[0] if __doc__ else ""
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--add", action="store_true")
    mode.add_argument("--approve", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--module", action="append", dest="modules", default=None)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    modules = tuple(args.modules) if args.modules is not None else FROZEN_MODULES
    cli = () if args.modules is not None else FROZEN_CLI
    new = collect(modules, cli)
    old = _read(args.snapshot)

    if args.approve:
        _write(args.snapshot, new)
        return 0
    if args.check:
        if old == new:
            return 0
        print("API snapshot differs from the code:", file=sys.stderr)
        print("\n".join(_diff(old, new)), file=sys.stderr)
        return 1
    # --add: only pure additions are accepted.
    if any(line not in new for line in old):
        print(APPROVAL_MESSAGE, file=sys.stderr)
        print("\n".join(_diff(old, new)), file=sys.stderr)
        return 1
    _write(args.snapshot, new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
