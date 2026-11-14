# pyright: basic
"""Snapshot of the frozen public API surface.

Writes one sorted line per public symbol of the frozen modules to
``tests/architecture/api_snapshot.txt``:

* ``module:Name(signature) -> return`` for functions and classes
  (``module:async name(...)`` / ``module:async Class.method(...)`` for coroutine functions)
* ``module:Name.field: type`` for dataclass and msgspec struct fields (inherited struct fields
  resolved across the MRO) and for class-level annotations of other classes (Protocol attributes)
* ``module:Name.prop: property -> type`` for properties
* ``module:Name [tag_field='cmd', tag='speak']`` for tagged msgspec structs
* ``module:Name: TypeAlias = body`` for ``type X = ...`` aliases and plain aliases
  (``X = A | B``, ``X = Literal[...]``, ``X = float``) assigned in the module itself
* ``module:Name.MEMBER = value`` for enum members
* ``cli:command --option (default)`` for the typer command tree

A signature that cannot be read is an error (the snapshot never records a ``(...)`` placeholder).

Modes:
  --check    exit 1 on any difference and print it
  --add      write only if every old line is still present unchanged
  --approve  write unconditionally (owner only)
"""

import argparse
import ast
import dataclasses
import enum
import importlib
import inspect
import sys
import types
import typing
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
    "voxlektor.contracts.protocol",
    "voxlektor.contracts.codec",
    "voxlektor.adapters.transport.mapping",
    "voxlektor.runtime.engine_worker",
    "voxlektor.runtime.buffers",
    "voxlektor.runtime.loop",
    "voxlektor.adapters.transport.unix_server",
    "voxlektor.adapters.audio.cues",
    "voxlektor.bootstrap.daemon",
    "voxlektor.entrypoints.daemon",
    "voxlektor.adapters.transport.unix_client",
    "voxlektor.adapters.procs.spawn",
    "voxlektor.bootstrap.client",
    "voxlektor.entrypoints.cli.app",
    "voxlektor.entrypoints.oneshot",
)
# Typer applications as "module:attribute", extended by the CLI task.
FROZEN_CLI: tuple[str, ...] = ("voxlektor.entrypoints.cli.app:app",)

APPROVAL_MESSAGE = "API change requires owner approval — file an API change request"


# Dunder methods that are part of a public interface (``EngineProvider.__call__``).
ALLOWED_DUNDERS: frozenset[str] = frozenset({"__call__"})


class SnapshotError(Exception):
    """The tool cannot describe a public symbol; fix the tool instead of recording a guess."""


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj, eval_str=True))
    except (TypeError, ValueError, NameError):
        pass
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        pass
    # Exception subclasses that keep the builtin constructor have no introspectable signature
    # (e.g. ``testing.simulator.SimulationError(AssertionError)``): they take ``*args``.
    if inspect.isclass(obj) and issubclass(obj, BaseException) and "__init__" not in vars(obj):
        return "(*args: object)"
    raise SnapshotError(f"cannot read the signature of {obj!r}; teach scripts/api_snapshot.py")


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _is_alias(value: Any) -> bool:
    """A type alias value: ``type X = ...``, a union, a subscripted generic or a plain class."""
    return (
        isinstance(value, typing.TypeAliasType | types.UnionType | type)
        or typing.get_origin(value) is not None
    )


def _assigned_names(module: Any) -> set[str]:
    """Names bound by a top-level assignment in the module's own source (not by an import)."""
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.TypeAlias):
            names.add(node.name.id)
    return names


def _public_names(module: Any) -> list[str]:
    declared: list[str] | None = getattr(module, "__all__", None)
    if declared is not None:
        return sorted(declared)
    assigned = _assigned_names(module)
    return sorted(
        name
        for name, value in vars(module).items()
        if _is_public(name)
        and not inspect.ismodule(value)
        and (
            getattr(value, "__module__", module.__name__) == module.__name__
            or (name in assigned and _is_alias(value))
        )
    )


def _annotation(tp: Any) -> str:
    return tp if isinstance(tp, str) else inspect.formatannotation(tp)


def _annotations(obj: Any) -> dict[str, Any]:
    try:
        return inspect.get_annotations(obj, eval_str=True)
    except NameError:  # names imported only under TYPE_CHECKING
        return inspect.get_annotations(obj)


def _async(func: Any) -> str:
    return "async " if inspect.iscoroutinefunction(func) else ""


def _struct_hint(cls: type, name: str) -> Any:
    for klass in cls.__mro__:
        hints = inspect.get_annotations(klass)
        if name in hints:
            return hints[name]
    raise SnapshotError(f"no annotation for struct field {cls.__qualname__}.{name}")


def _member_lines(module_name: str, cls_name: str, cls: type) -> list[str]:
    lines: list[str] = []
    prefix = f"{module_name}:{cls_name}"
    for name, member in sorted(vars(cls).items()):
        if not (_is_public(name) or name in ALLOWED_DUNDERS):
            continue
        if isinstance(member, staticmethod | classmethod):
            member = member.__func__  # noqa: PLW2901
        if isinstance(member, property):
            if member.fget is None:
                raise SnapshotError(f"property {cls.__qualname__}.{name} has no getter")
            ret = inspect.signature(member.fget, eval_str=True).return_annotation
            lines.append(f"{prefix}.{name}: property -> {_annotation(ret)}")
        elif inspect.isfunction(member):
            sig = _signature(member)
            lines.append(f"{module_name}:{_async(member)}{cls_name}.{name}{sig}")
    return lines


def _class_lines(module_name: str, name: str, cls: type) -> list[str]:
    prefix = f"{module_name}:{name}"
    lines = [f"{prefix}{_signature(cls)}"]
    if issubclass(cls, enum.Enum):
        lines += [f"{prefix}.{member.name} = {member.value!r}" for member in cls]
        return lines
    struct_fields: tuple[str, ...] | None = getattr(cls, "__struct_fields__", None)
    if dataclasses.is_dataclass(cls):
        lines += [f"{prefix}.{f.name}: {_annotation(f.type)}" for f in dataclasses.fields(cls)]
    elif struct_fields is not None:
        lines += [f"{prefix}.{n}: {_annotation(_struct_hint(cls, n))}" for n in struct_fields]
        config = getattr(cls, "__struct_config__")  # noqa: B009 (msgspec dunder, untyped)
        if config.tag is not None:
            lines.append(f"{prefix} [tag_field={config.tag_field!r}, tag={config.tag!r}]")
    else:
        lines += [
            f"{prefix}.{attr}: {_annotation(tp)}"
            for attr, tp in _annotations(cls).items()
            if _is_public(attr)
        ]
    lines += _member_lines(module_name, name, cls)
    return lines


def module_lines(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    assigned = _assigned_names(module)
    lines: list[str] = []
    for name in _public_names(module):
        value = getattr(module, name)
        prefix = f"{module_name}:{name}"
        if isinstance(value, typing.TypeAliasType):
            lines.append(f"{prefix}: TypeAlias = {_annotation(value.__value__)}")
        elif inspect.isclass(value) and value.__module__ != module_name and name in assigned:
            lines.append(f"{prefix}: TypeAlias = {_annotation(value)}")
        elif inspect.isclass(value):
            lines += _class_lines(module_name, name, value)
        elif typing.get_origin(value) is not None or isinstance(value, types.UnionType):
            lines.append(f"{prefix}: TypeAlias = {_annotation(value)}")
        elif callable(value):
            lines.append(f"{module_name}:{_async(value)}{name}{_signature(value)}")
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
