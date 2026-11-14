"""The API snapshot tool: --add accepts pure additions only, --check fails on any diff."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "api_snapshot.py"

V1 = "def alpha(x: int) -> str:\n    return str(x)\n"
V2 = V1 + "\n\ndef beta(y: str) -> int:\n    return len(y)\n"
V3 = "def alpha(x: str) -> str:\n    return x\n"


def _run(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(SCRIPT),
            mode,
            "--snapshot",
            str(tmp_path / "snap.txt"),
            "--module",
            "fakeapi",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _set_module(tmp_path: Path, source: str) -> None:
    (tmp_path / "fakeapi.py").write_text(source)


def test_approve_then_check_passes(tmp_path: Path) -> None:
    _set_module(tmp_path, V1)
    assert _run(tmp_path, "--approve").returncode == 0
    assert (tmp_path / "snap.txt").read_text() == "fakeapi:alpha(x: int) -> str\n"
    assert _run(tmp_path, "--check").returncode == 0


def test_add_accepts_a_new_line(tmp_path: Path) -> None:
    _set_module(tmp_path, V1)
    _run(tmp_path, "--approve")
    _set_module(tmp_path, V2)
    assert _run(tmp_path, "--check").returncode == 1
    assert _run(tmp_path, "--add").returncode == 0
    assert _run(tmp_path, "--check").returncode == 0
    assert "fakeapi:beta(y: str) -> int" in (tmp_path / "snap.txt").read_text()


def test_add_rejects_a_changed_line(tmp_path: Path) -> None:
    _set_module(tmp_path, V1)
    _run(tmp_path, "--approve")
    before = (tmp_path / "snap.txt").read_text()
    _set_module(tmp_path, V3)
    result = _run(tmp_path, "--add")
    assert result.returncode == 1
    assert "API change requires owner approval" in result.stderr
    assert (tmp_path / "snap.txt").read_text() == before


def test_check_fails_on_diff(tmp_path: Path) -> None:
    _set_module(tmp_path, V1)
    _run(tmp_path, "--approve")
    _set_module(tmp_path, V3)
    result = _run(tmp_path, "--check")
    assert result.returncode == 1
    assert "- fakeapi:alpha(x: int) -> str" in result.stderr
    assert "+ fakeapi:alpha(x: str) -> str" in result.stderr


RICH = """
from typing import Literal, Protocol

import msgspec


class Base(msgspec.Struct, tag_field="cmd"):
    v: int
    id: str


class Child(Base, tag="child"):
    text: str


class Port(Protocol):
    spec: int

    def __call__(self, name: str) -> int: ...


class Queue:
    @property
    def size(self) -> int:
        return 0

    async def drain(self) -> None:
        return None


async def serve(path: str) -> None:
    return None


type Command = Child | Base
Mode = Literal["a", "b"]
Pair = int | str
Seconds = float


class Boom(AssertionError):
    pass
"""


def _snapshot_lines(tmp_path: Path) -> list[str]:
    result = _run(tmp_path, "--approve")
    assert result.returncode == 0, result.stderr
    return (tmp_path / "snap.txt").read_text().splitlines()


def test_records_the_whole_surface(tmp_path: Path) -> None:
    _set_module(tmp_path, RICH)
    lines = _snapshot_lines(tmp_path)
    expected = [
        # class property with its return annotation
        "fakeapi:Queue.size: property -> int",
        # coroutine functions and methods
        "fakeapi:async Queue.drain(self) -> None",
        "fakeapi:async serve(path: str) -> None",
        # Protocol attribute and the allowlisted __call__
        "fakeapi:Port.spec: int",
        "fakeapi:Port.__call__(self, name: str) -> int",
        # type alias bodies, plain aliases included
        "fakeapi:Command: TypeAlias = fakeapi.Child | fakeapi.Base",
        "fakeapi:Mode: TypeAlias = Literal['a', 'b']",
        "fakeapi:Pair: TypeAlias = int | str",
        "fakeapi:Seconds: TypeAlias = float",
        # inherited struct fields resolved across the MRO, and the msgspec tag
        "fakeapi:Child.v: int",
        "fakeapi:Child.id: str",
        "fakeapi:Child.text: str",
        "fakeapi:Child [tag_field='cmd', tag='child']",
        # an exception that keeps the builtin constructor
        "fakeapi:Boom(*args: object)",
    ]
    for line in expected:
        assert line in lines, line
    assert not [line for line in lines if line.endswith(": ?") or "(...)" in line]


def test_imported_names_are_not_recorded_as_aliases(tmp_path: Path) -> None:
    _set_module(tmp_path, "from pathlib import Path\n\nHome = Path\n")
    assert _snapshot_lines(tmp_path) == ["fakeapi:Home: TypeAlias = pathlib.Path"]


def test_module_values_of_foreign_classes_are_recorded_with_their_repr(tmp_path: Path) -> None:
    """A frozen value such as ``omnivoice_model.MODEL`` (a ``contracts.engine.ModelInfo``)."""
    source = (
        "from dataclasses import dataclass\n"
        "from fractions import Fraction\n"
        "from pathlib import PurePosixPath\n\n"
        "HALF = Fraction(1, 2)\n"
        "ROOT = PurePosixPath('/srv')\n"
        "_PRIVATE = Fraction(1, 3)\n"
    )
    _set_module(tmp_path, source)
    assert _snapshot_lines(tmp_path) == [
        "fakeapi:HALF = Fraction(1, 2)",
        "fakeapi:ROOT = PurePosixPath('/srv')",
    ]


def test_unreadable_signature_fails_loudly(tmp_path: Path) -> None:
    _set_module(tmp_path, "class Table(dict[str, int]):\n    pass\n")
    result = _run(tmp_path, "--approve")
    assert result.returncode != 0
    assert "cannot read the signature" in result.stderr
    assert not (tmp_path / "snap.txt").exists()


def test_committed_snapshot_is_valid_when_empty() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
