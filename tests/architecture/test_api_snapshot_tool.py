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


def test_committed_snapshot_is_valid_when_empty() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
