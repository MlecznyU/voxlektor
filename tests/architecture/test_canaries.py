"""Canaries: prove that the static architecture gates really bite."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(sys.executable).parent


def _copy_project(tmp_path: Path) -> Path:
    """Copy the config and the package into ``tmp_path`` (src layout)."""
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copytree(
        ROOT / "src" / "voxlektor",
        tmp_path / "src" / "voxlektor",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return tmp_path


def _lint_imports(project: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(project / "src")}
    return subprocess.run(  # noqa: S603
        [str(BIN / "lint-imports")],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_linter_catches_internal_violation(tmp_path: Path) -> None:
    project = _copy_project(tmp_path)
    assert _lint_imports(project).returncode == 0
    (project / "src" / "voxlektor" / "domain" / "_bad.py").write_text("from voxlektor import app\n")
    result = _lint_imports(project)
    assert result.returncode != 0, result.stdout


def test_import_linter_catches_stdlib_violation(tmp_path: Path) -> None:
    project = _copy_project(tmp_path)
    assert _lint_imports(project).returncode == 0
    (project / "src" / "voxlektor" / "domain" / "_bad.py").write_text("import threading\n")
    result = _lint_imports(project)
    assert result.returncode != 0, result.stdout
    assert "Domain is pure and synchronous BROKEN" in result.stdout


def test_ruff_bans_threading_in_domain(tmp_path: Path) -> None:
    """Second line of defence: Ruff TID251 rejects the same stdlib imports."""
    project = _copy_project(tmp_path)
    (project / "src" / "voxlektor" / "domain" / "_bad.py").write_text("import threading\n")
    result = subprocess.run(  # noqa: S603
        [str(BIN / "ruff"), "check", "--no-cache", "--select", "TID251", "src"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "TID251" in result.stdout


def test_cli_import_is_light() -> None:
    code = (
        "import voxlektor.entrypoints.cli.app, sys; "
        "bad={'mlx','numpy','sounddevice','mlx_audio','asyncio','voxlektor.runtime'} "
        "& set(sys.modules); "
        "assert not bad, bad"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_cli_status_run_is_light(short_tmp: Path) -> None:
    """Running a control command (not just importing the CLI) loads nothing heavy either.

    This is the run-time proof behind the ``contracts.engine -> numpy`` ignore of the
    "CLI control path is lightweight" import-linter contract (that import is typing-only).
    """
    code = (
        "import sys; from voxlektor.entrypoints.cli.app import main; "
        "rc = main(['status']); "
        "heavy = ('mlx', 'numpy', 'sounddevice', 'mlx_audio', 'asyncio', 'voxlektor.runtime'); "
        "bad = {m for m in sys.modules if m.split('.')[0] in heavy or m.startswith(heavy)}; "
        "assert rc == 0, rc; assert not bad, sorted(bad)"
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("VOXLEKTOR_")}
    env["VOXLEKTOR_HOME"] = str(short_tmp)  # no daemon there: the socket is missing
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OFF"
