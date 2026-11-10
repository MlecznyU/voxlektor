"""Canaries: prove that the static architecture gates really bite."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.mark.xfail(strict=True, reason="entrypoints/cli/app.py arrives in Task 10")
def test_cli_import_is_light() -> None:
    code = (
        "import voxlektor.entrypoints.cli.app, sys; "
        "bad={'mlx','numpy','sounddevice','mlx_audio'} & set(sys.modules); "
        "assert not bad, bad"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
