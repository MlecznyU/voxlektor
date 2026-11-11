import stat
from pathlib import Path

import pytest

from voxlektor.adapters.platform.macos import MacOSPlatform


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_relocated_dirs_are_private(tmp_path: Path) -> None:
    platform = MacOSPlatform(tmp_path / "home")
    assert platform.config_dir() == tmp_path / "home" / "config"
    assert platform.runtime_dir() == tmp_path / "home" / "run"
    assert mode(platform.config_dir()) == 0o700
    assert mode(platform.runtime_dir()) == 0o700


def test_existing_loose_dir_is_tightened(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir(mode=0o755)
    (tmp_path / "run").chmod(0o755)
    MacOSPlatform(tmp_path).runtime_dir()
    assert mode(tmp_path / "run") == 0o700


def test_file_paths(tmp_path: Path) -> None:
    platform = MacOSPlatform(tmp_path)
    run = tmp_path / "run"
    assert platform.socket_path() == run / "voxlektor.sock"
    assert platform.log_path() == run / "voxlektor.log"
    assert platform.pid_path() == run / "voxlektor.pid"


def test_env_home_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXLEKTOR_HOME", str(tmp_path))
    assert MacOSPlatform().config_dir() == tmp_path / "config"


def test_default_locations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXLEKTOR_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    platform = MacOSPlatform()
    assert platform.config_dir() == tmp_path / ".config" / "voxlektor"
    assert platform.runtime_dir() == tmp_path / "Library" / "Application Support" / "voxlektor"
    assert mode(platform.runtime_dir()) == 0o700
