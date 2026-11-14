import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def short_tmp() -> Iterator[Path]:
    """A temporary directory with a short path: macOS limits socket paths to 104 bytes."""
    path = Path(tempfile.mkdtemp(prefix="vxl-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def mark_setup_completed(home: Path) -> Path:
    """Write a config under ``VOXLEKTOR_HOME=home`` that says ``vxl setup`` has finished.

    Speaking is gated on that marker; CLI and e2e tests that are not about setup start from
    here. Returns the config path (tests may add keys with ``vxl config`` or the store).
    """
    config = home / "config" / "config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("config_version = 1\nsetup_completed = true\n")
    return config


@pytest.fixture
def setup_completed(short_tmp: Path) -> Path:
    """``short_tmp`` as a home whose setup has completed (the speak gate is open)."""
    mark_setup_completed(short_tmp)
    return short_tmp
