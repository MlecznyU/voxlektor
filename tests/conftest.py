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
