from importlib.metadata import metadata
from pathlib import Path

import voxlektor

_ROOT = Path(__file__).resolve().parent.parent
_REPO = "https://github.com/MlecznyU/voxlektor"


def test_version_is_exposed_without_side_effects() -> None:
    assert voxlektor.__version__ == "0.1.0"


def test_package_metadata_declares_license_and_urls() -> None:
    meta = metadata("voxlektor")
    assert meta["License-Expression"] == "Apache-2.0"
    urls = meta.get_all("Project-URL") or []
    assert f"Homepage, {_REPO}" in urls
    assert f"Issues, {_REPO}/issues" in urls


def test_license_file_is_apache_2_with_copyright() -> None:
    text = (_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "Copyright 2026 MlecznyU" in text
