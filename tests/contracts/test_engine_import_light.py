"""numpy must not be imported by the contract or by the ports that mention it only in types."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["voxlektor.contracts.engine", "voxlektor.app.ports"])
def test_import_does_not_load_numpy(module: str) -> None:
    code = f"import sys, {module}; sys.exit(1 if 'numpy' in sys.modules else 0)"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, f"{module} imported numpy: {result.stderr}"
