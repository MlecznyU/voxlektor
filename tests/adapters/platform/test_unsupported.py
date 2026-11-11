from collections.abc import Callable
from pathlib import Path

import pytest

from voxlektor.adapters.platform.unsupported import UnsupportedPlatform
from voxlektor.domain.errors import PlatformError


@pytest.mark.parametrize(
    "call",
    [
        UnsupportedPlatform().config_dir,
        UnsupportedPlatform().runtime_dir,
        UnsupportedPlatform().socket_path,
        UnsupportedPlatform().log_path,
        UnsupportedPlatform().pid_path,
    ],
)
def test_every_call_raises(call: Callable[[], Path]) -> None:
    with pytest.raises(PlatformError, match="platform not supported yet") as info:
        call()
    assert info.value.code == "unsupported_platform"
