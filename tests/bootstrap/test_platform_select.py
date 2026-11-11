import pytest

from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.domain.errors import PlatformError


def test_apple_silicon_gets_macos() -> None:
    assert isinstance(select_platform("darwin", "arm64"), MacOSPlatform)


@pytest.mark.parametrize(("sys_platform", "machine"), [("linux", "x86_64"), ("darwin", "x86_64")])
def test_other_systems_are_unsupported(sys_platform: str, machine: str) -> None:
    platform = select_platform(sys_platform, machine)
    with pytest.raises(PlatformError):
        platform.config_dir()
    with pytest.raises(PlatformError):
        platform.socket_path()
