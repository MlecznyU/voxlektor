import pytest

from voxlektor.adapters.platform.macos import MacOSPlatform
from voxlektor.bootstrap.platform_select import select_platform
from voxlektor.domain.errors import PlatformError


@pytest.mark.parametrize("version", ["14", "14.0", "14.5", "15.1.2"])
def test_apple_silicon_with_supported_macos_gets_macos(version: str) -> None:
    assert isinstance(select_platform("darwin", "arm64", version), MacOSPlatform)


@pytest.mark.parametrize(
    ("sys_platform", "machine", "version"),
    [
        ("darwin", "arm64", "13.6"),
        ("darwin", "arm64", ""),
        ("darwin", "arm64", "abc"),
        ("darwin", "x86_64", "14.0"),
        ("linux", "x86_64", ""),
        ("linux", "arm64", "15.0"),
    ],
)
def test_other_systems_are_unsupported(sys_platform: str, machine: str, version: str) -> None:
    platform = select_platform(sys_platform, machine, version)
    with pytest.raises(PlatformError, match="platform not supported yet"):
        platform.config_dir()
    with pytest.raises(PlatformError):
        platform.socket_path()
