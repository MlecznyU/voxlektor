import tomllib
from pathlib import Path

import pytest

from voxlektor.adapters.storage.settings_toml import TomlSettingsStore
from voxlektor.domain.errors import InvalidSetting
from voxlektor.domain.settings import merge_layers


def store(tmp_path: Path, **env: str) -> TomlSettingsStore:
    return TomlSettingsStore(tmp_path / "config.toml", env)


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    assert store(tmp_path).read_file() == {}


def test_read_file_layout(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        'config_version = 1\nengine = "fake"\nvoice = "ola"\nlang = "pl"\nrate = 1.5\n'
        'history = 7\nfoo = 1\n\n[service]\nidle_timeout = "5m"\nautostart = false\n'
    )
    assert store(tmp_path).read_file() == {
        "engine": "fake",
        "voice": "ola",
        "lang": "pl",
        "rate": 1.5,
        "history": 7,
        "idle_timeout": "5m",
        "autostart": False,
    }


def test_malformed_file_raises(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("this is = = not toml")
    with pytest.raises(InvalidSetting):
        store(tmp_path).read_file()


def test_service_must_be_table(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("service = 3\n")
    with pytest.raises(InvalidSetting):
        store(tmp_path).read_file()


def test_env_parsing(tmp_path: Path) -> None:
    env = {
        "VOXLEKTOR_ENGINE": "fake",
        "VOXLEKTOR_VOICE": "ola",
        "VOXLEKTOR_LANG": "en",
        "VOXLEKTOR_RATE": "1.25",
        "VOXLEKTOR_IDLE_TIMEOUT": "10m",
        "VOXLEKTOR_AUTOSTART": "false",
        "OTHER": "x",
    }
    assert TomlSettingsStore(tmp_path / "c.toml", env).read_env() == {
        "engine": "fake",
        "voice": "ola",
        "lang": "en",
        "rate": 1.25,
        "idle_timeout": "10m",
        "autostart": False,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("TRUE", True), ("on", True), ("0", False), ("No", False), ("off", False)],
)
def test_env_autostart_forms(tmp_path: Path, raw: str, expected: bool) -> None:
    assert store(tmp_path, VOXLEKTOR_AUTOSTART=raw).read_env() == {"autostart": expected}


def test_env_idle_timeout_seconds(tmp_path: Path) -> None:
    assert store(tmp_path, VOXLEKTOR_IDLE_TIMEOUT="90").read_env() == {"idle_timeout": 90}


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("VOXLEKTOR_AUTOSTART", "maybe"),
        ("VOXLEKTOR_RATE", "fast"),
        ("VOXLEKTOR_IDLE_TIMEOUT", "soon"),
    ],
)
def test_env_invalid_values_raise(tmp_path: Path, name: str, raw: str) -> None:
    with pytest.raises(InvalidSetting):
        store(tmp_path, **{name: raw}).read_env()


def test_layers_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('voice = "file"\n[service]\nautostart = true\n')
    s = store(tmp_path, VOXLEKTOR_VOICE="env", VOXLEKTOR_AUTOSTART="false")
    settings = merge_layers(s.read_file(), s.read_env())
    assert settings.voice == "env"
    assert settings.autostart is False


def test_write_value_keeps_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('foo = 1\n[extra]\nbar = "x"\n[service]\nautostart = true\nzed = 2\n')
    s = store(tmp_path)
    s.write_value("voice", "ola")
    s.write_value("rate", "1.5")
    s.write_value("history", "9")
    s.write_value("idle_timeout", "1h")
    s.write_value("autostart", "false")
    data = tomllib.loads(path.read_text())
    assert data == {
        "config_version": 1,
        "foo": 1,
        "extra": {"bar": "x"},
        "voice": "ola",
        "rate": 1.5,
        "history": 9,
        "service": {"autostart": False, "zed": 2, "idle_timeout": "1h"},
    }
    assert not (tmp_path / "config.toml.tmp").exists()


def test_write_value_creates_file_and_parent(tmp_path: Path) -> None:
    s = TomlSettingsStore(tmp_path / "nested" / "config.toml", {})
    s.write_value("lang", "pl")
    assert tomllib.loads((tmp_path / "nested" / "config.toml").read_text()) == {
        "config_version": 1,
        "lang": "pl",
    }
    assert s.read_file() == {"lang": "pl"}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rate", "9"),
        ("rate", "x"),
        ("lang", "de"),
        ("nope", "1"),
        ("autostart", "maybe"),
        ("idle_timeout", "soon"),
    ],
)
def test_write_value_rejects_invalid(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises(InvalidSetting):
        store(tmp_path).write_value(key, value)
    assert not (tmp_path / "config.toml").exists()


def test_write_value_rejects_non_table_service(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("service = 3\n")
    with pytest.raises(InvalidSetting):
        store(tmp_path).write_value("autostart", "true")
