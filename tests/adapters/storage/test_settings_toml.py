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
        'config_version = 1\nengine = "fake"\nvoice = "ola"\nlang = "pl"\n'
        'base_lang = "en"\nrate = 1.5\nvolume = 70\nhistory = 7\nfoo = 1\n\n'
        '[service]\nidle_timeout = "5m"\nautostart = false\n'
    )
    assert store(tmp_path).read_file() == {
        "engine": "fake",
        "voice": "ola",
        "lang": "pl",
        "base_lang": "en",
        "rate": 1.5,
        "volume": 70,
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
        "VOXLEKTOR_BASE_LANG": "pl",
        "VOXLEKTOR_RATE": "1.25",
        "VOXLEKTOR_VOLUME": "70",
        "VOXLEKTOR_IDLE_TIMEOUT": "10m",
        "VOXLEKTOR_AUTOSTART": "false",
        "OTHER": "x",
    }
    assert TomlSettingsStore(tmp_path / "c.toml", env).read_env() == {
        "engine": "fake",
        "voice": "ola",
        "lang": "en",
        "base_lang": "pl",
        "rate": 1.25,
        "volume": 70,
        "idle_timeout": "10m",
        "autostart": False,
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", True), ("TRUE", True), ("on", True), ("0", False), ("No", False), ("off", False)],
)
def test_env_autostart_forms(tmp_path: Path, raw: str, expected: bool) -> None:
    assert store(tmp_path, VOXLEKTOR_AUTOSTART=raw).read_env() == {"autostart": expected}


def test_env_volume(tmp_path: Path) -> None:
    assert store(tmp_path, VOXLEKTOR_VOLUME="70").read_env() == {"volume": 70}
    with pytest.raises(InvalidSetting) as info:
        store(tmp_path, VOXLEKTOR_VOLUME="loud").read_env()
    assert "VOXLEKTOR_VOLUME" in str(info.value)


def test_write_value_volume_round_trip(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.write_value("volume", "35")
    assert s.read_file() == {"volume": 35}
    assert merge_layers(s.read_file()).volume == 35


def test_env_idle_timeout_seconds(tmp_path: Path) -> None:
    assert store(tmp_path, VOXLEKTOR_IDLE_TIMEOUT="90").read_env() == {"idle_timeout": 90}


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("VOXLEKTOR_AUTOSTART", "maybe"),
        ("VOXLEKTOR_RATE", "fast"),
        ("VOXLEKTOR_BASE_LANG", "auto"),
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


def test_write_value_base_lang_round_trip(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.write_value("base_lang", "en")
    assert s.read_file() == {"base_lang": "en"}
    assert merge_layers(s.read_file()).base_lang == "en"


def test_env_invalid_base_lang_names_variable(tmp_path: Path) -> None:
    with pytest.raises(InvalidSetting) as info:
        store(tmp_path, VOXLEKTOR_BASE_LANG="de").read_env()
    assert "VOXLEKTOR_BASE_LANG" in str(info.value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("rate", "9"),
        ("rate", "x"),
        ("volume", "101"),
        ("volume", "5.5"),
        ("lang", "de"),
        ("base_lang", "de"),
        ("nope", "1"),
        ("autostart", "maybe"),
        ("idle_timeout", "soon"),
    ],
)
def test_write_value_rejects_invalid(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises(InvalidSetting):
        store(tmp_path).write_value(key, value)
    assert not (tmp_path / "config.toml").exists()


def test_setup_marker_is_absent_without_a_file(tmp_path: Path) -> None:
    assert store(tmp_path).setup_completed() is False


def test_mvp_config_without_marker_is_not_set_up(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('config_version = 1\nvoice = "ola"\n')
    assert store(tmp_path).setup_completed() is False


def test_mark_setup_completed_keeps_unknown_keys_and_tables(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('foo = 1\nvoice = "ola"\n[service]\nautostart = false\nzed = 2\n')
    s = store(tmp_path)
    s.mark_setup_completed()
    assert s.setup_completed() is True
    assert tomllib.loads(path.read_text()) == {
        "config_version": 1,
        "foo": 1,
        "voice": "ola",
        "setup_completed": True,
        "service": {"autostart": False, "zed": 2},
    }
    assert s.read_file() == {"voice": "ola", "autostart": False}  # the marker is no setting
    assert not (tmp_path / "config.toml.tmp").exists()


def test_only_a_true_marker_counts(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('setup_completed = "yes"\n')
    assert store(tmp_path).setup_completed() is False


def test_marker_cannot_be_written_as_a_setting(tmp_path: Path) -> None:
    with pytest.raises(InvalidSetting):
        store(tmp_path).write_value("setup_completed", "true")


def test_write_value_rejects_non_table_service(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("service = 3\n")
    with pytest.raises(InvalidSetting):
        store(tmp_path).write_value("autostart", "true")
