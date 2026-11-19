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
        '[service]\nidle_timeout = "5m"\nautostart = false\nfloor_grace = "15s"\n'
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
        "floor_grace": "15s",
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
        "VOXLEKTOR_FLOOR_GRACE": "20s",
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
        "floor_grace": "20s",
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


def test_env_steps(tmp_path: Path) -> None:
    assert store(tmp_path, VOXLEKTOR_STEPS="12").read_env() == {"steps": 12}
    with pytest.raises(InvalidSetting) as info:
        store(tmp_path, VOXLEKTOR_STEPS="many").read_env()
    assert "VOXLEKTOR_STEPS" in str(info.value)


def test_steps_file_key_and_round_trip(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("steps = 8\n")
    assert store(tmp_path).read_file() == {"steps": 8}
    s = store(tmp_path)
    s.write_value("steps", "24")
    assert s.read_file() == {"steps": 24}
    assert merge_layers(s.read_file()).steps == 24
    with pytest.raises(InvalidSetting):
        s.write_value("steps", "40")


def test_env_floor_grace_seconds_and_override(tmp_path: Path) -> None:
    assert store(tmp_path, VOXLEKTOR_FLOOR_GRACE="0").read_env() == {"floor_grace": 0}
    (tmp_path / "config.toml").write_text('[service]\nfloor_grace = "15s"\n')
    s = store(tmp_path, VOXLEKTOR_FLOOR_GRACE="30")
    assert merge_layers(s.read_file(), s.read_env()).floor_grace_s == 30


def test_env_idle_timeout_seconds(tmp_path: Path) -> None:
    assert store(tmp_path, VOXLEKTOR_IDLE_TIMEOUT="90").read_env() == {"idle_timeout": 90}


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("VOXLEKTOR_AUTOSTART", "maybe"),
        ("VOXLEKTOR_RATE", "fast"),
        ("VOXLEKTOR_BASE_LANG", "auto"),
        ("VOXLEKTOR_IDLE_TIMEOUT", "soon"),
        ("VOXLEKTOR_FLOOR_GRACE", "soon"),
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
        "config_version": 2,
        "foo": 1,
        "extra": {"bar": "x"},
        "history": 9,
        "service": {"autostart": False, "zed": 2, "idle_timeout": "1h"},
        "profiles": {"default": {"voice": "ola", "rate": 1.5}},
    }
    assert not (tmp_path / "config.toml.tmp").exists()


def test_write_value_creates_file_and_parent(tmp_path: Path) -> None:
    s = TomlSettingsStore(tmp_path / "nested" / "config.toml", {})
    s.write_value("lang", "pl")
    assert tomllib.loads((tmp_path / "nested" / "config.toml").read_text()) == {
        "config_version": 2,
        "profiles": {"default": {"lang": "pl"}},
    }
    assert s.read_file() == {"lang": "pl"}


V1_FILE = (
    'config_version = 1\nengine = "fake"\nvoice = "ola"\nrate = 1.5\nhistory = 7\nfoo = 1\n'
    'setup_completed = true\n\n[service]\nidle_timeout = "5m"\nzed = 2\n\n[extra]\nbar = "x"\n'
)
V2_FILE = (
    'config_version = 2\nengine = "fake"\nhistory = 7\nsetup_completed = true\n\n'
    '[service]\nidle_timeout = "5m"\n\n'
    '[profiles.default]\nvoice = "zoe"\nrate = 1.0\nsteps = 12\n\n'
    "[profiles.claude]\nrate = 1.1\nbogus = 3\n"
)


def test_v1_file_reads_as_default_profile(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V1_FILE)
    s = store(tmp_path)
    assert s.read_profiles() == {"default": {"voice": "ola", "rate": 1.5}}
    assert s.read_globals() == {"engine": "fake", "history": 7, "idle_timeout": "5m"}
    assert s.read_file() == {
        "engine": "fake",
        "voice": "ola",
        "rate": 1.5,
        "history": 7,
        "idle_timeout": "5m",
    }
    assert path.read_text() == V1_FILE  # reading never rewrites


def test_v2_file_reads_profiles_and_globals(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(V2_FILE)
    s = store(tmp_path)
    assert s.read_profiles() == {
        "default": {"voice": "zoe", "rate": 1.0, "steps": 12},
        "claude": {"rate": 1.1},
    }
    assert s.read_globals() == {"engine": "fake", "history": 7, "idle_timeout": "5m"}
    assert s.read_file() == {
        "engine": "fake",
        "history": 7,
        "idle_timeout": "5m",
        "voice": "zoe",
        "rate": 1.0,
        "steps": 12,
    }


def test_default_profile_is_implicit(tmp_path: Path) -> None:
    assert store(tmp_path).read_profiles() == {"default": {}}
    (tmp_path / "config.toml").write_text("config_version = 2\n[profiles.claude]\nrate = 1.1\n")
    assert store(tmp_path).read_profiles() == {"default": {}, "claude": {"rate": 1.1}}


def test_v2_default_profile_wins_over_stray_top_level_keys(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        'voice = "top"\nlang = "en"\n[profiles.default]\nvoice = "table"\n'
    )
    assert store(tmp_path).read_profiles() == {"default": {"voice": "table", "lang": "en"}}


_MIXED = (
    "config_version = 2\n"
    'voice = "top"\n'
    "rate = 1.2\n"
    '[profiles.default]\nvoice = "table"\n'
)  # a v2 file to which an older CLI (0.2.x) wrote top-level profile keys


def test_a_mixed_file_warns_once_on_read(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "config.toml").write_text(_MIXED)
    s = store(tmp_path)
    assert s.read_profiles() == {"default": {"voice": "table", "rate": 1.2}}
    s.read_file()
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "top-level" in warnings[0]
    assert "rate, voice" in warnings[0]
    assert "[profiles.default]" in warnings[0]


def test_a_v1_file_reads_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "config.toml").write_text(V1_FILE)
    store(tmp_path).read_profiles()
    assert caplog.records == []


def test_saving_a_mixed_file_merges_only_missing_keys_and_names_the_dropped_ones(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_MIXED)
    store(tmp_path).write_value("engine", "fake")
    data = tomllib.loads(path.read_text())
    assert data["profiles"]["default"] == {"voice": "table", "rate": 1.2}
    assert "voice" not in data
    assert "rate" not in data
    assert "dropped top-level voice = 'top'" in caplog.text


@pytest.mark.parametrize("text", ["profiles = 3\n", "[profiles]\nclaude = 3\n"])
def test_profiles_must_be_tables(tmp_path: Path, text: str) -> None:
    (tmp_path / "config.toml").write_text(text)
    s = store(tmp_path)
    with pytest.raises(InvalidSetting):
        s.read_profiles()
    with pytest.raises(InvalidSetting):
        s.read_file()
    with pytest.raises(InvalidSetting):
        s.write_value("voice", "ola")


def test_first_write_converts_v1_to_v2_preserving_unknowns(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V1_FILE)
    s = store(tmp_path)
    s.write_profile_value("claude", "rate", "1.1")
    assert tomllib.loads(path.read_text()) == {
        "config_version": 2,
        "engine": "fake",
        "history": 7,
        "foo": 1,
        "setup_completed": True,
        "service": {"idle_timeout": "5m", "zed": 2},
        "extra": {"bar": "x"},
        "profiles": {"default": {"voice": "ola", "rate": 1.5}, "claude": {"rate": 1.1}},
    }
    assert s.setup_completed() is True
    assert s.read_profiles() == {
        "default": {"voice": "ola", "rate": 1.5},
        "claude": {"rate": 1.1},
    }


def test_write_value_of_a_global_key_converts_v1(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V1_FILE)
    store(tmp_path).write_value("engine", "omnivoice")
    data = tomllib.loads(path.read_text())
    assert data["config_version"] == 2
    assert data["engine"] == "omnivoice"
    assert "voice" not in data
    assert data["profiles"] == {"default": {"voice": "ola", "rate": 1.5}}


def test_write_value_of_a_profile_key_writes_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V2_FILE)
    s = store(tmp_path)
    s.write_value("rate", "1.25")
    assert s.read_profiles()["default"] == {"voice": "zoe", "rate": 1.25, "steps": 12}
    assert s.read_profiles()["claude"] == {"rate": 1.1}
    assert tomllib.loads(path.read_text())["profiles"]["claude"] == {"rate": 1.1, "bogus": 3}


def test_write_profile_value_coerces_and_validates(tmp_path: Path) -> None:
    s = store(tmp_path)
    s.write_profile_value("claude", "volume", "40")
    s.write_profile_value("claude", "base_lang", "en")
    assert s.read_profiles() == {"default": {}, "claude": {"volume": 40, "base_lang": "en"}}
    for key, value in [("volume", "101"), ("rate", "x"), ("nope", "1")]:
        with pytest.raises(InvalidSetting):
            s.write_profile_value("claude", key, value)


@pytest.mark.parametrize("key", ["engine", "history", "idle_timeout", "setup_completed"])
def test_global_keys_are_not_profile_keys(tmp_path: Path, key: str) -> None:
    s = store(tmp_path)
    with pytest.raises(InvalidSetting) as info:
        s.write_profile_value("claude", key, "1")
    assert "profile" in str(info.value)
    with pytest.raises(InvalidSetting):
        s.unset_profile_value("default", key)
    assert not (tmp_path / "config.toml").exists()


def test_unset_profile_value(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V2_FILE)
    s = store(tmp_path)
    s.unset_profile_value("claude", "rate")
    s.unset_profile_value("default", "voice")
    s.unset_profile_value("default", "lang")  # not set: nothing to remove
    assert s.read_profiles() == {
        "default": {"rate": 1.0, "steps": 12},
        "claude": {},  # the profile itself stays
    }
    assert tomllib.loads(path.read_text())["profiles"]["claude"] == {"bogus": 3}


def test_unset_converts_v1_default(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V1_FILE)
    store(tmp_path).unset_profile_value("default", "voice")
    data = tomllib.loads(path.read_text())
    assert data["config_version"] == 2
    assert "voice" not in data
    assert data["profiles"] == {"default": {"rate": 1.5}}


def test_unset_of_an_unknown_profile_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidSetting) as info:
        store(tmp_path).unset_profile_value("ghost", "rate")
    assert str(info.value) == "unknown profile 'ghost'"
    assert not (tmp_path / "config.toml").exists()


def test_remove_profile(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(V2_FILE)
    s = store(tmp_path)
    s.remove_profile("claude")
    assert s.read_profiles() == {"default": {"voice": "zoe", "rate": 1.0, "steps": 12}}
    assert "claude" not in tomllib.loads(path.read_text())["profiles"]


def test_remove_profile_rejects_default_and_unknown(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(V2_FILE)
    s = store(tmp_path)
    with pytest.raises(InvalidSetting) as info:
        s.remove_profile("default")
    assert "default" in str(info.value)
    with pytest.raises(InvalidSetting) as info:
        s.remove_profile("ghost")
    assert str(info.value) == "unknown profile 'ghost'"
    assert s.read_profiles()["claude"] == {"rate": 1.1}


def test_create_profile_adds_an_empty_table_once(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('voice = "ola"\n[profiles.claude]\nrate = 1.1\n')  # v1 + a profile
    s = store(tmp_path)
    s.create_profile("claude")  # exists: nothing is written
    s.create_profile("default")  # implicit: nothing is written
    assert "config_version" not in tomllib.loads(path.read_text())
    s.create_profile("audiobook")
    data = tomllib.loads(path.read_text())
    assert data["config_version"] == 2
    assert data["profiles"] == {
        "default": {"voice": "ola"},
        "claude": {"rate": 1.1},
        "audiobook": {},
    }
    assert s.read_profiles()["audiobook"] == {}


def test_profile_keys_add_no_new_names(tmp_path: Path) -> None:
    """Profile keys are the v1 setting names; no new config keys or env names appear."""
    path = tmp_path / "config.toml"
    s = store(tmp_path)
    for key, value in [
        ("voice", "ola"),
        ("lang", "pl"),
        ("base_lang", "en"),
        ("rate", "1.5"),
        ("volume", "30"),
        ("steps", "8"),
    ]:
        s.write_profile_value("p1", key, value)
    assert set(s.read_profiles()["p1"]) == {"voice", "lang", "base_lang", "rate", "volume", "steps"}
    assert set(tomllib.loads(path.read_text())) == {"config_version", "profiles"}


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
        "config_version": 2,
        "foo": 1,
        "setup_completed": True,
        "service": {"autostart": False, "zed": 2},
        "profiles": {"default": {"voice": "ola"}},
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
