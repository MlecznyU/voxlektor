import os
from pathlib import Path

import pytest

from voxlektor.adapters.storage import voices_fs
from voxlektor.adapters.storage.voices_fs import FsVoiceRepository
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import UnknownVoice

BUNDLED = [
    "f1-warm",
    "f2-narrator",
    "f3-bright",
    "m1-calm",
    "m2-narrator",
    "m3-casual",
    "v0-default",
    "zoe-spark",
]


def installed(tmp_path: Path) -> FsVoiceRepository:
    repo = FsVoiceRepository(tmp_path / "voices")
    repo.install_bundled()
    return repo


def test_names_empty_for_missing_root(tmp_path: Path) -> None:
    assert FsVoiceRepository(tmp_path / "absent").names() == []


def test_install_bundled_creates_all_seven(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    assert repo.names() == BUNDLED
    for name in BUNDLED:
        voice_dir = tmp_path / "voices" / name
        assert (voice_dir / "ref.wav").stat().st_size > 0
        assert (voice_dir / "voice.toml").is_file()
        assert voice_dir.stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "voices").stat().st_mode & 0o777 == 0o700


def test_ref_reads_text_exactly(tmp_path: Path) -> None:
    ref = installed(tmp_path).ref("m1-calm")
    assert ref == VoiceRef("m1-calm", tmp_path / "voices" / "m1-calm" / "ref.wav", ref.ref_text)
    assert ref.ref_text.startswith("Dzień dobry, nazywam się")
    assert ref.ref_text.endswith("przez dłuższy czas.")


def test_ref_unknown_voice(tmp_path: Path) -> None:
    with pytest.raises(UnknownVoice) as info:
        installed(tmp_path).ref("nope")
    assert info.value.available == tuple(BUNDLED)


def test_ref_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(UnknownVoice):
        installed(tmp_path).ref("../voices/m1-calm")


def test_names_ignores_stray_files_and_dirs(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    (tmp_path / "voices" / "stray.txt").write_text("x")
    (tmp_path / "voices" / "empty-dir").mkdir()
    assert repo.names() == BUNDLED


@pytest.mark.parametrize(
    "breakage", ["no-ref-txt", "no-ref-wav", "undecodable-ref-txt", "ref-txt-is-a-dir"]
)
def test_broken_voice_dir_is_not_listed(tmp_path: Path, breakage: str) -> None:
    """M28: a voice.toml alone does not make a usable voice."""
    repo = installed(tmp_path)
    broken = tmp_path / "voices" / "broken"
    broken.mkdir()
    (broken / "voice.toml").write_text('format = 1\nname = "broken"\n')
    (broken / "ref.wav").write_bytes(b"RIFF")
    (broken / "ref.txt").write_text("ok", encoding="utf-8")
    match breakage:
        case "no-ref-txt":
            (broken / "ref.txt").unlink()
        case "no-ref-wav":
            (broken / "ref.wav").unlink()
        case "undecodable-ref-txt":
            (broken / "ref.txt").write_bytes(b"\xff\xfe\xfa")
        case _:
            (broken / "ref.txt").unlink()
            (broken / "ref.txt").mkdir()
    assert repo.names() == BUNDLED
    with pytest.raises(UnknownVoice):
        repo.ref("broken")


def test_complete_custom_voice_is_listed(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    voice = tmp_path / "voices" / "mine"
    voice.mkdir()
    for name, data in (("voice.toml", b"format = 1\n"), ("ref.wav", b"RIFF"), ("ref.txt", b"hi")):
        (voice / name).write_bytes(data)
    assert "mine" in repo.names()
    assert repo.ref("mine").ref_text == "hi"


def test_ref_text_broken_after_listing_is_unknown_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race: names() saw a usable voice, then its ref.txt vanished before ref() read it."""
    repo = installed(tmp_path)

    def listed_anyway(voice_dir: Path) -> bool:
        return True

    monkeypatch.setattr(voices_fs, "_usable", listed_anyway)
    (tmp_path / "voices" / "m1-calm" / "ref.txt").unlink()
    with pytest.raises(UnknownVoice) as info:
        repo.ref("m1-calm")
    assert info.value.name == "m1-calm"


def test_install_bundled_never_overwrites_edited_text(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    edited = tmp_path / "voices" / "f1-warm" / "ref.txt"
    edited.write_text("my words", encoding="utf-8")
    repo.install_bundled()
    assert edited.read_text(encoding="utf-8") == "my words"


def test_install_bundled_keeps_unknown_extra_file(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    extra = tmp_path / "voices" / "f1-warm" / "notes.md"
    extra.write_text("keep me", encoding="utf-8")
    repo.install_bundled()
    assert extra.read_text(encoding="utf-8") == "keep me"


def test_install_bundled_restores_a_missing_file(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    (tmp_path / "voices" / "f1-warm" / "voice.toml").unlink()
    repo.install_bundled()
    assert (tmp_path / "voices" / "f1-warm" / "voice.toml").is_file()


def test_cache_round_trip(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    repo.write_cache("m1-calm", "omnivoice-01234567-p1.bin", b"\x00\x01data")
    assert repo.read_cache("m1-calm", "omnivoice-01234567-p1.bin") == b"\x00\x01data"
    cache = tmp_path / "voices" / "m1-calm" / "cache"
    assert cache.stat().st_mode & 0o777 == 0o700
    assert [p.name for p in cache.iterdir()] == ["omnivoice-01234567-p1.bin"]


def test_cache_overwrite_replaces(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    repo.write_cache("m1-calm", "k.bin", b"one")
    repo.write_cache("m1-calm", "k.bin", b"two")
    assert repo.read_cache("m1-calm", "k.bin") == b"two"


def test_read_cache_missing_key_is_none(tmp_path: Path) -> None:
    assert installed(tmp_path).read_cache("m1-calm", "absent.bin") is None


def test_cache_unknown_voice(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    with pytest.raises(UnknownVoice):
        repo.read_cache("nope", "k.bin")
    with pytest.raises(UnknownVoice):
        repo.write_cache("nope", "k.bin", b"x")


@pytest.mark.parametrize(
    "key", ["../ref.txt", "a/b.bin", "", ".", "..", ".tmp-abc123", ".hidden.bin"]
)
def test_cache_key_must_be_a_plain_file_name(tmp_path: Path, key: str) -> None:
    repo = installed(tmp_path)
    with pytest.raises(ValueError, match="cache key"):
        repo.write_cache("m1-calm", key, b"x")
    with pytest.raises(ValueError, match="cache key"):
        repo.read_cache("m1-calm", key)
    assert (tmp_path / "voices" / "m1-calm" / "ref.txt").is_file()


def test_read_cache_of_a_directory_is_none(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    (tmp_path / "voices" / "m1-calm" / "cache" / "sub.bin").mkdir(parents=True)
    assert repo.read_cache("m1-calm", "sub.bin") is None


def test_write_cache_leaves_no_temp_files(tmp_path: Path) -> None:
    repo = installed(tmp_path)
    repo.write_cache("m1-calm", "a.bin", b"1")
    repo.write_cache("m1-calm", "b.bin", b"2")
    cache = tmp_path / "voices" / "m1-calm" / "cache"
    assert sorted(p.name for p in cache.iterdir()) == ["a.bin", "b.bin"]


def make_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bundle = tmp_path / "bundle"
    voice = bundle / "x-voice"
    voice.mkdir(parents=True)
    (voice / "ref.wav").write_bytes(b"wav")
    (voice / "ref.txt").write_text("text", encoding="utf-8")
    (voice / "voice.toml").write_text("format = 1\n", encoding="utf-8")
    (voice / ".DS_Store").write_bytes(b"junk")
    (bundle / "stray.txt").write_text("not a voice", encoding="utf-8")

    def fake_files(package: str) -> Path:
        assert package == "voxlektor.bundled_voices"
        return bundle

    monkeypatch.setattr(voices_fs.resources, "files", fake_files)
    return bundle


def test_install_bundled_skips_dotfiles_and_non_voices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_bundle(tmp_path, monkeypatch)
    repo = FsVoiceRepository(tmp_path / "voices")
    repo.install_bundled()
    installed_dir = tmp_path / "voices" / "x-voice"
    files = sorted(p.name for p in installed_dir.iterdir() if p.is_file())
    assert files == ["ref.txt", "ref.wav", "voice.toml"]
    assert not (tmp_path / "voices" / "stray.txt").exists()
    assert repo.names() == ["x-voice"]


def test_leftover_temp_file_does_not_make_a_voice_look_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_bundle(tmp_path, monkeypatch)
    repo = FsVoiceRepository(tmp_path / "voices")
    voice_dir = tmp_path / "voices" / "x-voice"
    (voice_dir / "cache").mkdir(parents=True)
    leftover = voice_dir / "cache" / ".tmp-interrupted"
    leftover.write_bytes(b"trunc")
    assert repo.names() == []
    repo.install_bundled()
    assert repo.names() == ["x-voice"]
    assert (voice_dir / "ref.wav").read_bytes() == b"wav"
    assert leftover.read_bytes() == b"trunc"
    with pytest.raises(ValueError, match="cache key"):
        repo.read_cache("x-voice", ".tmp-interrupted")


def test_voice_toml_is_installed_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_bundle(tmp_path, monkeypatch)
    repo = FsVoiceRepository(tmp_path / "voices")
    real_link = os.link

    def failing_link(src: Path, dst: Path) -> None:
        if Path(dst).name == "ref.wav":
            raise OSError("disk full")
        real_link(src, dst)

    monkeypatch.setattr(voices_fs.os, "link", failing_link)
    with pytest.raises(OSError, match="disk full"):
        repo.install_bundled()
    voice_dir = tmp_path / "voices" / "x-voice"
    assert not (voice_dir / "ref.wav").exists()
    assert not (voice_dir / "voice.toml").exists()
    assert repo.names() == []
    assert sorted(p.name for p in voice_dir.iterdir()) == ["cache", "ref.txt"]
    assert list((voice_dir / "cache").iterdir()) == []


def test_install_never_overwrites_a_file_that_appears_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_bundle(tmp_path, monkeypatch)
    repo = FsVoiceRepository(tmp_path / "voices")
    real_link = os.link

    def racing_link(src: Path, dst: Path) -> None:
        if Path(dst).name == "ref.txt":
            Path(dst).write_text("user", encoding="utf-8")
        real_link(src, dst)

    monkeypatch.setattr(voices_fs.os, "link", racing_link)
    repo.install_bundled()
    voice_dir = tmp_path / "voices" / "x-voice"
    assert (voice_dir / "ref.txt").read_text(encoding="utf-8") == "user"
    files = sorted(p.name for p in voice_dir.iterdir() if p.is_file())
    assert files == ["ref.txt", "ref.wav", "voice.toml"]
    assert list((voice_dir / "cache").iterdir()) == []


def test_install_removes_nothing_outside_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_bundle(tmp_path, monkeypatch)
    repo = FsVoiceRepository(tmp_path / "voices")
    voice_dir = tmp_path / "voices" / "x-voice"
    voice_dir.mkdir(parents=True)
    (voice_dir / "notes.md").write_text("mine", encoding="utf-8")
    (voice_dir / ".hidden").write_text("mine", encoding="utf-8")
    (voice_dir / "ref.txt").write_text("edited", encoding="utf-8")
    real_link = os.link

    def failing_link(src: Path, dst: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(voices_fs.os, "link", failing_link)
    with pytest.raises(OSError, match="disk full"):
        repo.install_bundled()
    monkeypatch.setattr(voices_fs.os, "link", real_link)
    assert sorted(p.name for p in voice_dir.iterdir() if p.is_file()) == [
        ".hidden",
        "notes.md",
        "ref.txt",
    ]
    repo.install_bundled()
    assert (voice_dir / "ref.txt").read_text(encoding="utf-8") == "edited"
    assert (voice_dir / "notes.md").read_text(encoding="utf-8") == "mine"
    assert (voice_dir / ".hidden").read_text(encoding="utf-8") == "mine"
    assert (voice_dir / "ref.wav").read_bytes() == b"wav"
