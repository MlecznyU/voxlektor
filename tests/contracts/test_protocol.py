import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from voxlektor.contracts.codec import ProtocolError, decode_reply, decode_request, encode_line
from voxlektor.contracts.protocol import (
    MAX_FRAME,
    PROTOCOL_VERSION,
    ErrorCode,
    Event,
    Reply,
    ReplyError,
    SpeakRequest,
    StatusRequest,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "protocol_v1"


def _lines(name: str) -> list[bytes]:
    return [ln for ln in (FIXTURES / name).read_bytes().split(b"\n") if ln]


def _fail(line: bytes) -> ProtocolError:
    with pytest.raises(ProtocolError) as info:
        decode_request(line)
    return info.value


def test_constants() -> None:
    assert PROTOCOL_VERSION == 1
    assert MAX_FRAME == 1 << 20


def test_error_codes_are_closed_and_stable() -> None:
    assert [c.value for c in ErrorCode] == [
        "bad_frame",
        "bad_version",
        "unknown_command",
        "invalid_argument",
        "unknown_voice",
        "unknown_engine",
        "invalid_duration",
        "invalid_setting",
        "engine_failed",
        "audio_failed",
        "unsupported",
        "internal",
    ]


@pytest.mark.parametrize("line", _lines("requests.jsonl"))
def test_golden_requests_round_trip(line: bytes) -> None:
    req = decode_request(line)
    assert encode_line(req) == line + b"\n"


def test_golden_requests_cover_every_command() -> None:
    cmds = {json.loads(ln)["cmd"] for ln in _lines("requests.jsonl")}
    assert cmds == {
        "speak",
        "pause",
        "resume",
        "skip",
        "back",
        "stop_reading",
        "status",
        "idle",
        "unload",
        "shutdown",
        "subscribe",
    }


@pytest.mark.parametrize("line", _lines("replies.jsonl"))
def test_golden_replies_round_trip(line: bytes) -> None:
    assert encode_line(decode_reply(line)) == line + b"\n"


def test_golden_replies_cover_every_error_code() -> None:
    codes = {json.loads(ln)["error"]["code"] for ln in _lines("replies.jsonl") if b'"error"' in ln}
    assert codes == {c.value for c in ErrorCode}


def test_encode_line_appends_newline_and_is_compact() -> None:
    line = encode_line(StatusRequest(v=1, id="7"))
    assert line == b'{"cmd":"status","v":1,"id":"7"}\n'


def test_event_envelope_encodes() -> None:
    line = encode_line(Event(v=1, event="x", data={"a": 1}))
    assert line == b'{"v":1,"event":"x","data":{"a":1}}\n'


def test_speak_defaults() -> None:
    req = decode_request(b'{"v":1,"id":"1","cmd":"speak","text":"hi"}')
    assert req == SpeakRequest(v=1, id="1", text="hi")
    assert isinstance(req, SpeakRequest)
    assert (req.voice, req.lang, req.rate, req.now) == (None, None, None, False)


def test_speak_volume_decodes() -> None:
    req = decode_request(b'{"v":1,"id":"1","cmd":"speak","text":"x","volume":70}')
    assert isinstance(req, SpeakRequest)
    assert req.volume == 70
    assert SpeakRequest(v=1, id="1", text="x").volume is None


@pytest.mark.parametrize("volume", [b"101", b"-1", b"50.5", b'"50"', b"true"])
def test_speak_volume_out_of_range_is_invalid_argument(volume: bytes) -> None:
    err = _fail(b'{"v":1,"id":"1","cmd":"speak","text":"x","volume":%s}' % volume)
    assert err.code is ErrorCode.invalid_argument


def test_reply_with_error_decodes() -> None:
    rep = decode_reply(b'{"v":1,"id":null,"ok":false,"error":{"code":"bad_frame","message":"m"}}')
    assert rep == Reply(v=1, id=None, ok=False, error=ReplyError(ErrorCode.bad_frame, "m"))


def test_unknown_extra_fields_are_ignored() -> None:
    req = decode_request(b'{"v":1,"id":"1","cmd":"pause","future":{"x":1}}')
    assert req.id == "1"


def test_invalid_json_is_bad_frame() -> None:
    assert _fail(b"{bad").code is ErrorCode.bad_frame


def test_non_object_is_bad_frame() -> None:
    assert _fail(b"[1]").code is ErrorCode.bad_frame


def test_oversize_frame_is_bad_frame() -> None:
    assert _fail(b" " * (MAX_FRAME + 1)).code is ErrorCode.bad_frame


_PAUSE = b'{"v":1,"id":"1","cmd":"pause"}'


def _padded(total: int) -> bytes:
    return _PAUSE + b" " * (total - len(_PAUSE))


def test_max_frame_counts_the_newline() -> None:
    assert decode_request(_padded(MAX_FRAME - 1) + b"\n").id == "1"  # MAX_FRAME with newline
    assert decode_request(_padded(MAX_FRAME - 1)).id == "1"  # newline implied
    assert _fail(_padded(MAX_FRAME) + b"\n").code is ErrorCode.bad_frame  # MAX_FRAME + 1
    assert _fail(_padded(MAX_FRAME)).code is ErrorCode.bad_frame  # newline implied


def test_encode_line_rejects_oversize_frames() -> None:
    event = Event(v=1, event="x", data={"blob": "a" * MAX_FRAME})
    with pytest.raises(ProtocolError) as info:
        encode_line(event)
    assert info.value.code is ErrorCode.bad_frame


def test_encode_line_accepts_a_frame_of_exactly_max_size() -> None:
    empty = len(encode_line(Event(v=1, event="x", data={"b": ""})))
    line = encode_line(Event(v=1, event="x", data={"b": "a" * (MAX_FRAME - empty)}))
    assert len(line) == MAX_FRAME


@pytest.mark.parametrize(
    "line",
    [
        b'{"v":1,"id":"a","cmd":"speak","text":"\xc3\x28"}',
        b'{"v":1,"id":"a","cmd":"speak","text":"\xed\xa0\x80"}',
    ],
)
def test_invalid_utf8_is_bad_frame_and_keeps_the_id(line: bytes) -> None:
    err = _fail(line)
    assert err.code is ErrorCode.bad_frame
    assert err.request_id == "a"


def test_invalid_utf8_without_a_readable_id() -> None:
    err = _fail(b'{"text":"\xc3\x28"')
    assert err.code is ErrorCode.bad_frame
    assert err.request_id is None


@pytest.mark.parametrize("version", ["true", "1.0", '"1"', "null"])
def test_version_must_be_the_integer_one(version: str) -> None:
    err = _fail(b'{"v":%s,"id":"1","cmd":"pause"}' % version.encode())
    assert err.code is ErrorCode.bad_version
    assert err.request_id == "1"


def _fail_reply(line: bytes) -> ProtocolError:
    with pytest.raises(ProtocolError) as info:
        decode_reply(line)
    return info.value


def test_decode_reply_maps_an_unknown_error_code_to_internal() -> None:
    line = b'{"v":1,"id":"a","ok":false,"data":{"k":1},"error":{"code":"brand_new","message":"m"}}'
    rep = decode_reply(line)
    assert rep == Reply(
        v=1, id="a", ok=False, data={"k": 1}, error=ReplyError(ErrorCode.internal, "m")
    )


def test_decode_reply_keeps_known_error_codes() -> None:
    line = b'{"v":1,"id":"a","ok":false,"error":{"code":"unknown_voice","message":"m"}}'
    assert decode_reply(line).error == ReplyError(ErrorCode.unknown_voice, "m")


def test_decode_reply_error_needs_a_string_code() -> None:
    line = b'{"v":1,"id":"a","ok":false,"error":{"code":5,"message":"m"}}'
    assert _fail_reply(line).code is ErrorCode.invalid_argument


def test_decode_reply_invalid_utf8_is_bad_frame() -> None:
    line = b'{"v":1,"id":"a","ok":true,"data":{"x":"\xc3\x28"}}'
    assert _fail_reply(line).code is ErrorCode.bad_frame


def test_decode_reply_checks_the_version_and_the_object() -> None:
    assert _fail_reply(b'{"v":2,"id":"a","ok":true}').code is ErrorCode.bad_version
    assert _fail_reply(b'{"v":true,"id":"a","ok":true}').code is ErrorCode.bad_version
    assert _fail_reply(b"[1]").code is ErrorCode.bad_frame
    assert _fail_reply(b'{"v":1,"id":"a"}').request_id == "a"


def test_wrong_version_is_bad_version() -> None:
    assert _fail(b'{"v":2,"id":"1","cmd":"pause"}').code is ErrorCode.bad_version


def test_missing_version_is_bad_version() -> None:
    assert _fail(b'{"id":"1","cmd":"pause"}').code is ErrorCode.bad_version


def test_unknown_command() -> None:
    assert _fail(b'{"v":1,"id":"1","cmd":"rm"}').code is ErrorCode.unknown_command


def test_missing_command_is_unknown_command() -> None:
    assert _fail(b'{"v":1,"id":"1"}').code is ErrorCode.unknown_command


def test_wrong_type_is_invalid_argument() -> None:
    err = _fail(b'{"v":1,"id":"1","cmd":"speak","text":"hi","rate":"x"}')
    assert err.code is ErrorCode.invalid_argument


def test_missing_required_field_is_invalid_argument() -> None:
    assert _fail(b'{"v":1,"id":"1","cmd":"idle"}').code is ErrorCode.invalid_argument


def test_bad_lang_is_invalid_argument() -> None:
    err = _fail(b'{"v":1,"id":"1","cmd":"speak","text":"hi","lang":"de"}')
    assert err.code is ErrorCode.invalid_argument


def test_error_carries_request_id_when_known() -> None:
    assert _fail(b'{"v":1,"id":"abc","cmd":"rm"}').request_id == "abc"
    assert _fail(b"{bad").request_id is None
    assert _fail(b'{"v":1,"id":5,"cmd":"rm"}').request_id is None


def test_decode_reply_rejects_garbage() -> None:
    with pytest.raises(ProtocolError) as info:
        decode_reply(b"{bad")
    assert info.value.code is ErrorCode.bad_frame
    with pytest.raises(ProtocolError) as info:
        decode_reply(b'{"v":1}')
    assert info.value.code is ErrorCode.invalid_argument


def test_decode_reply_rejects_oversize() -> None:
    with pytest.raises(ProtocolError) as info:
        decode_reply(b" " * (MAX_FRAME + 1))
    assert info.value.code is ErrorCode.bad_frame


def _gen_schema() -> ModuleType:
    path = ROOT / "scripts" / "gen_schema.py"
    spec = importlib.util.spec_from_file_location("gen_schema", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_is_current() -> None:
    fresh = _gen_schema().render_schema()
    assert (ROOT / "docs" / "protocol" / "v1.schema.json").read_text(encoding="utf-8") == fresh


def test_schema_describes_requests_replies_and_events() -> None:
    schema = json.loads(_gen_schema().render_schema())
    assert set(schema) == {"$defs", "request", "reply", "event"}
