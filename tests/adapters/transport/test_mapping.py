from pathlib import Path

import msgspec
import pytest

from voxlektor.adapters.transport.mapping import error_reply, to_command, to_wire_reply
from voxlektor.app.commands import (
    Back,
    Command,
    Pause,
    Resume,
    SetIdle,
    Shutdown,
    Skip,
    Speak,
    Status,
    StopReading,
    Unload,
)
from voxlektor.app.effects import Reply as AppReply
from voxlektor.app.session import Session
from voxlektor.app.voices import VoiceCatalog
from voxlektor.contracts import protocol as wire
from voxlektor.contracts.codec import ProtocolError, decode_reply, decode_request, encode_line
from voxlektor.contracts.engine import VoiceRef
from voxlektor.domain.errors import (
    DomainError,
    InvalidDuration,
    InvalidSetting,
    PlatformError,
    UnknownVoice,
)
from voxlektor.domain.settings import Settings
from voxlektor.domain.types import EngineId
from voxlektor.testing.fakes import FakeClock, FakeEngine, InMemoryVoiceRepository, MemoryOutput
from voxlektor.testing.simulator import Simulator


@pytest.mark.parametrize(
    ("req", "cmd"),
    [
        (wire.SpeakRequest(v=1, id="1", text="hi"), Speak("hi", None, None, None, False)),
        (
            wire.SpeakRequest(v=1, id="1", text="hi", voice="a", lang="pl", rate=1.5, now=True),
            Speak("hi", "a", "pl", 1.5, True),
        ),
        (
            wire.SpeakRequest(v=1, id="1", text="hi", owner="o", group="g", tag="t", preempt=True),
            Speak("hi", None, None, None, False, owner="o", group="g", tag="t", preempt=True),
        ),
        (wire.PauseRequest(v=1, id="1", owner="o"), Pause(owner="o")),
        (wire.ResumeRequest(v=1, id="1", owner="o"), Resume(owner="o")),
        (wire.SkipRequest(v=1, id="1", owner="o"), Skip(owner="o")),
        (wire.BackRequest(v=1, id="1", owner="o"), Back(owner="o")),
        (wire.StopReadingRequest(v=1, id="1", owner="o"), StopReading(owner="o")),
        (
            wire.SpeakRequest(v=1, id="1", text="hi", base_lang="en"),
            Speak("hi", None, None, None, False, base_lang="en"),
        ),
        (
            wire.SpeakRequest(v=1, id="1", text="hi", volume=70),
            Speak("hi", None, None, None, False, volume=70),
        ),
        (wire.PauseRequest(v=1, id="1"), Pause()),
        (wire.ResumeRequest(v=1, id="1"), Resume()),
        (wire.SkipRequest(v=1, id="1"), Skip()),
        (wire.BackRequest(v=1, id="1"), Back()),
        (wire.StopReadingRequest(v=1, id="1"), StopReading()),
        (wire.StatusRequest(v=1, id="1"), Status()),
        (wire.IdleRequest(v=1, id="1", timeout="10m"), SetIdle("10m")),
        (wire.UnloadRequest(v=1, id="1"), Unload()),
        (wire.ShutdownRequest(v=1, id="1"), Shutdown()),
    ],
)
def test_to_command(req: wire.Request, cmd: Command) -> None:
    assert to_command(req) == cmd


def test_subscribe_is_unsupported() -> None:
    with pytest.raises(ProtocolError) as info:
        to_command(wire.SubscribeRequest(v=1, id="9"))
    assert info.value.code is wire.ErrorCode.unsupported
    assert info.value.request_id == "9"


def test_ok_reply() -> None:
    rep = to_wire_reply("1", AppReply(ok=True, data={}, error=None))
    assert rep == wire.Reply(v=1, id="1", ok=True, data={})


def test_status_state_is_uppercased() -> None:
    rep = to_wire_reply("1", AppReply(ok=True, data={"state": "warm", "engine": "e"}, error=None))
    assert rep.data == {"state": "WARM", "engine": "e"}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (InvalidDuration("bad"), wire.ErrorCode.invalid_duration),
        (InvalidSetting("bad"), wire.ErrorCode.invalid_setting),
        (PlatformError(), wire.ErrorCode.unsupported),
        (DomainError("unknown_engine", "no such engine"), wire.ErrorCode.unknown_engine),
        (DomainError("something_new", "odd"), wire.ErrorCode.internal),
    ],
)
def test_domain_error_codes(error: DomainError, code: wire.ErrorCode) -> None:
    rep = to_wire_reply("2", AppReply(ok=False, data={}, error=error))
    assert rep == wire.Reply(
        v=1, id="2", ok=False, error=wire.ReplyError(code=code, message=error.message)
    )


def test_unknown_voice_message_lists_voices() -> None:
    err = UnknownVoice("zed", ("m1-calm", "v0-default"))
    rep = to_wire_reply("3", AppReply(ok=False, data={}, error=err))
    assert rep.error is not None
    assert rep.error.code is wire.ErrorCode.unknown_voice
    assert rep.error.message == "unknown voice 'zed'; available: m1-calm, v0-default"


def test_failed_reply_without_error_is_internal() -> None:
    rep = to_wire_reply("4", AppReply(ok=False, data={}, error=None))
    assert rep.error is not None
    assert rep.error.code is wire.ErrorCode.internal


def test_error_reply_from_protocol_error() -> None:
    rep = error_reply(ProtocolError(wire.ErrorCode.bad_version, "v 2", "5"))
    assert rep == wire.Reply(
        v=1, id="5", ok=False, error=wire.ReplyError(wire.ErrorCode.bad_version, "v 2")
    )
    assert error_reply(ProtocolError(wire.ErrorCode.bad_frame, "x")).id is None


def _supported_wire_commands() -> set[str]:
    union = msgspec.inspect.type_info(wire.Request)
    assert isinstance(union, msgspec.inspect.UnionType)
    tags: set[str] = set()
    for member in union.types:
        assert isinstance(member, msgspec.inspect.StructType)
        assert isinstance(member.tag, str)
        tags.add(member.tag)
    return tags - {"subscribe"}


def test_status_caps_equal_supported_wire_commands() -> None:
    clock = FakeClock()
    refs = [VoiceRef("a", Path("/v/a/ref.wav"), "ref a")]
    catalog = VoiceCatalog(InMemoryVoiceRepository(refs))
    session = Session(Settings(engine=EngineId("fake")), catalog, clock)
    sim = Simulator(session, FakeEngine(sample_rate=100), MemoryOutput(auto_complete=True), clock)
    rep = to_wire_reply("1", sim.handle(Status()))
    assert rep.data is not None
    caps: list[str] = rep.data["caps"]
    assert set(caps) == _supported_wire_commands()
    assert rep.data["state"] == "COLD"


def _protocol_failure(line: bytes) -> wire.Reply:
    try:
        decode_request(line)
    except ProtocolError as err:
        return error_reply(err)
    raise AssertionError("expected a ProtocolError")


def _real_error_replies() -> dict[str, wire.Reply]:
    """Error replies produced by the real code path, keyed by error code."""
    clock = FakeClock()
    catalog = VoiceCatalog(InMemoryVoiceRepository([VoiceRef("a", Path("/v/a.wav"), "a")]))
    session = Session(Settings(engine=EngineId("fake")), catalog, clock)
    sim = Simulator(session, FakeEngine(sample_rate=100), MemoryOutput(auto_complete=True), clock)
    unknown_engine = DomainError(
        "unknown_engine", "unknown engine 'nope'; available: fake, omnivoice"
    )
    with pytest.raises(ProtocolError) as subscribe:
        to_command(wire.SubscribeRequest(v=1, id="r12"))
    unsupported = error_reply(subscribe.value)
    return {
        "bad_frame": _protocol_failure(b"{bad"),
        "bad_version": _protocol_failure(b'{"v":2}'),
        "unknown_command": _protocol_failure(b'{"v":1,"id":"x1","cmd":"rm"}'),
        "invalid_argument": _protocol_failure(
            b'{"v":1,"id":"x2","cmd":"speak","text":"hi","rate":"x"}'
        ),
        "unknown_voice": to_wire_reply(
            "x3", AppReply(ok=False, data={}, error=UnknownVoice("zed", ("m1-calm", "v0-default")))
        ),
        "unknown_engine": to_wire_reply("x4", AppReply(ok=False, data={}, error=unknown_engine)),
        "invalid_duration": to_wire_reply("x5", sim.handle(SetIdle("5x"))),
        "invalid_setting": to_wire_reply("x6", sim.handle(Speak("hi", "a", None, 9.0, False))),
        "unsupported": unsupported,
    }


def _speak_sim() -> Simulator:
    clock = FakeClock()
    catalog = VoiceCatalog(InMemoryVoiceRepository([VoiceRef("a", Path("/v/a.wav"), "a")]))
    session = Session(Settings(engine=EngineId("fake")), catalog, clock)
    return Simulator(session, FakeEngine(sample_rate=100), MemoryOutput(auto_complete=True), clock)


def test_speak_reply_carries_the_planned_chunk_count() -> None:
    sim = _speak_sim()
    reply = to_wire_reply("r1", sim.handle(Speak("Hello world.", "a", None, None, False)))
    assert reply.data == {"chunks": 1, "group": "g1"}
    fixtures = Path(__file__).parents[2] / "contracts" / "protocol_v1" / "replies.jsonl"
    assert encode_line(reply) in fixtures.read_bytes().splitlines(keepends=True)
    assert to_wire_reply("r2", sim.handle(Speak("   ", "a", None, None, False))).data == {
        "chunks": 0
    }


def test_golden_error_messages_match_the_real_code_path() -> None:
    fixtures = Path(__file__).parents[2] / "contracts" / "protocol_v1" / "replies.jsonl"
    by_code = {
        reply.error.code: line
        for line in fixtures.read_bytes().splitlines()
        if (reply := decode_reply(line)).error is not None
    }
    for code, reply in _real_error_replies().items():
        assert encode_line(reply) == by_code[wire.ErrorCode(code)] + b"\n", code
