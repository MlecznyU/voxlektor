"""Translation between the wire protocol v1 and the application: the only place that knows both.

``to_command`` raises ``ProtocolError(unsupported)`` for ``subscribe`` (events are reserved);
the transport catches every ``ProtocolError`` and answers with ``error_reply``.
"""

from collections.abc import Callable

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
from voxlektor.contracts import protocol as wire
from voxlektor.contracts.codec import ProtocolError
from voxlektor.domain.errors import DomainError, UnknownVoice

# Domain codes whose wire name differs; names that match ``ErrorCode`` map to themselves.
_RENAMED = {"unsupported_platform": wire.ErrorCode.unsupported}


# Requests that carry no arguments.
_PLAIN: dict[type[wire.Request], Callable[[], Command]] = {
    wire.PauseRequest: Pause,
    wire.ResumeRequest: Resume,
    wire.SkipRequest: Skip,
    wire.BackRequest: Back,
    wire.StopReadingRequest: StopReading,
    wire.StatusRequest: Status,
    wire.UnloadRequest: Unload,
    wire.ShutdownRequest: Shutdown,
}


def to_command(req: wire.Request) -> Command:
    if isinstance(req, wire.SpeakRequest):
        return Speak(
            req.text,
            req.voice,
            req.lang,
            req.rate,
            req.now,
            base_lang=req.base_lang,
            volume=req.volume,
        )
    if isinstance(req, wire.IdleRequest):
        return SetIdle(req.timeout)
    if isinstance(req, wire.SubscribeRequest):
        raise ProtocolError(wire.ErrorCode.unsupported, "subscribe is not supported yet", req.id)
    return _PLAIN[type(req)]()


def _error_code(code: str) -> wire.ErrorCode:
    renamed = _RENAMED.get(code)
    if renamed is not None:
        return renamed
    try:
        return wire.ErrorCode(code)
    except ValueError:
        return wire.ErrorCode.internal


def _message(err: DomainError) -> str:
    if isinstance(err, UnknownVoice):
        return f"{err.message}; available: {', '.join(err.available)}"
    return err.message


def to_wire_reply(req_id: str, reply: AppReply) -> wire.Reply:
    """Wire form of an application reply.

    Any ok reply whose data holds a string ``state`` gets it uppercased; only the status reply has
    that key. (The frozen signature does not carry the request type, so this is not scoped further.)
    """
    if reply.ok:
        data = dict(reply.data)
        state = data.get("state")
        if isinstance(state, str):  # lifecycle names are uppercase on the wire
            data["state"] = state.upper()
        return wire.Reply(v=wire.PROTOCOL_VERSION, id=req_id, ok=True, data=data)
    err = reply.error
    if err is None:
        info = wire.ReplyError(wire.ErrorCode.internal, "command failed")
    else:
        info = wire.ReplyError(_error_code(err.code), _message(err))
    return wire.Reply(v=wire.PROTOCOL_VERSION, id=req_id, ok=False, error=info)


def error_reply(err: ProtocolError) -> wire.Reply:
    """Error reply for a frame that could not be served; the id is ``err.request_id`` only."""
    info = wire.ReplyError(err.code, err.message)
    return wire.Reply(v=wire.PROTOCOL_VERSION, id=err.request_id, ok=False, error=info)
