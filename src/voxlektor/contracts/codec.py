"""Encoding and decoding of protocol v1 frames (one JSON object per line).

Size rule: a frame, counted with its terminating newline, is at most ``MAX_FRAME`` bytes. The
decoders accept a line with or without the trailing newline (it is counted either way).
"""

from typing import Any, get_args

import msgspec

from voxlektor.contracts import protocol as p

_COMMANDS = frozenset(t.__struct_config__.tag for t in get_args(p.Request.__value__))


class ProtocolError(Exception):
    """A frame that cannot be served; ``request_id`` is the request's id when it was readable."""

    def __init__(self, code: p.ErrorCode, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id


def encode_line(msg: msgspec.Struct) -> bytes:
    line = msgspec.json.encode(msg) + b"\n"
    if len(line) > p.MAX_FRAME:
        raise ProtocolError(p.ErrorCode.bad_frame, f"frame larger than {p.MAX_FRAME} bytes")
    return line


def _id_of(raw: dict[str, object]) -> str | None:
    raw_id = raw.get("id")
    return raw_id if isinstance(raw_id, str) else None


def _salvage_id(line: bytes) -> str | None:
    """Best-effort request id from a frame that is not valid UTF-8."""
    try:
        raw = msgspec.json.decode(line.decode("utf-8", errors="replace"), type=dict[str, object])
    except msgspec.DecodeError:
        return None
    return _id_of(raw)


def _load(line: bytes) -> dict[str, object]:
    """Parse one frame into a JSON object, enforcing size, encoding and protocol version."""
    if len(line) + (0 if line.endswith(b"\n") else 1) > p.MAX_FRAME:
        raise ProtocolError(p.ErrorCode.bad_frame, f"frame larger than {p.MAX_FRAME} bytes")
    try:
        raw = msgspec.json.decode(line, type=dict[str, object])
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            p.ErrorCode.bad_frame, f"frame is not valid UTF-8: {exc}", _salvage_id(line)
        ) from None
    except msgspec.DecodeError:  # includes "not an object"
        raise ProtocolError(p.ErrorCode.bad_frame, "frame is not a valid JSON object") from None
    version = raw.get("v")
    if type(version) is not int or version != p.PROTOCOL_VERSION:
        raise ProtocolError(
            p.ErrorCode.bad_version, f"unsupported protocol version {version!r}", _id_of(raw)
        )
    return raw


def decode_request(line: bytes) -> p.Request:
    raw = _load(line)
    req_id = _id_of(raw)
    cmd = raw.get("cmd")
    if cmd not in _COMMANDS:
        raise ProtocolError(p.ErrorCode.unknown_command, f"unknown command {cmd!r}", req_id)
    try:
        return msgspec.convert(raw, p.Request)
    except msgspec.ValidationError as exc:
        raise ProtocolError(p.ErrorCode.invalid_argument, str(exc), req_id) from None


class _LooseError(msgspec.Struct):
    code: str
    message: str


class _LooseReply(msgspec.Struct):
    """A reply whose error code is not yet checked against the closed ``ErrorCode`` enum."""

    v: int
    id: str | None
    ok: bool
    data: dict[str, Any] | None = None
    error: _LooseError | None = None


def decode_reply(line: bytes) -> p.Reply:
    """Decode a reply; an ``error.code`` this version does not know becomes ``internal``."""
    raw = _load(line)
    try:
        loose = msgspec.convert(raw, _LooseReply)
    except msgspec.ValidationError as exc:
        raise ProtocolError(p.ErrorCode.invalid_argument, str(exc), _id_of(raw)) from None
    error = None
    if loose.error is not None:
        known = p.ErrorCode.__members__.get(loose.error.code, p.ErrorCode.internal)
        error = p.ReplyError(known, loose.error.message)
    return p.Reply(v=loose.v, id=loose.id, ok=loose.ok, data=loose.data, error=error)
