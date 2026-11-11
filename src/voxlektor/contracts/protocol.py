"""Socket protocol v1: one JSON object per line, requests, replies and (reserved) events.

Self-contained on purpose: no imports from ``domain`` or the other contracts, so that any
client can depend on this module alone. Changes in v1 are strictly additive.
"""

from enum import StrEnum
from typing import Any, Literal

import msgspec

PROTOCOL_VERSION = 1
MAX_FRAME = 1 << 20  # the whole line INCLUDING its terminating newline is at most this long


class ErrorCode(StrEnum):
    """Closed set of error codes; a new code needs a new protocol version."""

    bad_frame = "bad_frame"
    bad_version = "bad_version"
    unknown_command = "unknown_command"
    invalid_argument = "invalid_argument"
    unknown_voice = "unknown_voice"
    unknown_engine = "unknown_engine"
    invalid_duration = "invalid_duration"
    invalid_setting = "invalid_setting"
    engine_failed = "engine_failed"
    audio_failed = "audio_failed"
    unsupported = "unsupported"
    internal = "internal"


WireLang = Literal["auto", "pl", "en"]


class _Request(msgspec.Struct, tag_field="cmd", omit_defaults=True):
    """Every request carries the protocol version and a client-chosen ``id`` echoed in the reply."""

    v: int
    id: str


class SpeakRequest(_Request, tag="speak"):
    text: str
    voice: str | None = None
    lang: WireLang | None = None
    rate: float | None = None
    now: bool = False


class PauseRequest(_Request, tag="pause"):
    pass


class ResumeRequest(_Request, tag="resume"):
    pass


class SkipRequest(_Request, tag="skip"):
    pass


class BackRequest(_Request, tag="back"):
    pass


class StopReadingRequest(_Request, tag="stop_reading"):
    pass


class StatusRequest(_Request, tag="status"):
    pass


class IdleRequest(_Request, tag="idle"):
    timeout: str


class UnloadRequest(_Request, tag="unload"):
    pass


class ShutdownRequest(_Request, tag="shutdown"):
    pass


class SubscribeRequest(_Request, tag="subscribe"):
    """Reserved for events; the MVP daemon answers ``unsupported``."""


type Request = (
    SpeakRequest
    | PauseRequest
    | ResumeRequest
    | SkipRequest
    | BackRequest
    | StopReadingRequest
    | StatusRequest
    | IdleRequest
    | UnloadRequest
    | ShutdownRequest
    | SubscribeRequest
)


class ReplyError(msgspec.Struct, frozen=True):
    code: ErrorCode
    message: str


class Reply(msgspec.Struct, frozen=True, omit_defaults=True):
    """``id`` is null only when the request had no readable id."""

    v: int
    id: str | None
    ok: bool
    data: dict[str, Any] | None = None
    error: ReplyError | None = None


class Event(msgspec.Struct, frozen=True):
    """Envelope for server-pushed events; defined for v1, unused in the MVP."""

    v: int
    event: str
    data: dict[str, Any]
