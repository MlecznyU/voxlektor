"""Socket protocol v1: one JSON object per line, requests, replies and (reserved) events.

Self-contained on purpose: no imports from ``domain`` or the other contracts, so that any
client can depend on this module alone. Changes in v1 are strictly additive.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal

import msgspec

PROTOCOL_VERSION = 1
MAX_FRAME = 1 << 20  # the whole line INCLUDING its terminating newline is at most this long


class ErrorCode(StrEnum):
    """Error codes of v1; new codes may be added within v1; clients treat unknown codes as
    ``internal``."""

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


OWNER_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"

# Added within v1 (S1): identity of the client on the multi-client reading queue.
_Owner = Annotated[str, msgspec.Meta(pattern=OWNER_PATTERN)]
_Label = Annotated[str, msgspec.Meta(min_length=1, max_length=128)]


class SpeakRequest(_Request, tag="speak"):
    text: str
    voice: str | None = None
    lang: WireLang | None = None
    rate: float | None = None
    now: bool = False
    # Added within v1 (P1): last, so positional construction of this class is unchanged.
    base_lang: Literal["pl", "en"] | None = None  # tie fallback of `auto` detection
    # Added within v1 (Etap 1.1); daemons before Etap 1.1 ignore it. 50 = the even level.
    volume: Annotated[int, msgspec.Meta(ge=0, le=100)] | None = None
    # Added within v1 (S1): queue identity; daemons before S1 ignore these.
    owner: _Owner | None = None
    group: _Label | None = None
    tag: _Label | None = None
    preempt: bool = False


class PauseRequest(_Request, tag="pause"):
    owner: _Owner | None = None  # Added within v1 (S1)


class ResumeRequest(_Request, tag="resume"):
    owner: _Owner | None = None  # Added within v1 (S1)


class SkipRequest(_Request, tag="skip"):
    owner: _Owner | None = None  # Added within v1 (S1)


class BackRequest(_Request, tag="back"):
    owner: _Owner | None = None  # Added within v1 (S1)


class StopReadingRequest(_Request, tag="stop_reading"):
    owner: _Owner | None = None  # Added within v1 (S1)


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
