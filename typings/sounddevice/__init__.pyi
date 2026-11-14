"""Handwritten stub: exactly the sounddevice symbols voxlektor uses."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

class PortAudioError(Exception): ...
class CallbackAbort(Exception): ...
class CallbackStop(Exception): ...

class CallbackFlags:
    output_underflow: bool

class OutputStream:
    def __init__(
        self,
        samplerate: float | None = None,
        blocksize: int | None = None,
        device: int | str | None = None,
        channels: int | None = None,
        dtype: str | None = None,
        latency: float | str | None = None,
        callback: Callable[[npt.NDArray[np.float32], int, object, CallbackFlags], None]
        | None = None,
        finished_callback: Callable[[], None] | None = None,
    ) -> None: ...
    @property
    def active(self) -> bool: ...
    def start(self) -> None: ...
    def abort(self, ignore_errors: bool = True) -> None: ...
    def close(self, ignore_errors: bool = True) -> None: ...
