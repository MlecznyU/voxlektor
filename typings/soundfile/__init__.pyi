"""Handwritten stub: exactly the soundfile symbols voxlektor uses."""

from typing import BinaryIO

import numpy as np
import numpy.typing as npt

class _Info:
    samplerate: int
    channels: int
    subtype: str
    duration: float

def info(file: str | BinaryIO) -> _Info: ...
def read(
    file: str | BinaryIO,
    dtype: str = "float64",
    always_2d: bool = False,
) -> tuple[npt.NDArray[np.float32], int]: ...
def write(
    file: str | BinaryIO,
    data: npt.NDArray[np.float32],
    samplerate: int,
    subtype: str | None = None,
) -> None: ...
