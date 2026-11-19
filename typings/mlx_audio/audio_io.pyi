import numpy as np
import numpy.typing as npt

def read(
    file: str,
    always_2d: bool = ...,
    dtype: str = ...,
    sample_rate: int | None = ...,
    nchannels: int | None = ...,
    duration: float | None = ...,
) -> tuple[npt.NDArray[np.float64], int]: ...
