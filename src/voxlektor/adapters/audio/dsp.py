"""Signal processing helpers for playback."""

import numpy as np
import numpy.typing as npt
import pytsmod


def time_stretch(samples: npt.NDArray[np.float32], rate: float) -> npt.NDArray[np.float32]:
    """Change speed without changing pitch (``rate`` 1.5 = 1.5x faster); 1.0 is the identity."""
    if rate == 1.0:
        return samples
    stretched = pytsmod.wsola(samples, 1.0 / rate)
    return np.asarray(stretched, dtype=np.float32)
