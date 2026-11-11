"""Handwritten stub: exactly the pytsmod symbols voxlektor uses."""

import numpy as np
import numpy.typing as npt

def wsola(
    x: npt.NDArray[np.float32],
    s: float,
    win_type: str = "hann",
    win_size: int = 1024,
    syn_hop_size: int = 512,
    tolerance: int = 512,
) -> npt.NDArray[np.float64]: ...
