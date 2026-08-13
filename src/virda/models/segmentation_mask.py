from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray
