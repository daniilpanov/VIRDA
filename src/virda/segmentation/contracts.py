from typing import Protocol, runtime_checkable

import numpy as np

from virda.models.mri_volume import MRIVolume


@runtime_checkable
class HeadSegmenter(Protocol):
    def segment(self, volume: MRIVolume, closing_radius: int = 5) -> np.ndarray: ...
