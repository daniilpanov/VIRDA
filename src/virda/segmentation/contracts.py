from typing import Protocol, runtime_checkable

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


@runtime_checkable
class HeadSegmenter(Protocol):
    def run(self, volume: MRIVolume, closing_radius: int = 5) -> SegmentationMask: ...
