from abc import ABC, abstractmethod
from logging import Logger

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


class HeadSegmenter(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    @abstractmethod
    def process(self, volume: MRIVolume) -> SegmentationMask:
        raise NotImplementedError


class SegmentationMaskPostprocessor(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    @abstractmethod
    def process(self, mask: SegmentationMask) -> SegmentationMask:
        raise NotImplementedError
