from abc import ABC, abstractmethod
from logging import Logger

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda.pipeline_context import PipelineContext


class HeadSegmenter(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    def run(self, context: PipelineContext) -> SegmentationMask:
        self._logger = context.get_logger()
        return self._process(context.get_store_notnull(MRIVolume))

    @abstractmethod
    def _process(self, volume: MRIVolume) -> SegmentationMask:
        raise NotImplementedError


class SegmentationMaskPostprocessor(ABC):
    def __init__(self) -> None:
        self._logger: Logger | None = None

    def run(self, context: PipelineContext) -> SegmentationMask:
        self._logger = context.get_logger()
        return self._process(context.get_store_notnull(SegmentationMask))

    @abstractmethod
    def _process(self, mask: SegmentationMask) -> SegmentationMask:
        raise NotImplementedError
