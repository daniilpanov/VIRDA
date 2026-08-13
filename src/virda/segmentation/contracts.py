from abc import ABC, abstractmethod

from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask
from virda.pipeline_context import PipelineContext


class HeadSegmenter(ABC):
    def run(self, context: PipelineContext) -> SegmentationMask:
        return self._process(context.get_store_notnull(MRIVolume))

    @abstractmethod
    def _process(self, volume: MRIVolume) -> SegmentationMask:
        raise NotImplementedError


class SegmentationMaskPostprocessor(ABC):
    def run(self, context: PipelineContext) -> SegmentationMask:
        return self._process(context.get_store_notnull(SegmentationMask))

    @abstractmethod
    def _process(self, mask: SegmentationMask) -> SegmentationMask:
        raise NotImplementedError
