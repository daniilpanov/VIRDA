from abc import ABC, abstractmethod

from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.pipeline_context import PipelineContext


class MeshExtractor(ABC):
    def run(self, context: PipelineContext) -> ScalpMesh:
        return self._process(
            context.get_store_notnull(SegmentationMask),
            context.get_store_notnull(MRIVolume),
        )

    @abstractmethod
    def _process(self, mask: SegmentationMask, mri_volume: MRIVolume) -> ScalpMesh:
        raise NotImplementedError


class MeshPostprocessor(ABC):
    def run(self, context: PipelineContext) -> ScalpMesh:
        return self._process(context.get_store_notnull(ScalpMesh))

    @abstractmethod
    def _process(self, mesh: ScalpMesh) -> ScalpMesh:
        raise NotImplementedError
