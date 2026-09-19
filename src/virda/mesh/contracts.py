from abc import ABC, abstractmethod

from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


class MeshExtractor(ABC):
    @abstractmethod
    def process(self, mask: SegmentationMask, mri_volume: MRIVolume) -> ScalpMesh:
        raise NotImplementedError


class MeshPostprocessor(ABC):
    @abstractmethod
    def process(self, mesh: ScalpMesh) -> ScalpMesh:
        raise NotImplementedError
