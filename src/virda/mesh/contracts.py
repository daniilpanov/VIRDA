from typing import Protocol, runtime_checkable

from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


@runtime_checkable
class MeshCleaner(Protocol):
    def run(self, mesh: ScalpMesh) -> ScalpMesh: ...


@runtime_checkable
class MeshSmoother(Protocol):
    def run(self, mesh: ScalpMesh) -> ScalpMesh: ...


@runtime_checkable
class MeshExtractor(Protocol):
    def run(self, mask: SegmentationMask, mri_volume: MRIVolume) -> ScalpMesh: ...
