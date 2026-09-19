from dataclasses import dataclass

from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


@dataclass(frozen=True)
class ScalpSurface:
    mask: SegmentationMask
    mesh: ScalpMesh