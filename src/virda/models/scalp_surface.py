"""The scalp surface: the head mask together with the mesh extracted from it."""

from dataclasses import dataclass

from virda.models.mesh import Mesh
from virda.models.segmentation_mask import SegmentationMask


@dataclass(frozen=True)
class ScalpSurface:
    """Segmentation mask together with the mesh extracted from it."""

    mask: SegmentationMask
    mesh: Mesh
