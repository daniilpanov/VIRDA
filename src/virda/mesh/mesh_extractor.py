import numpy as np
from skimage.measure import marching_cubes

from virda.mesh import MeshExtractor
from virda.mesh.adjacency import build_scalp_mesh
from virda.mesh.density import step_size_for_voxel_size
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


class MarchingCubesExtractor(MeshExtractor):
    def __init__(self, voxel_size_mm: float | None = None) -> None:
        """Extract the scalp surface with a tunable density.

        ``voxel_size_mm`` is the desired effective voxel size of the output
        mesh (``None`` keeps the native NIfTI resolution / ``step_size=1``).
        """
        self._voxel_size_mm = voxel_size_mm

    def _process(self, mask: SegmentationMask, mri_volume: MRIVolume) -> ScalpMesh:
        affine = mri_volume.affine

        if self._voxel_size_mm is None:
            step_size = 1
        else:
            step_size, _ = step_size_for_voxel_size(self._voxel_size_mm, mri_volume.spacing)

        voxel_vertices, triangle_faces, _, _ = marching_cubes(
            mask.mask, level=0.5, step_size=step_size
        )
        world_vertices = voxel_vertices @ affine[:3, :3].T + affine[:3, 3]

        return build_scalp_mesh(
            vertices=np.asarray(world_vertices, dtype=np.float64),
            faces=np.asarray(triangle_faces, dtype=np.int64),
        )
