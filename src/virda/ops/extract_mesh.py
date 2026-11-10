"""Step: extract a world-space scalp mesh with marching cubes."""

import numpy as np
from skimage.measure import marching_cubes

from virda.mesh.density import step_size_for_voxel_size
from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


def extract_mesh(
    mask: SegmentationMask,
    mri_volume: MRIVolume,
    *,
    voxel_size_mm: float | None = None,
) -> Mesh:
    """Extract the mask surface and map it into world millimeters.

    ``voxel_size_mm`` is the desired effective voxel size of the output mesh;
    ``None`` keeps the native NIfTI resolution (``step_size=1``).
    """
    affine = mri_volume.affine

    if voxel_size_mm is None:
        step_size = 1
    else:
        step_size, _ = step_size_for_voxel_size(voxel_size_mm, mri_volume.spacing)

    voxel_vertices, triangle_faces, _, _ = marching_cubes(mask.mask, level=0.5, step_size=step_size)
    world_vertices = voxel_vertices @ affine[:3, :3].T + affine[:3, 3]

    return Mesh(
        vertices=np.asarray(world_vertices, dtype=np.float64),
        faces=np.asarray(triangle_faces, dtype=np.int64),
    )
