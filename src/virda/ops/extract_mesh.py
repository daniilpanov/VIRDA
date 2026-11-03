"""Step: extract a world-space scalp mesh with marching cubes."""

import numpy as np
from skimage.measure import marching_cubes

from virda.mesh.adjacency import build_scalp_mesh
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


def extract_mesh(mask: SegmentationMask, mri_volume: MRIVolume) -> ScalpMesh:
    """Extract the mask surface and map it into world millimeters."""
    affine = mri_volume.affine

    voxel_vertices, triangle_faces, _, _ = marching_cubes(mask.mask, level=0.5)
    world_vertices = voxel_vertices @ affine[:3, :3].T + affine[:3, 3]

    return build_scalp_mesh(
        vertices=np.asarray(world_vertices, dtype=np.float64),
        faces=np.asarray(triangle_faces, dtype=np.int64),
    )
