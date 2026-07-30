import numpy as np
from skimage.measure import marching_cubes

from virda.models.scalp_mesh import ScalpMesh


class MarchingCubesExtractor:
    def extract(self, mask: np.ndarray, affine: np.ndarray) -> ScalpMesh:
        voxel_vertices, triangle_faces, _, _ = marching_cubes(mask, level=0.5)

        world_vertices = voxel_vertices @ affine[:3, :3].T + affine[:3, 3]

        return ScalpMesh(
            vertices=np.asarray(world_vertices, dtype=np.float64),
            faces=np.asarray(triangle_faces, dtype=np.int64),
        )
