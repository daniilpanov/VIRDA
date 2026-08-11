import numpy as np
import trimesh

from virda.mesh.mesh_extractor import MarchingCubesExtractor
from virda.mesh.ray_casting import ray_internal_face_mask


def _box_with_inner_cavity_mesh() -> trimesh.Trimesh:
    """Solid 64^3 box with a spherical cavity open to the exterior via a tunnel.

    The cavity is 6-connected to the outside (through the tunnel), so its walls
    survive the raw-mesh component cleanup and look like an internal surface.
    """
    n = 64
    center = np.array([32.0, 32.0, 32.0])
    grid = np.indices((n, n, n)).astype(float)
    radius = np.linalg.norm(grid - center.reshape(-1, 1, 1, 1), axis=0)
    cavity = radius <= 16.0
    tunnel = (
        (np.abs(grid[0] - 32) <= 4)
        & (np.abs(grid[1] - 32) <= 4)
        & (grid[2] >= 32)
        & (grid[2] <= 56)
    )
    mask = np.ones((n, n, n), dtype=bool)
    mask &= ~(cavity | tunnel)
    mask[:8, :, :] = False
    mask[-8:, :, :] = False
    mask[:, :8, :] = False
    mask[:, -8:, :] = False
    mask[:, :, :8] = False
    mask[:, :, -8:] = False

    mesh = MarchingCubesExtractor().extract(mask, np.eye(4))
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)


def _shell_faces(mesh: trimesh.Trimesh, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    distance = np.linalg.norm(centroids - center, axis=1)
    cavity = np.asarray((distance >= 14.0) & (distance <= 20.0))
    outer = np.asarray(distance > 20.0)
    return cavity, outer


class TestRayInternalFaceMask:
    def test_flags_cavity_walls(self) -> None:
        mesh = _box_with_inner_cavity_mesh()
        center = np.array([32.0, 32.0, 32.0])
        cavity, outer = _shell_faces(mesh, center)
        assert cavity.sum() > 1000

        remove = ray_internal_face_mask(mesh, region=None)

        assert remove.dtype == bool
        assert remove.shape == (len(mesh.faces),)
        assert int((cavity & remove).sum()) > 0.8 * int(cavity.sum())
        assert int((outer & remove).sum()) < 50

    def test_empty_mesh_returns_empty_mask(self) -> None:
        result = ray_internal_face_mask(trimesh.Trimesh(vertices=[], faces=[]), region=None)

        assert result.dtype == bool
        assert len(result) == 0
