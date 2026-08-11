import numpy as np
import pytest
from dataclasses import replace

from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.scalp_mesh import ScalpMesh


@pytest.fixture
def sphere_mesh() -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = squared_distance <= 8**2

    from virda.mesh.mesh_extractor import MarchingCubesExtractor

    return MarchingCubesExtractor().extract(mask, np.eye(4))


class TestLaplacianSmoother:
    def test_smooth_preserves_mesh_structure(self, sphere_mesh: ScalpMesh) -> None:
        smoother = LaplacianSmoother(iterations=3, lamb=0.5)
        smoothed = smoother.smooth(sphere_mesh)

        assert isinstance(smoothed, ScalpMesh)
        assert smoothed.vertices.shape == sphere_mesh.vertices.shape
        assert smoothed.faces.shape == sphere_mesh.faces.shape
        assert np.array_equal(smoothed.faces, sphere_mesh.faces)

    def test_smooth_preserves_face_adjacency(self, sphere_mesh: ScalpMesh) -> None:
        sphere_mesh = replace(sphere_mesh, face_adjacency=np.array([[0, 1], [1, 2]]))
        smoother = LaplacianSmoother(iterations=3, lamb=0.5)
        smoothed = smoother.smooth(sphere_mesh)

        assert smoothed.face_adjacency is not None
        assert np.array_equal(smoothed.face_adjacency, sphere_mesh.face_adjacency)

    def test_smooth_moves_vertices(self, sphere_mesh: ScalpMesh) -> None:
        smoother = LaplacianSmoother(iterations=5, lamb=0.5)
        smoothed = smoother.smooth(sphere_mesh)

        vertex_displacement = np.abs(smoothed.vertices - sphere_mesh.vertices).max()
        assert vertex_displacement > 1e-7
        assert vertex_displacement < 5.0


class TestTaubinSmoother:
    def test_smooth_preserves_mesh_structure(self, sphere_mesh: ScalpMesh) -> None:
        smoother = TaubinSmoother(iterations=3, lamb=0.5, nu=-0.53)
        smoothed = smoother.smooth(sphere_mesh)

        assert isinstance(smoothed, ScalpMesh)
        assert smoothed.vertices.shape == sphere_mesh.vertices.shape
        assert smoothed.faces.shape == sphere_mesh.faces.shape
        assert np.array_equal(smoothed.faces, sphere_mesh.faces)

    def test_smooth_preserves_face_adjacency(self, sphere_mesh: ScalpMesh) -> None:
        sphere_mesh = replace(sphere_mesh, face_adjacency=np.array([[0, 1], [1, 2]]))
        smoother = TaubinSmoother(iterations=3, lamb=0.5, nu=-0.53)
        smoothed = smoother.smooth(sphere_mesh)

        assert smoothed.face_adjacency is not None
        assert np.array_equal(smoothed.face_adjacency, sphere_mesh.face_adjacency)

    def test_smooth_moves_vertices(self, sphere_mesh: ScalpMesh) -> None:
        smoother = TaubinSmoother(iterations=5, lamb=0.5, nu=-0.53)
        smoothed = smoother.smooth(sphere_mesh)

        vertex_displacement = np.abs(smoothed.vertices - sphere_mesh.vertices).max()
        assert vertex_displacement > 1e-7
        assert vertex_displacement < 5.0
