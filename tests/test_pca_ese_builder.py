import numpy as np
import pytest

from tests.helpers.meshes import make_plane, make_sphere
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh

ESE_OFFSET_MM = 1.0


def run_builder(mesh: ScalpMesh, **config) -> ESEMesh:
    builder = PCAESEBuilder(ese_offset_mm=ESE_OFFSET_MM, **config)
    return builder.process(mesh)


class TestPCAESEBuilder:
    def test_flat_plane_normals_along_z(self) -> None:
        result = run_builder(make_plane(), k_neighbors=20)

        assert result.vertices.shape == result.scalp_vertices.shape
        assert np.mean(np.abs(result.normals[:, 2])) > 0.9

    def test_sphere_normals_radial_outward(self) -> None:
        mesh = make_sphere()
        result = run_builder(mesh, k_neighbors=30)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.mean(dots) > 0.8
        assert np.median(result.quality) < 0.1

    def test_ese_offset_distance(self) -> None:
        mesh = make_sphere()
        result = run_builder(mesh, k_neighbors=30)

        offsets = np.linalg.norm(result.vertices - mesh.vertices, axis=1)
        assert np.allclose(offsets, ESE_OFFSET_MM, atol=1e-9)

    def test_quality_in_unit_interval(self) -> None:
        result = run_builder(make_sphere(), k_neighbors=30)

        assert np.all((result.quality >= 0.0) & (result.quality <= 1.0))

    def test_faces_mirror_scalp_mesh(self) -> None:
        mesh = make_sphere()
        result = run_builder(mesh, k_neighbors=30)

        assert np.array_equal(result.faces, mesh.faces)

    def test_rejects_k_larger_than_vertex_count(self) -> None:
        with pytest.raises(ValueError, match="k_neighbors must be less than"):
            run_builder(make_sphere(), k_neighbors=500)

    def test_flat_plane_normals_along_z_radius(self) -> None:
        result = run_builder(
            make_plane(), neighborhood_radius_mm=5.0, k_neighbors=None, min_neighbors=5
        )

        assert np.mean(np.abs(result.normals[:, 2])) > 0.9

    def test_sphere_normals_radial_outward_radius(self) -> None:
        mesh = make_sphere()
        result = run_builder(
            mesh, neighborhood_radius_mm=20.0, k_neighbors=None, min_neighbors=5
        )

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.mean(dots) > 0.8
        assert np.median(result.quality) < 0.1

    def test_radius_and_knn_consistency(self) -> None:
        radius_result = run_builder(make_sphere(), neighborhood_radius_mm=20.0, k_neighbors=None)
        knn_result = run_builder(make_sphere(), k_neighbors=30)

        dots = np.sum(radius_result.normals * knn_result.normals, axis=1)
        assert np.mean(dots) > 0.99

    def test_radius_falls_back_to_knn(self) -> None:
        mesh = make_sphere()
        result = run_builder(
            mesh, neighborhood_radius_mm=1.0, k_neighbors=None, min_neighbors=5
        )

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.allclose(np.linalg.norm(result.normals, axis=1), 1.0)
        assert np.all(np.isfinite(result.quality))
        assert np.mean(dots) > 0.8

    def test_weighted_pca_knn_consistent(self) -> None:
        unweighted = run_builder(make_sphere(), k_neighbors=30)
        weighted = run_builder(
            make_sphere(), k_neighbors=30, use_weighted_pca=True, pca_sigma_mm=10.0
        )

        dots = np.sum(unweighted.normals * weighted.normals, axis=1)
        assert np.mean(dots) > 0.99
        assert np.all(np.isfinite(weighted.quality))

    def test_weighted_pca_radius_mode(self) -> None:
        config = {
            "neighborhood_radius_mm": 20.0,
            "k_neighbors": None,
            "use_weighted_pca": True,
            "pca_sigma_mm": 10.0,
        }
        mesh = make_sphere()
        result = run_builder(mesh, **config)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.mean(dots) > 0.8
        assert np.all(np.isfinite(result.quality))