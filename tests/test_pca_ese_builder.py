import numpy as np
import pytest

from tests.helpers.meshes import make_plane, make_sphere
from tests.helpers.pipelines import build_context
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.ese_mesh import ESEMesh
from virda.models.scalp_mesh import ScalpMesh
from virda.models.stage2_config import Stage2Config

ESE_OFFSET_MM = 1.0


def run_builder(config: Stage2Config, mesh: ScalpMesh) -> ESEMesh:
    builder = PCAESEBuilder(config=config, ese_offset_mm=ESE_OFFSET_MM)
    return builder.run(build_context(mesh=mesh))


class TestPCAESEBuilder:
    def test_flat_plane_normals_along_z(self) -> None:
        config = Stage2Config(k_neighbors=20)
        result = run_builder(config, make_plane())

        assert result.vertices.shape == result.scalp_vertices.shape
        assert np.mean(np.abs(result.normals[:, 2])) > 0.9

    def test_sphere_normals_radial_outward(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()
        result = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.mean(dots) > 0.8
        assert np.median(result.quality) < 0.1

    def test_ese_offset_distance(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()
        result = run_builder(config, mesh)

        offsets = np.linalg.norm(result.vertices - mesh.vertices, axis=1)
        assert np.allclose(offsets, ESE_OFFSET_MM, atol=1e-9)

    def test_quality_in_unit_interval(self) -> None:
        config = Stage2Config(k_neighbors=30)
        result = run_builder(config, make_sphere())

        assert np.all((result.quality >= 0.0) & (result.quality <= 1.0))

    def test_faces_mirror_scalp_mesh(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()
        result = run_builder(config, mesh)

        assert np.array_equal(result.faces, mesh.faces)

    def test_rejects_k_larger_than_vertex_count(self) -> None:
        config = Stage2Config(k_neighbors=500)
        with pytest.raises(ValueError, match="k_neighbors must be less than"):
            run_builder(config, make_sphere())

    def test_flat_plane_normals_along_z_radius(self) -> None:
        config = Stage2Config(neighborhood_radius_mm=5.0, k_neighbors=None, min_neighbors=5)
        result = run_builder(config, make_plane())

        assert np.mean(np.abs(result.normals[:, 2])) > 0.9

    def test_sphere_normals_radial_outward_radius(self) -> None:
        config = Stage2Config(neighborhood_radius_mm=20.0, k_neighbors=None, min_neighbors=5)
        mesh = make_sphere()
        result = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.mean(dots) > 0.8
        assert np.median(result.quality) < 0.1

    def test_radius_and_knn_consistency(self) -> None:
        radius_result = run_builder(
            Stage2Config(neighborhood_radius_mm=20.0, k_neighbors=None), make_sphere()
        )
        knn_result = run_builder(Stage2Config(k_neighbors=30), make_sphere())

        dots = np.sum(radius_result.normals * knn_result.normals, axis=1)
        assert np.mean(dots) > 0.99

    def test_radius_falls_back_to_knn(self) -> None:
        config = Stage2Config(neighborhood_radius_mm=1.0, k_neighbors=None, min_neighbors=5)
        mesh = make_sphere()
        result = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.allclose(np.linalg.norm(result.normals, axis=1), 1.0)
        assert np.all(np.isfinite(result.quality))
        assert np.mean(dots) > 0.8

    def test_weighted_pca_knn_consistent(self) -> None:
        unweighted = run_builder(Stage2Config(k_neighbors=30), make_sphere())
        weighted = run_builder(
            Stage2Config(k_neighbors=30, use_weighted_pca=True, pca_sigma_mm=10.0),
            make_sphere(),
        )

        dots = np.sum(unweighted.normals * weighted.normals, axis=1)
        assert np.mean(dots) > 0.99
        assert np.all(np.isfinite(weighted.quality))

    def test_weighted_pca_radius_mode(self) -> None:
        config = Stage2Config(
            neighborhood_radius_mm=20.0,
            k_neighbors=None,
            use_weighted_pca=True,
            pca_sigma_mm=10.0,
        )
        mesh = make_sphere()
        result = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(result.normals * radial, axis=1)
        assert np.mean(dots) > 0.8
        assert np.all(np.isfinite(result.quality))
