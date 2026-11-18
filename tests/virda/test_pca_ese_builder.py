import numpy as np
import pytest

from tests.helpers.meshes import make_plane, make_sphere
from virda.models.mesh import Mesh
from virda.models.stage2_config import Stage2Config
from virda.ops.build_ese import build_ese, estimate_normals

ESE_OFFSET_MM = 1.0


def run_builder(config: Stage2Config, mesh: Mesh) -> tuple[Mesh, np.ndarray]:
    """Build the ESE surface and return it together with its normals."""
    return build_ese(mesh, config=config, ese_offset_mm=ESE_OFFSET_MM)


class TestEstimateNormals:
    def test_three_neighbors_are_accepted(self) -> None:
        mesh = make_sphere()
        normals = estimate_normals(mesh, Stage2Config(k_neighbors=3))

        assert normals.shape == mesh.vertices.shape

    def test_two_neighbors_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="k_neighbors must be >= 3"):
            Stage2Config(k_neighbors=2)

    def test_the_normals_are_the_ones_of_the_ese_build(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()

        _, built = run_builder(config, mesh)
        estimated = estimate_normals(mesh, config)

        assert np.allclose(built, estimated)

    def test_the_normals_point_outward(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()

        normals = estimate_normals(mesh, config)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        assert np.mean(np.sum(normals * radial, axis=1)) > 0.8

    def test_the_radius_mode_gives_the_same_normals(self) -> None:
        mesh = make_sphere()

        estimated = estimate_normals(
            mesh, Stage2Config(k_neighbors=None, neighborhood_radius_mm=5.0)
        )
        knn = estimate_normals(mesh, Stage2Config(k_neighbors=30))

        assert np.allclose(estimated, knn, atol=0.2)

    def test_it_rejects_k_larger_than_vertex_count(self) -> None:
        with pytest.raises(ValueError, match="k_neighbors must be less than"):
            estimate_normals(make_sphere(), Stage2Config(k_neighbors=500))


class TestBuildEse:
    def test_flat_plane_normals_along_z(self) -> None:
        config = Stage2Config(k_neighbors=20)
        mesh = make_plane()
        ese, normals = run_builder(config, mesh)

        assert ese.vertices.shape == mesh.vertices.shape
        assert np.mean(np.abs(normals[:, 2])) > 0.9

    def test_sphere_normals_radial_outward(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()
        ese, normals = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(normals * radial, axis=1)
        assert np.mean(dots) > 0.8

    def test_ese_offset_distance(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()
        ese, _ = run_builder(config, mesh)

        offsets = np.linalg.norm(ese.vertices - mesh.vertices, axis=1)
        assert np.allclose(offsets, ESE_OFFSET_MM, atol=1e-9)

    def test_faces_mirror_scalp_mesh(self) -> None:
        config = Stage2Config(k_neighbors=30)
        mesh = make_sphere()
        ese, _ = run_builder(config, mesh)

        assert np.array_equal(ese.faces, mesh.faces)

    def test_rejects_k_larger_than_vertex_count(self) -> None:
        config = Stage2Config(k_neighbors=500)
        with pytest.raises(ValueError, match="k_neighbors must be less than"):
            run_builder(config, make_sphere())

    def test_flat_plane_normals_along_z_radius(self) -> None:
        config = Stage2Config(neighborhood_radius_mm=5.0, k_neighbors=None, min_neighbors=5)
        _, normals = run_builder(config, make_plane())

        assert np.mean(np.abs(normals[:, 2])) > 0.9

    def test_sphere_normals_radial_outward_radius(self) -> None:
        config = Stage2Config(neighborhood_radius_mm=20.0, k_neighbors=None, min_neighbors=5)
        mesh = make_sphere()
        _, normals = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(normals * radial, axis=1)
        assert np.mean(dots) > 0.8

    def test_radius_and_knn_consistency(self) -> None:
        _, radius_normals = run_builder(
            Stage2Config(neighborhood_radius_mm=20.0, k_neighbors=None), make_sphere()
        )
        _, knn_normals = run_builder(Stage2Config(k_neighbors=30), make_sphere())

        dots = np.sum(radius_normals * knn_normals, axis=1)
        assert np.mean(dots) > 0.99

    def test_radius_falls_back_to_knn(self) -> None:
        config = Stage2Config(neighborhood_radius_mm=1.0, k_neighbors=None, min_neighbors=5)
        mesh = make_sphere()
        _, normals = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(normals * radial, axis=1)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
        assert np.mean(dots) > 0.8

    def test_weighted_pca_knn_consistent(self) -> None:
        _, unweighted_normals = run_builder(Stage2Config(k_neighbors=30), make_sphere())
        _, weighted_normals = run_builder(
            Stage2Config(k_neighbors=30, use_weighted_pca=True, pca_sigma_mm=10.0),
            make_sphere(),
        )

        dots = np.sum(unweighted_normals * weighted_normals, axis=1)
        assert np.mean(dots) > 0.99

    def test_weighted_pca_radius_mode(self) -> None:
        config = Stage2Config(
            neighborhood_radius_mm=20.0,
            k_neighbors=None,
            use_weighted_pca=True,
            pca_sigma_mm=10.0,
        )
        mesh = make_sphere()
        _, normals = run_builder(config, mesh)

        radial = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)
        dots = np.sum(normals * radial, axis=1)
        assert np.mean(dots) > 0.8
