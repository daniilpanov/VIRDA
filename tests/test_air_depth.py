import numpy as np
import pytest
import trimesh

from virda.mesh.air_depth import air_depth_score, internal_face_mask
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.models.scalp_mesh import ScalpMesh


def _box_with_inner_cavity() -> tuple[np.ndarray, np.ndarray]:
    """Solid box surrounded by thick air, with a spherical cavity open to the
    exterior via a narrow deep tunnel (mimics a head with a cavity).

    The thick air layer guarantees a deep-exterior-air seed (air EDT >= 10 mm),
    so the cavity air is reached only through the long tunnel and scores as deep.
    """
    n = 96
    center = np.array([48.0, 48.0, 48.0])
    grid = np.indices((n, n, n)).astype(float)
    radius = np.linalg.norm(grid - center.reshape(-1, 1, 1, 1), axis=0)
    cavity = radius <= 9.0
    tunnel = (
        (np.abs(grid[0] - 48) <= 4)
        & (np.abs(grid[1] - 48) <= 4)
        & (grid[2] >= 48)
        & (grid[2] <= 63)
    )
    mask = np.ones((n, n, n), dtype=bool)
    mask &= ~(cavity | tunnel)
    mask[:32, :, :] = False
    mask[-32:, :, :] = False
    mask[:, :32, :] = False
    mask[:, -32:, :] = False
    mask[:, :, :32] = False
    mask[:, :, -32:] = False
    return mask, np.eye(4)


def _cavity_mesh() -> ScalpMesh:
    from virda.mesh.mesh_extractor import MarchingCubesExtractor

    mask, affine = _box_with_inner_cavity()
    return MarchingCubesExtractor().extract(mask, affine)


def _cavity_shell_faces(mesh: ScalpMesh, center: np.ndarray) -> np.ndarray:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    distance = np.linalg.norm(centroids - center, axis=1)
    return np.asarray((distance >= 7.0) & (distance <= 11.0))


def _outer_shell_faces(mesh: ScalpMesh, center: np.ndarray) -> np.ndarray:
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    distance = np.linalg.norm(centroids - center, axis=1)
    return np.asarray((distance >= 15.0) & (distance <= 17.0))


class TestAirDepthScore:
    def test_score_shape_and_dtype(self) -> None:
        mask, _ = _box_with_inner_cavity()
        score = air_depth_score(mask, wide_mm=10.0)

        assert score.shape == mask.shape
        assert score.dtype == np.float64
        assert np.all(score >= 0.0)

    def test_cavity_air_is_deep(self) -> None:
        mask, _ = _box_with_inner_cavity()
        score = air_depth_score(mask, wide_mm=10.0)

        cavity_voxel = score[48, 48, 42]
        assert cavity_voxel >= 20.0


class TestGeodesicInternalFaces:
    def test_removes_cavity_walls(self) -> None:
        mask, affine = _box_with_inner_cavity()
        mesh = _cavity_mesh()
        center = np.array([48.0, 48.0, 48.0])
        assert _cavity_shell_faces(mesh, center).sum() > 900

        cleaned = TrimeshCleaner().clean(mesh, mask=mask, affine=affine)

        assert _cavity_shell_faces(cleaned, center).sum() < 50
        assert cleaned.vertices.shape[0] > 1000

    def test_preserves_outer_surface(self) -> None:
        mask, affine = _box_with_inner_cavity()
        mesh = _cavity_mesh()
        center = np.array([48.0, 48.0, 48.0])
        outer_before = int(_outer_shell_faces(mesh, center).sum())

        cleaned = TrimeshCleaner().clean(mesh, mask=mask, affine=affine)
        outer_after = int(_outer_shell_faces(cleaned, center).sum())

        assert outer_after > outer_before * 0.9

    def test_keeps_cavity_walls_when_disabled(self) -> None:
        mask, affine = _box_with_inner_cavity()
        mesh = _cavity_mesh()
        center = np.array([48.0, 48.0, 48.0])
        shell_before = int(_cavity_shell_faces(mesh, center).sum())

        cleaned = TrimeshCleaner(remove_internal_faces=False).clean(mesh, mask=mask, affine=affine)

        assert int(_cavity_shell_faces(cleaned, center).sum()) == shell_before

    def test_geodesic_requires_mask_and_affine(self) -> None:
        mesh = _cavity_mesh()
        with pytest.raises(ValueError, match="segmentation mask"):
            TrimeshCleaner().clean(mesh)


class TestInternalFaceMask:
    def test_empty_mesh_returns_empty_mask(self) -> None:
        mask, affine = _box_with_inner_cavity()

        result = internal_face_mask(trimesh.Trimesh(vertices=[], faces=[]), mask, affine)

        assert result.dtype == bool
        assert len(result) == 0
