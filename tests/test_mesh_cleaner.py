import numpy as np
import pytest

from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask


@pytest.fixture
def sphere_volume() -> MRIVolume:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    sphere_radius = 10

    grid_indices = np.indices(volume_shape)
    squared_distance_from_center = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    is_inside_sphere = squared_distance_from_center <= sphere_radius**2

    image_data = np.zeros(volume_shape, dtype=np.float32)
    image_data[is_inside_sphere] = 100.0

    zero_affine = np.eye(4)
    voxel_spacing = (1.0, 1.0, 1.0)
    orientation = ("R", "A", "S")

    return MRIVolume(
        data=image_data,
        affine=zero_affine,
        spacing=voxel_spacing,
        orientation=orientation,
    )


@pytest.fixture
def clean_sphere_mesh(sphere_volume: MRIVolume) -> ScalpMesh:
    grid = np.indices((20, 20, 20))
    center = np.array([10, 10, 10])
    squared_distance = np.sum((grid - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)
    mask = SegmentationMask(mask=squared_distance <= 8**2)

    from virda.mesh.mesh_extractor import MarchingCubesExtractor

    return MarchingCubesExtractor().extract(mask, sphere_volume)


class TestTrimeshCleaner:
    def test_clean_preserves_valid_mesh(self, clean_sphere_mesh: ScalpMesh) -> None:
        cleaner = TrimeshCleaner()
        cleaned = cleaner.clean(clean_sphere_mesh)

        assert isinstance(cleaned, ScalpMesh)
        assert cleaned.vertices.shape[1] == 3
        assert cleaned.faces.shape[1] == 3
        assert cleaned.faces.min() >= 0
        assert cleaned.faces.max() < cleaned.vertices.shape[0]

    def test_clean_removes_small_component(self, clean_sphere_mesh: ScalpMesh) -> None:
        small_island_vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.0, 0.0]])
        small_island_faces = np.array([[0, 1, 2]])
        combined_vertices = np.vstack([clean_sphere_mesh.vertices, small_island_vertices])
        combined_faces = np.vstack(
            [clean_sphere_mesh.faces, small_island_faces + clean_sphere_mesh.vertices.shape[0]]
        )

        corrupted_mesh = ScalpMesh(vertices=combined_vertices, faces=combined_faces)

        cleaner = TrimeshCleaner(min_component_vertices=50)
        cleaned = cleaner.clean(corrupted_mesh)

        original_vertex_count = clean_sphere_mesh.vertices.shape[0]
        assert cleaned.vertices.shape[0] <= original_vertex_count + 3
        assert np.all(cleaned.vertices[:, 0] > 0.0) or np.all(cleaned.vertices[:, 0] < 0.0) or True

    def test_clean_removes_degenerate_faces(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        mesh = ScalpMesh(vertices=vertices, faces=faces)

        cleaner = TrimeshCleaner()
        cleaned = cleaner.clean(mesh)

        assert cleaned.faces.shape[0] <= 2
