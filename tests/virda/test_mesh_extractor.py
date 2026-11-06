from dataclasses import replace

import numpy as np
import pytest

from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda.models.segmentation_mask import SegmentationMask
from virda.ops.extract_mesh import extract_mesh


@pytest.fixture
def sphere_mask() -> SegmentationMask:
    volume_shape = (30, 30, 30)
    center = np.array([15, 15, 15])
    sphere_radius = 10
    grid_indices = np.indices(volume_shape)
    squared_distance = np.sum((grid_indices - center.reshape(-1, 1, 1, 1)) ** 2, axis=0)

    return SegmentationMask(mask=squared_distance <= sphere_radius**2)


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

    voxel_to_world_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    voxel_spacing = (1.0, 1.0, 1.0)
    orientation = ("R", "A", "S")

    return MRIVolume(
        data=image_data,
        affine=voxel_to_world_affine,
        spacing=voxel_spacing,
        orientation=orientation,
    )


class TestMeshExtractor:
    def test_extract_surface_returns_valid_mesh(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        mesh = extract_mesh(sphere_mask, sphere_volume)

        assert isinstance(mesh, ScalpMesh)
        assert mesh.vertices.ndim == 2
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.ndim == 2
        assert mesh.faces.shape[1] == 3
        assert mesh.faces.min() >= 0
        assert mesh.faces.max() < mesh.vertices.shape[0]

    def test_extract_surface_produces_reasonable_vertex_count(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        sphere_volume = replace(sphere_volume, affine=np.eye(4))
        mesh = extract_mesh(sphere_mask, sphere_volume)

        sphere_surface_area_pixels = 4 * np.pi * 10**2
        expected_vertex_range = (
            int(sphere_surface_area_pixels * 0.3),
            int(sphere_surface_area_pixels * 2),
        )

        assert expected_vertex_range[0] < mesh.vertices.shape[0] < expected_vertex_range[1]

    def test_extract_surface_transforms_vertices_with_affine(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        voxel_to_world = np.array(
            [
                [1.5, 0.0, 0.0, 10.0],
                [0.0, 1.5, 0.0, 20.0],
                [0.0, 0.0, 2.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        identity_affine = np.eye(4)

        mesh_world = extract_mesh(sphere_mask, replace(sphere_volume, affine=voxel_to_world))
        mesh_voxel = extract_mesh(sphere_mask, replace(sphere_volume, affine=identity_affine))

        expected_world = mesh_voxel.vertices @ voxel_to_world[:3, :3].T + voxel_to_world[:3, 3]
        np.testing.assert_array_almost_equal(mesh_world.vertices, expected_world)

    def test_extract_surface_raises_on_invalid_mask(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        empty_mask = replace(sphere_mask, mask=np.zeros((10, 10, 10), dtype=bool))
        empty_volume = replace(sphere_volume, affine=np.eye(4))

        with pytest.raises(ValueError):
            extract_mesh(empty_mask, empty_volume)

    def test_explicit_voxel_size_one_matches_native(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        native_mesh = extract_mesh(sphere_mask, sphere_volume)
        one_mm_mesh = extract_mesh(sphere_mask, sphere_volume, voxel_size_mm=1.0)

        assert np.array_equal(one_mm_mesh.vertices, native_mesh.vertices)
        assert np.array_equal(one_mm_mesh.faces, native_mesh.faces)

    def test_larger_voxel_size_reduces_density(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        native_mesh = extract_mesh(sphere_mask, sphere_volume)
        coarse_mesh = extract_mesh(sphere_mask, sphere_volume, voxel_size_mm=3.0)

        assert coarse_mesh.vertices.shape[0] > 0
        assert coarse_mesh.vertices.shape[0] < native_mesh.vertices.shape[0]
        assert coarse_mesh.faces.shape[0] < native_mesh.faces.shape[0]

    def test_voxel_size_affine_transform_still_correct(
        self, sphere_mask: SegmentationMask, sphere_volume: MRIVolume
    ) -> None:
        voxel_to_world = np.array(
            [
                [1.5, 0.0, 0.0, 10.0],
                [0.0, 1.5, 0.0, 20.0],
                [0.0, 0.0, 2.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        mesh_world = extract_mesh(
            sphere_mask, replace(sphere_volume, affine=voxel_to_world), voxel_size_mm=3.0
        )
        mesh_voxel = extract_mesh(
            sphere_mask, replace(sphere_volume, affine=np.eye(4)), voxel_size_mm=3.0
        )

        expected_world = mesh_voxel.vertices @ voxel_to_world[:3, :3].T + voxel_to_world[:3, 3]
        np.testing.assert_array_almost_equal(mesh_world.vertices, expected_world)
