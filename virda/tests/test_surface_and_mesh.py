"""Tests for core.surface_extractor and core.mesh_cleaner modules."""

import numpy as np
import pytest

from virda.core.mesh_cleaner import clean_mesh
from virda.core.surface_extractor import extract_surface
from virda.core.types import MeshData


def _sphere_mask(radius: int = 20, size: int = 60) -> np.ndarray:
    mask = np.zeros((size, size, size), dtype=np.int32)
    cx = cy = cz = size // 2
    xx, yy, zz = np.mgrid[:size, :size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    mask[dist <= radius] = 1
    return mask


class TestSurfaceExtractor:
    def test_extract_surface(self):
        mask = _sphere_mask()
        voxel_size = np.array([1.0, 1.0, 1.0])
        mesh = extract_surface(mask, voxel_size)

        assert mesh.num_vertices > 0
        assert mesh.num_faces > 0
        assert mesh.coordinate_system == "MRI_world_mm"

    def test_extract_with_affine(self):
        mask = _sphere_mask()
        voxel_size = np.array([1.0, 1.0, 1.0])
        affine = np.eye(4)
        affine[0, 0] = 2.0
        mesh = extract_surface(mask, voxel_size, affine=affine)

        assert mesh.num_vertices > 0
        center = mesh.vertices.mean(axis=0)
        assert np.all(np.abs(center) < 200)


class TestMeshCleaner:
    def test_clean_mesh(self):
        mask = _sphere_mask()
        mesh = extract_surface(mask, np.array([1.0, 1.0, 1.0]))
        cleaned, stats = clean_mesh(mesh)

        assert cleaned.num_vertices > 0
        assert cleaned.num_faces > 0
        assert "original_vertices" in stats
        assert "final_vertices" in stats
