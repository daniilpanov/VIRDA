"""Tests for core.types module."""

import numpy as np
import pytest

from virda.core.types import (
    ESEResult,
    ESEConfigData,
    ElectrodeLocalization,
    Fiducial,
    LocalizationResult,
    MeshData,
    MRIData,
    NormalResult,
    SegmentationResult,
)


class TestMRIData:
    def test_shape(self):
        vol = np.zeros((10, 10, 10))
        affine = np.eye(4)
        voxel_size = np.array([1.0, 1.0, 1.0])
        mri = MRIData(volume=vol, affine=affine, voxel_size=voxel_size)
        assert mri.volume.shape == (10, 10, 10)
        assert mri.affine.shape == (4, 4)

    def test_voxel_to_world(self):
        vol = np.zeros((10, 10, 10))
        affine = np.eye(4)
        affine[0, 0] = 2.0
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        mri = MRIData(
            volume=vol, affine=affine, voxel_size=np.array([2.0, 2.0, 2.0])
        )
        world = mri.affine @ np.array([5, 5, 5, 1])
        np.testing.assert_allclose(world[:3], [10.0, 10.0, 10.0])


class TestMeshData:
    def test_empty_mesh(self):
        mesh = MeshData(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=np.int64),
        )
        assert mesh.num_vertices == 0
        assert mesh.num_faces == 0

    def test_simple_mesh(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = MeshData(vertices=verts, faces=faces)
        assert mesh.num_vertices == 3
        assert mesh.num_faces == 1


class TestESEConfigData:
    def test_defaults(self):
        cfg = ESEConfigData()
        assert cfg.offset_mm == 5.0
        assert cfg.reference_point == "center_of_external_surface"
