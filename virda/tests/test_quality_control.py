"""Tests for core.quality_control module."""

import numpy as np
import pytest

from virda.core.quality_control import validate_stage1, validate_stage2, validate_stage3
from virda.core.types import (
    ESEResult,
    ElectrodeLocalization,
    LocalizationResult,
    MeshData,
    MRIData,
)


class TestQualityControl:
    def test_stage1_qc_with_affine(self):
        mri = MRIData(
            volume=np.zeros((10, 10, 10)),
            affine=np.eye(4),
            voxel_size=np.array([1.0, 1.0, 1.0]),
        )
        verts = np.random.default_rng(42).uniform(0, 100, (50, 3))
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = MeshData(vertices=verts, faces=faces)

        msgs = validate_stage1(mri=mri, mesh=mesh, ese_offset_mm=5.0)
        errors = [m for m in msgs if "ERROR" in m]
        assert len(errors) == 0

    def test_stage1_qc_empty_mesh(self):
        mri = MRIData(
            volume=np.zeros((10, 10, 10)),
            affine=np.eye(4),
            voxel_size=np.array([1.0, 1.0, 1.0]),
        )
        mesh = MeshData(vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int64))
        msgs = validate_stage1(mri=mri, mesh=mesh)
        assert any("no vertices" in m for m in msgs)

    def test_stage2_qc(self):
        n = 50
        scalp = np.random.default_rng(42).uniform(0, 100, (n, 3))
        normals = np.tile([0.0, 0.0, 1.0], (n, 1))
        quality = np.zeros(n)
        ese = ESEResult(
            scalp_vertices=scalp,
            ese_vertices=scalp + 5.0 * normals,
            normals=normals,
            quality=quality,
            head_centroid=scalp.mean(axis=0),
            num_points=n,
        )
        msgs = validate_stage2(ese)
        errors = [m for m in msgs if "ERROR" in m]
        assert len(errors) == 0

    def test_stage3_qc(self):
        loc = ElectrodeLocalization(
            electrode_id="E1",
            measured_distances={},
            ese_coords=np.array([1.0, 2.0, 3.0]),
            scalp_coords=np.array([1.0, 2.0, 3.0]),
            residual_error=0.5,
            confidence=0.9,
        )
        result = LocalizationResult(electrodes=[loc], num_electrodes=1)
        msgs = validate_stage3(result, max_residual_threshold=5.0)
        errors = [m for m in msgs if "ERROR" in m]
        assert len(errors) == 0

    def test_stage3_qc_high_residual(self):
        loc = ElectrodeLocalization(
            electrode_id="E1",
            measured_distances={},
            ese_coords=np.array([1.0, 2.0, 3.0]),
            scalp_coords=np.array([1.0, 2.0, 3.0]),
            residual_error=10.0,
            confidence=0.1,
        )
        result = LocalizationResult(electrodes=[loc], num_electrodes=1)
        msgs = validate_stage3(result, max_residual_threshold=5.0)
        assert any("exceeds threshold" in m for m in msgs)
