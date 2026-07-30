"""Tests for core.electrode_localizer module."""

import numpy as np
import pytest

from virda.core.electrode_localizer import localize_electrodes
from virda.core.fiducial_manager import FiducialManager
from virda.core.types import ESEResult


class TestElectrodeLocalizer:
    def _make_ese_and_fiducials(self):
        rng = np.random.default_rng(42)
        n = 100
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = np.arccos(2 * rng.random(n) - 1)
        r = 30.0
        scalp = np.column_stack([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
        ])
        offset = 5.0
        radial = scalp / np.linalg.norm(scalp, axis=1, keepdims=True)
        ese = scalp + offset * radial
        centroid = scalp.mean(axis=0)

        fiducial_mgr = FiducialManager(
            head_centroid=centroid,
            surface_vertices=scalp,
        )
        fiducial_mgr.add_fiducial("NAS", "Nasion", scalp[0])
        fiducial_mgr.add_fiducial("LPA", "Left", scalp[1])
        fiducial_mgr.add_fiducial("RPA", "Right", scalp[2])

        normal_result_normals = radial
        quality = np.zeros(n)

        ese_result = ESEResult(
            scalp_vertices=scalp,
            ese_vertices=ese,
            normals=normal_result_normals,
            quality=quality,
            head_centroid=centroid,
            num_points=n,
        )
        return ese_result, fiducial_mgr

    def test_localize_electrodes(self):
        ese, fiducial_mgr = self._make_ese_and_fiducials()
        fid_coords = fiducial_mgr.get_coordinates_matrix(["NAS", "LPA", "RPA"])

        target_idx = 50
        target_ese = ese.ese_vertices[target_idx]
        dists = {
            "NAS": float(np.linalg.norm(target_ese - fid_coords[0])),
            "LPA": float(np.linalg.norm(target_ese - fid_coords[1])),
            "RPA": float(np.linalg.norm(target_ese - fid_coords[2])),
        }

        measurements = {"E1": dists}
        result = localize_electrodes(
            ese=ese,
            fiducial_mgr=fiducial_mgr,
            measurements=measurements,
        )

        assert result.num_electrodes == 1
        assert result.electrodes[0].residual_error < 1e-6
        np.testing.assert_allclose(
            result.electrodes[0].ese_coords, target_ese, atol=1e-6
        )
