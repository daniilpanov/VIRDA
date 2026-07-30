"""Tests for core.fiducial_manager module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from virda.core.fiducial_manager import FiducialManager


class TestFiducialManager:
    def _make_manager(self):
        centroid = np.array([40.0, 40.0, 40.0])
        verts = np.random.default_rng(42).uniform(20, 60, (100, 3))
        return FiducialManager(head_centroid=centroid, surface_vertices=verts)

    def test_add_fiducial(self):
        mgr = self._make_manager()
        fid = mgr.add_fiducial("NAS", "Nasion", np.array([60.0, 40.0, 40.0]))
        assert fid.fiducial_id == "NAS"
        assert fid.name == "Nasion"

    def test_remove_fiducial(self):
        mgr = self._make_manager()
        mgr.add_fiducial("NAS", "Nasion", np.array([60.0, 40.0, 40.0]))
        mgr.remove_fiducial("NAS")
        assert len(mgr.get_all_fiducials()) == 0

    def test_remove_nonexistent(self):
        mgr = self._make_manager()
        with pytest.raises(KeyError):
            mgr.remove_fiducial("INVALID")

    def test_get_coordinates_matrix(self):
        mgr = self._make_manager()
        mgr.add_fiducial("NAS", "Nasion", np.array([60.0, 40.0, 40.0]))
        mgr.add_fiducial("LPA", "Left", np.array([40.0, 60.0, 40.0]))
        coords = mgr.get_coordinates_matrix(["NAS", "LPA"])
        assert coords.shape == (2, 3)

    def test_validate_too_few(self):
        mgr = self._make_manager()
        mgr.add_fiducial("NAS", "Nasion", np.array([60.0, 40.0, 40.0]))
        msgs = mgr.validate()
        assert any("at least 3" in m for m in msgs)

    def test_save_load(self):
        mgr = self._make_manager()
        mgr.add_fiducial("NAS", "Nasion", np.array([60.0, 40.0, 40.0]))
        mgr.add_fiducial("LPA", "Left", np.array([40.0, 60.0, 40.0]))
        mgr.add_fiducial("RPA", "Right", np.array([40.0, 20.0, 40.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fiducials.json"
            mgr.save(path)

            mgr2 = self._make_manager()
            mgr2.load(path)
            assert len(mgr2.get_all_fiducials()) == 3
            np.testing.assert_allclose(
                mgr2.get_fiducial("NAS").coordinates,
                [60.0, 40.0, 40.0],
            )
