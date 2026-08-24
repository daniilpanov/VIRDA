import json
from pathlib import Path

import numpy as np
import pytest

from virda_gui.scene import compute_normal_lines, load_fiducial_points, load_normals, sample_normals


class TestLoadFiducialPoints:
    def test_reads_native_format(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        path.write_text(
            """
            {"fiducials": [
                {"fiducial_id": "NAS", "name": "Nasion",
                 "coordinates": [1.0, 2.0, 3.0],
                 "coordinate_system": "world", "definition_method": "manual"}
            ]}
            """,
            encoding="utf-8",
        )

        points, labels = load_fiducial_points(path)

        np.testing.assert_array_equal(points, [[1.0, 2.0, 3.0]])
        assert labels == ["NAS (Nasion)"]

    def test_reads_mne_coordsystem_json(self, tmp_path: Path) -> None:
        path = tmp_path / "coordsystem.json"
        path.write_text(
            json.dumps(
                {
                    "CoordinateSystem": "RAS",
                    "CoordinateUnits": "m",
                    "FiducialsCoordinates": {
                        "NASION": {"Head": [0.0, 0.0, 0.0], "MRI": [0.01, 0.02, 0.03]}
                    },
                }
            ),
            encoding="utf-8",
        )

        points, labels = load_fiducial_points(path)

        np.testing.assert_allclose(points, [[10.0, 20.0, 30.0]])
        assert labels == ["NAS (NASION)"]


class TestLoadNormals:
    def test_round_trip(self, tmp_path: Path) -> None:
        normals = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        path = tmp_path / "normals.npy"
        np.save(path, normals)
        loaded = load_normals(path)
        np.testing.assert_array_equal(loaded, normals)

    def test_rejects_wrong_shape(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.npy"
        np.save(path, np.ones((5,)))
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            load_normals(path)

    def test_rejects_4_columns(self, tmp_path: Path) -> None:
        path = tmp_path / "bad4.npy"
        np.save(path, np.ones((3, 4)))
        with pytest.raises(ValueError, match="\\(N, 3\\)"):
            load_normals(path)


class TestSampleNormals:
    def test_density_1_returns_all(self) -> None:
        normals = np.ones((10, 3))
        idx, sampled = sample_normals(normals, 1)
        assert len(idx) == 10
        np.testing.assert_array_equal(sampled, normals)

    def test_density_3_returns_every_third(self) -> None:
        normals = np.arange(30).reshape(10, 3).astype(np.float64)
        idx, sampled = sample_normals(normals, 3)
        np.testing.assert_array_equal(idx, [0, 3, 6, 9])
        assert sampled.shape == (4, 3)
        np.testing.assert_array_equal(sampled, normals[[0, 3, 6, 9]])


class TestComputeNormalLines:
    def test_endpoints(self) -> None:
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        nrm = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float64)
        origins, tips = compute_normal_lines(pts, nrm, scale=2.0)
        np.testing.assert_array_equal(origins, pts)
        expected_tips = np.array([[0, 0, 2], [1, 2, 0]], dtype=np.float64)
        np.testing.assert_allclose(tips, expected_tips)
