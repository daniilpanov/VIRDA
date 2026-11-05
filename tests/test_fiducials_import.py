import json
from pathlib import Path

import numpy as np
import pytest

from virda.io.importers.fiducials import import_fiducials
from virda.models.fiducial import Fiducials

PROJECT_FORMAT = {
    "fiducials": [
        {
            "fiducial_id": "NAS",
            "name": "Nasion",
            "coordinates": [0.0, 88.0, -10.0],
            "coordinate_system": "world",
            "definition_method": "manual",
            "weight": 2.0,
        },
        {
            "fiducial_id": "LPA",
            "name": "Left pre-auricular",
            "coordinates": [-75.0, -1.0, -14.0],
            "coordinate_system": "world",
            "definition_method": "imported",
        },
    ]
}


class TestImportFiducialsProjectFormat:
    def test_reads_fids(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        path.write_text(json.dumps(PROJECT_FORMAT), encoding="utf-8")

        fiducials = import_fiducials(path)

        assert isinstance(fiducials, Fiducials)
        assert [f.fiducial_id for f in fiducials.items] == ["NAS", "LPA"]
        np.testing.assert_array_almost_equal(fiducials.items[0].coordinates, [0.0, 88.0, -10.0])
        assert fiducials.items[0].definition_method == "manual"

    def test_default_weight_is_one(self, tmp_path: Path) -> None:
        path = tmp_path / "fiducials.json"
        path.write_text(json.dumps(PROJECT_FORMAT), encoding="utf-8")

        fiducials = import_fiducials(path)

        assert fiducials.items[0].weight == 2.0
        assert fiducials.items[1].weight == 1.0


class TestImportFiducialsCoordsystem:
    def test_mne_coordsystem_converted(self, tmp_path: Path) -> None:
        data = {
            "CoordinateSystem": "MRI",
            "CoordinateUnits": "mm",
            "FiducialsCoordinates": {
                "NASION": {"Head": [0.0, 1.0, 0.0], "MRI": [0.0, 88.0, -10.0]},
                "LPA": {"Head": [-1.0, 0.0, 0.0], "MRI": [-75.0, -1.0, -14.0]},
            },
        }
        path = tmp_path / "coordsystem.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        fiducials = import_fiducials(path)

        assert [f.fiducial_id for f in fiducials.items] == ["NAS", "LPA"]
        assert fiducials.items[0].definition_method == "imported"
        np.testing.assert_array_almost_equal(fiducials.items[0].coordinates, [0.0, 88.0, -10.0])

    def test_metre_units_are_scaled(self, tmp_path: Path) -> None:
        data = {
            "CoordinateUnits": "m",
            "FiducialsCoordinates": {
                "NASION": {"Head": [0.0, 0.0, 0.0], "MRI": [0.0, 0.088, -0.01]},
            },
        }
        path = tmp_path / "coordsystem.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        fiducials = import_fiducials(path)

        np.testing.assert_array_almost_equal(fiducials.items[0].coordinates, [0.0, 88.0, -10.0])


class TestImportFiducialsErrors:
    def test_invalid_nested_payload(self, tmp_path: Path) -> None:
        bad = {"fiducials": [{"fiducial_id": "NAS"}]}
        path = tmp_path / "fiducials.json"
        path.write_text(json.dumps(bad), encoding="utf-8")

        with pytest.raises((KeyError, ValueError)):
            import_fiducials(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_fiducials(tmp_path / "missing.json")
