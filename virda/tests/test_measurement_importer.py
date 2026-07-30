"""Tests for core.measurement_importer module."""

import tempfile
from pathlib import Path

import pytest

from virda.core.measurement_importer import MeasurementImporter


class TestMeasurementImporter:
    def test_add_and_get(self):
        importer = MeasurementImporter(["NAS", "LPA", "RPA"])
        importer.add_measurement("E1", {"NAS": 45.2, "LPA": 38.7, "RPA": 42.1})
        m = importer.get_measurement("E1")
        assert m["NAS"] == 45.2
        assert len(importer.get_all_measurements()) == 1

    def test_invalid_fiducial(self):
        importer = MeasurementImporter(["NAS", "LPA", "RPA"])
        with pytest.raises(ValueError, match="Unknown fiducial"):
            importer.add_measurement("E1", {"NAS": 10.0, "INVALID": 20.0})

    def test_csv_import_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "measurements.csv"
            importer = MeasurementImporter(["NAS", "LPA", "RPA"])
            importer.add_measurement("E1", {"NAS": 45.2, "LPA": 38.7, "RPA": 42.1})
            importer.add_measurement("E2", {"NAS": 51.3, "LPA": 33.9, "RPA": 39.8})
            importer.export_csv(path)

            importer2 = MeasurementImporter(["NAS", "LPA", "RPA"])
            importer2.import_csv(path)
            assert len(importer2.get_all_measurements()) == 2
            assert importer2.get_measurement("E1")["NAS"] == pytest.approx(45.2)

    def test_json_import_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "measurements.json"
            importer = MeasurementImporter(["NAS", "LPA", "RPA"])
            importer.add_measurement("E1", {"NAS": 45.2, "LPA": 38.7, "RPA": 42.1})
            importer.export_json(path)

            importer2 = MeasurementImporter(["NAS", "LPA", "RPA"])
            importer2.import_json(path)
            assert len(importer2.get_all_measurements()) == 1
