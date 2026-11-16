"""Unit tests for the artifact file import into popup table inputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from virda_gui_new.services.artifact_importing import (
    import_fiducial_rows,
    import_measurement_table,
)


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_import_fiducial_rows_matches_fixed_slots(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "NAS",
                    "name": "NASION",
                    "coordinates": [3.0, 94.0, 32.0],
                    "coordinate_system": "world",
                    "definition_method": "imported",
                },
                {
                    "fiducial_id": "LPA",
                    "name": "LPA",
                    "coordinates": [-69.0, 10.0, -25.0],
                    "coordinate_system": "world",
                    "definition_method": "imported",
                },
                {
                    "fiducial_id": "RPA",
                    "name": "RPA",
                    "coordinates": [77.0, 12.0, -30.0],
                    "coordinate_system": "world",
                    "definition_method": "imported",
                },
                {
                    "fiducial_id": "Extra",
                    "name": "Extra",
                    "coordinates": [1.0, 1.0, 1.0],
                    "coordinate_system": "world",
                    "definition_method": "imported",
                },
            ]
        },
    )

    rows = import_fiducial_rows(path, None)

    assert [name for name, _xyz in rows] == ["NASION", "LPA", "RPA"]  # fixed slot order
    assert rows[0][1].tolist() == [3.0, 94.0, 32.0]
    assert rows[2][1].tolist() == [77.0, 12.0, -30.0]


def test_import_fiducial_rows_matches_aliases_case_insensitively(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "nz",
                    "name": "nasion",
                    "coordinates": [0.0, 0.0, 0.0],
                    "coordinate_system": "world",
                    "definition_method": "manual",
                },
                {
                    "fiducial_id": "lpa",
                    "name": "lpa",
                    "coordinates": [1.0, 0.0, 0.0],
                    "coordinate_system": "world",
                    "definition_method": "manual",
                },
            ]
        },
    )

    rows = import_fiducial_rows(path, None)

    assert [name for name, _xyz in rows] == ["nasion", "lpa"]
    assert rows[0][1].tolist() == [0.0, 0.0, 0.0]


def test_import_fiducial_rows_reads_mne_coordsystem(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "coordsystem.json",
        {
            "CoordinateUnits": "mm",
            "FiducialsCoordinates": {
                "NAS": {"Head": [0.0, 0.1, 0.2], "MRI": [1.0, 2.0, 3.0]},
                "LPA": {"Head": [0.0, 0.1, 0.2], "MRI": [4.0, 5.0, 6.0]},
                "RPA": {"Head": [0.0, 0.1, 0.2], "MRI": [7.0, 8.0, 9.0]},
            },
        },
    )

    rows = import_fiducial_rows(path, None)

    assert [name for name, _xyz in rows] == ["NAS", "LPA", "RPA"]
    assert rows[0][1].tolist() == [1.0, 2.0, 3.0]


def test_import_fiducial_rows_rejects_unknown_fiducials(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "Extra",
                    "name": "Extra",
                    "coordinates": [1.0, 1.0, 1.0],
                    "coordinate_system": "world",
                    "definition_method": "manual",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="no NAS/LPA/RPA fiducials"):
        import_fiducial_rows(path, None)


def test_import_fiducial_rows_rejects_invalid_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        import_fiducial_rows(broken, None)
    with pytest.raises(OSError):
        import_fiducial_rows(tmp_path / "missing.json", None)


def test_import_measurement_table_reads_electrode_distances(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "measurements.json",
        {
            "electrodes": [
                {"electrode_id": "EEG 001", "measured_distances": {"NAS": 62.5, "LPA": 111.25}},
                {"electrode_id": "EEG 002", "measured_distances": {"NAS": 53.0}},
            ]
        },
    )

    table = import_measurement_table(path)

    assert table == {
        "EEG 001": {"NAS": 62.5, "LPA": 111.25},
        "EEG 002": {"NAS": 53.0},
    }


def test_import_measurement_table_assigns_missing_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "measurements.json",
        {"electrodes": [{"measured_distances": {"NAS": 1.0}}]},
    )

    table = import_measurement_table(path)

    assert table == {"E001": {"NAS": 1.0}}


def test_import_measurement_table_rejects_invalid_or_empty_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    empty = _write(tmp_path / "empty.json", {"electrodes": []})

    with pytest.raises(ValueError):
        import_measurement_table(broken)
    with pytest.raises(ValueError, match="no electrodes"):
        import_measurement_table(empty)
    with pytest.raises(OSError):
        import_measurement_table(tmp_path / "missing.json")


def test_import_fiducial_rows_values_are_float_arrays(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "NAS",
                    "name": "NASION",
                    "coordinates": [1, 2, 3],
                    "coordinate_system": "world",
                    "definition_method": "manual",
                }
            ]
        },
    )

    rows = import_fiducial_rows(path, None)

    assert rows[0][1].dtype == np.float64


_AFFINE = np.array(
    [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_import_fiducial_rows_converts_voxel_via_affine(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "NAS",
                    "name": "NAS",
                    "coordinates": [1.0, 2.0, 3.0],
                    "coordinate_system": "voxel",
                    "definition_method": "manual",
                }
            ]
        },
    )

    rows = import_fiducial_rows(path, _AFFINE)

    assert rows[0][1].tolist() == [12.0, 24.0, 36.0]


def test_import_fiducial_rows_voxel_without_affine_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "NAS",
                    "name": "NAS",
                    "coordinates": [1.0, 2.0, 3.0],
                    "coordinate_system": "voxel",
                    "definition_method": "manual",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="NIfTI volume"):
        import_fiducial_rows(path, None)


def test_import_fiducial_rows_rejects_unknown_coordinate_system(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "fiducials.json",
        {
            "fiducials": [
                {
                    "fiducial_id": "NAS",
                    "name": "NAS",
                    "coordinates": [1.0, 2.0, 3.0],
                    "coordinate_system": "mri",
                    "definition_method": "manual",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="coordinate_system"):
        import_fiducial_rows(path, _AFFINE)
