"""Unit tests for :mod:`virda.io.exporters.measurements`.

The exporter must produce documents the measurements importer reads
back into an equal :class:`Electrodes` model (byte-level round trip).
"""

import json
from pathlib import Path

from virda.io.exporters.measurements import export_measurements
from virda.io.importers.measurements import import_measurements
from virda.models.electrode import Electrode, Electrodes


def _electrodes() -> Electrodes:
    return Electrodes(
        items=[
            Electrode("Fz", {"NAS": 120.5, "LPA": 131.2}),
            Electrode("Cz", {"NAS": 95.0}),
        ]
    )


def test_round_trip_returns_equal_model(tmp_path: Path) -> None:
    path = export_measurements(tmp_path / "measurements.json", _electrodes())

    assert import_measurements(path) == _electrodes()


def test_round_trip_is_byte_stable(tmp_path: Path) -> None:
    """Export -> import -> export produces the same bytes."""
    path = export_measurements(tmp_path / "measurements.json", _electrodes())
    imported = import_measurements(path)

    reexported = export_measurements(tmp_path / "again.json", imported)

    assert reexported.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_document_holds_electrode_ids_and_distances(tmp_path: Path) -> None:
    path = export_measurements(tmp_path / "measurements.json", _electrodes())

    document = json.loads(path.read_text(encoding="utf-8"))

    assert document == {
        "electrodes": [
            {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}},
            {"electrode_id": "Cz", "measured_distances": {"NAS": 95.0}},
        ]
    }


def test_no_fiducial_weights_are_written(tmp_path: Path) -> None:
    path = export_measurements(tmp_path / "measurements.json", _electrodes())

    assert "fiducial_weights" not in json.loads(path.read_text(encoding="utf-8"))


def test_empty_electrodes_round_trip(tmp_path: Path) -> None:
    path = export_measurements(tmp_path / "measurements.json", Electrodes(items=[]))

    assert import_measurements(path) == Electrodes(items=[])


def test_missing_ids_are_written_empty(tmp_path: Path) -> None:
    """Id-less electrodes keep an empty id; the importer assigns one."""
    path = export_measurements(
        tmp_path / "measurements.json", Electrodes(items=[Electrode(None, {"NAS": 1.0})])
    )

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["electrodes"][0]["electrode_id"] == ""

    imported = import_measurements(path)
    assert imported.items[0].electrode_id == "E001"


def test_creates_parent_directory(tmp_path: Path) -> None:
    path = export_measurements(tmp_path / "localization" / "measurements.json", _electrodes())

    assert path.is_file()
