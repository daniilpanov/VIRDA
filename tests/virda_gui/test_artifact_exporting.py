"""Unit tests for :mod:`virda_gui_new.services.artifact_exporting`.

Every export writes a real file through a virda exporter into
``tmp_path``; the written files are read back (trimesh, CSV, JSON) and
the converted numbers are asserted against expected values. All source
values stay world-canonical in the manager: conversion happens on the
exported copy only.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from virda.models.electrode import Electrode, Electrodes
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import STEPS
from virda_gui_new.dto.viewer3d import LabeledPoints
from virda_gui_new.services.artifact_exporting import (
    ExportKind,
    MeshExportKind,
    default_export_path,
    export_artifact,
)
from virda_gui_new.services.artifact_loading import load_fiducial_points
from virda_gui_new.services.coordinates import CoordinateFrame
from virda_gui_new.services.steps_manager import StepsManager

# Mesh whose vertex centroid is (7.5, 7.5, 7.5): the head frame shifts by that.
_MESH = Mesh(
    np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]]),
    np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
)
_ESE_MESH = Mesh(np.eye(3) * 10.0, np.array([[0, 1, 2]]))
_CENTROID = np.array([7.5, 7.5, 7.5])
# World fiducials whose frame comes out axis-aligned: NAS at the origin,
# right = +x, anterior = +y, up = +z.
_WORLD_FIDUCIALS = np.array([[0.0, 100.0, 0.0], [-70.0, 0.0, 0.0], [70.0, 0.0, 0.0]])
_FIDUCIAL_LABELS = ("NASION", "LPA", "RPA")
_MEASUREMENTS = Electrodes([Electrode("Fz", {"NAS": 120.5, "LPA": 131.2})])
_ELECTRODES = Electrodes(
    [
        Electrode(
            "A",
            {"nas": 10.0, "lpa": 20.0},
            ese_coords=np.array([10.0, 0.0, 0.0]),
            scalp_coords=np.array([0.0, 10.0, 0.0]),
            residual_error=1.5,
            flagged=True,
        ),
        Electrode("B", {"nas": 5.0}),
    ]
)


def _make_manager(
    *,
    meshes: bool = True,
    electrodes: Electrodes | None = _ELECTRODES,
    measurements: Electrodes | None = _MEASUREMENTS,
) -> StepsManager:
    """Manager at step 4, optionally with the geometry contexts of 1-3."""
    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    steps.next_step()
    if meshes:
        steps.set_step_context_item(1, "scalp_surface", _MESH)
    steps.next_step()
    if meshes:
        steps.set_step_context_item(2, "mesh", _MESH)
    steps.next_step()
    if meshes:
        steps.set_step_context_item(3, "ese_mesh", _ESE_MESH)
    steps.next_step()
    if electrodes is not None:
        steps.set_step_context_item(4, "electrodes", electrodes)
    if measurements is not None:
        steps.set_step_context_item(4, "measurements", measurements)
    return steps


def _set_fiducials(manager: StepsManager) -> None:
    manager.set_global_item("fiducials", LabeledPoints(_WORLD_FIDUCIALS, _FIDUCIAL_LABELS))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- measurements ---


def test_export_measurements_round_trips_the_model(tmp_path: Path) -> None:
    path = export_artifact(_make_manager(), ExportKind.MEASUREMENTS, tmp_path / "measurements.json")

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "electrodes": [{"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}}]
    }


def test_export_measurements_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No measurements to export"):
        export_artifact(
            _make_manager(measurements=None), ExportKind.MEASUREMENTS, tmp_path / "m.json"
        )


# --- electrodes ---


def test_export_electrodes_world_frame_keeps_coordinates(tmp_path: Path) -> None:
    path = export_artifact(_make_manager(), ExportKind.ELECTRODES, tmp_path / "electrodes.csv")

    rows = _read_csv(path)
    assert [row["electrode_id"] for row in rows] == ["A", "B"]
    assert [float(rows[0][axis]) for axis in ("ese_x", "ese_y", "ese_z")] == [10.0, 0.0, 0.0]
    assert [float(rows[0][axis]) for axis in ("scalp_x", "scalp_y", "scalp_z")] == [
        0.0,
        10.0,
        0.0,
    ]
    # Frame-independent scalars stay, and the full-table columns exist.
    assert float(rows[0]["nas"]) == 10.0
    assert float(rows[0]["lpa"]) == 20.0
    assert float(rows[0]["residual_mm"]) == 1.5
    assert rows[0]["flagged"] == "yes"
    # The unlocalized row keeps empty coordinate cells.
    assert rows[1]["ese_x"] == ""
    assert rows[1]["scalp_x"] == ""
    assert rows[1]["flagged"] == "no"


def test_export_electrodes_head_frame_shifts_by_centroid(tmp_path: Path) -> None:
    path = export_artifact(
        _make_manager(),
        ExportKind.ELECTRODES,
        tmp_path / "electrodes.csv",
        frame=CoordinateFrame.HEAD,
    )

    row = _read_csv(path)[0]
    assert [float(row[axis]) for axis in ("ese_x", "ese_y", "ese_z")] == [2.5, -7.5, -7.5]
    assert [float(row[axis]) for axis in ("scalp_x", "scalp_y", "scalp_z")] == [
        -7.5,
        2.5,
        -7.5,
    ]
    # Distances and residual are frame-independent scalars.
    assert float(row["nas"]) == 10.0
    assert float(row["residual_mm"]) == 1.5


def test_export_electrodes_fiducials_frame_applies_full_transform(tmp_path: Path) -> None:
    manager = _make_manager()
    _set_fiducials(manager)
    path = export_artifact(
        manager,
        ExportKind.ELECTRODES,
        tmp_path / "electrodes.csv",
        frame=CoordinateFrame.FIDUCIALS,
    )

    row = _read_csv(path)[0]
    # NAS-origin frame with world-aligned axes: (10,0,0) -> (10,-100,0).
    assert [float(row[axis]) for axis in ("ese_x", "ese_y", "ese_z")] == [10.0, -100.0, 0.0]
    assert [float(row[axis]) for axis in ("scalp_x", "scalp_y", "scalp_z")] == [
        0.0,
        -90.0,
        0.0,
    ]


def test_export_electrodes_head_frame_needs_a_scalp_mesh(tmp_path: Path) -> None:
    manager = _make_manager(meshes=False)

    with pytest.raises(ValueError, match="head frame needs the scalp mesh centroid"):
        export_artifact(
            manager,
            ExportKind.ELECTRODES,
            tmp_path / "electrodes.csv",
            frame=CoordinateFrame.HEAD,
        )


def test_export_electrodes_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No localized electrodes to export"):
        export_artifact(_make_manager(electrodes=None), ExportKind.ELECTRODES, tmp_path / "e.csv")


def test_export_electrodes_leaves_the_stored_copy_world_canonical(tmp_path: Path) -> None:
    """Conversion happens on the exported copy; the context stays world."""
    manager = _make_manager()

    export_artifact(
        manager,
        ExportKind.ELECTRODES,
        tmp_path / "electrodes.csv",
        frame=CoordinateFrame.HEAD,
    )

    stored = manager.get_step_context_item(4, "electrodes")
    assert isinstance(stored, Electrodes)
    assert stored.items[0].ese_coords.tolist() == [10.0, 0.0, 0.0]


# --- fiducials ---


def test_export_fiducials_world_frame_matches_the_store(tmp_path: Path) -> None:
    manager = _make_manager()
    _set_fiducials(manager)

    path = export_artifact(manager, ExportKind.FIDUCIALS, tmp_path / "fiducials.json")

    points = load_fiducial_points(path)
    assert points.labels == _FIDUCIAL_LABELS
    assert points.points.tolist() == _WORLD_FIDUCIALS.tolist()


def test_export_fiducials_head_frame_subtracts_the_centroid(tmp_path: Path) -> None:
    manager = _make_manager()
    _set_fiducials(manager)

    path = export_artifact(
        manager, ExportKind.FIDUCIALS, tmp_path / "fiducials.json", frame=CoordinateFrame.HEAD
    )

    points = load_fiducial_points(path)
    expected = _WORLD_FIDUCIALS - _CENTROID
    assert points.points.tolist() == expected.tolist()


def test_export_fiducials_in_fiducials_frame_is_rejected(tmp_path: Path) -> None:
    manager = _make_manager()
    _set_fiducials(manager)

    with pytest.raises(ValueError, match="fiducials frame"):
        export_artifact(
            manager,
            ExportKind.FIDUCIALS,
            tmp_path / "fiducials.json",
            frame=CoordinateFrame.FIDUCIALS,
        )


def test_export_fiducials_missing_store_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No fiducials to export"):
        export_artifact(_make_manager(), ExportKind.FIDUCIALS, tmp_path / "f.json")


# --- mesh ---


def test_export_mesh_head_frame_writes_shifted_ply(tmp_path: Path) -> None:
    for mesh_kind in (MeshExportKind.BASE, MeshExportKind.POSTPROCESSED):
        path = export_artifact(
            _make_manager(),
            ExportKind.MESH,
            tmp_path / f"{mesh_kind.value}.ply",
            frame=CoordinateFrame.HEAD,
            mesh_kind=mesh_kind,
        )

        loaded = trimesh.load(path)
        assert np.allclose(loaded.vertices, _MESH.vertices - _CENTROID)
        assert loaded.faces.tolist() == _MESH.faces.tolist()


def test_export_mesh_ese_uses_the_ese_source(tmp_path: Path) -> None:
    path = export_artifact(
        _make_manager(),
        ExportKind.MESH,
        tmp_path / "ese.ply",
        frame=CoordinateFrame.HEAD,
        mesh_kind=MeshExportKind.ESE,
    )

    loaded = trimesh.load(path)
    assert np.allclose(loaded.vertices, np.eye(3) * 10.0 - _CENTROID)


def test_export_mesh_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No ese mesh to export"):
        export_artifact(
            _make_manager(meshes=False),
            ExportKind.MESH,
            tmp_path / "ese.ply",
            mesh_kind=MeshExportKind.ESE,
        )


def test_export_mesh_leaves_the_context_world_canonical(tmp_path: Path) -> None:
    manager = _make_manager()

    export_artifact(
        manager,
        ExportKind.MESH,
        tmp_path / "base.ply",
        frame=CoordinateFrame.HEAD,
    )

    stored = manager.get_step_context_item(2, "mesh")
    assert isinstance(stored, Mesh)
    assert stored.vertices.tolist() == _MESH.vertices.tolist()


# --- default paths ---


def test_default_export_paths_follow_the_canonical_names(tmp_path: Path) -> None:
    assert default_export_path(tmp_path, ExportKind.MEASUREMENTS) == (
        tmp_path / "localization" / "measurements.json"
    )
    assert default_export_path(tmp_path, ExportKind.ELECTRODES) == (
        tmp_path / "localization" / "electrodes.csv"
    )
    assert default_export_path(tmp_path, ExportKind.FIDUCIALS) == (
        tmp_path / "localization" / "fiducials.json"
    )
    assert default_export_path(tmp_path, ExportKind.MESH, MeshExportKind.BASE) == (
        tmp_path / "scalp" / "base_mesh.ply"
    )
    assert default_export_path(tmp_path, ExportKind.MESH, MeshExportKind.POSTPROCESSED) == (
        tmp_path / "scalp" / "final_mesh.ply"
    )
    assert default_export_path(tmp_path, ExportKind.MESH, MeshExportKind.ESE) == (
        tmp_path / "ese" / "mesh.ply"
    )


def test_default_export_path_without_project_is_none() -> None:
    assert default_export_path(None, ExportKind.FIDUCIALS) is None
