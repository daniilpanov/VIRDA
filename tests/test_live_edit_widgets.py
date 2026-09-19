"""Unit tests for the live fiducials and measurements editors.

The pure round-trip helpers run without a Qt platform; the offscreen tests
build the real widgets and the :class:`IdeWindow` to catch signal-wiring
regressions that pure helpers cannot see.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings

from tests.helpers.measurements import make_fiducials
from tests.helpers.pipelines import build_context
from virda.io.fiducial_helpers import load_fiducials, save_fiducials
from virda.io.loader.measurements_loader import MeasurementsLoaderFromJson
from virda.models.fiducial import Fiducial, Fiducials
from virda.models.path import MeasurementsPath
from virda_gui.main_window import IdeWindow
from virda_gui.preferences import Preferences
from virda_gui.tabs.editors_tab import (
    COL_X,
    FiducialRow,
    FiducialsEditor,
    MeasurementRow,
    MeasurementsEditor,
    fiducials_to_rows,
    measurements_fiducial_ids,
    measurements_rows_to_schema,
    measurements_schema_to_rows,
    rows_to_fiducials,
)


def _qt_app():
    """Return a shared QApplication on the offscreen platform (or skip)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"

    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------
# Pure fiducials helpers
# ----------------------------------------------------------------------


def test_fiducials_rows_round_trip_through_file(tmp_path: Path) -> None:
    fiducials = make_fiducials()
    target = tmp_path / "fiducials.json"
    save_fiducials(target, rows_to_fiducials(fiducials_to_rows(fiducials)))

    loaded = load_fiducials(target)

    assert loaded.ids == fiducials.ids
    for original in fiducials.items:
        restored = loaded.get(original.fiducial_id)
        assert restored is not None
        assert restored.name == original.name
        assert np.allclose(restored.coordinates, original.coordinates)
        assert restored.coordinate_system == original.coordinate_system
        assert restored.definition_method == original.definition_method
        assert restored.weight == pytest.approx(original.weight)


def test_fiducials_rows_preserve_voxel_and_auto_method() -> None:
    fiducials = Fiducials(
        items=[
            Fiducial(
                fiducial_id="F1",
                name="Fiducial 1",
                coordinates=np.array([1.0, 2.0, 3.0]),
                coordinate_system="voxel",
                definition_method="auto",
                weight=2.5,
            )
        ]
    )

    restored = rows_to_fiducials(fiducials_to_rows(fiducials)).items[0]

    assert restored.coordinate_system == "voxel"
    assert restored.definition_method == "auto"
    assert restored.weight == pytest.approx(2.5)


def test_rows_to_fiducials_rejects_duplicate_ids() -> None:
    rows = [
        FiducialRow("NAS", "a", (0.0, 0.0, 0.0), "world", "manual", 1.0),
        FiducialRow("NAS", "b", (1.0, 0.0, 0.0), "world", "manual", 1.0),
    ]

    with pytest.raises(ValueError, match="must be unique"):
        rows_to_fiducials(rows)


# ----------------------------------------------------------------------
# Pure measurements helpers
# ----------------------------------------------------------------------


def test_measurements_rows_round_trip_through_loader(tmp_path: Path) -> None:
    fiducials = make_fiducials()
    points = np.array([[0.0, 88.0, -10.0], [10.0, 20.0, 30.0]])
    rows = [
        MeasurementRow(
            electrode_id=f"E{index}",
            measured_distances={
                fiducial.fiducial_id: float(np.linalg.norm(point - fiducial.coordinates))
                for fiducial in fiducials.items
            },
        )
        for index, point in enumerate(points)
    ]
    target = tmp_path / "measurements.json"
    target.write_text(json.dumps(measurements_rows_to_schema(rows)), encoding="utf-8")

    loaded = MeasurementsLoaderFromJson().run(
        build_context(measurements_path=MeasurementsPath(target))
    )

    assert [electrode.electrode_id for electrode in loaded.items] == ["E0", "E1"]
    assert loaded.items[0].measured_distances == rows[0].measured_distances


def test_measurements_rows_empty_ids_generate_electrode_ids(tmp_path: Path) -> None:
    schema = measurements_rows_to_schema(
        [
            MeasurementRow(electrode_id="", measured_distances={"NAS": 1.0}),
            MeasurementRow(electrode_id="Fz", measured_distances={"NAS": 2.0}),
            MeasurementRow(electrode_id="", measured_distances={"NAS": 3.0}),
        ]
    )
    target = tmp_path / "measurements.json"
    target.write_text(json.dumps(schema), encoding="utf-8")

    loaded = MeasurementsLoaderFromJson().run(
        build_context(measurements_path=MeasurementsPath(target))
    )

    assert [electrode.electrode_id for electrode in loaded.items] == ["E001", "Fz", "E003"]


def test_measurements_rows_apply_weights_through_loader(tmp_path: Path) -> None:
    fiducials = make_fiducials()
    schema = measurements_rows_to_schema(
        [MeasurementRow(electrode_id="Fz", measured_distances={"NAS": 1.0, "LPA": 2.0})]
    )
    schema["fiducial_weights"] = {"NAS": 3.0}
    target = tmp_path / "measurements.json"
    target.write_text(json.dumps(schema), encoding="utf-8")

    context = build_context(fiducials=fiducials, measurements_path=MeasurementsPath(target))
    MeasurementsLoaderFromJson().run(context)

    nas = context.get_store_notnull(Fiducials).get("NAS")
    lpa = context.get_store_notnull(Fiducials).get("LPA")
    assert nas is not None
    assert lpa is not None
    assert nas.weight == pytest.approx(3.0)
    assert lpa.weight == pytest.approx(1.0)


def test_measurements_schema_to_rows_and_back() -> None:
    data = {
        "electrodes": [
            {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}},
            {"electrode_id": "", "measured_distances": {"NAS": 140.0}},
        ],
        "fiducial_weights": {"NAS": 2.0},
    }

    rows = measurements_schema_to_rows(data)
    schema = measurements_rows_to_schema(rows)

    assert rows == [
        MeasurementRow(electrode_id="Fz", measured_distances={"NAS": 120.5, "LPA": 131.2}),
        MeasurementRow(electrode_id="", measured_distances={"NAS": 140.0}),
    ]
    assert schema == {
        "electrodes": [
            {"electrode_id": "Fz", "measured_distances": {"NAS": 120.5, "LPA": 131.2}},
            {"electrode_id": "", "measured_distances": {"NAS": 140.0}},
        ]
    }


def test_measurements_fiducial_ids_unions_distances_and_weights() -> None:
    rows = [
        MeasurementRow(electrode_id="E0", measured_distances={"NAS": 1.0}),
        MeasurementRow(electrode_id="E1", measured_distances={"RPA": 2.0}),
    ]

    assert measurements_fiducial_ids(rows, {"LPA": 1.5}) == ["NAS", "RPA", "LPA"]
    assert measurements_fiducial_ids(rows) == ["NAS", "RPA"]


# ----------------------------------------------------------------------
# Offscreen widget tests
# ----------------------------------------------------------------------


def test_fiducials_editor_saves_loaded_rows_offscreen(tmp_path: Path) -> None:
    app = _qt_app()
    fiducials = make_fiducials()
    editor = FiducialsEditor()
    try:
        editor.set_rows(fiducials_to_rows(fiducials))
        assert editor.fiducial_ids() == fiducials.ids

        target = tmp_path / "out.json"
        assert editor.save_to(target) is True
        loaded = load_fiducials(target)
        assert loaded.ids == fiducials.ids

        item = editor._table.item(0, COL_X)
        assert item is not None
        item.setText("abc")
        with pytest.raises(ValueError, match="X must be a number"):
            editor.fiducial_rows()
    finally:
        editor.close()
        app.quit()


def test_measurements_editor_saves_round_trip_offscreen(tmp_path: Path) -> None:
    app = _qt_app()
    editor = MeasurementsEditor()
    try:
        editor.set_fiducial_ids(["NAS", "LPA", "RPA"])
        editor.set_measurement_rows(
            [
                MeasurementRow(
                    electrode_id="E0",
                    measured_distances={"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
                )
            ],
            weights={"NAS": 1.5},
        )

        schema = editor.collected_schema()
        assert schema["fiducial_weights"] == {"NAS": 1.5}

        target = tmp_path / "out.json"
        assert editor.save_to(target) is True
        context = build_context(
            fiducials=make_fiducials(), measurements_path=MeasurementsPath(target)
        )
        loaded = MeasurementsLoaderFromJson().run(context)
        assert loaded.items[0].measured_distances == {"NAS": 1.0, "LPA": 2.0, "RPA": 3.0}
        nas = context.get_store_notnull(Fiducials).get("NAS")
        assert nas is not None
        assert nas.weight == pytest.approx(1.5)
    finally:
        editor.close()
        app.quit()


def test_ide_window_opens_live_editing_tab_offscreen(tmp_path: Path) -> None:
    app = _qt_app()
    prefs = Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "recorded-project"
        inputs = project / "input"
        inputs.mkdir(parents=True)
        save_fiducials(inputs / "fiducials.json", make_fiducials())
        (inputs / "measurements.json").write_text(
            json.dumps(
                measurements_rows_to_schema(
                    [
                        MeasurementRow(
                            electrode_id="E0",
                            measured_distances={"NAS": 120.0, "LPA": 131.0, "RPA": 134.0},
                        ),
                        MeasurementRow(
                            electrode_id="E1",
                            measured_distances={"NAS": 121.0, "LPA": 130.0, "RPA": 135.0},
                        ),
                    ]
                )
            ),
            encoding="utf-8",
        )

        window.open_project(project)
        window._show_editors_tab()
        index = window._tabs.indexOf(window._editors_tab)
        assert index >= 0
        assert window._tabs.tabText(index) == "Live Editing"
        assert window._editors_tab._fiducials.fiducial_ids() == ["NAS", "LPA", "RPA"]
        assert window._editors_tab._measurements._table.rowCount() == 2

        window.close_project()
        assert window.project() is None
    finally:
        window._on_close()
        app.quit()