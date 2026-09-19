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


def test_measurements_editor_rejects_electrode_without_distances_offscreen(
    tmp_path: Path,
) -> None:
    app = _qt_app()
    editor = MeasurementsEditor()
    try:
        editor.set_fiducial_ids(["NAS"])
        editor.set_measurement_rows([MeasurementRow(electrode_id="E9", measured_distances={})])

        with pytest.raises(ValueError, match="E9.*no measured distances"):
            editor.fiducial_rows()
        editor._table.item(0, 0).setText("")
        assert editor.fiducial_rows() == []
    finally:
        editor.close()
        app.quit()


def test_fiducials_editor_load_invalid_file_returns_false(tmp_path: Path) -> None:
    app = _qt_app()
    editor = FiducialsEditor()
    try:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"fiducials": [{"id": {"nas": [0.0, 0.0]}, "name": None}]}), encoding="utf-8")
        assert editor.load(bad, interactive=False) is False

        not_json = tmp_path / "garbage.json"
        not_json.write_text("{not json", encoding="utf-8")
        assert editor.load(not_json, interactive=False) is False
    finally:
        editor.close()
        app.quit()


def test_measurements_editor_load_invalid_schema_returns_false(tmp_path: Path) -> None:
    app = _qt_app()
    editor = MeasurementsEditor()
    try:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"electrodes": [{"electrode_id": "E0"}]}), encoding="utf-8")
        assert editor.load(bad, interactive=False) is False

        non_dict = tmp_path / "list.json"
        non_dict.write_text("[1, 2]", encoding="utf-8")
        assert editor.load(non_dict, interactive=False) is False
    finally:
        editor.close()
        app.quit()


def test_editors_clear_resets_rows_and_path_offscreen(tmp_path: Path) -> None:
    app = _qt_app()
    fiducials_editor = FiducialsEditor()
    measurements_editor = MeasurementsEditor()
    try:
        fiducials_editor.set_rows(fiducials_to_rows(make_fiducials()))
        fiducials_editor.save_to(tmp_path / "f.json")
        measurements_editor.set_fiducial_ids(["NAS", "LPA", "RPA"])
        measurements_editor.set_measurement_rows(
            [MeasurementRow(electrode_id="E0", measured_distances={"NAS": 1.0})]
        )
        measurements_editor._path = tmp_path / "m.json"

        fiducials_editor.clear()
        measurements_editor.clear()

        assert fiducials_editor.fiducial_ids() == []
        assert fiducials_editor._table.rowCount() == 0
        assert fiducials_editor._path is None
        assert measurements_editor._table.rowCount() == 0
        assert measurements_editor._table.columnCount() == 1
        assert measurements_editor._path is None
        assert measurements_editor._weights == {}
    finally:
        fiducials_editor.close()
        measurements_editor.close()
        app.quit()


def test_measurements_rows_without_distances_validate_on_save(tmp_path: Path) -> None:
    app = _qt_app()
    editor = MeasurementsEditor()
    try:
        editor.set_fiducial_ids(["NAS"])
        editor.set_measurement_rows([MeasurementRow(electrode_id="E0", measured_distances={})])
        editor._table.item(0, 0).setText("E7")

        with pytest.raises(ValueError, match="no measured distances"):
            editor.collected_schema()
        assert editor.save_to(tmp_path / "m.json") is False
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


def test_measurements_editor_emits_rows_changed_offscreen() -> None:
    app = _qt_app()
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QTableWidgetItem

    editor = MeasurementsEditor()
    try:
        spy = QSignalSpy(editor.rowsChanged)

        editor.set_fiducial_ids(["NAS", "LPA"])
        assert spy.count() == 1
        spy.clear()

        editor.add_row()
        assert spy.count() == 1
        spy.clear()

        editor._table.setItem(0, 0, QTableWidgetItem("E0"))
        assert spy.count() == 1
        spy.clear()

        editor.remove_selected()
        assert spy.count() == 1
    finally:
        editor.close()
        app.quit()


def test_measurements_editor_parsed_weights_offscreen() -> None:
    app = _qt_app()
    editor = MeasurementsEditor()
    try:
        editor.set_fiducial_ids(["NAS", "LPA"])
        editor.set_weights({"NAS": 1.5})
        assert editor.parsed_weights() == {"NAS": 1.5}

        editor._weights["LPA"].setText("abc")
        with pytest.raises(ValueError, match="Invalid weight"):
            editor.parsed_weights()
    finally:
        editor.close()
        app.quit()


def test_editors_tab_localize_button_emits_signal_offscreen() -> None:
    app = _qt_app()
    from PySide6.QtTest import QSignalSpy
    from PySide6.QtWidgets import QPushButton

    from virda_gui.state import AppState
    from virda_gui.tabs.editors_tab import EditorsTab

    tab = EditorsTab(AppState())
    try:
        labels = [button.text() for button in tab.findChildren(QPushButton)]
        assert "Localize measurements" in labels

        spy = QSignalSpy(tab.localizeRequested)
        tab._on_localize_clicked()
        assert spy.count() == 1
    finally:
        tab.close()
        app.quit()