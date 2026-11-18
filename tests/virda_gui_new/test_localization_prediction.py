"""Unit tests for the predicted measurements of the localization popup."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox, QPushButton

from virda.models.electrode import Electrodes
from virda.models.fiducial import FiducialType
from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import LOCALIZATION, STEPS
from virda_gui_new.services.steps_manager import StepsManager
from virda_gui_new.widgets.popup.localization import LocalizationPopup

_RADIUS = 90.0


def _sphere(rings: int = 20, segments: int = 40) -> Mesh:
    """A triangulated sphere; the vertices lie on the sphere."""
    points = [np.array([0.0, 0.0, _RADIUS])]
    for ring in range(1, rings):
        phi = np.pi * ring / rings
        z = _RADIUS * np.cos(phi)
        circle = _RADIUS * np.sin(phi)
        for segment in range(segments):
            theta = 2 * np.pi * segment / segments
            points.append(np.array([circle * np.cos(theta), circle * np.sin(theta), z]))
    points.append(np.array([0.0, 0.0, -_RADIUS]))
    vertices = np.asarray(points)
    faces: list[tuple[int, int, int]] = []
    for segment in range(segments):
        faces.append((0, 1 + segment, 1 + (segment + 1) % segments))
    for ring in range(rings - 2):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            following = (segment + 1) % segments
            faces.append((first + segment, second + segment, second + following))
            faces.append((first + segment, second + following, first + following))
    last_ring = 1 + (rings - 2) * segments
    south = len(vertices) - 1
    for segment in range(segments):
        faces.append((south, last_ring + (segment + 1) % segments, last_ring + segment))
    return Mesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def _on_sphere(direction: np.ndarray) -> np.ndarray:
    """The point of the sphere along a direction."""
    unit = direction / np.linalg.norm(direction)
    return np.asarray(unit * _RADIUS, dtype=np.float64)


_NAS = _on_sphere(np.array([0.0, 0.89, 0.45]))
_LPA = _on_sphere(np.array([-0.92, -0.15, 0.35]))
_RPA = _on_sphere(np.array([0.92, -0.15, 0.35]))
_ELECTRODE = _on_sphere(np.array([-0.5, 0.25, 0.83]))
_FIDUCIAL_ROWS = [("NAS", _NAS), ("LPA", _LPA), ("RPA", _RPA)]


def _distance(point: np.ndarray, fiducial: np.ndarray) -> float:
    """The distance between two points."""
    return float(np.linalg.norm(point - fiducial))


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def popups():
    """Collect the popups so they are closed after the test."""
    created: list[LocalizationPopup] = []
    yield created
    for popup in created:
        popup.shutdown()


def _make_manager(mesh: Mesh) -> StepsManager:
    """A manager at the localization step with the given ESE mesh."""
    from tests.virda_gui_new.fixtures.steps import advance_to

    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    advance_to(steps, 4)
    steps.set_step_context_item(1, "scalp_surface", mesh)
    steps.set_step_context_item(2, "mesh", mesh)
    steps.set_step_context_item(3, "ese_mesh", mesh)
    steps.set_step_context_item(3, "normals", np.ones((mesh.vertices.shape[0], 3)))
    return steps


def _settle(qapp, popup: LocalizationPopup, timeout_s: float = 5.0) -> None:
    """Pump events until the debounced localization run has finished."""
    deadline = time.monotonic() + timeout_s
    while popup._manager.get_global_item("measurements") is None and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def _wait_for_prediction(qapp, popup: LocalizationPopup, timeout_s: float = 5.0) -> None:
    """Pump events until the prediction is computed and written."""
    deadline = time.monotonic() + timeout_s
    while popup._prediction_busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def _predict(qapp, popup: LocalizationPopup, row: int = 0) -> None:
    """Press the predict button of a row and wait for its result."""
    button = popup._measurements_table._table.cellWidget(row, 4)
    assert isinstance(button, QPushButton)
    button.click()
    _wait_for_prediction(qapp, popup)


def _popup(qapp, popups, mesh: Mesh) -> LocalizationPopup:
    """A popup with the fiducials and two measured distances filled in."""
    popup = LocalizationPopup(LOCALIZATION, _make_manager(mesh))
    popups.append(popup)
    popup._fiducials_table.set_rows(_FIDUCIAL_ROWS)
    popup._measurements_table.set_rows(
        {
            "A": {
                FiducialType.NAS.value: _distance(_ELECTRODE, _NAS),
                FiducialType.LPA.value: _distance(_ELECTRODE, _LPA),
            }
        }
    )
    return popup


class TestPrediction:
    def test_the_button_fills_the_missing_cell(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)

        _predict(qapp, popup)

        table = popup.collect_measurement_table()
        assert "RPA" in table["A"]
        predicted = table["A"]["RPA"]
        assert predicted == pytest.approx(_distance(_ELECTRODE, _RPA), abs=5.0)

    def test_the_predicted_cell_is_italic(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)

        _predict(qapp, popup)

        item = popup._measurements_table._table.item(0, 3)
        assert item is not None
        assert item.font().italic()
        assert "predicted" in item.toolTip()

    def test_the_import_predicts_the_third_measurement(
        self, qapp, popups, monkeypatch, tmp_path: Path
    ) -> None:
        mesh = _sphere()
        source = tmp_path / "measurements.json"
        source.write_text(
            json.dumps(
                {
                    "electrodes": [
                        {
                            "electrode_id": "A",
                            "measured_distances": {
                                FiducialType.NAS.value: _distance(_ELECTRODE, _NAS),
                                FiducialType.LPA.value: _distance(_ELECTRODE, _LPA),
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        popup = _popup(qapp, popups, mesh)
        monkeypatch.setattr(
            "virda_gui_new.widgets.popup.localization.popup.QFileDialog.getOpenFileName",
            lambda *args, **kwargs: (str(source), ""),
        )

        popup._measurements_table.importRequested.emit()
        _wait_for_prediction(qapp, popup)

        table = popup.collect_measurement_table()
        assert table["A"]["RPA"] == pytest.approx(_distance(_ELECTRODE, _RPA), abs=5.0)
        item = popup._measurements_table._table.item(0, 3)
        assert item is not None
        assert item.font().italic()

    def test_the_import_keeps_an_impossible_cell_empty(
        self, qapp, popups, monkeypatch, tmp_path: Path
    ) -> None:
        mesh = _sphere()
        source = tmp_path / "measurements.json"
        source.write_text(
            json.dumps(
                {
                    "electrodes": [
                        {
                            "electrode_id": "A",
                            "measured_distances": {
                                FiducialType.NAS.value: 1.0,
                                FiducialType.LPA.value: 2.0,
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        popup = _popup(qapp, popups, mesh)
        monkeypatch.setattr(
            "virda_gui_new.widgets.popup.localization.popup.QFileDialog.getOpenFileName",
            lambda *args, **kwargs: (str(source), ""),
        )

        popup._measurements_table.importRequested.emit()
        _wait_for_prediction(qapp, popup)

        table = popup.collect_measurement_table()
        assert "RPA" not in table["A"]
        assert popup._measurements_table._table.item(0, 3) is None


class TestStalePredictions:
    def test_the_dialog_appears_when_the_inputs_move(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)
        _predict(qapp, popup)

        popup._fiducials_table.set_rows([("NAS", _NAS + 20.0), ("LPA", _LPA), ("RPA", _RPA)])
        popup._check_stale_predictions()

        boxes = popup.findChildren(QMessageBox)
        assert len(boxes) == 1
        boxes[0].close()

    def test_no_dialog_without_predictions(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)

        popup._check_stale_predictions()

        assert popup.findChildren(QMessageBox) == []

    def test_the_predictions_are_recalculated_on_yes(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)
        _predict(qapp, popup)
        before = popup.collect_measurement_table()["A"]["RPA"]

        # The fiducials move a little: a bigger move makes the measured
        # distances impossible, and there is nothing to predict then.
        popup._fiducials_table.set_rows([("NAS", _NAS + 5.0), ("LPA", _LPA), ("RPA", _RPA)])
        popup._check_stale_predictions()
        box = popup.findChildren(QMessageBox)[0]
        box.button(QMessageBox.StandardButton.Yes).click()
        _wait_for_prediction(qapp, popup)

        after = popup.collect_measurement_table()["A"]["RPA"]
        assert after != before


class TestPredictionState:
    def test_the_button_is_disabled_without_the_ese_mesh(self, qapp, popups) -> None:
        from tests.virda_gui_new.fixtures.steps import advance_to

        steps = StepsManager(tuple(step.required_keys for step in STEPS))
        advance_to(steps, 4)
        steps.set_step_context_item(3, "ese_mesh", None)
        popup = LocalizationPopup(LOCALIZATION, steps)
        popups.append(popup)
        popup._fiducials_table.set_rows(_FIDUCIAL_ROWS)
        popup._measurements_table.set_rows(
            {"A": {FiducialType.NAS.value: 10.0, FiducialType.LPA.value: 11.0}}
        )

        button = popup._measurements_table._table.cellWidget(0, 4)
        assert isinstance(button, QPushButton)
        assert button.isEnabled() is False
        assert "ESE mesh" in button.toolTip()


def test_the_prediction_names_the_missing_fiducials(qapp, popups) -> None:
    """A prediction without the three fiducials says so."""
    popup = _popup(qapp, popups, _sphere())
    popup._fiducials_table.set_rows([])

    popup._measurements_table.predictRequested.emit("A", FiducialType.RPA.value)

    assert popup._hint_label.text() == "The prediction needs the three fiducials."


def test_the_prediction_reports_why_it_failed(qapp, popups, monkeypatch) -> None:
    """The reason of the geometry service reaches the hint."""

    def refuse(*_args, **_kwargs):
        raise ValueError("the measured distances cannot belong to one point")

    monkeypatch.setattr(
        "virda_gui_new.widgets.popup.localization.popup.predict_missing_measurement",
        refuse,
    )
    popup = _popup(qapp, popups, _sphere())

    popup._measurements_table.predictRequested.emit("A", FiducialType.RPA.value)
    _wait_for_prediction(qapp, popup)

    assert popup._hint_label.text() == (
        "Prediction failed: the measured distances cannot belong to one point"
    )


def test_the_inputs_are_blocked_while_a_prediction_runs(qapp, popups, monkeypatch) -> None:
    """The tables cannot change while the prediction is computed."""
    release: list[bool] = []

    def slow(*_args, **_kwargs):
        while not release:
            time.sleep(0.01)
        return None

    monkeypatch.setattr(
        "virda_gui_new.widgets.popup.localization.popup.predict_missing_measurement",
        slow,
    )
    popup = _popup(qapp, popups, _sphere())

    popup._measurements_table.predictRequested.emit("A", FiducialType.RPA.value)
    qapp.processEvents()

    assert popup._prediction_busy is True
    assert popup._measurements_table.isEnabled() is False
    assert popup._fiducials_table.isEnabled() is False

    release.append(True)
    _wait_for_prediction(qapp, popup)

    assert popup._prediction_busy is False
    assert popup._measurements_table.isEnabled() is True


def test_the_store_keeps_the_predicted_measurements(qapp, popups) -> None:
    """The predicted measurement reaches the measurements store."""
    mesh = _sphere()
    popup = _popup(qapp, popups, mesh)
    _predict(qapp, popup)
    _settle(qapp, popup)

    stored = popup._manager.get_global_item("measurements")
    assert isinstance(stored, Electrodes)
    assert stored.items[0].electrode_id == "A"
    assert FiducialType.RPA.value in stored.items[0].measured_distances


def test_the_electrode_model_keeps_the_fiducial_names(qapp, popups) -> None:
    """The canonical slot names reach the electrode model."""
    mesh = _sphere()
    popup = _popup(qapp, popups, mesh)
    _predict(qapp, popup)
    _settle(qapp, popup)

    stored = popup._manager.get_global_item("measurements")
    assert isinstance(stored, Electrodes)
    assert set(stored.items[0].measured_distances) == {"NAS", "LPA", "RPA"}


def test_an_ambiguous_prediction_is_not_written(qapp, popups, monkeypatch) -> None:
    """An ambiguous prediction leaves the cell empty and is reported."""
    from virda.models.measurement import MeasurementPrediction

    monkeypatch.setattr(
        "virda_gui_new.widgets.popup.localization.popup.predict_missing_measurement",
        lambda *args, **kwargs: MeasurementPrediction(
            point=None,
            distance_mm=100.0,
            ambiguous=True,
            residual_mm=0.5,
            vertex_index=None,
            approximate=True,
        ),
    )
    popup = _popup(qapp, popups, _sphere())

    popup._measurements_table.predictRequested.emit("A", FiducialType.RPA.value)
    _wait_for_prediction(qapp, popup)

    assert "RPA" not in popup.collect_measurement_table()["A"]
    boxes = popup.findChildren(QMessageBox)
    assert len(boxes) == 1
    boxes[0].close()


def test_a_changed_measurement_does_not_ask_to_recalculate(qapp, popups) -> None:
    """The stale dialog waits for a change of the fiducials."""
    popup = _popup(qapp, popups, _sphere())
    _predict(qapp, popup)

    popup._measurements_table.set_rows({"A": {"NAS": 10.0, "LPA": 11.0}})
    popup._check_stale_predictions()

    assert popup.findChildren(QMessageBox) == []
