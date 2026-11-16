"""Unit tests for the scalp-surface check of the fiducials."""

import numpy as np
import pytest

from virda.models.mesh import Mesh
from virda_gui_new.dto.steps import LOCALIZATION, STEPS
from virda_gui_new.widgets.popup.localization import LocalizationPopup

_RADIUS = 90.0


def _sphere(rings: int = 12, segments: int = 24) -> Mesh:
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


def _popup(qapp, popups, mesh: Mesh) -> LocalizationPopup:
    """A popup over a manager that carries the given mesh."""
    from tests.virda_gui_new.fixtures.steps import advance_to
    from virda_gui_new.services.steps_manager import StepsManager

    steps = StepsManager(tuple(step.required_keys for step in STEPS))
    advance_to(steps, 4)
    steps.set_step_context_item(1, "scalp_surface", mesh)
    steps.set_step_context_item(2, "mesh", mesh)
    steps.set_step_context_item(3, "ese_mesh", mesh)
    steps.set_step_context_item(3, "normals", np.ones((mesh.vertices.shape[0], 3)))
    popup = LocalizationPopup(LOCALIZATION, steps)
    popups.append(popup)
    return popup


class TestSurfaceCheck:
    def test_a_fiducial_off_the_surface_is_flagged(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)

        # The centre of the sphere is far from its surface.
        popup._fiducials_table.set_rows([("NAS", np.zeros(3))])

        status = popup._fiducials_table._table.item(0, 0)
        assert status is not None
        assert status.text() == "●"
        assert "mm from the scalp surface" in status.toolTip()

    def test_a_fiducial_on_the_surface_is_not_flagged(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)

        popup._fiducials_table.set_rows([("NAS", np.array([0.0, 0.0, _RADIUS]))])

        status = popup._fiducials_table._table.item(0, 0)
        assert status is not None
        assert status.text() == ""
        assert "mm from the scalp surface" in status.toolTip()

    def test_the_tolerance_decides_what_is_flagged(self, qapp, popups) -> None:
        mesh = _sphere()
        popup = _popup(qapp, popups, mesh)
        popup._fiducials_table.set_rows([("NAS", np.array([0.0, 0.0, _RADIUS + 5.0]))])

        flagged = popup._fiducials_table._table.item(0, 0)
        assert flagged is not None
        assert flagged.text() == "●"

        popup._tolerance_spin.setValue(10.0)

        quiet = popup._fiducials_table._table.item(0, 0)
        assert quiet is not None
        assert quiet.text() == ""

    def test_without_a_mesh_nothing_is_flagged(self, qapp, popups) -> None:
        from tests.virda_gui_new.fixtures.steps import advance_to
        from virda_gui_new.services.steps_manager import StepsManager

        steps = StepsManager(tuple(step.required_keys for step in STEPS))
        advance_to(steps, 4)
        popup = LocalizationPopup(LOCALIZATION, steps)
        popups.append(popup)
        popup._fiducials_table.set_rows([("NAS", np.zeros(3))])

        status = popup._fiducials_table._table.item(0, 0)
        assert status is not None
        assert status.text() == ""
