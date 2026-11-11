"""Unit tests for the :class:`SceneViewer3D` visibility toggles.

Runs under the offscreen Qt platform. Only the mesh/electrode layer flags
are exercised: the mesh checkbox hides the active mesh but keeps points,
the electrodes checkbox hides the ese/scalp points but keeps mesh and
fiducials, and both flags survive a surface switch and a scene rebuild.
"""

import os
from typing import cast

import numpy as np
import pytest
import pyvista as pv
from PySide6.QtWidgets import QCheckBox, QRadioButton

from virda_gui_new.dto.viewer3d import LabeledPoints


@pytest.fixture
def qapp():
    """Yield an offscreen ``QApplication``, skipping without a Qt install."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")

    app = QApplication.instance() or QApplication([])
    return app


def _points(n: int) -> LabeledPoints:
    rng = np.random.default_rng(n)
    return LabeledPoints(rng.random((n, 3)), tuple(f"p{i}" for i in range(n)))


@pytest.fixture
def viewer(qapp):
    """A scalp + ESE scene with points on both and two fiducials."""
    from virda_gui_new.widgets.viewer3d import SceneViewer3D

    widget = SceneViewer3D(
        scalp=pv.Sphere(),
        scalp_points=_points(3),
        ese=pv.Sphere(radius=0.4),
        ese_points=_points(2),
        fiducials=_points(2),
    )
    yield widget
    widget.deleteLater()


def _checkbox(viewer, text: str) -> QCheckBox:
    found = [c for c in viewer._panel.findChildren(QCheckBox) if c.text() == text]  # noqa: SLF001
    assert len(found) == 1, f"expected exactly one {text!r} checkbox"
    return cast("QCheckBox", found[0])


def _radio(viewer, text: str) -> QRadioButton:
    found = [r for r in viewer._panel.findChildren(QRadioButton) if r.text() == text]  # noqa: SLF001
    assert len(found) == 1, f"expected exactly one {text!r} radio"
    return cast("QRadioButton", found[0])


def _visible(actor) -> bool:
    return bool(actor.GetVisibility())


def test_show_mesh_unchecked_hides_active_mesh_keeps_points(viewer) -> None:
    mesh_check = _checkbox(viewer, "Show mesh")
    assert mesh_check.isChecked()

    mesh_check.setChecked(False)

    assert viewer._mode == "scalp"  # noqa: SLF001
    assert not _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001
    assert _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001
    assert _visible(viewer._fid_points_actor)  # noqa: SLF001

    mesh_check.setChecked(True)
    assert _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001


def test_show_electrodes_unchecked_hides_points_keeps_mesh(viewer) -> None:
    electrodes_check = _checkbox(viewer, "Show electrodes")
    assert electrodes_check.isChecked()
    # The reworked checkbox replaced "Show points" one-to-one.
    texts = [c.text() for c in viewer._panel.findChildren(QCheckBox)]  # noqa: SLF001
    assert "Show points" not in texts

    electrodes_check.setChecked(False)

    assert not _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001
    assert not _visible(viewer._mesh_label_actors["scalp"])  # noqa: SLF001
    assert _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001
    assert _visible(viewer._fid_points_actor)  # noqa: SLF001

    electrodes_check.setChecked(True)
    assert _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001


def test_both_flags_off_leaves_only_fiducials(viewer) -> None:
    _checkbox(viewer, "Show mesh").setChecked(False)
    _checkbox(viewer, "Show electrodes").setChecked(False)

    assert not _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001
    assert not _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001
    assert not _visible(viewer._mesh_label_actors["scalp"])  # noqa: SLF001
    assert _visible(viewer._fid_points_actor)  # noqa: SLF001


def test_flags_survive_surface_switch(viewer) -> None:
    _checkbox(viewer, "Show mesh").setChecked(False)
    _checkbox(viewer, "Show electrodes").setChecked(False)

    _radio(viewer, "ESE").setChecked(True)

    assert viewer._mode == "ese"  # noqa: SLF001
    assert not _visible(viewer._mesh_actors["ese"])  # noqa: SLF001
    assert not _visible(viewer._mesh_points_actors["ese"])  # noqa: SLF001

    _radio(viewer, "Scalp").setChecked(True)

    assert not _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001
    assert not _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001


def test_flags_applied_on_switch_when_enabled(viewer) -> None:
    _radio(viewer, "ESE").setChecked(True)

    assert viewer._mode == "ese"  # noqa: SLF001
    assert _visible(viewer._mesh_actors["ese"])  # noqa: SLF001
    assert _visible(viewer._mesh_points_actors["ese"])  # noqa: SLF001
    assert not _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001
    assert not _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001


def test_flags_survive_scene_rebuild(viewer) -> None:
    _checkbox(viewer, "Show mesh").setChecked(False)
    _checkbox(viewer, "Show electrodes").setChecked(False)

    viewer.set_scene(scalp=pv.Sphere(), scalp_points=_points(4), fiducials=_points(1))

    assert not _visible(viewer._mesh_actors["scalp"])  # noqa: SLF001
    assert not _visible(viewer._mesh_points_actors["scalp"])  # noqa: SLF001
    assert _visible(viewer._fid_points_actor)  # noqa: SLF001
