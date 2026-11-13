"""Unit tests for the :class:`SceneViewer3D` visibility toggles.

Runs under the offscreen Qt platform. Only the mesh/electrode layer flags
are exercised: the mesh checkbox hides the active mesh but keeps points,
the electrodes checkbox hides the ese/scalp points but keeps mesh and
fiducials, and both flags survive a surface switch and a scene rebuild.
"""

import pyvista as pv
from PySide6.QtWidgets import QCheckBox, QRadioButton

from tests.virda_gui_new.fixtures.viewer3d import points


def _checkbox(viewer, text: str) -> QCheckBox:
    found: list[QCheckBox] = [c for c in viewer._panel.findChildren(QCheckBox) if c.text() == text]
    assert len(found) == 1, f"expected exactly one {text!r} checkbox"
    return found[0]


def _radio(viewer, text: str) -> QRadioButton:
    found: list[QRadioButton] = [
        r for r in viewer._panel.findChildren(QRadioButton) if r.text() == text
    ]
    assert len(found) == 1, f"expected exactly one {text!r} radio"
    return found[0]


def _visible(actor) -> bool:
    return bool(actor.GetVisibility())


def test_show_mesh_unchecked_hides_active_mesh_keeps_points(viewer) -> None:
    mesh_check = _checkbox(viewer, "Show mesh")
    assert mesh_check.isChecked()

    mesh_check.setChecked(False)

    assert viewer._mode == "scalp"
    assert not _visible(viewer._mesh_actors["scalp"])
    assert _visible(viewer._mesh_points_actors["scalp"])
    assert _visible(viewer._fid_points_actor)

    mesh_check.setChecked(True)
    assert _visible(viewer._mesh_actors["scalp"])


def test_show_electrodes_unchecked_hides_points_keeps_mesh(viewer) -> None:
    electrodes_check = _checkbox(viewer, "Show electrodes")
    assert electrodes_check.isChecked()
    # The reworked checkbox replaced "Show points" one-to-one.
    texts = [c.text() for c in viewer._panel.findChildren(QCheckBox)]
    assert "Show points" not in texts

    electrodes_check.setChecked(False)

    assert not _visible(viewer._mesh_points_actors["scalp"])
    assert not _visible(viewer._mesh_label_actors["scalp"])
    assert _visible(viewer._mesh_actors["scalp"])
    assert _visible(viewer._fid_points_actor)

    electrodes_check.setChecked(True)
    assert _visible(viewer._mesh_points_actors["scalp"])


def test_both_flags_off_leaves_only_fiducials(viewer) -> None:
    _checkbox(viewer, "Show mesh").setChecked(False)
    _checkbox(viewer, "Show electrodes").setChecked(False)

    assert not _visible(viewer._mesh_actors["scalp"])
    assert not _visible(viewer._mesh_points_actors["scalp"])
    assert not _visible(viewer._mesh_label_actors["scalp"])
    assert _visible(viewer._fid_points_actor)


def test_flags_survive_surface_switch(viewer) -> None:
    _checkbox(viewer, "Show mesh").setChecked(False)
    _checkbox(viewer, "Show electrodes").setChecked(False)

    _radio(viewer, "ESE").setChecked(True)

    assert viewer._mode == "ese"
    assert not _visible(viewer._mesh_actors["ese"])
    assert not _visible(viewer._mesh_points_actors["ese"])

    _radio(viewer, "Scalp").setChecked(True)

    assert not _visible(viewer._mesh_actors["scalp"])
    assert not _visible(viewer._mesh_points_actors["scalp"])


def test_flags_applied_on_switch_when_enabled(viewer) -> None:
    _radio(viewer, "ESE").setChecked(True)

    assert viewer._mode == "ese"
    assert _visible(viewer._mesh_actors["ese"])
    assert _visible(viewer._mesh_points_actors["ese"])
    assert not _visible(viewer._mesh_actors["scalp"])
    assert not _visible(viewer._mesh_points_actors["scalp"])


def test_flags_survive_scene_rebuild(viewer) -> None:
    _checkbox(viewer, "Show mesh").setChecked(False)
    _checkbox(viewer, "Show electrodes").setChecked(False)

    viewer.set_scene(scalp=pv.Sphere(), scalp_points=points(4), fiducials=points(1))

    assert not _visible(viewer._mesh_actors["scalp"])
    assert not _visible(viewer._mesh_points_actors["scalp"])
    assert _visible(viewer._fid_points_actor)
