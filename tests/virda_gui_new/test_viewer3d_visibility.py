"""Unit tests for the :class:`SceneViewer3D` visibility toggles.

Runs under the offscreen Qt platform. Only the mesh/electrode layer flags
are exercised: the mesh checkbox hides the active mesh but keeps points,
the electrodes checkbox hides the ese/scalp points but keeps mesh and
fiducials, and both flags survive a surface switch and a scene rebuild.
"""

import pytest
import pyvista as pv
from PySide6.QtWidgets import QCheckBox, QRadioButton, QSlider

from tests.virda_gui_new.fixtures.viewer3d import points
from virda_gui_new.dto.viewer3d import SceneSources


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


def _slider(viewer) -> QSlider:
    found: list[QSlider] = viewer._panel.findChildren(QSlider)
    assert len(found) == 1, "expected exactly one opacity slider"
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

    viewer.set_scene(SceneSources(scalp=pv.Sphere(), scalp_points=points(4), fiducials=points(1)))

    assert not _visible(viewer._mesh_actors["scalp"])
    assert not _visible(viewer._mesh_points_actors["scalp"])
    assert _visible(viewer._fid_points_actor)


def test_the_mesh_edges_can_be_toggled(viewer) -> None:
    """The layers panel switches the edges of the mesh on and off."""
    edges = _checkbox(viewer, "Show mesh edges")
    assert edges.isChecked() is False

    edges.setChecked(True)

    scalp = viewer._mesh_actors["scalp"].GetProperty()
    ese = viewer._mesh_actors["ese"].GetProperty()
    assert scalp.GetEdgeVisibility() == 1
    assert ese.GetEdgeVisibility() == 1

    edges.setChecked(False)

    assert scalp.GetEdgeVisibility() == 0


def test_the_mesh_is_transparent_by_default(viewer) -> None:
    """The meshes are drawn semi-transparent, so the inside stays visible."""
    scalp = viewer._mesh_actors["scalp"].GetProperty()
    ese = viewer._mesh_actors["ese"].GetProperty()

    assert scalp.GetOpacity() == pytest.approx(0.6)
    assert ese.GetOpacity() == pytest.approx(0.6)


def test_the_opacity_slider_changes_the_meshes(viewer) -> None:
    """The slider makes the meshes more or less transparent."""
    slider = _slider(viewer)
    assert slider.value() == 60

    slider.setValue(20)

    scalp = viewer._mesh_actors["scalp"].GetProperty()
    assert scalp.GetOpacity() == pytest.approx(0.2)


def test_the_opacity_survives_a_scene_rebuild(viewer) -> None:
    """A rebuilt scene keeps the opacity the user chose."""
    _slider(viewer).setValue(25)

    viewer.set_scene(SceneSources(scalp=pv.Sphere(), scalp_points=points(4), fiducials=points(1)))

    scalp = viewer._mesh_actors["scalp"].GetProperty()
    assert scalp.GetOpacity() == pytest.approx(0.25)


def test_the_panel_holds_one_opacity_slider(viewer, qapp) -> None:
    """A rebuilt scene leaves exactly one slider in the panel."""
    viewer.set_scene(SceneSources(scalp=pv.Sphere(), scalp_points=points(4), fiducials=points(1)))
    qapp.processEvents()

    sliders = viewer._panel.findChildren(QSlider)
    assert len(sliders) == 1
    assert sliders[0].value() == 60


def test_the_mesh_edges_survive_a_scene_rebuild(viewer) -> None:
    """A rebuilt scene keeps the edges the user switched on."""
    edges = _checkbox(viewer, "Show mesh edges")
    edges.setChecked(True)

    viewer.set_scene(SceneSources(scalp=pv.Sphere(), scalp_points=points(4), fiducials=points(1)))

    scalp = viewer._mesh_actors["scalp"].GetProperty()
    assert scalp.GetEdgeVisibility() == 1
