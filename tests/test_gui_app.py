"""Unit tests for the PySide6 application logic that does not need a display.

These tests exercise the pure helper logic of :mod:`virda_gui` by calling
the functions directly, so no ``QApplication`` is created (CI runners have no
display).  The one exception is the offscreen smoke tests, which construct the
real :class:`IdeWindow` to catch signal wiring regressions that the stubs
cannot.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from virda.ops.options import CleanOptions, SealingOptions
from virda_gui.constants import ADVANCED_FIELD_DEFAULTS
from virda_gui.dialogs.advanced_settings import AdvancedSettingsDialog
from virda_gui.dialogs.project_dialog import (
    ProjectStartDialog,
    ask_create_project_folder,
)
from virda_gui.importing import (
    IMPORT_FALLBACK_ROLES,
    ROLE_REGISTRY,
    detect_role,
    import_file,
    import_target,
    validate_import_source,
)
from virda_gui.main_window import IdeWindow
from virda_gui.preferences import Preferences
from virda_gui.viewer.frames import (
    frame_to_frame_matrix,
    frame_to_world_matrix,
    world_to_frame_matrix,
)
from virda_gui.viewer.scene import transform_points


def _make_prefs(tmp_path: Path) -> Preferences:
    """Preferences backed by an isolated INI file so user config stays clean."""
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


def _write_triangle_ply(path: Path) -> Path:
    """Write a minimal valid ASCII PLY triangle."""
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 1\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
        "0 0 0\n"
        "1 0 0\n"
        "0 1 0\n"
        "3 0 1 2\n",
        encoding="utf-8",
    )
    return path


def _write_mini_nifti(path: Path) -> Path:
    """Write a tiny NIfTI volume with a solid central block."""
    import nibabel as nib

    data = np.zeros((16, 16, 16), dtype=np.uint8)
    data[4:12, 4:12, 4:12] = 200
    nib.Nifti1Image(data, np.eye(4)).to_filename(path)
    return path


# ----------------------------------------------------------------------
# import role detection
# ----------------------------------------------------------------------


def test_detect_role_by_extension_and_name() -> None:
    nifti = detect_role("scan.nii")
    assert nifti is not None
    assert nifti.key == "nifti"
    nifti_gz = detect_role("scan.nii.gz")
    assert nifti_gz is not None
    assert nifti_gz.key == "nifti"
    ese_mesh = detect_role("ese_mesh.ply")
    assert ese_mesh is not None
    assert ese_mesh.key == "ese_mesh"
    assert detect_role("final_mesh.ply") is None  # ambiguous between mesh / ese_mesh
    assert detect_role("mesh.ply") is None
    normals = detect_role("normals.npy")
    assert normals is not None
    assert normals.key == "normals"
    assert detect_role("volume_002.npy") is None
    assert detect_role("blob.bin") is None


def test_detect_role_npy_mesh_part_is_not_a_mesh() -> None:
    """A single NPY array (faces/vertices/edges) is never auto-detected as mesh."""
    assert detect_role("scalp_vertices.npy") is None
    assert detect_role("scalp_faces.npy") is None
    assert detect_role("adjacency.npy") is None


def test_detect_role_json_by_structure(tmp_path: Path) -> None:
    electrodes = tmp_path / "electrodes.json"
    electrodes.write_text(
        json.dumps([{"name": "Cz", "ese_coords": [1.0, 2.0, 3.0]}]), encoding="utf-8"
    )
    electrodes_role = detect_role(electrodes)
    assert electrodes_role is not None
    assert electrodes_role.key == "electrodes"

    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"electrodes": []}), encoding="utf-8")
    measurements_role = detect_role(measurements)
    assert measurements_role is not None
    assert measurements_role.key == "measurements"

    fiducials = tmp_path / "fiducials.json"
    fiducials.write_text(json.dumps({"fiducials": []}), encoding="utf-8")
    fiducials_role = detect_role(fiducials)
    assert fiducials_role is not None
    assert fiducials_role.key == "fiducials"

    unknown = tmp_path / "config.json"
    unknown.write_text(json.dumps({"nifti_path": "x"}), encoding="utf-8")
    assert detect_role(unknown) is None


def test_import_fallback_roles_cover_seven_artifact_kinds() -> None:
    assert [role.key for role in IMPORT_FALLBACK_ROLES] == [
        "nifti",
        "mesh",
        "ese_mesh",
        "normals",
        "electrodes",
        "measurements",
        "fiducials",
    ]


# ----------------------------------------------------------------------
# import validation
# ----------------------------------------------------------------------


def test_validate_mesh_source_accepts_ply(tmp_path: Path) -> None:
    mesh = _write_triangle_ply(tmp_path / "final_mesh.ply")
    validate_import_source(next(role for role in ROLE_REGISTRY if role.key == "mesh"), mesh)


def test_validate_mesh_source_rejects_npy_mesh_part(tmp_path: Path) -> None:
    """A mesh part array cannot rebuild a mesh, so it is rejected with an error."""
    source = tmp_path / "faces.npy"
    np.save(source, np.zeros((4, 3), dtype=np.int64))
    role = next(role for role in ROLE_REGISTRY if role.key == "mesh")
    with pytest.raises(ValueError, match="cannot"):
        validate_import_source(role, source)


def test_validate_mesh_source_rejects_bad_ply(tmp_path: Path) -> None:
    source = tmp_path / "broken.ply"
    source.write_bytes(b"not a ply")
    role = next(role for role in ROLE_REGISTRY if role.key == "mesh")
    with pytest.raises(ValueError):
        validate_import_source(role, source)


def test_validate_nifti_source(tmp_path: Path) -> None:
    nifti = _write_mini_nifti(tmp_path / "head.nii.gz")
    role = next(role for role in ROLE_REGISTRY if role.key == "nifti")
    validate_import_source(role, nifti)

    broken = tmp_path / "broken.nii"
    broken.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError):
        validate_import_source(role, broken)


def test_validate_normals_source(tmp_path: Path) -> None:
    role = next(role for role in ROLE_REGISTRY if role.key == "normals")
    good = tmp_path / "normals.npy"
    np.save(good, np.ones((5, 3)))
    validate_import_source(role, good)

    bad_shape = tmp_path / "bad.npy"
    np.save(bad_shape, np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="shape"):
        validate_import_source(role, bad_shape)


def test_validate_electrodes_source(tmp_path: Path) -> None:
    role = next(role for role in ROLE_REGISTRY if role.key == "electrodes")
    good = tmp_path / "electrodes.json"
    good.write_text(json.dumps([{"name": "Cz", "coords": [0.0, 0.0, 0.0]}]), encoding="utf-8")
    validate_import_source(role, good)

    not_a_list = tmp_path / "bad.json"
    not_a_list.write_text(json.dumps({"electrodes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="array"):
        validate_import_source(role, not_a_list)

    missing_coords = tmp_path / "no-coords.json"
    missing_coords.write_text(json.dumps([{"name": "Cz"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="coordinate"):
        validate_import_source(role, missing_coords)


def test_validate_measurements_and_fiducials_sources(tmp_path: Path) -> None:
    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"electrodes": []}), encoding="utf-8")
    validate_import_source(
        next(role for role in ROLE_REGISTRY if role.key == "measurements"), measurements
    )

    fiducials = tmp_path / "fiducials.json"
    fiducials.write_text(json.dumps({"fiducials": []}), encoding="utf-8")
    validate_import_source(
        next(role for role in ROLE_REGISTRY if role.key == "fiducials"), fiducials
    )

    with pytest.raises(ValueError):
        validate_import_source(
            next(role for role in ROLE_REGISTRY if role.key == "fiducials"),
            tmp_path / "empty.json",
        )


# ----------------------------------------------------------------------
# import target / file copying
# ----------------------------------------------------------------------


def test_import_target_resolves_canonical_locations(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    source = tmp_path / "final_mesh.ply"
    assert (
        import_target(next(r for r in ROLE_REGISTRY if r.key == "mesh"), project, source)
        == project / "mesh" / "final_mesh.ply"
    )
    assert (
        import_target(next(r for r in ROLE_REGISTRY if r.key == "ese_mesh"), project, source)
        == project / "ese" / "mesh.ply"
    )
    assert (
        import_target(next(r for r in ROLE_REGISTRY if r.key == "nifti"), project, source)
        == project / "input" / "final_mesh.ply"
    )


def test_import_file_copies_and_respects_overwrite(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    source = _write_triangle_ply(tmp_path / "final_mesh.ply")
    role = next(r for r in ROLE_REGISTRY if r.key == "mesh")

    target = import_file(project, source, role)
    assert target.exists()

    source.write_bytes(b"newer")
    with pytest.raises(FileExistsError):
        import_file(project, source, role)

    target = import_file(project, source, role, overwrite=True)
    assert target.read_bytes() == b"newer"


# ----------------------------------------------------------------------
# coordinate-frame conversion for the fiducials editor
# ----------------------------------------------------------------------

_AFFINE = np.asarray(
    [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 2.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_CRAS = np.asarray([10.0, 20.0, 30.0])


def test_frame_to_frame_matrix_same_frame_is_identity() -> None:
    for frame in ("scanner_ras", "voxel", "cras"):
        matrix = frame_to_frame_matrix(frame, frame, _AFFINE, _CRAS)
        assert np.allclose(matrix, np.eye(4))


def test_frame_to_frame_matrix_scanner_to_voxel() -> None:
    to_voxel = frame_to_frame_matrix("scanner_ras", "voxel", _AFFINE, _CRAS)
    point = np.asarray([[10.0, 20.0, 30.0]])
    assert np.allclose(transform_points(point, to_voxel), [[0.0, 0.0, 0.0]])


def test_frame_to_frame_matrix_voxel_to_scanner_round_trips() -> None:
    to_scanner = frame_to_frame_matrix("voxel", "scanner_ras", _AFFINE, _CRAS)
    point = np.asarray([[1.0, 2.0, 3.0]])
    assert np.allclose(transform_points(point, to_scanner), [[12.0, 24.0, 36.0]])


def test_frame_to_frame_matrix_cras_to_voxel() -> None:
    to_voxel = frame_to_frame_matrix("cras", "voxel", _AFFINE, _CRAS)
    point = np.asarray([[0.0, 0.0, 0.0]])
    assert np.allclose(transform_points(point, to_voxel), [[0.0, 0.0, 0.0]])


def test_frame_to_frame_matrix_matches_manual_composition() -> None:
    composed = frame_to_frame_matrix("cras", "voxel", _AFFINE, _CRAS)
    expected = world_to_frame_matrix("voxel", _AFFINE, _CRAS) @ frame_to_world_matrix(
        "cras", _AFFINE, _CRAS
    )
    assert np.allclose(composed, expected)


def test_frame_to_frame_matrix_requires_loaded_params() -> None:
    with pytest.raises(ValueError):
        frame_to_frame_matrix("scanner_ras", "voxel", None, _CRAS)
    with pytest.raises(ValueError):
        frame_to_frame_matrix("scanner_ras", "cras", _AFFINE, None)


# ----------------------------------------------------------------------
# mesh generation atoms (pure, no Qt)
# ----------------------------------------------------------------------


def test_generate_and_clean_scalp_mesh_from_mini_nifti(tmp_path: Path) -> None:
    from virda_gui.tabs.base_generation_tab import generate_mesh_from_nifti

    mesh = generate_mesh_from_nifti(
        _write_mini_nifti(tmp_path / "mini.nii.gz"),
        SealingOptions(seal_enabled=True, seal_radius=1),
        CleanOptions(min_component_vertices=1, merge_digits=7),
    )

    assert len(mesh.vertices) > 0
    assert mesh.faces.size > 0


# ----------------------------------------------------------------------
# offscreen Qt smoke tests
# ----------------------------------------------------------------------


def _offscreen_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.environ.get("PYVISTA_OFF_SCREEN") is None:
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on local Qt install
        pytest.skip(f"Qt platform unavailable: {exc}")
    return QApplication.instance() or QApplication([])


def test_advanced_defaults_cover_generation_and_localization() -> None:
    assert ADVANCED_FIELD_DEFAULTS["residual_threshold_mm"] == "10.0"
    assert ADVANCED_FIELD_DEFAULTS["calibrate_ese_offset"] == "true"


def test_viewer_widget_importable_from_main_window() -> None:
    """The 3D viewer tab embeds ``ViewerWidget`` from ``virda_gui.viewer.viewer``."""
    import virda_gui.main_window as main_window_module
    from virda_gui.viewer.viewer import ViewerWidget

    assert vars(main_window_module)["ViewerWidget"] is ViewerWidget


def test_hud_panel_drags_between_sides_offscreen() -> None:
    """Dragging the HUD header re-anchors it; a click without drag does not move it."""
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QLabel

    from virda_gui.viewer.hud import HUDContainer, HudPanel

    app = _offscreen_app()
    hud = HUDContainer()
    hud.resize(1000, 600)
    panel = HudPanel("Live editing", hud)
    panel.set_body(QLabel("body", panel))
    hud.add_overlay(panel, Qt.AlignmentFlag.AlignLeft, fixed_width=430)
    hud.show()
    try:
        assert panel.property("hud_side") == "left"

        press_global = panel.mapToGlobal(QPoint(10, 5))
        drop_global = hud.mapToGlobal(QPoint(800, 300))

        def _send(
            kind: QEvent.Type, local: QPoint, global_pos: QPoint, buttons: Qt.MouseButton
        ) -> None:
            QApplication.sendEvent(
                panel,
                QMouseEvent(
                    kind,
                    QPointF(local),
                    global_pos,
                    Qt.MouseButton.LeftButton,
                    buttons,
                    Qt.KeyboardModifier.NoModifier,
                ),
            )

        _send(QEvent.Type.MouseButtonPress, QPoint(10, 5), press_global, Qt.MouseButton.NoButton)
        _send(QEvent.Type.MouseMove, QPoint(400, 150), drop_global, Qt.MouseButton.LeftButton)
        _send(
            QEvent.Type.MouseButtonRelease, QPoint(400, 150), drop_global, Qt.MouseButton.NoButton
        )
        assert panel.property("hud_side") == "right"

        _send(QEvent.Type.MouseButtonPress, QPoint(10, 5), press_global, Qt.MouseButton.NoButton)
        _send(QEvent.Type.MouseButtonRelease, QPoint(10, 5), press_global, Qt.MouseButton.NoButton)
        assert panel.property("hud_side") == "right"
    finally:
        hud.close()
        app.quit()


def test_live_overlay_rebuild_purges_stale_actors_offscreen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuilding the live overlay removes flagged actors and drops dead refs."""
    from virda_gui.viewer.viewer import ViewerWidget

    app = _offscreen_app()
    viewer = ViewerWidget()
    try:

        class _Plotter:
            def __init__(self) -> None:
                self.removed: list[object] = []
                self.created = 0

            def _actor(self) -> object:
                self.created += 1
                return object()

            def add_points(self, *args: object, **kwargs: object) -> object:
                return self._actor()

            def add_point_labels(self, *args: object, **kwargs: object) -> object:
                return self._actor()

            def add_mesh(self, *args: object, **kwargs: object) -> object:
                return self._actor()

            def remove_actor(self, actor: object) -> None:
                self.removed.append(actor)

            def close(self) -> None:
                return None

        plotter = _Plotter()
        monkeypatch.setattr(viewer, "_plotter", plotter)
        viewer.set_live_electrodes(["E1"], np.array([[1.0, 0.0, 0.0]]), np.array([True]))

        viewer._rebuild_live_overlay(np.eye(4))
        first_point_actors = list(viewer._point_actors)
        assert len(first_point_actors) == 3  # lime, flagged red, labels
        assert plotter.removed == []

        viewer._rebuild_live_overlay(np.eye(4))
        assert len(viewer._point_actors) == 3  # no accumulation
        assert len(plotter.removed) == 3  # lime, flagged red, labels
        assert all(actor not in viewer._point_actors for actor in plotter.removed)
    finally:
        viewer.shutdown()
        viewer.close()
        app.quit()


def test_advanced_dialog_exposes_only_gui_keys_offscreen() -> None:
    """The advanced dialog is trimmed to GUI-only mesh/localization knobs."""
    app = _offscreen_app()
    dialog = AdvancedSettingsDialog(None, dict(ADVANCED_FIELD_DEFAULTS))
    try:
        assert set(dialog._fields) == {
            "residual_threshold_mm",
            "calibrate_ese_offset",
        }
    finally:
        dialog.close()
        app.quit()


def test_ide_window_constructs_and_manages_project_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IdeWindow builds, opens/closes a project and closes tabs without a run tab."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        assert window.project() is None
        assert window._sidebar.project is None
        assert window._sidebar._tree.topLevelItemCount() == 0
        assert window._tabs.count() == 0

        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        (project / "note.txt").write_text("x", encoding="utf-8")

        window.open_project(project)
        assert window.project() == project
        assert window._sidebar.project == project
        assert window._sidebar._tree.topLevelItemCount() == 1
        root = window._sidebar._tree.topLevelItem(0)
        assert root.text(0) == "sample-project"
        assert root.childCount() == 2  # mesh group + loose note.txt
        assert window._tabs.count() == 0  # opening a project auto-opens no tab

        from PySide6.QtWidgets import QWidget

        tab = QWidget()
        window._tabs.addTab(tab, "Untitled")
        assert window._tabs.count() == 1
        window._close_tab(0)
        assert window._tabs.count() == 0
        assert window.project() == project

        window.close_project()
        assert window.project() is None
        assert window._sidebar._tree.topLevelItemCount() == 0
    finally:
        window.close()
        app.quit()


def test_ide_window_does_not_auto_restore_offscreen(tmp_path: Path) -> None:
    """A fresh IdeWindow starts empty — auto-restore now lives in the dialog."""
    app = _offscreen_app()
    project = tmp_path / "remembered"
    (project / "note.txt").parent.mkdir(parents=True)
    (project / "note.txt").write_text("x", encoding="utf-8")

    prefs = _make_prefs(tmp_path)
    first = IdeWindow(prefs=prefs)
    try:
        first.open_project(project)
    finally:
        first.close()
    assert project in prefs.recent_projects()

    second = IdeWindow(prefs=prefs)
    try:
        assert second.project() is None
    finally:
        second.close()
    app.quit()


def test_project_start_dialog_picks_path_offscreen(tmp_path: Path) -> None:
    """The startup dialog returns the project the user accepted."""
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    chosen = tmp_path / "published"
    (chosen / "note.txt").parent.mkdir(parents=True)
    (chosen / "note.txt").write_text("x", encoding="utf-8")
    prefs.note_project_opened(chosen)

    dialog = ProjectStartDialog(prefs=prefs)
    try:
        dialog._accept(chosen)
        assert dialog.project() == chosen
    finally:
        dialog.close()
    app.quit()


def test_ask_create_project_folder_warns_if_non_empty(tmp_path: Path, monkeypatch) -> None:
    """Creating a project in a non-empty folder asks before opening it."""
    app = _offscreen_app()
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    target = tmp_path / "wanted"
    (target / "existing.txt").parent.mkdir(parents=True)
    (target / "existing.txt").write_text("y", encoding="utf-8")

    monkeypatch.setattr(QFileDialog, "exec", lambda self: True)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda self: [str(target)])

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert ask_create_project_folder(parent=None) == target

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert ask_create_project_folder(parent=None) is None
    app.quit()


def test_pipeline_bar_tracks_active_tab_offscreen(tmp_path: Path) -> None:
    """Switching tabs highlights the matching pipeline chip; off-pipeline clears it."""
    from PySide6.QtWidgets import QWidget

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        assert window._pipeline_bar.current_key() is None

        window._show_base_tab()
        assert window._pipeline_bar.current_key() == "base"

        window._show_mesh_processing_tab()
        assert window._pipeline_bar.current_key() == "mesh"

        window._show_ese_tab()
        assert window._pipeline_bar.current_key() == "ese"

        window._show_editors_tab()
        assert window._pipeline_bar.current_key() == "points"

        other = QWidget()
        window._tabs.addTab(other, "Other")
        window._tabs.setCurrentWidget(other)
        assert window._pipeline_bar.current_key() is None
    finally:
        window.close()
        app.quit()


class _StubViewer:
    """Stand-in for ``ViewerWidget`` used by the offscreen file-tab tests."""

    def __init__(self, log=None) -> None:
        self.log = log
        self.load_calls: list[dict[str, str]] = []
        self.shut_down = False

    def load(self, **kwargs: str) -> None:
        self.load_calls.append(dict(kwargs))

    def shutdown(self) -> None:
        self.shut_down = True


def test_ide_window_prefills_editors_from_project_offscreen(tmp_path: Path) -> None:
    """Opening a project loads its canonical fiducials/measurements into the tables."""
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "ready-project"
        inputs = project / "input"
        inputs.mkdir(parents=True)
        (inputs / "head.nii.gz").write_bytes(b"\x00")
        (inputs / "fiducials.json").write_text(
            json.dumps(
                {
                    "fiducials": [
                        {
                            "fiducial_id": "nas",
                            "name": "Nasion",
                            "coordinates": [1.0, 2.0, 3.0],
                            "coordinate_system": "world",
                            "definition_method": "manual",
                            "weight": 1.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (inputs / "measurements.json").write_text(
            json.dumps(
                {
                    "electrodes": [{"electrode_id": "Cz", "measured_distances": {"nas": 100.0}}],
                    "fiducial_weights": {"nas": 1.0},
                }
            ),
            encoding="utf-8",
        )

        window.open_project(project)

        rows = window._editors_tab.fiducials.fiducial_rows()
        assert [row.fiducial_id for row in rows] == ["NAS"]  # legacy "nas" is canonicalised
        assert window._editors_tab.measurements.measurement_rows()[0].electrode_id == "Cz"
        assert window._tabs.count() == 0  # no tab is forced open
    finally:
        window._on_close()
        app.quit()


def test_prefill_ignores_final_mesh_as_base_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A final mesh on disk never becomes the base mesh on project open."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "final-only-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")

        window.open_project(project)

        mesh_tab = window._mesh_processing_tab
        assert mesh_tab._base_mesh is None
        assert mesh_tab._base_path is None
        assert mesh_tab._base_label.text() == "No base mesh loaded"
        assert mesh_tab._final_mesh is not None  # final still loads as final
    finally:
        window._on_close()
        app.quit()


def test_load_base_recomputes_preview_offscreen(tmp_path: Path) -> None:
    """Loading a base mesh rebuilds the postprocessed preview, not just base."""
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        mesh_tab = window._mesh_processing_tab
        assert mesh_tab._smoother_combo.currentData() == "taubin"
        assert mesh_tab._iterations_spin.isEnabled()
        project = tmp_path / "preview-project"
        (project / "mesh").mkdir(parents=True)
        base_path = project / "mesh" / "base_mesh.ply"
        _write_triangle_ply(base_path)
        window.open_project(project)

        mesh_tab = window._mesh_processing_tab
        mesh_tab._smoother_combo.setCurrentIndex(1)  # laplacian
        emitted: list[object] = []
        mesh_tab.previewMesh.connect(emitted.append)
        assert mesh_tab.load_base(base_path)

        assert mesh_tab._preview_mesh is not None
        assert emitted and emitted[-1] is mesh_tab._preview_mesh
        assert mesh_tab._preview_label.text().startswith("Preview:")
    finally:
        window._on_close()
        app.quit()


def test_perform_import_copies_into_project_and_refreshes_sidebar(
    tmp_path: Path,
) -> None:
    """Importing a role copies the file and repopulates the sidebar tree."""
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    role = next(role for role in ROLE_REGISTRY if role.key == "mesh")
    try:
        assert window._perform_import(role, tmp_path / "mesh.ply") is None  # no project yet

        project = tmp_path / "sample-project"
        project.mkdir()
        window.open_project(project)
        source = _write_triangle_ply(tmp_path / "final_mesh.ply")

        target = window._perform_import(role, source)

        assert target == project / "mesh" / "final_mesh.ply"
        assert target.exists()
        root = window._sidebar._tree.topLevelItem(0)
        assert root is not None
        mesh_group = None
        for i in range(root.childCount()):
            child = root.child(i)
            if child is not None and child.text(0) == "mesh":
                mesh_group = child
                break
        assert mesh_group is not None
        assert mesh_group.childCount() == 1
    finally:
        window._on_close()
        app.quit()


def test_ide_window_deletes_no_config_or_run_tab(tmp_path: Path) -> None:
    """The window has no pipeline runner, config tab or log wiring."""
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        assert not hasattr(window, "_config_tab")
        assert not hasattr(window, "_pipe_runner")
        assert not hasattr(window, "_poll_timer")
        assert not hasattr(window._state, "log_queue")
        assert not hasattr(window._state, "stage3_summary")
        assert not hasattr(window._state, "coordsystem")
    finally:
        window.close()
        app.quit()


def test_advanced_dialog_updates_state_offscreen(tmp_path: Path) -> None:
    """Accepting the advanced dialog writes its values back into state."""
    app = _offscreen_app()

    class _Dialog:
        def __init__(self, parent, values) -> None:
            self.result_values = {"seal_enabled": "false", "seal_radius": "9"}
            self._exec = True

        def exec(self):
            return True

    original = AdvancedSettingsDialog
    import virda_gui.main_window as main_window_module

    try:
        vars(main_window_module)["AdvancedSettingsDialog"] = _Dialog
        prefs = _make_prefs(tmp_path)
        window = IdeWindow(prefs=prefs)
        try:
            window._on_show_advanced_settings()
            assert window._state.advanced["seal_enabled"] == "false"
            assert window._state.advanced["seal_radius"] == "9"
        finally:
            window.close()
    finally:
        vars(main_window_module)["AdvancedSettingsDialog"] = original
        app.quit()


def test_npy_export_dialogs_default_to_project_dir_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NPY save dialogs open in the current project folder by default."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "base_mesh.ply")
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        window.open_project(project)

        captured: list[str] = []

        def _cancelled(*args: object, **_kwargs: object) -> tuple[str, str]:
            assert len(args) >= 3
            assert isinstance(args[2], str)
            captured.append(args[2])
            return "", ""

        monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getSaveFileName", _cancelled)

        mesh_tab = window._mesh_processing_tab
        assert mesh_tab._active_mesh() is not None
        mesh_tab._on_export_vertices()
        mesh_tab._on_export_faces()
        assert captured == [
            str(project / "scalp_vertices.npy"),
            str(project / "scalp_faces.npy"),
        ]

        import pyvista as pv

        from virda_gui.viewer.viewer import ViewerWidget

        viewer = ViewerWidget()
        try:
            viewer.set_project_dir(project)
            viewer.set_extra_mesh(pv.PolyData(np.array([[0.0, 0.0, 0.0]])), "scalp")
            viewer._on_export_vertices()
        finally:
            viewer.shutdown()
            viewer.close()
        assert captured[-1] == str(project / "scalp_vertices.npy")
    finally:
        window.close()
        app.quit()


def test_ply_export_writes_active_scalp_mesh_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PLY export writes the active scalp mesh instead of failing to unpack it."""
    import numpy as np

    from virda.io.importers.scalp_mesh import import_scalp_mesh

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "ply-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "base_mesh.ply")
        window.open_project(project)

        mesh_tab = window._mesh_processing_tab
        active = mesh_tab._active_mesh()
        assert active is not None

        target = tmp_path / "exported" / "scalp_mesh.ply"
        captured: list[str] = []

        def _save(*args: object, **_kwargs: object) -> tuple[str, str]:
            captured.append(args[2])  # type: ignore[arg-type]
            return str(target), "PLY (*.ply)"

        monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getSaveFileName", _save)
        mesh_tab._on_export_mesh_file()

        assert captured == [str(project / "mesh" / "scalp_mesh.ply")]
        assert target.is_file()
        exported = import_scalp_mesh(target)
        assert np.allclose(exported.vertices, active.vertices)
    finally:
        window.close()
        app.quit()


def test_ese_ply_export_writes_sensor_surface_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ESE tab PLY export writes the in-memory sensor surface."""
    import numpy as np

    from virda.io.importers.scalp_mesh import import_scalp_mesh

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "ese-ply-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "base_mesh.ply")

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ese_dir = project / "ese"
        ese_dir.mkdir()
        np.save(ese_dir / "ese_vertices.npy", np.array(vertices))
        np.save(ese_dir / "ese_faces.npy", np.array([[0, 1, 2]]))
        np.save(ese_dir / "normals.npy", np.array([[0.0, 0.0, 1.0]] * 3))
        np.save(ese_dir / "quality.npy", np.array([1.0, 1.0, 1.0]))
        (ese_dir / "point_pairs.json").write_text(
            json.dumps(
                {
                    "n_points": 3,
                    "scalp_vertices": vertices,
                    "ese_vertices": vertices,
                    "normals": [[0.0, 0.0, 1.0]] * 3,
                    "quality": [1.0, 1.0, 1.0],
                }
            ),
            encoding="utf-8",
        )
        (ese_dir / "mesh.ply").write_text("placeholder", encoding="utf-8")
        window.open_project(project)

        ese_tab = window._ese_tab
        ese = ese_tab.current_ese_mesh()
        assert ese is not None

        target = tmp_path / "exported" / "ese_mesh.ply"
        captured: list[str] = []

        def _save(*args: object, **_kwargs: object) -> tuple[str, str]:
            captured.append(args[2])  # type: ignore[arg-type]
            return str(target), "PLY (*.ply)"

        monkeypatch.setattr("PySide6.QtWidgets.QFileDialog.getSaveFileName", _save)
        ese_tab._on_export_mesh_file()

        assert captured == [str(project / "ese" / "ese_mesh.ply")]
        assert target.is_file()
        exported = import_scalp_mesh(target)
        assert np.allclose(exported.vertices, ese.vertices)
    finally:
        window.close()
        app.quit()


def test_ese_display_checkboxes_toggle_layers_offscreen(tmp_path: Path) -> None:
    """The ESE tab display checkboxes show and hide the preview layers."""
    import numpy as np

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "ese-display-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "base_mesh.ply")

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ese_dir = project / "ese"
        ese_dir.mkdir()
        np.save(ese_dir / "ese_vertices.npy", np.array(vertices))
        np.save(ese_dir / "ese_faces.npy", np.array([[0, 1, 2]]))
        np.save(ese_dir / "normals.npy", np.array([[0.0, 0.0, 1.0]] * 3))
        np.save(ese_dir / "quality.npy", np.array([1.0, 1.0, 1.0]))
        (ese_dir / "point_pairs.json").write_text(
            json.dumps(
                {
                    "n_points": 3,
                    "scalp_vertices": vertices,
                    "ese_vertices": vertices,
                    "normals": [[0.0, 0.0, 1.0]] * 3,
                    "quality": [1.0, 1.0, 1.0],
                }
            ),
            encoding="utf-8",
        )
        (ese_dir / "mesh.ply").write_text("placeholder", encoding="utf-8")
        window.open_project(project)

        ese_tab = window._ese_tab
        assert ese_tab.current_ese_mesh() is not None
        assert ese_tab._final_mesh is not None  # working mesh pushed as the backdrop
        assert ese_tab._final_actor is not None
        assert ese_tab._ese_actor is not None

        ese_tab._show_ese_chk.setChecked(False)
        assert ese_tab._ese_actor is None
        assert ese_tab._final_actor is not None

        ese_tab._show_final_chk.setChecked(False)
        assert ese_tab._final_actor is None

        ese_tab._show_final_chk.setChecked(True)
        ese_tab._show_ese_chk.setChecked(True)
        assert ese_tab._final_actor is not None
        assert ese_tab._ese_actor is not None
    finally:
        window.close()
        app.quit()


def test_ese_continue_reaches_live_editing_offscreen(tmp_path: Path) -> None:
    """The live-editing handoff lives in the ESE tab, not in mesh processing."""
    from PySide6.QtWidgets import QPushButton

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        mesh_buttons = [
            button.text() for button in window._mesh_processing_tab.findChildren(QPushButton)
        ]
        assert not any("Live Editing" in text for text in mesh_buttons)

        window._show_ese_tab()
        continue_btn = next(
            button
            for button in window._ese_tab.findChildren(QPushButton)
            if "Live Editing" in button.text()
        )
        continue_btn.click()
        assert window._tabs.currentWidget() is window._editors_tab
    finally:
        window.close()
        app.quit()


def test_localization_blocked_until_ese_mesh_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Localization stays blocked with an explanation until the ESE mesh exists."""
    from PySide6.QtWidgets import QMessageBox

    from virda_gui.tabs.editors_tab import FiducialRow, MeasurementRow

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        window.open_project(project)

        window._editors_tab.fiducials.set_rows(
            [
                FiducialRow(
                    fiducial_id=fiducial_id,
                    name=fiducial_id,
                    coordinates=(0.0, 0.0, 0.0),
                    coordinate_system="world",
                    definition_method="manual",
                    weight=1.0,
                )
                for fiducial_id in ("NAS", "LPA", "RPA")
            ]
        )
        window._editors_tab.measurements.set_measurement_rows(
            [
                MeasurementRow(
                    electrode_id="E1",
                    measured_distances={"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
                )
            ]
        )
        window._run_localize(interactive=False)
        assert window._localized_electrodes is None
        assert window._localize_thread is None
        assert "ESE mesh" in window._editors_tab.localization._hint.text()
    finally:
        window.close()
        app.quit()


def test_open_project_autoloads_ese_mesh_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a project with ese/mesh.ply restores the in-memory ESE mesh."""
    import json

    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ese_dir = project / "ese"
        ese_dir.mkdir()
        np.save(ese_dir / "ese_vertices.npy", np.array(vertices))
        np.save(ese_dir / "ese_faces.npy", np.array([[0, 1, 2]]))
        np.save(ese_dir / "normals.npy", np.array([[0.0, 0.0, 1.0]] * 3))
        np.save(ese_dir / "quality.npy", np.array([1.0, 1.0, 1.0]))
        (ese_dir / "point_pairs.json").write_text(
            json.dumps(
                {
                    "n_points": 3,
                    "scalp_vertices": vertices,
                    "ese_vertices": vertices,
                    "normals": [[0.0, 0.0, 1.0]] * 3,
                    "quality": [1.0, 1.0, 1.0],
                }
            ),
            encoding="utf-8",
        )
        (ese_dir / "mesh.ply").write_text("placeholder", encoding="utf-8")

        window.open_project(project)
        ese = window._ese_tab.current_ese_mesh()
        assert ese is not None
        assert len(ese.vertices) == 3
    finally:
        window.close()
        app.quit()


def test_viewer_open_restores_ese_overlay_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A freshly loaded viewer scene re-applies the scalp and ESE mesh overlays."""
    import json

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "base_mesh.ply")
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ese_dir = project / "ese"
        ese_dir.mkdir()
        np.save(ese_dir / "ese_vertices.npy", np.array(vertices))
        np.save(ese_dir / "ese_faces.npy", np.array([[0, 1, 2]]))
        np.save(ese_dir / "normals.npy", np.array([[0.0, 0.0, 1.0]] * 3))
        np.save(ese_dir / "quality.npy", np.array([1.0, 1.0, 1.0]))
        (ese_dir / "point_pairs.json").write_text(
            json.dumps({"scalp_vertices": vertices}), encoding="utf-8"
        )
        (ese_dir / "mesh.ply").write_text("placeholder", encoding="utf-8")
        window.open_project(project)
        assert window._ese_tab.current_ese_mesh() is not None

        class _Viewer:
            def __init__(self) -> None:
                self.scene_frame_params = (None, None, True)
                self.extra_meshes: list[str] = []

            def set_live_fiducials(self, *args: object, **kwargs: object) -> None:
                return None

            def shutdown(self) -> None:
                return None

            def set_extra_mesh(self, poly: object, kind: str = "scalp") -> None:
                self.extra_meshes.append(kind)

        viewer = _Viewer()
        monkeypatch.setattr(window, "_viewer_widget", viewer)
        window._on_viewer_scene_loaded(object())
        assert viewer.extra_meshes == ["scalp", "ese"]
    finally:
        window.close()
        app.quit()


def test_ese_signal_schedules_localization_without_viewer_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mesh/ESE signals schedule localization even when the 3D viewer is closed."""
    import json

    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")

        vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ese_dir = project / "ese"
        ese_dir.mkdir()
        np.save(ese_dir / "ese_vertices.npy", np.array(vertices))
        np.save(ese_dir / "ese_faces.npy", np.array([[0, 1, 2]]))
        np.save(ese_dir / "normals.npy", np.array([[0.0, 0.0, 1.0]] * 3))
        np.save(ese_dir / "quality.npy", np.array([1.0, 1.0, 1.0]))
        (ese_dir / "point_pairs.json").write_text(
            json.dumps({"scalp_vertices": vertices}), encoding="utf-8"
        )
        (ese_dir / "mesh.ply").write_text("placeholder", encoding="utf-8")

        inputs = project / "input"
        inputs.mkdir()
        (inputs / "fiducials.json").write_text(
            json.dumps(
                {
                    "fiducials": [
                        {
                            "fiducial_id": fiducial_id,
                            "name": fiducial_id,
                            "coordinates": [1.0, 2.0, 3.0],
                            "coordinate_system": "world",
                            "definition_method": "manual",
                            "weight": 1.0,
                        }
                        for fiducial_id in ("NAS", "LPA", "RPA")
                    ]
                }
            ),
            encoding="utf-8",
        )
        (inputs / "measurements.json").write_text(
            json.dumps(
                {
                    "electrodes": [
                        {
                            "electrode_id": "E1",
                            "measured_distances": {"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        window.open_project(project)
        assert window._viewer_widget is None
        ese = window._ese_tab.current_ese_mesh()
        assert ese is not None

        window._localize_timer.stop()
        window._on_ese_mesh(ese)
        assert window._localize_timer.isActive()
    finally:
        window.close()
        app.quit()


def test_blocked_localization_drops_cached_result_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the ESE mesh clears the cached result so stale rows cannot resurface."""
    from PySide6.QtWidgets import QMessageBox

    from virda.models.electrode import Electrode, Electrodes
    from virda_gui.tabs.editors_tab import FiducialRow, MeasurementRow

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        window.open_project(project)

        window._editors_tab.fiducials.set_rows(
            [
                FiducialRow(
                    fiducial_id=fiducial_id,
                    name=fiducial_id,
                    coordinates=(0.0, 0.0, 0.0),
                    coordinate_system="world",
                    definition_method="manual",
                    weight=1.0,
                )
                for fiducial_id in ("NAS", "LPA", "RPA")
            ]
        )
        window._editors_tab.measurements.set_measurement_rows(
            [
                MeasurementRow(
                    electrode_id="E1",
                    measured_distances={"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
                )
            ]
        )
        window._localized_electrodes = Electrodes(
            items=[
                Electrode(
                    electrode_id="E1",
                    measured_distances={"NAS": 1.0},
                    ese_coords=np.array([1.0, 0.0, 0.0]),
                    scalp_coords=np.array([1.0, 0.0, 0.0]),
                )
            ]
        )
        window._ese_tab._ese_mesh = None  # as after clearing the ESE tab
        window._schedule_localization()
        assert window._localized_electrodes is None
        assert "ESE mesh" in window._editors_tab.localization._hint.text()
    finally:
        window.close()
        app.quit()


def test_tables_autosave_to_project_on_edit_offscreen(tmp_path: Path) -> None:
    """Table edits are written back to input/*.json without a Save click."""
    import json

    from virda_gui.tabs.editors_tab import FiducialRow, MeasurementRow

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "autosave-project"
        (project / "input").mkdir(parents=True)
        window.open_project(project)

        tab = window._editors_tab
        tab.fiducials.set_rows(
            [
                FiducialRow(
                    fiducial_id=fiducial_id,
                    name=fiducial_id,
                    coordinates=(1.0, 2.0, 3.0),
                    coordinate_system="world",
                    definition_method="manual",
                    weight=1.0,
                )
                for fiducial_id in ("NAS", "LPA", "RPA")
            ]
        )
        tab.measurements.set_measurement_rows(
            [
                MeasurementRow(
                    electrode_id="E1",
                    measured_distances={"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
                )
            ]
        )
        assert tab._save_timer.isActive()  # edits arm the debounced save
        tab._autosave_tables()

        fiducials_path = project / "input" / "fiducials.json"
        measurements_path = project / "input" / "measurements.json"
        assert fiducials_path.is_file()
        assert measurements_path.is_file()
        assert [
            item["fiducial_id"] for item in json.loads(fiducials_path.read_text())["fiducials"]
        ] == [
            "NAS",
            "LPA",
            "RPA",
        ]
        assert json.loads(measurements_path.read_text())["electrodes"][0]["electrode_id"] == "E1"
        assert not tab.is_dirty()
    finally:
        window.close()
        app.quit()


def test_autosave_skips_invalid_tables_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half-filled tables are skipped silently instead of clobbering the files."""
    from PySide6.QtWidgets import QMessageBox

    from virda_gui.tabs.editors_tab import FiducialRow

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "invalid-project"
        (project / "input").mkdir(parents=True)
        window.open_project(project)

        tab = window._editors_tab
        tab.fiducials.set_rows(
            [
                FiducialRow(
                    fiducial_id="NAS",
                    name="NAS",
                    coordinates=(0.0, 0.0, 0.0),
                    coordinate_system="world",
                    definition_method="manual",
                    weight=1.0,
                )
            ]
        )
        tab._autosave_tables()
        assert (project / "input" / "fiducials.json").is_file()

        tab.fiducials.set_rows(
            [
                FiducialRow(
                    fiducial_id="NAS",
                    name="NAS",
                    coordinates=("", "", ""),  # type: ignore[arg-type]
                    coordinate_system="world",
                    definition_method="manual",
                    weight=1.0,
                )
            ]
        )
        before = (project / "input" / "fiducials.json").read_bytes()
        tab._autosave_tables()  # must not raise, warn, or clobber
        assert (project / "input" / "fiducials.json").read_bytes() == before
    finally:
        window.close()
        app.quit()


def test_base_tab_step_labels_follow_voxel_spin_offscreen(tmp_path: Path) -> None:
    """Editing voxel size recomputes only the step/real-voxel labels."""
    from virda_gui.state import AppState
    from virda_gui.tabs.base_generation_tab import BaseGenerationTab

    app = _offscreen_app()
    tab = BaseGenerationTab(AppState())
    try:
        assert "marching cube step = --" in tab._real_label.text()
        tab.set_source(_write_mini_nifti(tmp_path / "mini.nii.gz"))
        assert tab._voxel_spin.value() == 1.0
        assert "marching cube step = 1" in tab._real_label.text()

        tab._voxel_spin.setValue(2.5)
        assert "marching cube step = 2" in tab._real_label.text()
        assert "2 x 2 x 2" in tab._real_label.text()
        assert tab._seal_chk.isChecked()
        assert tab._seal_radius_spin.value() == 4
        assert tab._cleaner_min_vertices_spin.value() == 100
        assert tab._cleaner_merge_digits_spin.value() == 7
    finally:
        tab.shutdown()
        tab.close()
        app.quit()


def test_base_tab_store_result_writes_meshes_and_sidecar_offscreen(
    tmp_path: Path,
) -> None:
    """Storing a generated mesh writes base, final and the source sidecar."""
    from virda.ops.options import CleanOptions, SealingOptions
    from virda_gui.scan_hash import read_source_hash, sha256_file
    from virda_gui.state import AppState
    from virda_gui.tabs.base_generation_tab import generate_mesh_from_nifti

    app = _offscreen_app()
    state = AppState()
    tab = None
    try:
        from virda_gui.tabs.base_generation_tab import BaseGenerationTab

        project = tmp_path / "gen-project"
        (project / "input").mkdir(parents=True)
        source = _write_mini_nifti(project / "input" / "head.nii.gz")
        state.last_project_dir = str(project)

        mesh = generate_mesh_from_nifti(
            source,
            SealingOptions(seal_enabled=True, seal_radius=1),
            CleanOptions(min_component_vertices=1, merge_digits=7),
            voxel_size_mm=2.0,
        )
        tab = BaseGenerationTab(state)
        fired: list[str] = []
        tab.baseMesh.connect(lambda _m: fired.append("base"))
        tab.saved.connect(lambda: fired.append("saved"))
        tab._store_result(mesh, source)

        assert (project / "mesh" / "base_mesh.ply").is_file()
        assert (project / "mesh" / "final_mesh.ply").is_file()
        assert read_source_hash(project / "mesh") == sha256_file(source)
        assert tab.current_base_mesh() is mesh
        assert fired == ["base", "saved"]
    finally:
        if tab is not None:
            tab.shutdown()
            tab.close()
        app.quit()


def test_source_paths_are_read_only_offscreen(tmp_path: Path) -> None:
    """Pipeline source rows show paths without file-picker buttons."""
    from PySide6.QtWidgets import QPushButton

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        base_buttons = [b.text() for b in window._base_tab.findChildren(QPushButton)]
        assert not any("Pick" in text for text in base_buttons)
        mesh_buttons = [b.text() for b in window._mesh_processing_tab.findChildren(QPushButton)]
        assert "Load..." not in mesh_buttons
    finally:
        window.close()
        app.quit()


def test_base_mesh_handoff_buttons_switch_tabs_offscreen(tmp_path: Path) -> None:
    """Regenerate-base and open-in-mesh buttons navigate between the tabs."""
    from PySide6.QtWidgets import QPushButton

    def _click(widget: QWidget, text: str) -> None:
        button = next(child for child in widget.findChildren(QPushButton) if text in child.text())
        button.click()

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        window._show_mesh_processing_tab()
        _click(window._mesh_processing_tab, "Regenerate base")
        assert window._tabs.currentWidget() is window._base_tab

        _click(window._base_tab, "Open in Mesh Processing")
        assert window._tabs.currentWidget() is window._mesh_processing_tab
    finally:
        window.close()
        app.quit()


def _write_full_project(project: Path) -> None:
    """Project with scan, base/final meshes, ESE companions and point tables."""
    import json

    import numpy as np

    (project / "mesh").mkdir(parents=True)
    (project / "input").mkdir(parents=True)
    _write_mini_nifti(project / "input" / "head.nii.gz")
    _write_triangle_ply(project / "mesh" / "base_mesh.ply")
    _write_triangle_ply(project / "mesh" / "final_mesh.ply")

    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ese_dir = project / "ese"
    ese_dir.mkdir()
    np.save(ese_dir / "ese_vertices.npy", np.array(vertices))
    np.save(ese_dir / "ese_faces.npy", np.array([[0, 1, 2]]))
    np.save(ese_dir / "normals.npy", np.array([[0.0, 0.0, 1.0]] * 3))
    np.save(ese_dir / "quality.npy", np.array([1.0, 1.0, 1.0]))
    (ese_dir / "point_pairs.json").write_text(
        json.dumps(
            {
                "n_points": 3,
                "scalp_vertices": vertices,
                "ese_vertices": vertices,
                "normals": [[0.0, 0.0, 1.0]] * 3,
                "quality": [1.0, 1.0, 1.0],
            }
        ),
        encoding="utf-8",
    )
    (ese_dir / "mesh.ply").write_text("placeholder", encoding="utf-8")

    (project / "input" / "fiducials.json").write_text(
        json.dumps(
            {
                "fiducials": [
                    {
                        "fiducial_id": fiducial_id,
                        "name": fiducial_id,
                        "coordinates": [1.0, 2.0, 3.0],
                        "coordinate_system": "world",
                        "definition_method": "manual",
                        "weight": 1.0,
                    }
                    for fiducial_id in ("NAS", "LPA", "RPA")
                ]
            }
        ),
        encoding="utf-8",
    )
    (project / "input" / "measurements.json").write_text(
        json.dumps(
            {
                "electrodes": [
                    {
                        "electrode_id": "E1",
                        "measured_distances": {"NAS": 1.0, "LPA": 2.0, "RPA": 3.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _chip_text(window: IdeWindow, key: str) -> str:
    chips = {
        button.text().split("\n")[0]: button.text() for button in window._pipeline_bar._buttons
    }
    return next(text for title, text in chips.items() if key in title)


def test_import_new_scan_invalidates_meshes_but_keeps_tables_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different scan drops base/final/ESE while fiducials/measurements survive."""
    from PySide6.QtWidgets import QMessageBox

    from virda_gui.scan_hash import sha256_file, write_source_hash

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "full-project"
        _write_full_project(project)
        scan = project / "input" / "head.nii.gz"
        write_source_hash(project / "mesh", scan, sha256_file(scan))
        window.open_project(project)
        assert window._ese_tab.current_ese_mesh() is not None

        monkeypatch.setattr(window, "_confirm_nifti_replace", lambda: True)
        questions: list[str] = []

        def _record_question(*args: object, **_kwargs: object) -> object:
            questions.append(str(args[2]))
            return QMessageBox.StandardButton.Yes

        monkeypatch.setattr(QMessageBox, "question", _record_question)
        other = tmp_path / "head2.nii.gz"
        other.write_bytes(b"a different scan")
        role = next(r for r in ROLE_REGISTRY if r.key == "nifti")
        window._import_nifti_scan(role, other)

        assert questions == []  # replace-confirm covers the overwrite
        assert not (project / "mesh" / "base_mesh.ply").exists()
        assert not (project / "mesh" / "final_mesh.ply").exists()
        assert list((project / "ese").glob("*")) == []
        assert window._mesh_processing_tab.base_mesh() is None
        assert window._ese_tab.current_ese_mesh() is None
        assert window._localized_electrodes is None
        assert [row.fiducial_id for row in window._editors_tab.fiducials.fiducial_rows()] == [
            "NAS",
            "LPA",
            "RPA",
        ]
        assert (project / "input" / "fiducials.json").is_file()
        assert (project / "input" / "measurements.json").is_file()
        assert window._base_tab._source_path == project / "input" / "head2.nii.gz"
        assert window._tabs.currentWidget() is window._base_tab
        assert "[ok]" not in _chip_text(window, "2. Base")
        assert "[ok]" not in _chip_text(window, "5. Points")
    finally:
        window.close()
        app.quit()


def test_import_identical_scan_skips_with_warning_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-importing the same scan warns and keeps every derived mesh."""
    from PySide6.QtWidgets import QMessageBox

    from virda_gui.scan_hash import sha256_file, write_source_hash

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "full-project"
        _write_full_project(project)
        scan = project / "input" / "head.nii.gz"
        write_source_hash(project / "mesh", scan, sha256_file(scan))
        window.open_project(project)

        confirms: list[str] = []

        def _no_confirm() -> bool:
            confirms.append("asked")
            return True

        monkeypatch.setattr(window, "_confirm_nifti_replace", _no_confirm)
        role = next(r for r in ROLE_REGISTRY if r.key == "nifti")
        window._import_nifti_scan(role, scan)

        assert confirms == []
        assert (project / "mesh" / "base_mesh.ply").is_file()
        assert window._ese_tab.current_ese_mesh() is not None
        assert "identical" in window.statusBar().currentMessage()
    finally:
        window.close()
        app.quit()


def test_import_identical_scan_shows_modal_dialog_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The identical-scan warning is a modal dialog, not just a status line."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "scan-project"
        (project / "input").mkdir(parents=True)
        scan = _write_mini_nifti(project / "input" / "head.nii.gz")
        window.open_project(project)

        shown: list[str] = []

        def _shown(*args: object, **_kwargs: object) -> object:
            shown.append(str(args[2]))
            return QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QMessageBox, "information", _shown)
        role = next(r for r in ROLE_REGISTRY if r.key == "nifti")
        window._import_nifti_scan(role, scan)

        assert len(shown) == 1
        assert "identical" in shown[0]
    finally:
        window.close()
        app.quit()


def test_scan_chip_imports_identical_scan_with_dialog_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clicking the Scan chip and picking the same file warns and stops."""
    from PySide6.QtWidgets import QFileDialog, QMessageBox, QPushButton

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "scan-project"
        (project / "input").mkdir(parents=True)
        scan = _write_mini_nifti(project / "input" / "head.nii.gz")
        window.open_project(project)

        monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(scan)], ""))
        shown: list[str] = []

        def _shown(*args: object, **_kwargs: object) -> object:
            shown.append(str(args[2]))
            return QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QMessageBox, "information", _shown)
        confirms: list[str] = []

        def _no_confirm() -> bool:
            confirms.append("asked")
            return True

        monkeypatch.setattr(window, "_confirm_nifti_replace", _no_confirm)
        chip = next(
            button
            for button in window._pipeline_bar.findChildren(QPushButton)
            if button.text().startswith("1. Brain")
        )
        assert chip.isEnabled()
        chip.click()

        assert len(shown) == 1
        assert "identical" in shown[0]
        assert confirms == []
    finally:
        window.close()
        app.quit()


def test_import_same_name_scan_skips_without_sidecar_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-importing the same file warns even when no sidecar was recorded."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "scan-project"
        (project / "input").mkdir(parents=True)
        scan = _write_mini_nifti(project / "input" / "head.nii.gz")
        window.open_project(project)

        confirms: list[str] = []

        def _no_confirm() -> bool:
            confirms.append("asked")
            return True

        monkeypatch.setattr(window, "_confirm_nifti_replace", _no_confirm)
        role = next(r for r in ROLE_REGISTRY if r.key == "nifti")
        window._import_nifti_scan(role, scan)

        assert confirms == []
        assert "identical" in window.statusBar().currentMessage()
    finally:
        window.close()
        app.quit()


def test_base_generated_drops_ese_but_keeps_tables_offscreen(tmp_path: Path) -> None:
    """A fresh base invalidates the ESE step while tables and base survive."""
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "full-project"
        _write_full_project(project)
        window.open_project(project)
        assert window._ese_tab.current_ese_mesh() is not None

        window._on_base_generated(object())

        assert (project / "mesh" / "base_mesh.ply").is_file()
        assert (project / "mesh" / "final_mesh.ply").is_file()
        assert list((project / "ese").glob("*")) == []
        assert window._ese_tab.current_ese_mesh() is None
        assert window._localized_electrodes is None
        assert "ESE mesh" in window._editors_tab.localization._hint.text()
        assert [row.fiducial_id for row in window._editors_tab.fiducials.fiducial_rows()] == [
            "NAS",
            "LPA",
            "RPA",
        ]
        assert "[ok]" not in _chip_text(window, "4. Sensor")
        assert "[ok]" not in _chip_text(window, "5. Points")
    finally:
        window.close()
        app.quit()


def test_legacy_hole_project_shows_linear_prefix_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final+ESE without a base: only scan is [ok], base is active, rest locked."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "hole-project"
        _write_full_project(project)
        (project / "mesh" / "base_mesh.ply").unlink()
        window.open_project(project)

        assert "[ok]" in _chip_text(window, "1. Brain")
        assert "[>]" in _chip_text(window, "2. Base")
        assert "[ok]" not in _chip_text(window, "3. Skin")
        assert "[ok]" not in _chip_text(window, "4. Sensor")
        assert "[ok]" not in _chip_text(window, "5. Points")
        assert window._pipeline_bar.is_enabled("base")
        assert not window._pipeline_bar.is_enabled("mesh")
        assert not window._pipeline_bar.is_enabled("ese")
        assert not window._pipeline_bar.is_enabled("points")
        assert [row.fiducial_id for row in window._editors_tab.fiducials.fiducial_rows()] == [
            "NAS",
            "LPA",
            "RPA",
        ]
    finally:
        window.close()
        app.quit()


def test_hole_dialog_offers_rebuild_from_last_valid_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accepting the data-loss dialog navigates to the first missing step."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "hole-project"
        _write_full_project(project)
        (project / "mesh" / "base_mesh.ply").unlink()
        window.open_project(project)

        assert window._tabs.currentWidget() is window._base_tab
    finally:
        window.close()
        app.quit()


def test_hole_dialog_decline_stays_put_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the data-loss dialog keeps the current tab and the tables."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "hole-project"
        _write_full_project(project)
        (project / "mesh" / "base_mesh.ply").unlink()
        window.open_project(project)

        assert window._tabs.count() == 0  # no tab is forced open
        assert [row.fiducial_id for row in window._editors_tab.fiducials.fiducial_rows()] == [
            "NAS",
            "LPA",
            "RPA",
        ]
        assert (project / "input" / "fiducials.json").is_file()
    finally:
        window.close()
        app.quit()


class _ReloadViewerStub(QWidget):
    """Stand-in viewer recording scene reloads without threads."""

    def __init__(self) -> None:
        super().__init__()
        self.loads: list[dict[str, str]] = []
        self.dirs: list[str] = []

    def set_project_dir(self, project: object) -> None:
        self.dirs.append(str(project))

    def load(self, **kwargs: str) -> None:
        self.loads.append(dict(kwargs))

    def shutdown(self) -> None:
        return None


def test_reload_viewer_if_open_reloads_scene_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regenerated data reopens the 3D viewer scene instead of going stale."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "reload-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        window.open_project(project)

        window._reload_viewer_if_open()  # no viewer yet: no-op, no crash
        assert window._viewer_widget is None

        stub = _ReloadViewerStub()
        window._viewer_widget = stub  # type: ignore[assignment]
        window._viewer_tab_widget = stub
        window._reload_viewer_if_open()
        assert len(stub.loads) == 1
        assert stub.loads[0]["mesh_path"] == str(project / "mesh" / "final_mesh.ply")

        window._state.viewer_loading = True
        window._reload_viewer_if_open()  # loading: skipped, no second load
        assert len(stub.loads) == 1
        window._state.viewer_loading = False
    finally:
        window._viewer_widget = None
        window._viewer_tab_widget = None
        window._on_close()
        app.quit()


def test_invalidate_records_and_reopens_file_tabs_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File tabs closed by invalidation come back once the meshes are back."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "tabs-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        ese_file = project / "ese" / "normals.npy"
        ese_file.parent.mkdir(parents=True)
        import numpy as np

        np.save(ese_file, np.ones((2, 3)))
        window.open_project(project)

        window._file_tabs[str(ese_file)] = QWidget()
        window._invalidate_derived_meshes(delete_base=False)
        assert not ese_file.exists()
        assert window._invalidated_file_tabs == [str(ese_file)]

        opened: list[str] = []
        import unittest.mock as mock

        with mock.patch.object(window, "_open_file_tab", lambda p: opened.append(str(p))):
            window._reopen_invalidated_file_tabs()
        assert opened == [str(ese_file)]
        assert window._invalidated_file_tabs == []
    finally:
        window._on_close()
        app.quit()


def test_electrode_group_edits_sync_and_reload_offscreen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group joins the scene after a valid path; a bad path warns and drops it."""
    from PySide6.QtWidgets import QMessageBox

    app = _offscreen_app()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    warnings: list[str] = []

    def _warn(*args: object, **_kwargs: object) -> object:
        warnings.append(str(args[2]))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", _warn)
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "groups-project"
        (project / "mesh").mkdir(parents=True)
        _write_triangle_ply(project / "mesh" / "final_mesh.ply")
        window.open_project(project)
        window._build_viewer_widget()

        reloaded: list[str] = []
        import unittest.mock as mock

        with mock.patch.object(window, "_open_viewer", lambda p: reloaded.append(str(p))):
            window._on_add_electrode_group()  # empty row: nothing synced, no reload
            assert reloaded == []
            assert window._state.electrode_rows == []

            group_file = tmp_path / "group.tsv"
            group_file.write_text("id\tx\ty\tz\n", encoding="utf-8")
            window._electrode_group_widgets[-1].set(str(group_file))
            assert (str(group_file), "yellow") in window._state.electrode_rows
            assert reloaded == [str(project)]

            window._electrode_group_widgets[-1].set(str(tmp_path / "missing.tsv"))
            assert warnings and "not found" in warnings[-1]
            assert window._electrode_group_widgets == []
            assert window._state.electrode_rows == []
            assert reloaded == [str(project)]  # invalid path: no reload
    finally:
        window._viewer_widget = None
        window._viewer_tab_widget = None
        window._on_close()
        app.quit()


def test_no_surface_picking_controls_offscreen(tmp_path: Path) -> None:
    """The viewer panel offers groups only: no picking, no extra fiducials."""
    from PySide6.QtWidgets import QPushButton

    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        window._build_viewer_widget()
        panel = window._viewer_tab_widget
        assert panel is not None
        labels = [button.text() for button in panel.findChildren(QPushButton)]
        assert not any("Pick point" in text for text in labels)
        assert not any("Clear points" in text for text in labels)
        assert not hasattr(window, "_pick_btn")
        assert not hasattr(window, "_on_surface_picked")
        assert not hasattr(window, "_next_fiducial_id")
    finally:
        window._viewer_widget = None
        window._viewer_tab_widget = None
        window._on_close()
        app.quit()


def test_cleared_scene_ignores_live_points_offscreen() -> None:
    """Live points landing mid-reload must not touch deleted buttons."""
    import numpy as np
    from PySide6.QtWidgets import QApplication

    from virda_gui.viewer.viewer import ViewerWidget

    app = _offscreen_app()
    viewer = ViewerWidget()
    try:
        viewer._add_export_controls(None)  # type: ignore[arg-type]
        assert viewer._electrodes_export_button is not None
        viewer.clear_scene()
        QApplication.processEvents()  # flush deleteLater: C++ buttons die here
        assert viewer._electrodes_export_button is None
        assert viewer._fiducials_export_button is None
        viewer.set_live_electrodes(["E1"], np.array([[1.0, 0.0, 0.0]]), np.array([True]))
        viewer.clear_live_points()
    finally:
        viewer.shutdown()
        viewer.close()
        app.quit()


def test_ese_offset_allows_zero_offscreen() -> None:
    """The ESE offset spin accepts 0 mm (sensor surface on the scalp)."""
    from virda_gui.state import AppState
    from virda_gui.tabs.ese_tab import EseTab

    app = _offscreen_app()
    tab = EseTab(AppState(), base_provider=lambda: None)
    try:
        assert tab._ese_offset_spin.minimum() == 0.0
        tab._ese_offset_spin.setValue(0.0)
        assert tab._ese_options().ese_offset_mm == 0.0
    finally:
        tab.shutdown()
        tab.close()
        app.quit()
