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
from virda_gui.state import AppState
from virda_gui.viewer.frames import frame_to_frame_matrix, frame_to_world_matrix, world_to_frame_matrix
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
    assert detect_role("scan.nii").key == "nifti"
    assert detect_role("scan.nii.gz").key == "nifti"
    assert detect_role("ese_mesh.ply").key == "ese_mesh"
    assert detect_role("final_mesh.ply") is None  # ambiguous between mesh / ese_mesh
    assert detect_role("mesh.ply") is None
    assert detect_role("normals.npy").key == "normals"
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
    assert detect_role(electrodes).key == "electrodes"

    measurements = tmp_path / "measurements.json"
    measurements.write_text(json.dumps({"electrodes": []}), encoding="utf-8")
    assert detect_role(measurements).key == "measurements"

    fiducials = tmp_path / "fiducials.json"
    fiducials.write_text(json.dumps({"fiducials": []}), encoding="utf-8")
    assert detect_role(fiducials).key == "fiducials"

    unknown = tmp_path / "config.json"
    unknown.write_text(json.dumps({"nifti_path": "x"}), encoding="utf-8")
    assert detect_role(unknown) is None


def test_import_fallback_roles_cover_six_artifact_kinds() -> None:
    assert [role.key for role in IMPORT_FALLBACK_ROLES] == [
        "nifti",
        "mesh",
        "ese_mesh",
        "normals",
        "electrodes",
        "measurements",
    ]


# ----------------------------------------------------------------------
# import validation
# ----------------------------------------------------------------------


def test_validate_mesh_source_accepts_ply(tmp_path: Path) -> None:
    mesh = _write_triangle_ply(tmp_path / "final_mesh.ply")
    validate_import_source(
        next(role for role in ROLE_REGISTRY if role.key == "mesh"), mesh
    )


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
        == project / "ese" / "ese_mesh.ply"
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
    from virda.io.importers.nifti import import_nifti
    from virda.ops.atoms import clean, generate_scalp_surface

    mri = import_nifti(_write_mini_nifti(tmp_path / "mini.nii.gz"))
    surface = generate_scalp_surface(mri, SealingOptions(seal_enabled=True, seal_radius=1))
    mesh = clean(surface.mesh, CleanOptions(min_component_vertices=1, merge_digits=7))

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
    assert ADVANCED_FIELD_DEFAULTS["seal_enabled"] == "true"
    assert ADVANCED_FIELD_DEFAULTS["seal_radius"] == "4"
    assert ADVANCED_FIELD_DEFAULTS["cleaner_min_vertices"] == "100"
    assert ADVANCED_FIELD_DEFAULTS["cleaner_merge_digits"] == "7"
    assert ADVANCED_FIELD_DEFAULTS["residual_threshold_mm"] == "10.0"
    assert ADVANCED_FIELD_DEFAULTS["calibrate_ese_offset"] == "true"


def test_no_pipeline_or_config_surfaces_in_gui() -> None:
    """The pipeline/config/log layers are gone (files and runtime attributes)."""
    import importlib
    import virda_gui as gui

    root = Path(gui.__file__).parent
    assert not (root / "services").exists()
    assert not (root / "services" / "pipeline_runner.py").exists()
    assert not (root / "services" / "logging.py").exists()
    assert not (root / "tabs" / "config_tab.py").exists()
    assert not (root / "constants.py").read_text(encoding="utf-8").count(
        "DEFAULT_PIPELINE_CONFIG_FILENAME"
    )

    removed = {
        "pipeline_runner",
        "logging",
        "ConfigTab",
        "LogViewer",
        "PipelineRunner",
        "runPipelineRequested",
    }
    for module_name in (
        "virda_gui.sidebar",
        "virda_gui.main_window",
        "virda_gui.state",
        "virda_gui.importing",
        "virda_gui.tabs.mesh_processing_tab",
        "virda_gui.tabs.editors_tab",
    ):
        module = importlib.import_module(module_name)
        assert removed.isdisjoint(vars(module))
    assert not hasattr(AppState(), "log_queue")
    assert not hasattr(AppState(), "stage3_summary")
    assert not hasattr(AppState(), "coordsystem")


def test_viewer_widget_importable_from_main_window() -> None:
    """The 3D viewer tab embeds ``ViewerWidget`` from ``virda_gui.viewer.viewer``."""
    import virda_gui.main_window as main_window_module
    from virda_gui.viewer.viewer import ViewerWidget

    assert vars(main_window_module)["ViewerWidget"] is ViewerWidget


def test_advanced_dialog_exposes_only_gui_keys_offscreen() -> None:
    """The advanced dialog is trimmed to GUI-only mesh/localization knobs."""
    app = _offscreen_app()
    dialog = AdvancedSettingsDialog(None, dict(ADVANCED_FIELD_DEFAULTS))
    try:
        assert set(dialog._fields) == {
            "seal_enabled",
            "seal_radius",
            "cleaner_min_vertices",
            "cleaner_merge_digits",
            "residual_threshold_mm",
            "calibrate_ese_offset",
        }
    finally:
        dialog.close()
        app.quit()


def test_ide_window_constructs_and_manages_project_offscreen(tmp_path: Path) -> None:
    """IdeWindow builds, opens/closes a project and closes tabs without a run tab."""
    app = _offscreen_app()
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

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(target))

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    assert ask_create_project_folder(parent=None) == target

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    assert ask_create_project_folder(parent=None) is None
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


def test_ide_window_opens_project_files_in_tabs_offscreen(
    tmp_path: Path, monkeypatch
) -> None:
    """Double-clicking a sidebar artifact opens a viewer/preview tab."""
    import virda_gui.main_window as main_window_module

    monkeypatch.setattr(main_window_module, "ViewerWidget", _StubViewer)
    app = _offscreen_app()
    prefs = _make_prefs(tmp_path)
    window = IdeWindow(prefs=prefs)
    try:
        project = tmp_path / "sample-project"
        mesh = project / "mesh" / "final_mesh.ply"
        mesh.parent.mkdir(parents=True)
        _write_triangle_ply(mesh)
        (project / "note.txt").write_text("hello", encoding="utf-8")
        window.open_project(project)

        window._sidebar.fileActivated.emit(mesh)
        mesh_tab = window._tabs.widget(window._tabs.count() - 1)
        assert isinstance(mesh_tab, _StubViewer)
        assert mesh_tab.load_calls == [{"mesh_path": str(mesh)}]

        window._sidebar.fileActivated.emit(mesh)
        assert window._tabs.count() == 1  # file tabs de-duplicate by path

        window._sidebar.fileActivated.emit(project / "note.txt")
        from virda_gui.tabs.preview_tab import PreviewTab

        preview_index = next(
            i for i in range(window._tabs.count()) if isinstance(window._tabs.widget(i), PreviewTab)
        )
        assert window._tabs.tabText(preview_index) == "note.txt"

        window._close_tab(preview_index)
        assert not any(
            isinstance(window._tabs.widget(i), PreviewTab) for i in range(window._tabs.count())
        )

        mesh_index = next(
            i
            for i in range(window._tabs.count())
            if isinstance(window._tabs.widget(i), _StubViewer)
        )
        window._close_tab(mesh_index)
        assert mesh_tab.shut_down
    finally:
        window._on_close()
        app.quit()


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
                    "electrodes": [
                        {"electrode_id": "Cz", "measured_distances": {"nas": 100.0}}
                    ],
                    "fiducial_weights": {"nas": 1.0},
                }
            ),
            encoding="utf-8",
        )

        window.open_project(project)

        rows = window._editors_tab.fiducials.fiducial_rows()
        assert [row.fiducial_id for row in rows] == ["nas"]
        assert window._editors_tab.measurements.measurement_rows()[0].electrode_id == "Cz"
        assert window._tabs.count() == 0  # no tab is forced open
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
    state = AppState(advanced=dict(ADVANCED_FIELD_DEFAULTS))

    class _Dialog:
        def __init__(self, parent, values) -> None:
            self.result_values = {"seal_enabled": "false", "seal_radius": "9"}
            self._exec = True

        def exec(self):
            return True

    original = AdvancedSettingsDialog
    import virda_gui.main_window as main_window_module

    try:
        main_window_module.AdvancedSettingsDialog = _Dialog  # type: ignore[assignment]
        prefs = _make_prefs(tmp_path)
        window = IdeWindow(prefs=prefs)
        try:
            window._on_show_advanced_settings()
            assert window._state.advanced["seal_enabled"] == "false"
            assert window._state.advanced["seal_radius"] == "9"
        finally:
            window.close()
    finally:
        main_window_module.AdvancedSettingsDialog = original
        app.quit()