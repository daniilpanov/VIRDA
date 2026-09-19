"""Embeddable interactive 3D viewer: scalp mesh over the MRI volume.

:class:`ViewerWidget` renders the scalp mesh on top of a semi-transparent MRI
volume and places both objects in the same coordinate frame using the NIfTI
affine, so the mesh overlays the actual scalp in all three views. Qt
checkboxes toggle the visibility of the mesh, the MRI, the fiducial points,
the normal glyphs and per-electrode labels, and a "Boost contrast" checkbox
sharpens the MRI and the mesh.  :class:`ViewerWidget` wraps the same scene
that the standalone ``virda-gui-viewer`` CLI (module ``virda_gui.viewer.viewer_cli``)
shows in a blocking window; the GUI embeds it on its 3D Viewer tab.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .frames import (
    FRAME_IDS,
    FRAME_SCANNER,
    collect_electrodes_export,
    collect_fiducials_export,
    collect_mesh_export,
    frame_available,
    frame_label,
    natural_frame,
    scene_to_frame_matrix,
    write_mesh_obj,
    write_points_tsv,
)
from .scene import transform_points
from .viewer_loaders import (
    SceneData,
    build_electrode_links,
    collect_scene_data,
    intensify_color,
)


class _SceneLoader(QObject):
    """Collect scene data off the GUI thread.

    Lives on a dedicated :class:`QThread`; ``loaded``/``failed`` are delivered
    back to the main thread because the widget (the receiver) lives there.
    """

    loaded = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        log: Callable[[str], None],
        kwargs: dict[str, Any],
        seq: int,
    ) -> None:
        super().__init__()
        self._log = log
        self._kwargs = kwargs
        self._seq = seq

    def run(self) -> None:
        try:
            scene = collect_scene_data(log=self._log, **self._kwargs)
        except Exception as exc:
            self.failed.emit(self._seq, str(exc))
        else:
            self.loaded.emit(self._seq, scene)


class ViewerWidget(QWidget):
    """Embeddable Qt 3D viewer.

    Renders the scalp mesh over the MRI volume inside a pyvista
    :class:`pyvistaqt.QtInteractor` next to a native Qt panel of layer
    checkboxes.  :meth:`load` collects the scene on a worker thread and builds
    the VTK actors on the GUI thread; the :attr:`sceneLoaded` and
    :attr:`sceneFailed` signals notify embedders when rendering is ready or a
    load/validation error occurred.
    """

    sceneLoaded = Signal(object)  # noqa: N815
    sceneFailed = Signal(str)  # noqa: N815

    def __init__(self, parent: QWidget | None = None, log: Callable[[str], None] = print) -> None:
        super().__init__(parent)
        self._log = log
        self._load_seq = 0
        self._thread: QThread | None = None
        self._worker: _SceneLoader | None = None

        self._plotter = QtInteractor(parent=self)
        self._volume: pv.ImageData | None = None
        self._scene: SceneData | None = None
        self._affine: np.ndarray | None = None
        self._cras_offset: np.ndarray | None = None
        self._mm_scene = True
        self._current_frame = FRAME_SCANNER
        self._point_actors: list[Any] = []
        self._frame_combo: QComboBox | None = None
        self._mri_actor: Any = None
        self._mesh_actor: Any = None
        self._fiducial_actor: Any = None
        self._fiducial_label_actor: Any = None
        self._normals_actor: Any = None
        self._electrode_actors_per_group: list[list[Any]] = []
        self._flagged_actors_per_group: list[list[Any]] = []
        self._link_actors_per_group: list[list[Any]] = []
        self._label_actors_per_group: list[list[Any]] = []
        self._group_states: list[bool] = []
        self._mri_visible = True
        self._mesh_visible = True
        self._fiducials_visible = True
        self._normals_visible = True
        self._labels_visible = True
        self._mesh_opacity = 0.6
        self._hi_clim: tuple[float, float] | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._plotter, 1)

        self._layers_panel = QGroupBox("Layers", self)
        self._layers_layout = QVBoxLayout(self._layers_panel)
        self._layers_layout.setContentsMargins(4, 4, 4, 4)
        self._layers_layout.setSpacing(4)
        layout.addWidget(self._layers_panel, 0)

    # ---- scene loading (background thread -> GUI thread) ----

    def load(self, **kwargs: Any) -> None:
        if not kwargs.get("nifti_path") and not kwargs.get("mesh_path"):
            message = "at least one of nifti_path or mesh_path is required"
            self._log(f"ERROR: viewer failed: {message}")
            self.sceneFailed.emit(message)
            return
        if kwargs.get("electrodes_cras") and not kwargs.get("nifti_path"):
            message = "electrodes_cras requires nifti_path"
            self._log(f"ERROR: viewer failed: {message}")
            self.sceneFailed.emit(message)
            return

        self._load_seq += 1
        seq = self._load_seq
        self.clear_scene()
        self._log("Loading 3D scene...")
        self._cancel_running_load()

        self._thread = QThread(self)
        self._worker = _SceneLoader(self._log, kwargs, seq)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.loaded.connect(self._on_scene_loaded)
        self._worker.failed.connect(self._on_scene_failed)
        self._thread.start()

    def _on_scene_loaded(self, seq: int, scene: object) -> None:
        if seq != self._load_seq:
            return
        self._finish_loading()
        self.set_scene(scene)  # type: ignore[arg-type]
        self.sceneLoaded.emit(scene)

    def _on_scene_failed(self, seq: int, message: str) -> None:
        if seq != self._load_seq:
            return
        self._finish_loading()
        self._log(f"ERROR: viewer failed: {message}")
        self.sceneFailed.emit(message)

    def _finish_loading(self) -> None:
        # deleteLater() must be posted while the worker's event loop is still
        # live, otherwise the DeferredDelete event is never processed.
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        self._worker = None

    def _cancel_running_load(self) -> None:
        """Stop an in-flight scene load so a newer request replaces it.

        A previous ``load`` may still be collecting scene data on its worker
        thread; retiring it here prevents the abandoned thread (and its
        ``loaded``/``failed`` emissions) from lingering after ``load`` is
        called again.
        """
        thread = self._thread
        if thread is None:
            return
        worker = self._worker
        if worker is not None:
            worker.loaded.disconnect(self._on_scene_loaded)
            worker.failed.disconnect(self._on_scene_failed)
            # deleteLater() must be posted while the worker's event loop is
            # still live, otherwise the DeferredDelete event is never processed.
            worker.deleteLater()
        thread.quit()
        thread.wait(3000)
        # Never delete a thread that is still running (wait timed out);
        # destroying a live QThread is undefined behaviour.  In that case the
        # thread keeps running under its parent until it finishes.
        if thread.isFinished():
            thread.deleteLater()
        self._thread = None
        self._worker = None

    def clear_scene(self) -> None:
        self._plotter.clear()
        self._clear_layers()
        self._scene = None
        self._point_actors = []
        self._frame_combo = None
        self._current_frame = FRAME_SCANNER
        self._affine = None
        self._cras_offset = None

    def _clear_layers(self) -> None:
        while self._layers_layout.count():
            item = self._layers_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # ---- actor building (GUI thread) ----

    def set_scene(self, scene: SceneData) -> None:
        self._plotter.clear()
        self._clear_layers()

        self._scene = scene
        self._volume = scene.volume
        self._affine = scene.affine
        self._cras_offset = scene.cras_offset
        self._mm_scene = scene.mm_scene
        self._mesh_opacity = scene.mesh_opacity
        self._hi_clim = scene.hi_clim

        self._group_states = [True] * len(scene.electrode_groups)
        self._mri_visible = True
        self._mesh_visible = True
        self._fiducials_visible = True
        self._normals_visible = True
        self._labels_visible = True
        self._current_frame = natural_frame(scene.mm_scene)

        self._mri_actor = None
        if scene.volume is not None:
            self._mri_actor = self._plotter.add_volume(
                scene.volume, cmap="bone", opacity="sigmoid", mapper="smart"
            )

        # The scene data is already expressed in its natural frame, so the
        # initial actor pass uses the identity transform.
        self._rebuild_point_actors(np.eye(4))
        self._build_layers(scene)
        self._plotter.add_axes(interactive=False)
        self._plotter.render()

    def _remove_point_actors(self) -> None:
        for actor in self._point_actors:
            self._plotter.remove_actor(actor)
        self._point_actors = []
        self._mesh_actor = None
        self._fiducial_actor = None
        self._fiducial_label_actor = None
        self._normals_actor = None
        self._electrode_actors_per_group = []
        self._flagged_actors_per_group = []
        self._link_actors_per_group = []
        self._label_actors_per_group = []

    def _rebuild_point_actors(self, matrix: np.ndarray) -> None:
        """(Re)build the point-based actors transformed by 4x4 *matrix*.

        Runs at load time with the identity matrix (the scene data already is
        in its natural frame) and again whenever the user selects another
        display frame via the coordinate-frame combo.  The NIfTI volume is
        never reloaded and every actor is rebuilt from the original scene
        points, so switching frames does not accumulate floating-point drift.
        """
        self._remove_point_actors()
        scene = self._scene
        if scene is None:
            return

        self._mesh_actor = None
        if scene.scene_mesh is not None:
            poly = scene.scene_mesh.copy()
            poly.transform(matrix, inplace=True)
            self._mesh_actor = self._plotter.add_mesh(
                poly, color="salmon", opacity=self._mesh_opacity
            )
            self._point_actors.append(self._mesh_actor)

        self._fiducial_actor = None
        self._fiducial_label_actor = None
        if scene.fiducial_points is not None and len(scene.fiducial_points) > 0:
            pts = transform_points(scene.fiducial_points, matrix)
            self._fiducial_actor = self._plotter.add_points(
                pts, color="red", point_size=10, render_points_as_spheres=True
            )
            self._fiducial_label_actor = self._plotter.add_point_labels(
                pts,
                scene.fiducial_labels,
                font_size=12,
                text_color="white",
                show_points=False,
                shape="rounded_rect",
                shape_color="black",
                shape_opacity=0.65,
                always_visible=True,
            )
            self._point_actors.extend([self._fiducial_actor, self._fiducial_label_actor])

        self._normals_actor = None
        if scene.normals_poly is not None:
            poly = scene.normals_poly.copy()
            poly.transform(matrix, inplace=True)
            self._normals_actor = self._plotter.add_mesh(
                poly, color="cyan", opacity=0.8, line_width=2
            )
            self._point_actors.append(self._normals_actor)

        # Fiducial anchors used by the electrode links must move with the
        # same transform as the electrode points they connect to.
        fiducial_anchor = {
            key: transform_points(point[np.newaxis, :], matrix)[0]
            for key, point in scene.fiducial_id_to_point.items()
        }

        self._electrode_actors_per_group = []
        self._flagged_actors_per_group = []
        self._link_actors_per_group = []
        self._label_actors_per_group = []
        for group in scene.electrode_groups:
            pts = group["points"]
            flg = group["flags"]
            meas = group["measured"]
            color = group["color"]
            e_actors: list[Any] = []
            f_actors: list[Any] = []
            l_actors: list[Any] = []
            n_actors: list[Any] = []
            if pts is not None and len(pts) > 0:
                pts = transform_points(pts, matrix)
                healthy = ~flg
                if healthy.any():
                    actor = self._plotter.add_points(
                        pts[healthy],
                        color=color,
                        point_size=12,
                        render_points_as_spheres=True,
                    )
                    e_actors.append(actor)
                    self._point_actors.append(actor)
                if (~healthy).any():
                    actor = self._plotter.add_points(
                        pts[~healthy],
                        color=intensify_color(color),
                        point_size=17,
                        render_points_as_spheres=True,
                    )
                    f_actors.append(actor)
                    self._point_actors.append(actor)
                links = build_electrode_links(pts, meas, fiducial_anchor)
                if len(links) > 0:
                    actor = self._plotter.add_lines(links.reshape(-1, 3), color=color, width=1)
                    l_actors.append(actor)
                    self._point_actors.append(actor)
                actor = self._plotter.add_point_labels(
                    pts,
                    group["names"],
                    font_size=12,
                    text_color="white",
                    show_points=False,
                    shape="rounded_rect",
                    shape_color="black",
                    shape_opacity=0.65,
                    always_visible=True,
                )
                n_actors.append(actor)
                self._point_actors.append(actor)
            self._electrode_actors_per_group.append(e_actors)
            self._flagged_actors_per_group.append(f_actors)
            self._link_actors_per_group.append(l_actors)
            self._label_actors_per_group.append(n_actors)

        self._apply_visibility_states()

    def _apply_visibility_states(self) -> None:
        """Restore the layer visibility after a frame-switch actor rebuild."""
        if self._mesh_actor is not None:
            self._mesh_actor.SetVisibility(self._mesh_visible)
        if self._normals_actor is not None:
            self._normals_actor.SetVisibility(self._normals_visible)
        if self._fiducial_actor is not None:
            self._fiducial_actor.SetVisibility(self._fiducials_visible)
        self._apply_labels_visibility()
        for gi in range(len(self._group_states)):
            self._apply_group_visibility(gi)

    def _volume_frame_aligned(self) -> bool:
        """Whether the static volume actor lines up with the selected frame."""
        return self._current_frame == natural_frame(self._mm_scene)

    def _on_frame_selected(self) -> None:
        """Re-render the point actors in the coordinate frame the user picked."""
        combo = self._frame_combo
        if combo is None:
            return
        frame = combo.currentData()
        frame = frame if isinstance(frame, str) else None
        if frame is None or frame == self._current_frame:
            return
        try:
            matrix = scene_to_frame_matrix(frame, self._affine, self._cras_offset, self._mm_scene)
        except ValueError as exc:
            self._log(f"ERROR: cannot switch to {frame_label(frame)}: {exc}")
            combo.setCurrentIndex(FRAME_IDS.index(self._current_frame))
            return
        self._current_frame = frame
        self._rebuild_point_actors(matrix)
        # The volume is only meaningful in the scene's natural frame; hide it
        # in the alternative frames where it would otherwise float misaligned.
        if self._mri_actor is not None:
            self._mri_actor.SetVisibility(self._mri_visible and self._volume_frame_aligned())
        self._plotter.render()

    def _add_frame_controls(self) -> None:
        box = QGroupBox("Coordinate frame", self._layers_panel)
        frame_layout = QVBoxLayout(box)
        frame_layout.setContentsMargins(4, 4, 4, 4)
        frame_layout.setSpacing(4)
        combo = QComboBox(box)
        for frame_id in FRAME_IDS:
            combo.addItem(frame_label(frame_id), frame_id)
        combo.setCurrentIndex(FRAME_IDS.index(self._current_frame))
        for index, frame_id in enumerate(FRAME_IDS):
            if not frame_available(frame_id, self._affine, self._cras_offset):
                combo.setItemData(index, False, role=int(Qt.ItemDataRole.DisableRole))
        combo.currentIndexChanged.connect(lambda _index: self._on_frame_selected())
        frame_layout.addWidget(combo)
        frame_layout.addWidget(
            QLabel(
                "The MRI volume is shown only in the scene's native frame; "
                "the overlay survives all frames.",
                box,
            )
        )
        self._frame_combo = combo
        self._layers_layout.addWidget(box)

    def _add_export_controls(self) -> None:
        box = QGroupBox("Export to coordinate system", self._layers_panel)
        export_layout = QVBoxLayout(box)
        export_layout.setContentsMargins(4, 4, 4, 4)
        export_layout.setSpacing(4)
        export_layout.addWidget(
            QLabel("Writes the loaded data re-expressed in the selected frame.", box)
        )
        for text, slot in (
            ("Export mesh (OBJ)...", self._on_export_mesh),
            ("Export electrodes (TSV)...", self._on_export_electrodes),
            ("Export fiducials (TSV)...", self._on_export_fiducials),
        ):
            button = QPushButton(text, box)
            button.clicked.connect(slot)
            export_layout.addWidget(button)
        self._layers_layout.addWidget(box)

    def _current_export_matrix(self) -> np.ndarray:
        return scene_to_frame_matrix(
            self._current_frame, self._affine, self._cras_offset, self._mm_scene
        )

    def _on_export_mesh(self) -> None:
        if self._scene is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export scalp mesh",
            f"scalp_mesh_{self._current_frame}.obj",
            "OBJ (*.obj);;All files (*)",
        )
        if not path:
            return
        try:
            points, faces = collect_mesh_export(self._scene, self._current_export_matrix())
            write_mesh_obj(path, points, faces, self._current_frame)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export scalp mesh", f"Could not export mesh:\n{exc}")
            return
        self._log(f"Exported scalp mesh to {path} in {frame_label(self._current_frame)}")

    def _on_export_electrodes(self) -> None:
        if self._scene is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export electrodes",
            f"electrodes_{self._current_frame}.tsv",
            "TSV/CSV (*.tsv *.csv);;All files (*)",
        )
        if not path:
            return
        try:
            names, points = collect_electrodes_export(self._scene, self._current_export_matrix())
            write_points_tsv(path, names, points)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export electrodes", f"Could not export electrodes:\n{exc}")
            return
        self._log(f"Exported electrodes to {path} in {frame_label(self._current_frame)}")

    def _on_export_fiducials(self) -> None:
        if self._scene is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export fiducials",
            f"fiducials_{self._current_frame}.tsv",
            "TSV/CSV (*.tsv *.csv);;All files (*)",
        )
        if not path:
            return
        try:
            names, points = collect_fiducials_export(self._scene, self._current_export_matrix())
            write_points_tsv(path, names, points)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Export fiducials", f"Could not export fiducials:\n{exc}")
            return
        self._log(f"Exported fiducials to {path} in {frame_label(self._current_frame)}")

    def _build_layers(self, scene: SceneData) -> None:
        if self._mesh_actor is not None:
            self._add_layer_check("Show mesh", True, self._set_mesh_visibility)
        if self._mri_actor is not None:
            self._add_layer_check("Show MRI", True, self._set_mri_visibility)
        if self._fiducial_actor is not None:
            self._add_layer_check("Show fiducials", True, self._set_fiducials_visibility)
        if self._normals_actor is not None:
            self._add_layer_check("Show normals", True, self._set_normals_visibility)
        if self._fiducial_label_actor is not None or any(
            group_actors for group_actors in self._label_actors_per_group
        ):
            self._add_layer_check("Show labels", True, self._set_labels_visibility)
        for gi, group in enumerate(scene.electrode_groups):
            all_actors = (
                self._electrode_actors_per_group[gi]
                + self._flagged_actors_per_group[gi]
                + self._link_actors_per_group[gi]
            )
            if all_actors or self._label_actors_per_group[gi]:
                self._add_layer_check(
                    f"Show {group['label']}",
                    True,
                    lambda flag, gi=gi: self._set_group_visibility(gi, flag),
                )
        self._add_layer_check("Boost contrast", False, self._set_contrast)
        self._add_frame_controls()
        self._add_export_controls()
        self._layers_layout.addStretch(1)

    def _add_layer_check(self, text: str, checked: bool, slot: Callable[[bool], None]) -> None:
        check = QCheckBox(text, self._layers_panel)
        check.setChecked(checked)
        check.toggled.connect(slot)
        self._layers_layout.addWidget(check)

    # ---- layer visibility ----

    def _apply_labels_visibility(self) -> None:
        for gi in range(len(self._group_states)):
            for actor in self._label_actors_per_group[gi]:
                actor.SetVisibility(self._group_states[gi] and self._labels_visible)
        if self._fiducial_label_actor is not None:
            self._fiducial_label_actor.SetVisibility(
                self._fiducials_visible and self._labels_visible
            )

    def _apply_group_visibility(self, gi: int) -> None:
        visible = self._group_states[gi]
        for actor in (
            self._electrode_actors_per_group[gi]
            + self._flagged_actors_per_group[gi]
            + self._link_actors_per_group[gi]
        ):
            actor.SetVisibility(visible)
        for actor in self._label_actors_per_group[gi]:
            actor.SetVisibility(visible and self._labels_visible)

    def _set_mesh_visibility(self, flag: bool) -> None:
        self._mesh_visible = bool(flag)
        if self._mesh_actor is not None:
            self._mesh_actor.SetVisibility(flag)

    def _set_mri_visibility(self, flag: bool) -> None:
        self._mri_visible = bool(flag)
        if self._mri_actor is not None:
            self._mri_actor.SetVisibility(flag and self._volume_frame_aligned())

    def _set_fiducials_visibility(self, flag: bool) -> None:
        self._fiducials_visible = bool(flag)
        if self._fiducial_actor is not None:
            self._fiducial_actor.SetVisibility(flag)
        self._apply_labels_visibility()

    def _set_normals_visibility(self, flag: bool) -> None:
        self._normals_visible = bool(flag)
        if self._normals_actor is not None:
            self._normals_actor.SetVisibility(flag)

    def _set_labels_visibility(self, flag: bool) -> None:
        self._labels_visible = bool(flag)
        self._apply_labels_visibility()

    def _set_group_visibility(self, gi: int, flag: bool) -> None:
        self._group_states[gi] = bool(flag)
        self._apply_group_visibility(gi)

    def _set_contrast(self, flag: bool) -> None:
        if self._mesh_actor is not None:
            if flag:
                self._mesh_actor.prop.opacity = 0.95
                self._mesh_actor.prop.diffuse = 1.0
                self._mesh_actor.prop.specular = 0.6
                self._mesh_actor.prop.specular_power = 40.0
                self._mesh_actor.prop.ambient = 0.2
            else:
                self._mesh_actor.prop.opacity = self._mesh_opacity
                self._mesh_actor.prop.diffuse = 1.0
                self._mesh_actor.prop.specular = 0.0
                self._mesh_actor.prop.specular_power = 100.0
                self._mesh_actor.prop.ambient = 0.0
        if self._volume is not None and self._mri_actor is not None:
            if "intensity" in self._plotter.scalar_bars:
                self._plotter.remove_scalar_bar("intensity")
            self._plotter.remove_actor(self._mri_actor)
            if flag:
                self._mri_actor = self._plotter.add_volume(
                    self._volume,
                    cmap="bone",
                    opacity="sigmoid_10",
                    mapper="smart",
                    clim=self._hi_clim,
                    opacity_unit_distance=0.5,
                )
            else:
                self._mri_actor = self._plotter.add_volume(
                    self._volume, cmap="bone", opacity="sigmoid", mapper="smart"
                )
            self._mri_actor.SetVisibility(self._mri_visible and self._volume_frame_aligned())
        self._plotter.render()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Stop the scene-loading thread, if any.

        Safe to call at application teardown for embedded widgets whose
        :meth:`closeEvent` is never delivered (children of a main window).
        """
        if self._thread is not None and self._thread.isRunning():
            if self._worker is not None:
                self._worker.loaded.disconnect(self._on_scene_loaded)
                self._worker.failed.disconnect(self._on_scene_failed)
                # deleteLater() must be posted while the worker's event loop is
                # still live, otherwise the DeferredDelete event is never processed.
                self._worker.deleteLater()
            self._thread.quit()
            self._thread.wait(3000)
            # Never delete a thread that is still running (wait timed out);
            # destroying a live QThread is undefined behaviour.  In that case
            # the thread keeps running under its parent until it finishes.
            if self._thread.isFinished():
                self._thread.deleteLater()
            self._thread = None
            self._worker = None
