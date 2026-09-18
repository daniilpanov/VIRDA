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

import pyvista as pv
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

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
        self._fiducials_visible = True
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

        self._volume = scene.volume
        self._mesh_opacity = scene.mesh_opacity
        self._hi_clim = scene.hi_clim
        self._mri_actor = None
        self._mesh_actor = None
        if scene.volume is not None:
            self._mri_actor = self._plotter.add_volume(
                scene.volume, cmap="bone", opacity="sigmoid", mapper="smart"
            )
        if scene.scene_mesh is not None:
            self._mesh_actor = self._plotter.add_mesh(
                scene.scene_mesh, color="salmon", opacity=self._mesh_opacity
            )

        self._fiducial_actor = None
        self._fiducial_label_actor = None
        if scene.fiducial_points is not None and len(scene.fiducial_points) > 0:
            scene_points = scene.fiducial_points
            self._fiducial_actor = self._plotter.add_points(
                scene_points, color="red", point_size=10, render_points_as_spheres=True
            )
            self._fiducial_label_actor = self._plotter.add_point_labels(
                scene_points,
                scene.fiducial_labels,
                font_size=12,
                text_color="white",
                show_points=False,
                shape="rounded_rect",
                shape_color="black",
                shape_opacity=0.65,
                always_visible=True,
            )

        self._normals_actor = None
        if scene.normals_poly is not None:
            self._normals_actor = self._plotter.add_mesh(
                scene.normals_poly, color="cyan", opacity=0.8, line_width=2
            )

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
                healthy = ~flg
                if healthy.any():
                    e_actors.append(
                        self._plotter.add_points(
                            pts[healthy],
                            color=color,
                            point_size=12,
                            render_points_as_spheres=True,
                        )
                    )
                if (~healthy).any():
                    f_actors.append(
                        self._plotter.add_points(
                            pts[~healthy],
                            color=intensify_color(color),
                            point_size=17,
                            render_points_as_spheres=True,
                        )
                    )
                links = build_electrode_links(pts, meas, scene.fiducial_id_to_point)
                if len(links) > 0:
                    flat = links.reshape(-1, 3)
                    l_actors.append(self._plotter.add_lines(flat, color=color, width=1))
                n_actors.append(
                    self._plotter.add_point_labels(
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
                )
            self._electrode_actors_per_group.append(e_actors)
            self._flagged_actors_per_group.append(f_actors)
            self._link_actors_per_group.append(l_actors)
            self._label_actors_per_group.append(n_actors)

        self._group_states = [True] * len(scene.electrode_groups)
        self._mri_visible = True
        self._fiducials_visible = True
        self._labels_visible = True
        self._build_layers(scene)
        self._plotter.add_axes(interactive=False)
        self._plotter.render()

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
        if self._mesh_actor is not None:
            self._mesh_actor.SetVisibility(flag)

    def _set_mri_visibility(self, flag: bool) -> None:
        self._mri_visible = bool(flag)
        if self._mri_actor is not None:
            self._mri_actor.SetVisibility(flag)

    def _set_fiducials_visibility(self, flag: bool) -> None:
        self._fiducials_visible = bool(flag)
        if self._fiducial_actor is not None:
            self._fiducial_actor.SetVisibility(flag)
        self._apply_labels_visibility()

    def _set_normals_visibility(self, flag: bool) -> None:
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
            self._mri_actor.SetVisibility(self._mri_visible)
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
