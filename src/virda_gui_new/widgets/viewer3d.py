"""Embeddable 3D viewer widgets that can be dropped into any layout.

Two widgets share the same core:

* :class:`BaseViewer3D` - renders a single scene object (a NIfTI volume, a
  mesh, or labeled points) with no controls;
* :class:`AdvancedViewer3D` - full VIRDA viewer: NIfTI volume, fiducials,
  scalp mesh and ESE mesh, each of which may carry labeled points. Its right
  panel holds checkboxes to toggle the NIfTI, the fiducials, the points and
  their labels, plus a Scalp <-> ESE surface switcher when both meshes are
  present.

The widgets are pure display components: they accept only ready, already
loaded data (:class:`~virda_gui_new.dto.viewer3d.LabeledPoints`, PyVista
meshes, ``virda`` models, NumPy arrays) - never file paths - and they never
read, import or modify source data themselves, nor manage any loading
threads.  Building the renderable scene from the passed sources is
outsourced to :mod:`virda_gui_new.services.viewer_scene` and happens
synchronously right before rendering; when slow loading appears it will be
threaded by the embedder via
:class:`virda_gui_new.services.task_runner.TaskRunner`.  Here only VTK
actors are built and layer visibility is managed.  The
:attr:`AdvancedViewer3D.sceneLoaded` / :attr:`AdvancedViewer3D.sceneFailed`
signals notify embedders when the scene is ready or the scene collection
failed.
"""

from collections.abc import Callable
from typing import Any

import pyvista as pv
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from virda_gui_new.dto.viewer3d import (
    LabeledPoints,
    MeshBundle,
    MeshSource,
    MeshWithPoints,
    PointSource,
    ViewerScene,
    VolumeSource,
)
from virda_gui_new.services.viewer_scene import (
    collect_mesh_scene,
    collect_points_scene,
    collect_viewer_scene,
    collect_volume_scene,
)

_FIDUCIAL_COLOR = "red"
_MESH_COLOR = "salmon"
_SCALP_POINTS_COLOR = "yellow"
_ESE_POINTS_COLOR = "magenta"
_FIDUCIAL_POINT_SIZE = 10
_MESH_POINTS_POINT_SIZE = 10


# ---------------------------------------------------------------------------
# Widget-level rendering helpers
# ---------------------------------------------------------------------------


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item is None:
            break
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _build_label_actor(plotter: QtInteractor, points: LabeledPoints) -> Any:
    return plotter.add_point_labels(
        points.points,
        list(points.labels) if points.labels is not None else [],
        font_size=12,
        text_color="white",
        show_points=False,
        shape="rounded_rect",
        shape_color="black",
        shape_opacity=0.65,
        always_visible=True,
    )


def _add_point_actor(
    plotter: QtInteractor, points: LabeledPoints, *, color: str, point_size: int
) -> Any:
    return plotter.add_points(
        points.points,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True,
    )


class BaseViewer3D(QWidget):
    """Minimal embeddable 3D viewer that renders a single scene object.

    Shows one thing at a time - a NIfTI volume (:meth:`show_nifti`), a mesh
    (:meth:`show_mesh`, optionally with labeled points on it), or a labeled
    point set (:meth:`show_points`) - with no panels or checkboxes.  Calling
    another ``show_*`` replaces the current content.
    """

    sceneLoaded = Signal(object)  # noqa: N815
    sceneFailed = Signal(str)  # noqa: N815

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        log: Callable[[str], None] = print,
        background_color: str = "black",
    ) -> None:
        super().__init__(parent)
        self._log = log

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(parent=self)
        self._plotter.set_background(background_color)
        layout.addWidget(self._plotter, 1)

        self._kind: str | None = None
        self._props: dict[str, Any] = {}
        self._actors: list[Any] = []

    def show_nifti(
        self,
        source: VolumeSource,
        *,
        downsample_stride: int = 1,
        cmap: str = "bone",
        opacity: str = "sigmoid",
        clim: tuple[float, float] | None = None,
    ) -> None:
        """Show a volume: an MRIVolume model or a ready (data, affine) pair."""
        self._render(
            "volume",
            lambda: collect_volume_scene(source, downsample_stride),
            props={"cmap": cmap, "opacity": opacity, "clim": clim},
        )

    def show_mesh(
        self,
        source: MeshSource | MeshWithPoints,
        points: PointSource | None = None,
        *,
        color: str = _MESH_COLOR,
        opacity: float = 1.0,
        wireframe: bool = False,
    ) -> None:
        """Show a mesh; bundle or pass its labeled points to draw them too."""
        self._render(
            "mesh",
            lambda: collect_mesh_scene(source, points),
            props={"color": color, "opacity": opacity, "wireframe": wireframe},
        )

    def show_points(self, source: PointSource, *, color: str = _FIDUCIAL_COLOR) -> None:
        """Show a labeled point set (fiducials, contact points, ...)."""
        self._render("points", lambda: collect_points_scene(source), props={"color": color})

    def clear_scene(self) -> None:
        """Remove the current content."""
        self._kind = None
        self._props = {}
        self._actors = []
        self._plotter.clear()

    def _render(self, kind: str, fn: Callable[[], Any], props: dict[str, Any]) -> None:
        """Collect the scene synchronously, then build its actors."""
        self._log(f"Loading {kind}...")
        try:
            scene = fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the embedder as a signal
            self._log(f"ERROR: viewer failed: {exc}")
            self.sceneFailed.emit(str(exc))
            return

        self._kind = kind
        self._props = props
        self._apply_scene(scene)
        self.sceneLoaded.emit(scene)

    def _apply_scene(self, scene: Any) -> None:
        self._plotter.clear()
        self._actors = []
        plotter, props = self._plotter, self._props

        if self._kind == "volume" and isinstance(scene, pv.ImageData):
            self._actors.append(
                plotter.add_volume(
                    scene,
                    cmap=props["cmap"],
                    opacity=props["opacity"],
                    clim=props["clim"],
                    mapper="smart",
                )
            )
        elif self._kind == "mesh" and isinstance(scene, MeshBundle):
            self._apply_mesh_bundle(scene, props)
        elif self._kind == "points" and isinstance(scene, LabeledPoints) and len(scene) > 0:
            self._actors.append(
                _add_point_actor(
                    plotter, scene, color=props["color"], point_size=_FIDUCIAL_POINT_SIZE
                )
            )
            if scene.labels is not None:
                self._actors.append(_build_label_actor(plotter, scene))

        plotter.add_axes(interactive=False)
        plotter.reset_camera()
        plotter.render()

    def _apply_mesh_bundle(self, bundle: MeshBundle, props: dict[str, Any]) -> None:
        if bundle.mesh is not None:
            self._actors.append(
                self._plotter.add_mesh(
                    bundle.mesh,
                    color=props["color"],
                    opacity=props["opacity"],
                    show_edges=props["wireframe"],
                    edge_color="gray",
                )
            )

        if bundle.points is not None:
            self._actors.append(
                _add_point_actor(
                    self._plotter,
                    bundle.points,
                    color=_FIDUCIAL_COLOR,
                    point_size=_FIDUCIAL_POINT_SIZE,
                )
            )
            if bundle.points.labels is not None:
                self._actors.append(_build_label_actor(self._plotter, bundle.points))


class AdvancedViewer3D(QWidget):
    """Full VIRDA 3D viewer: NIfTI + fiducials + scalp/ESE meshes with points.

    Accepts four optional ready-data sources - *nifti*, *fiducials*, *scalp*
    and *ese*, where each mesh may bundle its labeled points via
    :class:`MeshWithPoints` or take them through ``scalp_points`` /
    ``ese_points``. By default, only the fiducials plus the scalp mesh with
    its points and labels are shown; the right panel hosts checkboxes for the
    NIfTI, the fiducials, the points and the labels, and a Scalp <-> ESE
    switcher when both meshes are loaded. Scenes with a single meaningful
    object (mesh-only, volume-only) render directly with no panel.
    """

    sceneLoaded = Signal(object)  # noqa: N815
    sceneFailed = Signal(str)  # noqa: N815

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        log: Callable[[str], None] = print,
        background_color: str = "black",
        nifti: VolumeSource | None = None,
        fiducials: PointSource | None = None,
        scalp: MeshSource | MeshWithPoints | None = None,
        scalp_points: PointSource | None = None,
        ese: MeshSource | MeshWithPoints | None = None,
        ese_points: PointSource | None = None,
    ) -> None:
        super().__init__(parent)
        self._log = log

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._plotter = QtInteractor(parent=self)
        self._plotter.set_background(background_color)
        layout.addWidget(self._plotter, 1)

        self._panel = QGroupBox("Layers", self)
        self._panel_layout = QVBoxLayout(self._panel)
        self._panel_layout.setContentsMargins(4, 4, 4, 4)
        self._panel_layout.setSpacing(4)
        layout.addWidget(self._panel, 0)
        self._panel.setVisible(False)

        # Render state / actors
        self._mesh_opacity = 1.0
        self._wireframe = False
        self._scene: ViewerScene | None = None
        self._mode = "scalp"
        self._nifti_visible = False
        self._fiducials_visible = True
        self._points_visible = True
        self._labels_visible = True

        self._volume_actor: Any = None
        self._fid_points_actor: Any = None
        self._fid_labels_actor: Any = None
        self._mesh_actors: dict[str, Any] = {}
        self._mesh_points_actors: dict[str, Any] = {}
        self._mesh_label_actors: dict[str, Any] = {}

        if any((nifti, fiducials, scalp, ese)):
            self.set_scene(
                nifti=nifti,
                fiducials=fiducials,
                scalp=scalp,
                scalp_points=scalp_points,
                ese=ese,
                ese_points=ese_points,
            )

    def set_scene(
        self,
        *,
        nifti: VolumeSource | None = None,
        fiducials: PointSource | None = None,
        scalp: MeshSource | MeshWithPoints | None = None,
        scalp_points: PointSource | None = None,
        ese: MeshSource | MeshWithPoints | None = None,
        ese_points: PointSource | None = None,
        downsample_stride: int = 1,
        mesh_opacity: float = 1.0,
        wireframe: bool = False,
    ) -> None:
        """Load a new scene from the given ready-data sources (all optional)."""
        self._mesh_opacity = float(mesh_opacity)
        self._wireframe = bool(wireframe)
        self._scene = None
        self._reset_actors()
        self._log("Loading 3D scene...")

        try:
            scene = collect_viewer_scene(
                nifti=nifti,
                fiducials=fiducials,
                scalp=scalp,
                scalp_points=scalp_points,
                ese=ese,
                ese_points=ese_points,
                downsample_stride=downsample_stride,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the embedder as a signal
            self._fail(str(exc))
            return

        self._scene = scene
        self._build_scene(scene)
        self.sceneLoaded.emit(scene)

    def clear_scene(self) -> None:
        """Remove the current scene."""
        self._scene = None
        self._reset_actors()
        self._build_panel()  # no scene: empty (and hidden) panel
        self._plotter.clear()
        self._plotter.render()

    def _fail(self, message: str) -> None:
        self._log(f"ERROR: viewer failed: {message}")
        self.sceneFailed.emit(message)

    def _reset_actors(self) -> None:
        self._volume_actor = None
        self._fid_points_actor = None
        self._fid_labels_actor = None
        self._mesh_actors = {}
        self._mesh_points_actors = {}
        self._mesh_label_actors = {}

    def _build_scene(self, scene: ViewerScene) -> None:
        self._plotter.clear()
        self._reset_actors()

        self._nifti_visible = (
            scene.volume is not None
            and scene.fiducials is None
            and (scene.scalp_mesh is None and scene.ese_mesh is None)
        )
        self._fiducials_visible = scene.fiducials is not None
        self._points_visible = True
        self._labels_visible = True
        self._mode = "scalp" if scene.scalp_mesh is not None else "ese"

        if scene.fiducials is not None:
            self._fid_points_actor = _add_point_actor(
                self._plotter,
                scene.fiducials,
                color=_FIDUCIAL_COLOR,
                point_size=_FIDUCIAL_POINT_SIZE,
            )
            if scene.fiducials.labels is not None:
                self._fid_labels_actor = _build_label_actor(self._plotter, scene.fiducials)

        for key, mesh, points, color in (
            ("scalp", scene.scalp_mesh, scene.scalp_points, _SCALP_POINTS_COLOR),
            ("ese", scene.ese_mesh, scene.ese_points, _ESE_POINTS_COLOR),
        ):
            if mesh is not None:
                self._mesh_actors[key] = self._plotter.add_mesh(
                    mesh,
                    color=_MESH_COLOR,
                    opacity=self._mesh_opacity,
                    show_edges=self._wireframe,
                    edge_color="gray",
                )
            if points is not None:
                points_actor = None
                if len(points) > 0:
                    points_actor = _add_point_actor(
                        self._plotter, points, color=color, point_size=_MESH_POINTS_POINT_SIZE
                    )
                self._mesh_points_actors[key] = points_actor
                self._mesh_label_actors[key] = (
                    _build_label_actor(self._plotter, points) if points.labels is not None else None
                )

        if scene.volume is not None:
            self._volume_actor = self._plotter.add_volume(
                scene.volume, cmap="bone", opacity="sigmoid", mapper="smart"
            )

        self._build_panel()
        self._apply_visibility()
        self._plotter.add_axes(interactive=False)
        self._plotter.reset_camera()
        self._plotter.render()

    def _build_panel(self) -> None:
        _clear_layout(self._panel_layout)
        scene = self._scene
        has_controls = False

        if scene is None:
            self._panel.setVisible(False)
            return

        if scene.volume is not None and (scene.fiducials is not None or self._mesh_actors):
            self._add_check("Show NIfTI", self._nifti_visible, self._set_nifti_visible)
            has_controls = True
        if scene.fiducials is not None:
            self._add_check("Show fiducials", self._fiducials_visible, self._set_fiducials_visible)
            has_controls = True
        if scene.scalp_points is not None or scene.ese_points is not None:
            self._add_check("Show points", self._points_visible, self._set_points_visible)
            has_controls = True
        if (
            (scene.fiducials is not None and scene.fiducials.labels is not None)
            or (scene.scalp_points is not None and scene.scalp_points.labels is not None)
            or (scene.ese_points is not None and scene.ese_points.labels is not None)
        ):
            self._add_check("Show labels", self._labels_visible, self._set_labels_visible)
            has_controls = True
        if scene.scalp_mesh is not None and scene.ese_mesh is not None:
            surface_box = QGroupBox("Surface", self._panel)
            surface_layout = QVBoxLayout(surface_box)
            surface_layout.setContentsMargins(4, 4, 4, 4)
            surface_layout.setSpacing(2)
            scalp_radio = QRadioButton("Scalp", surface_box)
            ese_radio = QRadioButton("ESE", surface_box)
            scalp_radio.setChecked(self._mode == "scalp")
            ese_radio.setChecked(self._mode == "ese")
            scalp_radio.toggled.connect(lambda on: self._on_mode_radio("scalp", on))
            ese_radio.toggled.connect(lambda on: self._on_mode_radio("ese", on))
            surface_layout.addWidget(scalp_radio)
            surface_layout.addWidget(ese_radio)
            self._panel_layout.addWidget(surface_box)
            has_controls = True

        self._panel.setVisible(has_controls)
        if has_controls:
            self._panel_layout.addStretch(1)

    def _add_check(self, text: str, checked: bool, slot: Callable[[bool], None]) -> None:
        check = QCheckBox(text, self._panel)
        check.setChecked(checked)
        check.toggled.connect(slot)
        self._panel_layout.addWidget(check)

    def _set_nifti_visible(self, flag: bool) -> None:
        self._nifti_visible = bool(flag)
        self._apply_visibility()

    def _set_fiducials_visible(self, flag: bool) -> None:
        self._fiducials_visible = bool(flag)
        self._apply_visibility()

    def _set_points_visible(self, flag: bool) -> None:
        self._points_visible = bool(flag)
        self._apply_visibility()

    def _set_labels_visible(self, flag: bool) -> None:
        self._labels_visible = bool(flag)
        self._apply_visibility()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._apply_visibility()

    def _on_mode_radio(self, mode: str, checked: bool) -> None:
        if checked:
            self._set_mode(mode)

    def _apply_visibility(self) -> None:
        if self._scene is None:
            return

        if self._volume_actor is not None:
            self._volume_actor.SetVisibility(self._nifti_visible)

        scalp_active = self._mode == "scalp"
        ese_active = self._mode == "ese"
        for key, active in (("scalp", scalp_active), ("ese", ese_active)):
            mesh_actor = self._mesh_actors.get(key)
            if mesh_actor is not None:
                mesh_actor.SetVisibility(active)
            visible = active and self._points_visible
            points_actor = self._mesh_points_actors.get(key)
            if points_actor is not None:
                points_actor.SetVisibility(visible)
            label_actor = self._mesh_label_actors.get(key)
            if label_actor is not None:
                label_actor.SetVisibility(visible and self._labels_visible)

        if self._fid_points_actor is not None:
            self._fid_points_actor.SetVisibility(self._fiducials_visible)
        if self._fid_labels_actor is not None:
            self._fid_labels_actor.SetVisibility(self._fiducials_visible and self._labels_visible)

        self._plotter.render()
