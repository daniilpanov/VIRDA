"""Embeddable 3D viewer widgets.

* :class:`BaseViewer3D` renders a single scene object;
* :class:`SceneViewer3D` renders the full VIRDA scene with a layers panel.
"""

from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvista import Actor
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from virda.geometry import apply_transform
from virda_gui_new.dto.viewer3d import (
    LabeledPoints,
    MeshBundle,
    MeshSource,
    MeshWithPoints,
    PointSource,
    SceneSources,
    ViewerScene,
    VolumeSource,
)
from virda_gui_new.services.camera import Bounds, camera_for_bounds, mesh_bounds
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
#: The color of the mesh edges.
_EDGE_COLOR = "gray"
_MESH_POINTS_POINT_SIZE = 10
#: How transparent the meshes are drawn by default: the ESE surface and
#: the electrode points stay visible through the scalp.
_DEFAULT_MESH_OPACITY = 0.6
#: How far from the surface a click may land, as a fraction of the
#: window diagonal (the VTK pick tolerance of the cell picker).
_PICK_TOLERANCE = 0.025


#: The state of a camera that has to survive a scene rebuild: its
#: position, its focal point, its up vector, its view angle and whether
#: it draws in parallel.
CameraState = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    bool,
]


def _camera_state(camera: Any) -> CameraState:
    """The state of a camera that has to survive a scene rebuild.

    Only the values pyvista reads back from VTK are used: ``azimuth`` and
    ``elevation`` are pyvista's own cached copies and do not follow the
    mouse.
    """
    return (
        tuple(camera.position),
        tuple(camera.focal_point),
        tuple(camera.up),
        float(camera.view_angle),
        bool(camera.parallel_projection),
    )


def _apply_camera_state(camera: Any, state: CameraState) -> None:
    """Restore a camera state captured by :func:`_camera_state`."""
    position, focal_point, up, view_angle, parallel_projection = state
    camera.position = position
    camera.focal_point = focal_point
    camera.up = up
    camera.view_angle = view_angle
    camera.parallel_projection = parallel_projection


def _add_mesh_actor(
    plotter: QtInteractor,
    mesh: pv.PolyData,
    *,
    color: str,
    opacity: float,
    wireframe: bool,
    pickable: bool = False,
) -> Actor:
    return plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        show_edges=wireframe,
        edge_color=_EDGE_COLOR,
        pickable=pickable,
    )


def _add_volume_actor(
    plotter: QtInteractor,
    volume: pv.ImageData,
    *,
    cmap: str = "bone",
    opacity: str = "sigmoid",
    clim: float | tuple[float, float] | None = None,
) -> Actor:
    # pyvista's stubs narrow cmap/opacity to Literal unions (a snapshot of
    # mpl names), while the plotter accepts any mpl cmap string at runtime -
    # keep the public API `str` and trust pyvista here.
    kwargs: dict[str, Any] = {"cmap": cmap, "opacity": opacity, "clim": clim}
    return cast(Actor, plotter.add_volume(volume, **kwargs))


def _add_point_actor(
    plotter: QtInteractor, points: LabeledPoints, *, color: str, point_size: int
) -> Actor:
    return plotter.add_points(
        points.points,
        color=color,
        point_size=point_size,
        render_points_as_spheres=True,
        pickable=False,
    )


def _build_label_actor(plotter: QtInteractor, points: LabeledPoints) -> Actor:
    return cast(
        Actor,
        plotter.add_point_labels(
            points.points,
            list(points.labels) if points.labels is not None else [],
            font_size=12,
            text_color="white",
            show_points=False,
            shape="rounded_rect",
            shape_color="black",
            shape_opacity=0.65,
            always_visible=True,
        ),
    )


def _bundle_bounds(scene: object) -> Bounds | None:
    """The bounding box of a collected mesh bundle, if it holds a mesh."""
    if isinstance(scene, MeshBundle) and scene.mesh is not None:
        return mesh_bounds([scene.mesh])
    return None


class _SceneRenderer(Protocol):
    """One scene kind: collect its data and build its actors."""

    label: str

    def collect(self) -> Any:
        """Collect the renderable scene (data is already ready in memory)."""
        ...

    def build(self, plotter: QtInteractor, scene: Any) -> None:
        """Build the VTK actors of the collected scene."""
        ...


class _VolumeRenderer:
    """Renderer for a single NIfTI volume scene."""

    label = "volume"

    def __init__(
        self,
        source: VolumeSource,
        *,
        downsample_stride: int = 1,
        cmap: str = "bone",
        opacity: str = "sigmoid",
        clim: float | tuple[float, float] | None = None,
    ) -> None:
        self.source = source
        self.downsample_stride = downsample_stride
        self.cmap = cmap
        self.opacity = opacity
        self.clim = clim

    def collect(self) -> pv.ImageData:
        return collect_volume_scene(self.source, self.downsample_stride)

    def build(self, plotter: QtInteractor, scene: pv.ImageData) -> None:
        _add_volume_actor(
            plotter,
            scene,
            cmap=self.cmap,
            opacity=self.opacity,
            clim=self.clim,
        )


class _MeshRenderer:
    """Renderer for a mesh, optionally with labeled points on it."""

    label = "mesh"

    def __init__(
        self,
        source: MeshSource | MeshWithPoints,
        points: PointSource | None = None,
        *,
        color: str = _MESH_COLOR,
        opacity: float = 1.0,
        wireframe: bool = False,
    ) -> None:
        self.source = source
        self.points = points
        self.color = color
        self.opacity = opacity
        self.wireframe = wireframe

    def collect(self) -> MeshBundle:
        return collect_mesh_scene(self.source, self.points)

    def build(self, plotter: QtInteractor, scene: MeshBundle) -> None:
        if scene.mesh is not None:
            _add_mesh_actor(
                plotter,
                scene.mesh,
                color=self.color,
                opacity=self.opacity,
                wireframe=self.wireframe,
            )

        if scene.points is not None:
            _add_point_actor(
                plotter,
                scene.points,
                color=_FIDUCIAL_COLOR,
                point_size=_FIDUCIAL_POINT_SIZE,
            )
            if scene.points.labels is not None:
                _build_label_actor(plotter, scene.points)


class _PointsRenderer:
    """Renderer for a labeled point set."""

    label = "points"

    def __init__(self, source: PointSource, *, color: str = _FIDUCIAL_COLOR) -> None:
        self.source = source
        self.color = color

    def collect(self) -> LabeledPoints | None:
        return collect_points_scene(self.source)

    def build(self, plotter: QtInteractor, scene: LabeledPoints | None) -> None:
        if scene is None:
            return

        _add_point_actor(
            plotter,
            scene,
            color=self.color,
            point_size=_FIDUCIAL_POINT_SIZE,
        )
        if scene.labels is not None:
            _build_label_actor(plotter, scene)


class _SceneViewBase(QWidget):
    """Shared plumbing: plotter assembly, signals, sync scene resolution."""

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

        self._outer_layout = QVBoxLayout(self)
        self._outer_layout.setContentsMargins(0, 0, 0, 0)
        self._layout = QHBoxLayout()
        self._outer_layout.addLayout(self._layout, 1)
        self._plotter = QtInteractor(parent=self)
        self._plotter.set_background(background_color)  # type: ignore[arg-type]  # mypy bug
        self._layout.addWidget(self._plotter, 1)
        self._camera_ready = False

    def clear_scene(self) -> None:
        """Remove everything the plotter currently draws."""
        self._plotter.clear()

    def _finalize_render(
        self,
        camera_state: CameraState | None = None,
        bounds: Bounds | None = None,
    ) -> None:
        """Finish a render pass: axes indicator, camera and immediate draw.

        ``camera_state`` restores the camera of the previous scene; without
        it the camera frames the meshes of the scene that was just built.
        """
        self._plotter.add_axes(interactive=False)  # type: ignore[call-arg]  # mypy bug
        if camera_state is None:
            self._apply_camera(bounds)
        else:
            _apply_camera_state(self._plotter.camera, camera_state)
        self._camera_ready = True
        self._plotter.render()

    def _apply_camera(self, bounds: Bounds | None) -> None:
        """Place the camera on the given bounds, or fit everything drawn."""
        if bounds is None:
            self._plotter.reset_camera()  # type: ignore[call-arg]  # mypy bug
            return
        position, focal_point, up, view_angle = camera_for_bounds(bounds)
        camera = self._plotter.camera
        camera.position = position
        camera.focal_point = focal_point
        camera.up = up
        camera.view_angle = view_angle

    def _camera_before_rebuild(self) -> CameraState | None:
        """The camera state to restore after the next scene rebuild.

        ``None`` before the first scene: there is no camera to keep yet.
        """
        if not self._camera_ready:
            return None
        return _camera_state(self._plotter.camera)

    def _reset_camera(self) -> None:
        """Frame the meshes again, as the "Reset view" button does."""
        self._apply_camera(self._bounds())
        self._plotter.render()

    def _bounds(self) -> Bounds | None:
        """The bounding box of the meshes the camera frames."""
        return None


class BaseViewer3D(_SceneViewBase):
    """Minimal embeddable viewer showing one scene object at a time.

    A NIfTI volume, a mesh with optional labeled points on it, or a standalone
    labeled point set; a newer ``show_*`` call replaces the previous content.
    """

    def show_nifti(
        self,
        source: VolumeSource,
        *,
        downsample_stride: int = 1,
        cmap: str = "bone",
        opacity: str = "sigmoid",
        clim: tuple[float, float] | None = None,
    ) -> None:
        """Show an MRIVolume model or a ready (data, affine) pair."""
        self._render(
            _VolumeRenderer(
                source,
                downsample_stride=downsample_stride,
                cmap=cmap,
                opacity=opacity,
                clim=clim,
            )
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
        """Show a mesh with optional labeled points (bundled or explicit)."""
        self._render(
            _MeshRenderer(source, points, color=color, opacity=opacity, wireframe=wireframe)
        )

    def show_points(self, source: PointSource, *, color: str = _FIDUCIAL_COLOR) -> None:
        """Show a labeled point set."""
        self._render(_PointsRenderer(source, color=color))

    def _render(self, renderer: _SceneRenderer) -> None:
        """Collect the scene synchronously, then build its actors."""
        self._log(f"Loading {renderer.label}...")

        try:
            scene = renderer.collect()
        except Exception as exc:  # noqa: BLE001 - surfaced to the embedder as a signal
            self._log(f"ERROR: viewer failed: {exc}")
            self.sceneFailed.emit(str(exc))
            return

        self._plotter.clear()
        renderer.build(self._plotter, scene)
        self._finalize_render(bounds=_bundle_bounds(scene))
        self.sceneLoaded.emit(scene)


class SceneViewer3D(_SceneViewBase):
    """Full VIRDA 3D viewer showing a whole scene at once.

    A NIfTI volume, fiducials, scalp and ESE mesh groups with optional
    labeled points, and a layers panel that toggles their visibility
    (with a Scalp <-> ESE surface switcher when both meshes are
    present). :meth:`start_surface_picking` arms the picker for one
    fiducial slot and :attr:`surfacePointPicked` reports the picked point
    in world millimetres. A click that moves the camera is a rotation,
    not a pick.
    """

    surfacePointPicked = Signal(str, object)  # noqa: N815 - the slot and the point

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        log: Callable[[str], None] = print,
        background_color: str = "black",
        sources: SceneSources | None = None,
    ) -> None:
        super().__init__(parent, log=log, background_color=background_color)

        self._layout.setSpacing(4)
        self._panel = QGroupBox("Layers", self)
        self._panel_layout = QVBoxLayout(self._panel)
        self._panel_layout.setContentsMargins(4, 4, 4, 4)
        self._panel_layout.setSpacing(4)
        self._layout.addWidget(self._panel, 0)
        self._panel.setVisible(False)
        # The reset button lives outside the layers panel: the panel is
        # rebuilt with every scene, the button is not.
        self._reset_button = QPushButton("Reset view", self)
        self._reset_button.clicked.connect(self._reset_camera)
        self._outer_layout.addWidget(self._reset_button)

        # Render state / actors
        self._mesh_opacity = _DEFAULT_MESH_OPACITY
        self._wireframe = False
        self._scene: ViewerScene | None = None
        self._mode = "scalp"
        self._nifti_visible = False
        self._fiducials_visible = True
        self._electrodes_visible = True
        self._labels_visible = True
        # Stable across scene rebuilds: they apply to whichever surface is
        # active, so a scalp <-> ESE switch must not reset them.
        self._mesh_visible = True

        self._volume_actor: Actor | None = None
        self._fid_points_actor: Actor | None = None
        self._fid_labels_actor: Actor | None = None
        self._mesh_actors: dict[str, Actor] = {}
        self._mesh_points_actors: dict[str, Actor | None] = {}
        self._mesh_label_actors: dict[str, Actor | None] = {}

        # Surface picking: the slot being picked and the camera before the
        # click (a click that moved the camera is a rotation, not a pick).
        self._pick_slot: str | None = None
        self._pick_camera: CameraState | None = None
        interactor = self._plotter.iren
        interactor.add_observer("LeftButtonPressEvent", self._on_pick_press)
        interactor.add_observer("LeftButtonReleaseEvent", self._on_pick_release)
        interactor.add_key_event("Escape", self.stop_surface_picking)
        # The cell picker returns the point on the surface that was
        # clicked, not the closest vertex; its tolerance is a fraction
        # of the window diagonal.
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(_PICK_TOLERANCE)
        self._picker.PickFromListOn()

        if sources is not None and not sources.is_empty():
            self.set_scene(sources)

    def set_scene(
        self,
        sources: SceneSources | None = None,
        *,
        downsample_stride: int = 1,
        mesh_opacity: float | None = None,
        wireframe: bool | None = None,
    ) -> None:
        """Load a new scene from the given sources.

        The mesh edges keep their state when ``wireframe`` is not given,
        so a rebuilt scene does not drop what the user switched on. The
        mesh opacity does the same: the value the user chose survives a
        rebuilt scene.
        """
        if mesh_opacity is not None:
            self._mesh_opacity = float(mesh_opacity)
        if wireframe is not None:
            self._wireframe = bool(wireframe)
        self._scene = None
        self._reset_actors()
        self._log("Loading 3D scene...")

        try:
            scene = collect_viewer_scene(
                sources if sources is not None else SceneSources(),
                downsample_stride=downsample_stride,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the embedder as a signal
            self._log(f"ERROR: viewer failed: {exc}")
            self.sceneFailed.emit(str(exc))
            return

        self._scene = scene
        self._build_scene(scene)
        self.sceneLoaded.emit(scene)

    def clear_scene(self) -> None:
        """Remove the current scene."""
        self._scene = None
        self._reset_actors()
        self._build_panel()  # no scene: empty (and hidden) panel
        super().clear_scene()
        self._plotter.render()

    # --- surface picking ---

    def start_surface_picking(self, slot: str) -> None:
        """Arm the picker for one fiducial slot on the scalp surface.

        The scalp surface is shown and the ESE surface hidden, so the
        click cannot land on the wrong mesh. The mode ends with the first
        pick, with Escape, or with another :meth:`stop_surface_picking`.
        """
        self._pick_slot = slot
        self._pick_camera = None
        self._set_mode("scalp")
        self._build_panel()
        self._plotter.render()

    def stop_surface_picking(self) -> None:
        """Leave the surface picking mode."""
        if self._pick_slot is None:
            return
        self._pick_slot = None
        self._pick_camera = None
        self._build_panel()
        self._plotter.render()

    def _on_pick_press(self, *_args: object) -> None:
        """Remember the camera before a possible rotation."""
        if self._pick_slot is None:
            return
        self._pick_camera = _camera_state(self._plotter.camera)

    def _on_pick_release(self, *_args: object) -> None:
        """Pick the point under the cursor when the camera did not move."""
        slot = self._pick_slot
        if slot is None or self._pick_camera is None:
            return
        if _camera_state(self._plotter.camera) != self._pick_camera:
            # The camera moved: this was a rotation, not a click.
            return
        self._pick_camera = None
        point = self._pick_surface_point()
        if point is None:
            return
        self.stop_surface_picking()
        self.surfacePointPicked.emit(slot, point)

    def _arm_picker(self) -> None:
        """Let the picker hit the scalp surface only.

        The pick list holds the scalp actor, so a click on the volume
        or on the ESE surface picks nothing.
        """
        self._picker.InitializePickList()
        scalp_actor = self._mesh_actors.get("scalp")
        if scalp_actor is not None:
            self._picker.AddPickList(scalp_actor)

    def _pick_surface_point(self) -> np.ndarray | None:
        """The picked point on the scalp surface, in world millimetres."""
        interactor = self._plotter.iren
        renderer = interactor.get_poked_renderer()
        x, y = interactor.get_event_position()
        picker = self._picker
        if not picker.Pick(x, y, 0, renderer):
            return None
        if picker.GetActor() is None:
            return None
        scene_point = np.asarray(picker.GetPickPosition(), dtype=np.float64)
        return self._world_point(scene_point)

    def _world_point(self, point: np.ndarray) -> np.ndarray:
        """The point of the scene frame in world millimetres."""
        scene = self._scene
        if scene is None:
            return point
        transform = scene.world_from_scene
        return apply_transform(point, transform)

    def _reset_actors(self) -> None:
        self._volume_actor = None
        self._fid_points_actor = None
        self._fid_labels_actor = None
        self._mesh_actors = {}
        self._mesh_points_actors = {}
        self._mesh_label_actors = {}

    def _build_scene(self, scene: ViewerScene) -> None:
        camera_state = self._camera_before_rebuild()
        self._plotter.clear()
        self._reset_actors()

        self._nifti_visible = (
            scene.volume is not None
            and scene.fiducials is None
            and (scene.scalp_mesh is None and scene.ese_mesh is None)
        )
        self._fiducials_visible = scene.fiducials is not None
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
                self._mesh_actors[key] = _add_mesh_actor(
                    self._plotter,
                    mesh,
                    color=_MESH_COLOR,
                    opacity=self._mesh_opacity,
                    wireframe=self._wireframe,
                    # Only the scalp surface carries the fiducial marks.
                    pickable=key == "scalp",
                )
            if points is not None:
                points_actor: Actor | None = None
                if len(points) > 0:
                    points_actor = _add_point_actor(
                        self._plotter,
                        points,
                        color=color,
                        point_size=_MESH_POINTS_POINT_SIZE,
                    )
                self._mesh_points_actors[key] = points_actor
                self._mesh_label_actors[key] = (
                    _build_label_actor(self._plotter, points) if points.labels is not None else None
                )

        if scene.volume is not None:
            self._volume_actor = _add_volume_actor(
                self._plotter,
                scene.volume,
            )

        self._build_panel()
        self._apply_visibility()
        self._arm_picker()
        self._finalize_render(camera_state, self._scene_bounds(scene))

    def _scene_bounds(self, scene: ViewerScene) -> Bounds | None:
        """The bounding box of the scene meshes; the volume is not one."""
        meshes = [mesh for mesh in (scene.scalp_mesh, scene.ese_mesh) if mesh is not None]
        return mesh_bounds(meshes)

    def _bounds(self) -> Bounds | None:
        """The bounding box of the meshes the camera frames."""
        scene = self._scene
        if scene is None:
            return None
        return self._scene_bounds(scene)

    def _clear_layout(self) -> None:
        """Remove the widgets of the panel.

        The panel is rebuilt with every scene, so every widget it holds
        has to go: a leftover slider would stay on the panel and
        overlap the new one. The widgets are plain items of the panel:
        a widget of a nested layout outlives its layout when the
        deferred deletion runs.
        """
        while self._panel_layout.count():
            item = self._panel_layout.takeAt(0)
            if item is None:
                return

            widget = item.widget()
            if widget is not None:
                # The widget leaves the panel before it is deleted: a
                # deferred deletion of a widget the panel still owns
                # frees it twice and crashes.
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()

    def _build_panel(self) -> None:
        self._clear_layout()

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
            self._add_check(
                "Show electrodes", self._electrodes_visible, self._set_electrodes_visible
            )
            has_controls = True

        if (
            (scene.fiducials is not None and scene.fiducials.labels is not None)
            or (scene.scalp_points is not None and scene.scalp_points.labels is not None)
            or (scene.ese_points is not None and scene.ese_points.labels is not None)
        ):
            self._add_check("Show labels", self._labels_visible, self._set_labels_visible)
            has_controls = True

        if self._mesh_actors:
            # Near the surface group: it acts on whichever surface is active.
            self._add_check("Show mesh", self._mesh_visible, self._set_mesh_visible)
            self._add_check("Show mesh edges", self._wireframe, self._set_mesh_edges)
            self._add_opacity_slider()
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

        if self._pick_slot is not None:
            hint = QLabel(f"Picking: {self._pick_slot} - click the scalp", self._panel)
            self._panel_layout.addWidget(hint)
            has_controls = True

        self._panel.setVisible(has_controls)
        if has_controls:
            self._panel_layout.addStretch(1)

    def _add_check(self, text: str, checked: bool, slot: Callable[[bool], None]) -> None:
        check = QCheckBox(text, self._panel)
        check.setChecked(checked)
        check.toggled.connect(slot)
        self._panel_layout.addWidget(check)

    def _add_opacity_slider(self) -> None:
        """Add the mesh opacity slider to the layers panel.

        The label and the slider are plain items of the panel: a nested
        layout would outlive its widgets when the panel is rebuilt.
        """
        label = QLabel("Mesh opacity", self._panel)
        self._panel_layout.addWidget(label)
        slider = QSlider(Qt.Orientation.Horizontal, self._panel)
        slider.setRange(0, 100)
        slider.setValue(int(round(self._mesh_opacity * 100)))
        slider.valueChanged.connect(self._set_mesh_opacity)
        self._panel_layout.addWidget(slider)

    def _set_mesh_opacity(self, value: int) -> None:
        """Change how transparent the meshes are drawn."""
        self._mesh_opacity = max(0.0, min(1.0, value / 100))
        for actor in self._mesh_actors.values():
            actor.GetProperty().SetOpacity(self._mesh_opacity)
        self._plotter.render()

    def _set_nifti_visible(self, flag: bool) -> None:
        self._nifti_visible = bool(flag)
        self._apply_visibility()

    def _set_fiducials_visible(self, flag: bool) -> None:
        self._fiducials_visible = bool(flag)
        self._apply_visibility()

    def _set_electrodes_visible(self, flag: bool) -> None:
        self._electrodes_visible = bool(flag)
        self._apply_visibility()

    def _set_mesh_visible(self, flag: bool) -> None:
        self._mesh_visible = bool(flag)
        self._apply_visibility()

    def _set_mesh_edges(self, flag: bool) -> None:
        """Show or hide the edges of the mesh."""
        self._wireframe = bool(flag)
        for actor in self._mesh_actors.values():
            actor.GetProperty().SetEdgeVisibility(self._wireframe)
            actor.GetProperty().SetEdgeColor(0.5, 0.5, 0.5)
        self._plotter.render()

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
                mesh_actor.SetVisibility(active and self._mesh_visible)

            visible = active and self._electrodes_visible
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
