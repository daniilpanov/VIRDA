"""Embeddable 3D viewer widget.

:class:`BaseViewer3D` renders a single scene object.
"""

from collections.abc import Callable
from typing import Any, Protocol, cast

import pyvista as pv
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget
from pyvista import Actor
from pyvistaqt import QtInteractor

from virda_gui_new.dto.viewer3d import (
    LabeledPoints,
    MeshBundle,
    MeshSource,
    MeshWithPoints,
    PointSource,
    VolumeSource,
)
from virda_gui_new.services.viewer_scene import (
    collect_mesh_scene,
    collect_points_scene,
    collect_volume_scene,
)

_FIDUCIAL_COLOR = "red"
_MESH_COLOR = "salmon"
_SCALP_POINTS_COLOR = "yellow"
_ESE_POINTS_COLOR = "magenta"
_FIDUCIAL_POINT_SIZE = 10
_MESH_POINTS_POINT_SIZE = 10


def _add_mesh_actor(
    plotter: QtInteractor,
    mesh: pv.PolyData,
    *,
    color: str,
    opacity: float,
    wireframe: bool,
) -> Actor:
    return plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        show_edges=wireframe,
        edge_color="gray",
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

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(parent=self)
        self._plotter.set_background(background_color)  # type: ignore[arg-type]  # mypy bug
        self._layout.addWidget(self._plotter, 1)

    def clear_scene(self) -> None:
        """Remove everything the plotter currently draws."""
        self._plotter.clear()

    def _finalize_render(self) -> None:
        """Finish a render pass: axes indicator, camera fit and immediate draw."""
        self._plotter.add_axes(interactive=False)  # type: ignore[call-arg]  # mypy bug
        self._plotter.reset_camera()  # type: ignore[call-arg]  # mypy bug
        self._plotter.render()


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
        self._finalize_render()
        self.sceneLoaded.emit(scene)
