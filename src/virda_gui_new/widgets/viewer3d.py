"""Embeddable 3D viewer widgets that can be dropped into any layout.

Two widgets share the same core:

* :class:`BaseViewer3D` — renders a single scene object (a NIfTI volume, a
  mesh, or labeled points) with no controls;
* :class:`AdvancedViewer3D` — full VIRDA viewer: NIfTI volume, fiducials,
  scalp mesh and ESE mesh, each of which may carry labeled points.  Its right
  panel holds checkboxes to toggle the NIfTI, the fiducials, the points and
  their labels, plus a Scalp <-> ESE surface switcher when both meshes are
  present.

Every input may be a file path or already-loaded data (NumPy arrays, PyVista
meshes, ``virda`` models), and each input is optional: a mesh-only or a
NIfTI-only scene renders directly without any extra checkboxes.  Sources are
normalized and placed in one coordinate frame on a worker thread (see
:func:`collect_viewer_scene`), then VTK actors are built on the GUI thread.
The :attr:`AdvancedViewer3D.sceneLoaded` / :attr:`AdvancedViewer3D.sceneFailed`
signals notify embedders when the scene is ready or the load failed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from virda.io.fiducial_helpers import load_fiducials
from virda.models.ese_mesh import ESEMesh
from virda.models.fiducial import Fiducials
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda_gui_new.dto.viewer3d import LabeledPoints

PointSource = Any  # LabeledPoints | (points, labels) | ndarray | str | Path
MeshSource = Any  # PolyData | (vertices, faces) | ScalpMesh | ESEMesh | str | Path

_FIDUCIAL_COLOR = "red"
_MESH_COLOR = "salmon"
_SCALP_POINTS_COLOR = "yellow"
_ESE_POINTS_COLOR = "magenta"
_FIDUCIAL_POINT_SIZE = 10
_MESH_POINTS_POINT_SIZE = 10

_COORD_KEYS = (
    "coords",
    "coordinates",
    "coordinate",
    "point",
    "position",
    "location",
    "xyz",
    "scalp_coords",
    "ese_coords",
)
_NAME_KEYS = ("electrode_id", "fiducial_id", "name", "label", "id")


# ---------------------------------------------------------------------------
# Frame placement: NIfTI affine <-> one shared scene frame
# ---------------------------------------------------------------------------


def downsample(data: np.ndarray, affine: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``data`` with a voxel ``stride`` and scale the affine accordingly."""
    if stride < 1:
        raise ValueError(f"downsample must be >= 1, got {stride}")
    if stride == 1:
        return data, affine
    sampled = data[::stride, ::stride, ::stride]
    scaled = affine @ np.diag([stride, stride, stride, 1.0])
    return sampled, scaled


def is_axis_aligned(affine: np.ndarray) -> bool:
    """True = no rotation or flip (the affine is a positive diagonal)."""
    linear = affine[:3, :3]
    return np.allclose(linear, np.diag(np.diag(linear))) and bool(np.all(np.diag(linear) > 0))


def scene_placement(affine: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return ``(spacing, origin, transform, mm_scene)`` for placing a scene.

    ``transform`` maps world coordinates into the scene frame: the identity in
    a world-millimeter scene, the inverse affine in a voxel-index scene.
    """
    if affine is not None and is_axis_aligned(affine):
        spacing = np.asarray(np.diag(affine)[:3], dtype=float)
        origin = np.asarray(affine[:3, 3], dtype=float)
        return spacing, origin, np.eye(4), True

    return (
        (np.ones(3), np.zeros(3), np.eye(4), True)
        if affine is None
        else (np.ones(3), np.zeros(3), np.linalg.inv(affine), False)
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine ``transform`` to an (N, 3) array of points."""
    return points @ transform[:3, :3].T + transform[:3, 3]


def as_labeled_points(source: PointSource) -> LabeledPoints:
    """Normalize any supported point source into :class:`LabeledPoints`.

    Accepts an existing :class:`LabeledPoints`, an ``(N, 3)`` NumPy array
    (unnamed points), an ``(array, labels)`` pair, or a ``.json`` / ``.csv`` /
    ``.tsv`` / ``.txt`` file path.
    """
    if isinstance(source, LabeledPoints):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        match path.suffix.lower():
            case ".json":
                return LabeledPoints.from_json(path)
            case ".npy":
                return LabeledPoints(np.load(path))
            case _:
                return LabeledPoints.from_table(path)
    if isinstance(source, np.ndarray):
        return LabeledPoints(source)
    is_pair = isinstance(source, (tuple, list)) and len(source) == 2
    if is_pair and not isinstance(source[0], (int, float)):
        points, labels = source
        return LabeledPoints(points, tuple(labels))
    raise ValueError(f"Unsupported point source: {type(source).__name__}")


def _faces_to_cells(faces: np.ndarray) -> np.ndarray:
    """Convert (M, 3) triangle indices into a PyVista ravelled face array."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    cells = np.empty((faces.shape[0], 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    return cells.ravel()


def polydata_from_triangles(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Build a :class:`pyvista.PolyData` from (N, 3) vertices and (M, 3) faces."""
    return pv.PolyData(np.asarray(vertices, dtype=np.float64), _faces_to_cells(faces))


def _load_mesh_poly(mesh_path: str | Path) -> pv.PolyData:
    loaded = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    return polydata_from_triangles(vertices, faces)


def as_polydata(source: MeshSource) -> pv.PolyData:
    """Normalize any supported mesh source into :class:`pyvista.PolyData`.

    Accepts a :class:`pyvista.PolyData`, a ``(vertices, faces)`` pair, a
    :class:`virda.models.scalp_mesh.ScalpMesh` or
    :class:`virda.models.ese_mesh.ESEMesh` model, or a mesh file path
    (``.ply`` / ``.stl`` / ``.obj`` / ... via trimesh).
    """
    if isinstance(source, pv.PolyData):
        return source.copy()
    if isinstance(source, (ScalpMesh, ESEMesh)):
        return polydata_from_triangles(source.vertices, source.faces)
    if isinstance(source, (str, Path)):
        return _load_mesh_poly(source)
    if isinstance(source, (tuple, list)) and len(source) == 2:
        vertices, faces = source
        return polydata_from_triangles(vertices, faces)
    raise ValueError(f"Unsupported mesh source: {type(source).__name__}")


@dataclass(frozen=True)
class MeshWithPoints:
    """A mesh together with labeled points lying on it (``points`` optional)."""

    mesh: MeshSource
    points: PointSource | None = None


@dataclass
class _MeshBundle:
    mesh: pv.PolyData | None
    points: LabeledPoints | None


def _load_mesh_bundle(source: Any, points: PointSource | None = None) -> _MeshBundle:
    """Split a mesh source (which may bundle its own points) into mesh + points.

    An explicit ``points`` argument wins over the points bundled inside a
    :class:`MeshWithPoints` source.
    """
    if isinstance(source, MeshWithPoints):
        points = source.points if points is None else points
        return _MeshBundle(as_polydata(source.mesh), _load_points(points))
    return _MeshBundle(as_polydata(source), _load_points(points))


def _load_points(source: PointSource | None) -> LabeledPoints | None:
    if source is None:
        return None
    points = as_labeled_points(source)
    if len(points) == 0:
        return None
    return points


def _load_volume_source(
    source: Any, downsample_stride: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a NIfTI volume source into ``(data (float32, 3D), affine)``.

    Accepts a ``.nii`` / ``.nii.gz`` path, a :class:`virda.models.mri_volume.MRIVolume`
    model, or an already-loaded ``(data, affine)`` pair.
    """
    if isinstance(source, MRIVolume):
        data = np.asarray(source.data, dtype=np.float32)
        affine = np.asarray(source.affine, dtype=np.float64)
    elif isinstance(source, (str, Path)):
        image = nib.load(str(source))
        data = image.get_fdata(dtype=np.float32)
        affine = image.affine
    elif isinstance(source, (tuple, list)) and len(source) == 2:
        data = np.asarray(source[0], dtype=np.float32)
        affine = np.asarray(source[1], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported NIfTI source: {type(source).__name__}")
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"NIfTI data must be 3D, got {data.ndim}D")
    if downsample_stride > 1:
        if affine is None:
            raise ValueError("downsample_stride > 1 requires an affine")
        data, affine = downsample(data, affine, downsample_stride)
    return data, affine


def _normalize_fiducials(source: PointSource) -> tuple[LabeledPoints, str]:
    """Normalize a fiducials source into points plus a coordinate frame.

    Accepts a :class:`virda.models.fiducial.Fiducials` model (its primary
    coordinate system becomes the returned frame), a fiducials JSON file path
    (world millimeters per the project format), or anything understood by
    :func:`as_labeled_points` (assumed to be in world millimeters).  Labels
    follow the classic ``"ID (name)"`` form when a model or file supplies both.
    """
    if isinstance(source, Fiducials):
        points = np.asarray([item.coordinates for item in source.items], dtype=np.float64)
        labels = tuple(f"{item.fiducial_id} ({item.name})" for item in source.items)
        frames = {item.coordinate_system for item in source.items}
        frame = frames.pop() if len(frames) == 1 else "world"
        if len(frames):
            raise ValueError("Fiducials must share a single coordinate system")
        return LabeledPoints(points, labels), frame
    if isinstance(source, (str, Path)):
        fiducials = load_fiducials(Path(source))
        return _normalize_fiducials(fiducials)
    return as_labeled_points(source), "world"


# ---------------------------------------------------------------------------
# Scene construction (no Qt; safe to run on a worker thread)
# ---------------------------------------------------------------------------


@dataclass
class ViewerScene:
    """Fully resolved advanced-viewer scene, ready to be rendered.

    Everything is already placed in the scene frame: in a world-millimeter
    scene (``mm_scene``) the volume keeps its spacing/origin from the NIfTI
    affine, otherwise every array was moved into voxel index space.
    """

    volume: pv.ImageData | None = None
    mm_scene: bool = True
    fiducials: LabeledPoints | None = None
    scalp_mesh: pv.PolyData | None = None
    scalp_points: LabeledPoints | None = None
    ese_mesh: pv.PolyData | None = None
    ese_points: LabeledPoints | None = None


def _build_volume(
    data: np.ndarray | None, affine: np.ndarray | None, placement: tuple[Any, Any, np.ndarray, bool]
) -> pv.ImageData | None:
    if data is None:
        return None
    volume = pv.ImageData(dimensions=data.shape)
    volume.point_data["intensity"] = data.ravel(order="F")
    _, _, _, mm_scene = placement
    if mm_scene and affine is not None:
        volume.spacing = tuple(np.asarray(np.diag(affine)[:3], dtype=float))
        volume.origin = tuple(np.asarray(affine[:3, 3], dtype=float))
    return volume


def _place_labeled_points(
    points: LabeledPoints, placement: tuple[Any, Any, np.ndarray, bool], frame: str = "world"
) -> LabeledPoints | None:
    """Move labeled points into the scene frame; ``None`` for empty sets."""
    if len(points) == 0:
        return None
    _, _, transform, mm_scene = placement
    if frame == "voxel":
        if mm_scene:
            raise ValueError("voxel-coordinate points require a NIfTI with an affine")
        return points
    if not mm_scene:
        return LabeledPoints(transform_points(points.points, transform), points.labels)
    return points


def collect_viewer_scene(
    nifti: Any | None = None,
    fiducials: PointSource | None = None,
    scalp: MeshSource | MeshWithPoints | None = None,
    scalp_points: PointSource | None = None,
    ese: MeshSource | MeshWithPoints | None = None,
    ese_points: PointSource | None = None,
    downsample_stride: int = 1,
) -> ViewerScene:
    """Load and resolve a full advanced-viewer scene without touching any display.

    Runs on a worker thread: NIfTI reading (with optional voxel downsampling),
    mesh and point normalization, and placement of every object into one
    coordinate frame using the NIfTI affine.  At least one source must be
    given.
    """
    if (
        nifti is None
        and fiducials is None
        and scalp is None
        and ese is None
        and scalp_points is None
        and ese_points is None
    ):
        raise ValueError("no scene sources given: pass at least one of nifti/fiducials/scalp/ese")

    data: np.ndarray | None = None
    affine: np.ndarray | None = None
    if nifti is not None:
        data, affine = _load_volume_source(nifti, downsample_stride)
    placement = scene_placement(affine)
    volume = _build_volume(data, affine, placement)

    fiducial_scene: LabeledPoints | None = None
    if fiducials is not None:
        points, frame = _normalize_fiducials(fiducials)
        fiducial_scene = _place_labeled_points(points, placement, frame)

    scalp_bundle = (
        _load_mesh_bundle(scalp, scalp_points) if scalp is not None else _MeshBundle(None, None)
    )
    ese_bundle = _load_mesh_bundle(ese, ese_points) if ese is not None else _MeshBundle(None, None)
    _, _, transform, mm_scene = placement
    if not mm_scene:
        for bundle in (scalp_bundle, ese_bundle):
            if bundle.mesh is not None:
                bundle.mesh.transform(transform, inplace=True)
        for bundle in (scalp_bundle, ese_bundle):
            if bundle.points is not None:
                bundle.points = LabeledPoints(
                    transform_points(bundle.points.points, transform), bundle.points.labels
                )

    return ViewerScene(
        volume=volume,
        mm_scene=mm_scene,
        fiducials=fiducial_scene,
        scalp_mesh=scalp_bundle.mesh,
        scalp_points=scalp_bundle.points,
        ese_mesh=ese_bundle.mesh,
        ese_points=ese_bundle.points,
    )


def collect_volume_scene(source: Any, downsample_stride: int = 1) -> pv.ImageData:
    """Read a NIfTI volume source and place it in its own scene frame."""
    data, affine = _load_volume_source(source, downsample_stride)
    volume = _build_volume(data, affine, scene_placement(affine))
    if volume is None:  # pragma: no cover - _load_volume_source always returns data
        raise ValueError("failed to load the volume")
    return volume


# ---------------------------------------------------------------------------
# Background loading: a scene-collection callable on a dedicated thread
# ---------------------------------------------------------------------------


class _SceneLoaderWorker(QObject):
    """Runs a ``fn()`` collection callable; results come back to the GUI thread."""

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, fn: Callable[[], Any], seq: int) -> None:
        super().__init__()
        self._fn = fn
        self._seq = seq

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the embedder as a signal
            self.failed.emit(self._seq, str(exc))
        else:
            self.done.emit(self._seq, result)


class _SceneLoaderRunner(QObject):
    """Thread lifecycle for scene collection, shared by both viewer widgets.

    :meth:`start` launches ``fn()`` on a fresh worker thread; a newer start
    cancels the still-running previous load, and :meth:`stop` tears everything
    down (call from application teardown for embedded widgets whose
    ``closeEvent`` is never delivered).
    """

    done = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._seq = 0
        self._thread: QThread | None = None
        self._worker: _SceneLoaderWorker | None = None

    @property
    def seq(self) -> int:
        """Sequence number of the last requested load."""
        return self._seq

    def start(self, fn: Callable[[], Any]) -> None:
        self._seq += 1
        self.stop()
        thread = QThread(self)
        worker = _SceneLoaderWorker(fn, self._seq)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.done.connect(self.done)
        worker.failed.connect(self.failed)
        thread.finished.connect(worker.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def stop(self) -> None:
        """Retire a running load so nothing stale is delivered afterwards."""
        thread = self._thread
        if thread is None:
            return
        worker = self._worker
        if worker is not None:
            worker.done.disconnect(self.done)
            worker.failed.disconnect(self.failed)
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


# ---------------------------------------------------------------------------
# Widgets
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


class BaseViewer3D(QWidget):
    """Minimal embeddable 3D viewer that renders a single scene object.

    Shows one thing at a time — a NIfTI volume (:meth:`show_nifti`), a mesh
    (:meth:`show_mesh`, optionally with labeled points on it), or a labeled
    point set (:meth:`show_points`) — with no panels or checkboxes.  Calling
    another ``show_*`` replaces the current content.  Sources load on a worker
    thread; :attr:`sceneLoaded` and :attr:`sceneFailed` report the outcome.
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
        self._loader = _SceneLoaderRunner(self)
        self._loader.done.connect(self._on_scene_done)
        self._loader.failed.connect(self._on_scene_failed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(parent=self)
        self._plotter.set_background(background_color)
        layout.addWidget(self._plotter, 1)

        self._kind: str | None = None
        self._props: dict[str, Any] = {}
        self._actors: list[Any] = []

    # ---- public API ----

    def show_nifti(
        self,
        source: Any,
        *,
        downsample_stride: int = 1,
        cmap: str = "bone",
        opacity: str = "sigmoid",
        clim: tuple[float, float] | None = None,
    ) -> None:
        """Show a NIfTI volume: path, MRIVolume model or a (data, affine) pair."""
        self._start_load(
            "volume",
            lambda: collect_volume_scene(source, downsample_stride),
            props={"cmap": cmap, "opacity": opacity, "clim": clim},
        )

    def show_mesh(
        self,
        source: Any,
        *,
        color: str = _MESH_COLOR,
        opacity: float = 1.0,
        wireframe: bool = False,
    ) -> None:
        """Show a mesh; a :class:`MeshWithPoints` source also draws its points."""
        self._start_load(
            "mesh",
            lambda: _MeshBundle(
                as_polydata(source),
                _load_points(source.points if isinstance(source, MeshWithPoints) else None),
            ),
            props={"color": color, "opacity": opacity, "wireframe": wireframe},
        )

    def show_points(self, source: PointSource, *, color: str = _FIDUCIAL_COLOR) -> None:
        """Show a labeled point set (fiducials, contact points, ...)."""
        self._start_load("points", lambda: _load_points(source), props={"color": color})

    def clear_scene(self) -> None:
        """Remove the current content and cancel any in-flight load."""
        self._loader.stop()
        self._kind = None
        self._props = {}
        self._actors = []
        self._plotter.clear()

    def shutdown(self) -> None:
        """Stop the scene-loading thread; safe to call at application teardown."""
        self._loader.stop()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)

    # ---- internals ----

    def _start_load(self, kind: str, fn: Callable[[], Any], props: dict[str, Any]) -> None:
        self._kind = kind
        self._props = props
        self._log(f"Loading {kind}...")
        self._loader.start(fn)

    def _on_scene_done(self, seq: int, scene: Any) -> None:
        if seq != self._loader.seq:
            return
        self._apply_scene(scene)
        self.sceneLoaded.emit(scene)

    def _on_scene_failed(self, seq: int, message: str) -> None:
        if seq != self._loader.seq:
            return
        self._log(f"ERROR: viewer failed: {message}")
        self.sceneFailed.emit(message)

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
        elif self._kind == "mesh" and isinstance(scene, _MeshBundle) and scene.mesh is not None:
            self._actors.append(
                plotter.add_mesh(
                    scene.mesh,
                    color=props["color"],
                    opacity=props["opacity"],
                    show_edges=props["wireframe"],
                    edge_color="gray",
                )
            )
            if scene.points is not None:
                self._add_points_actors(scene.points, color=_FIDUCIAL_COLOR)
        elif self._kind == "points" and isinstance(scene, LabeledPoints) and len(scene) > 0:
            self._add_points_actors(scene, color=props["color"])
        plotter.add_axes(interactive=False)
        plotter.reset_camera()
        plotter.render()

    def _add_points_actors(self, points: LabeledPoints, color: str) -> None:
        actor = self._plotter.add_points(
            points.points,
            color=color,
            point_size=_FIDUCIAL_POINT_SIZE,
            render_points_as_spheres=True,
        )
        self._actors.append(actor)
        if points.labels is not None:
            self._actors.append(_build_label_actor(self._plotter, points))


class AdvancedViewer3D(QWidget):
    """Full VIRDA 3D viewer: NIfTI + fiducials + scalp/ESE meshes with points.

    Accepts four optional sources (file paths or loaded data) — *nifti*,
    *fiducials*, *scalp* and *ese*, where each mesh may bundle labeled points
    via :class:`MeshWithPoints` or take them through ``scalp_points`` /
    ``ese_points``.  By default only the fiducials plus the scalp mesh with
    its points and labels are shown; the right panel hosts checkboxes for the
    NIfTI, the fiducials, the points and the labels, and a Scalp <-> ESE
    switcher when both meshes are loaded.  Scenes with a single meaningful
    object (mesh-only, NIfTI-only) render directly with no panel.
    """

    sceneLoaded = Signal(object)  # noqa: N815
    sceneFailed = Signal(str)  # noqa: N815

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        log: Callable[[str], None] = print,
        background_color: str = "black",
        nifti: Any | None = None,
        fiducials: PointSource | None = None,
        scalp: MeshSource | MeshWithPoints | None = None,
        scalp_points: PointSource | None = None,
        ese: MeshSource | MeshWithPoints | None = None,
        ese_points: PointSource | None = None,
    ) -> None:
        super().__init__(parent)
        self._log = log
        self._loader = _SceneLoaderRunner(self)
        self._loader.done.connect(self._on_scene_done)
        self._loader.failed.connect(self._on_scene_failed)

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

    # ---- public API ----

    def set_scene(
        self,
        *,
        nifti: Any | None = None,
        fiducials: PointSource | None = None,
        scalp: MeshSource | MeshWithPoints | None = None,
        scalp_points: PointSource | None = None,
        ese: MeshSource | MeshWithPoints | None = None,
        ese_points: PointSource | None = None,
        downsample_stride: int = 1,
        mesh_opacity: float = 1.0,
        wireframe: bool = False,
    ) -> None:
        """Load a new scene from the given sources (everything is optional).

        The previous scene (and any in-flight load) is replaced; sources load
        on a worker thread and the result renders once
        :attr:`sceneLoaded` fires.
        """
        self._mesh_opacity = float(mesh_opacity)
        self._wireframe = bool(wireframe)
        self._scene = None
        self._reset_actors()
        self._loader.start(
            lambda: collect_viewer_scene(
                nifti=nifti,
                fiducials=fiducials,
                scalp=scalp,
                scalp_points=scalp_points,
                ese=ese,
                ese_points=ese_points,
                downsample_stride=downsample_stride,
            )
        )
        self._log("Loading 3D scene...")

    def clear_scene(self) -> None:
        """Remove the current scene and cancel any in-flight load."""
        self._loader.stop()
        self._scene = None
        self._reset_actors()
        self._build_panel()  # no scene: empty (and hidden) panel
        self._plotter.clear()
        self._plotter.render()

    def shutdown(self) -> None:
        """Stop the scene-loading thread; safe to call at application teardown."""
        self._loader.stop()

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.shutdown()
        super().closeEvent(event)

    # ---- scene result handling (GUI thread) ----

    def _on_scene_done(self, seq: int, scene: Any) -> None:
        if seq != self._loader.seq:
            return
        if not isinstance(scene, ViewerScene):  # pragma: no cover - defensive
            self._fail("unexpected scene result")
            return
        self._scene = scene
        self._build_scene(scene)
        self.sceneLoaded.emit(scene)

    def _on_scene_failed(self, seq: int, message: str) -> None:
        if seq != self._loader.seq:
            return
        self._fail(message)

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

    # ---- actor building ----

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
            self._fid_points_actor = self._plotter.add_points(
                scene.fiducials.points,
                color=_FIDUCIAL_COLOR,
                point_size=_FIDUCIAL_POINT_SIZE,
                render_points_as_spheres=True,
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
                    points_actor = self._plotter.add_points(
                        points.points,
                        color=color,
                        point_size=_MESH_POINTS_POINT_SIZE,
                        render_points_as_spheres=True,
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

    # ---- right panel ----

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

    # ---- visibility control ----

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
            self._apply_points_visibility(key, active)
        if self._fid_points_actor is not None:
            self._fid_points_actor.SetVisibility(self._fiducials_visible)
        if self._fid_labels_actor is not None:
            self._fid_labels_actor.SetVisibility(self._fiducials_visible and self._labels_visible)
        self._plotter.render()

    def _apply_points_visibility(self, key: str, active: bool) -> None:
        points_actor = self._mesh_points_actors.get(key)
        label_actor = self._mesh_label_actors.get(key)
        visible = active and self._points_visible
        if points_actor is not None:
            points_actor.SetVisibility(visible)
        if label_actor is not None:
            label_actor.SetVisibility(visible and self._labels_visible)
