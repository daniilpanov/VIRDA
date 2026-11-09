"""Viewer scene collection: ready in-memory data -> renderable scene DTOs.

Turns the heterogeneous in-memory sources the GUI passes around
(``virda`` mesh/volume models, NumPy arrays, PyVista meshes, DTOs)
into uniform, display-ready DTOs.
"""

from typing import Any

import numpy as np
import pyvista as pv

from virda.models.ese_mesh import ESEMesh
from virda.models.mri_volume import MRIVolume
from virda.models.scalp_mesh import ScalpMesh
from virda_gui_new.dto.viewer3d import (
    LabeledPoints,
    MeshBundle,
    MeshSource,
    MeshWithPoints,
    PointSource,
    ViewerScene,
    VolumeSource,
)
from virda_gui_new.services.geometry import downsample, scene_placement, transform_points


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


def as_polydata(source: MeshSource) -> pv.PolyData:
    """Normalize an in-memory mesh source into :class:`pyvista.PolyData`.

    Accepts a :class:`pyvista.PolyData`, a ``(vertices, faces)`` pair, a
    :class:`virda.models.scalp_mesh.ScalpMesh` or
    :class:`virda.models.ese_mesh.ESEMesh` model.
    """
    if isinstance(source, pv.PolyData):
        return source.copy()

    if isinstance(source, (ScalpMesh, ESEMesh)):
        return polydata_from_triangles(source.vertices, source.faces)

    if isinstance(source, (tuple, list)) and len(source) == 2:
        vertices, faces = source
        return polydata_from_triangles(vertices, faces)

    raise ValueError(f"Unsupported mesh source: {type(source).__name__}")


def resolve_labeled_points(source: PointSource | None) -> LabeledPoints | None:
    """Validate a ready points DTO; ``None`` for absent or empty sets."""
    if source is None:
        return None
    if not isinstance(source, LabeledPoints):
        raise ValueError("Points must be a ready LabeledPoints DTO from the import layer")
    return source if len(source) > 0 else None


def _load_mesh_bundle(source: Any, points: PointSource | None = None) -> MeshBundle:
    """Split a mesh source (which may bundle its own points) into mesh and points.

    An explicit ``points`` argument wins over the points bundled inside a
    :class:`MeshWithPoints` source.
    """
    if isinstance(source, MeshWithPoints):
        points = source.points if points is None else points
        return MeshBundle(as_polydata(source.mesh), resolve_labeled_points(points))

    return MeshBundle(as_polydata(source), resolve_labeled_points(points))


def _load_volume_source(
    source: VolumeSource,
    downsample_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read an in-memory volume source into ``(data (float32, 3D), affine)``.

    Accepts a :class:`virda.models.mri_volume.MRIVolume` model or an
    already-loaded ``(data, affine)`` pair. ``downsample_stride`` reduces the
    voxel count for rendering only (see :func:`virda_gui_new.services.geometry.downsample`).
    """
    if isinstance(source, MRIVolume):
        data = np.asarray(source.data, dtype=np.float32)
        affine = np.asarray(source.affine, dtype=np.float64)
    elif isinstance(source, (tuple, list)) and len(source) == 2:
        data = np.asarray(source[0], dtype=np.float32)
        affine = np.asarray(source[1], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported volume source: {type(source).__name__}")

    if data.ndim == 4:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(f"Volume data must be 3D, got {data.ndim}D")

    if downsample_stride > 1:
        data, affine = downsample(data, affine, downsample_stride)

    return data, affine


def _build_volume(data: np.ndarray, affine: np.ndarray, mm_scene: bool) -> pv.ImageData:
    volume = pv.ImageData(dimensions=data.shape)
    volume.point_data["intensity"] = data.ravel(order="F")
    if mm_scene:
        volume.spacing = tuple(np.asarray(np.diag(affine)[:3], dtype=float))
        volume.origin = tuple(np.asarray(affine[:3, 3], dtype=float))
    return volume


def _place_labeled_points(
    points: LabeledPoints, mm_scene: bool, transform: np.ndarray
) -> LabeledPoints | None:
    """Move labeled points into the scene frame; ``None`` for empty sets."""
    if len(points) == 0:
        return None

    return (
        points
        if mm_scene
        else LabeledPoints(transform_points(points.points, transform), points.labels)
    )


def collect_viewer_scene(
    nifti: VolumeSource | None = None,
    fiducials: PointSource | None = None,
    scalp: MeshSource | MeshWithPoints | None = None,
    scalp_points: PointSource | None = None,
    ese: MeshSource | MeshWithPoints | None = None,
    ese_points: PointSource | None = None,
    downsample_stride: int = 1,
) -> ViewerScene:
    """Resolve a full advanced-viewer scene from in-memory sources.

    Normalizes meshes and labeled points and places everything into one
    coordinate frame using the volume affine;
    at least one source must be given.
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

    mm_scene = True
    volume: pv.ImageData | None = None
    transform: np.ndarray = np.eye(4)
    if nifti is not None:
        data, affine = _load_volume_source(nifti, downsample_stride)
        _, _, transform, mm_scene = scene_placement(affine)
        volume = _build_volume(data, affine, mm_scene)

    fiducial_scene = (
        _place_labeled_points(fiducials, mm_scene, transform) if fiducials is not None else None
    )
    scalp_bundle = (
        _load_mesh_bundle(scalp, scalp_points) if scalp is not None else MeshBundle(None, None)
    )
    ese_bundle = _load_mesh_bundle(ese, ese_points) if ese is not None else MeshBundle(None, None)

    if not mm_scene:
        for bundle in (scalp_bundle, ese_bundle):
            if bundle.mesh is not None:
                bundle.mesh.transform(transform, inplace=True)
            if bundle.points is not None:
                bundle.points = _place_labeled_points(bundle.points, mm_scene, transform)

    return ViewerScene(
        volume=volume,
        mm_scene=mm_scene,
        fiducials=fiducial_scene,
        scalp_mesh=scalp_bundle.mesh,
        scalp_points=scalp_bundle.points,
        ese_mesh=ese_bundle.mesh,
        ese_points=ese_bundle.points,
    )


def collect_volume_scene(source: VolumeSource, downsample_stride: int = 1) -> pv.ImageData:
    """Resolve a single volume source into a renderable :class:`pv.ImageData`."""
    data, affine = _load_volume_source(source, downsample_stride)
    return _build_volume(data, affine, scene_placement(affine)[3])


def collect_mesh_scene(
    source: MeshSource | MeshWithPoints, points: PointSource | None = None
) -> MeshBundle:
    """Resolve a single mesh source (optionally with labeled points)."""
    return _load_mesh_bundle(source, points)


def collect_points_scene(source: PointSource) -> LabeledPoints | None:
    """Resolve a single labeled point source into a renderable DTO."""
    return resolve_labeled_points(source)
