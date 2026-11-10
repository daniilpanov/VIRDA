"""Viewer scene collection: ready in-memory data -> display-ready scene DTOs."""

from typing import Any

import numpy as np
import pyvista as pv

from virda.models.mesh import Mesh
from virda.models.mri_volume import MRIVolume
from virda_gui_new.dto.viewer3d import (
    LabeledPoints,
    MeshBundle,
    MeshSource,
    MeshWithPoints,
    PointSource,
    VolumeSource,
)
from virda_gui_new.services.geometry import downsample, scene_placement


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
    """Normalize a mesh source into :class:`pyvista.PolyData`.

    Accepts a :class:`pyvista.PolyData`, a ``(vertices, faces)`` pair, or a
    :class:`virda.models.mesh.Mesh` model.
    """
    if isinstance(source, pv.PolyData):
        return source.copy()

    if isinstance(source, Mesh):
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

    An explicit ``points`` argument overrides points bundled in a
    ``MeshWithPoints`` source.
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

    Accepts an :class:`virda.models.mri_volume.MRIVolume` model or a ready
    ``(data, affine)`` pair.
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
