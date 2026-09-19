"""Coordinate-frame math and frame-scoped exporters for the 3D viewer.

The interactive viewer renders its scene in one of three frames:

* ``scanner_ras`` -- native scanner RAS world coordinates in millimetres
  (the NIfTI affine world frame);
* ``voxel`` -- NIfTI voxel grid indices (the inverse affine applied to world
  points);
* ``cras`` -- FreeSurfer surface RAS / centred RAS, i.e. scanner RAS minus
  the volume-centre offset that
  :func:`virda_gui.viewer.viewer_loaders._cras_to_scanner_ras_offset` and the
  tabular electrode loader use for their cRAS -> scanner RAS conversion.

Everything in this module is Qt-free so the transforms and the exporters can
be unit tested without a display.  The scene data produced by
``collect_scene_data`` lives in its *natural* frame -- world mm when the
NIfTI affine is axis-aligned, voxel indices otherwise;
:func:`scene_to_frame_matrix` maps those scene points into any of the three
frames through a single code path, and every transform is reversible via
:func:`world_to_frame_matrix` / :func:`frame_to_world_matrix`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .scene import transform_points
from .viewer_loaders import SceneData

FRAME_SCANNER = "scanner_ras"
FRAME_VOXEL = "voxel"
FRAME_CRAS = "cras"
FRAME_IDS: tuple[str, str, str] = (FRAME_SCANNER, FRAME_VOXEL, FRAME_CRAS)

_FRAME_LABELS: dict[str, str] = {
    FRAME_SCANNER: "Scanner RAS (world mm)",
    FRAME_VOXEL: "Voxel indices",
    FRAME_CRAS: "FreeSurfer cRAS",
}


def frame_label(frame: str) -> str:
    """Human-readable name of *frame*, or the identifier itself when unknown."""
    return _FRAME_LABELS.get(frame, frame)


def natural_frame(mm_scene: bool) -> str:
    """The frame the scene data is already expressed in.

    World millimetres for axis-aligned NIfTI affines (and mesh-only scenes),
    voxel indices when the affine carries rotation or a flip.
    """
    return FRAME_SCANNER if mm_scene else FRAME_VOXEL


def frame_available(frame: str, affine: np.ndarray | None, cras_offset: np.ndarray | None) -> bool:
    """Whether the scene can be shown or exported in *frame*."""
    if frame == FRAME_VOXEL:
        return affine is not None
    if frame == FRAME_CRAS:
        return cras_offset is not None
    return True


def _validate_frame(frame: str) -> None:
    if frame not in FRAME_IDS:
        raise ValueError(f"unknown coordinate frame {frame!r}, expected one of {FRAME_IDS}")


def world_to_frame_matrix(
    frame: str, affine: np.ndarray | None, cras_offset: np.ndarray | None
) -> np.ndarray:
    """Return the 4x4 transform mapping scanner-RAS world-mm points into *frame*."""
    _validate_frame(frame)
    if frame == FRAME_VOXEL:
        if affine is None:
            raise ValueError("the voxel frame requires the NIfTI affine, which is not loaded")
        return np.asarray(np.linalg.inv(affine), dtype=np.float64)
    if frame == FRAME_CRAS:
        if cras_offset is None:
            raise ValueError("the cRAS frame requires the NIfTI cRAS offset, which is not loaded")
        matrix = np.eye(4)
        matrix[:3, 3] = -np.asarray(cras_offset, dtype=np.float64)
        return matrix
    return np.eye(4)


def frame_to_world_matrix(
    frame: str, affine: np.ndarray | None, cras_offset: np.ndarray | None
) -> np.ndarray:
    """Return the inverse of :func:`world_to_frame_matrix`, *frame* back to world mm."""
    _validate_frame(frame)
    if frame == FRAME_VOXEL:
        if affine is None:
            raise ValueError("the voxel frame requires the NIfTI affine, which is not loaded")
        return np.asarray(affine, dtype=np.float64)
    if frame == FRAME_CRAS:
        if cras_offset is None:
            raise ValueError("the cRAS frame requires the NIfTI cRAS offset, which is not loaded")
        matrix = np.eye(4)
        matrix[:3, 3] = np.asarray(cras_offset, dtype=np.float64)
        return matrix
    return np.eye(4)


def scene_to_world_matrix(affine: np.ndarray | None, mm_scene: bool) -> np.ndarray:
    """Return the 4x4 transform mapping natural-frame scene points to world mm."""
    if mm_scene or affine is None:
        return np.eye(4)
    return np.asarray(affine, dtype=np.float64)


def scene_to_frame_matrix(
    frame: str, affine: np.ndarray | None, cras_offset: np.ndarray | None, mm_scene: bool
) -> np.ndarray:
    """One shared code path: natural-frame scene points -> selected *frame* points.

    Composes the scene -> world conversion with
    :func:`world_to_frame_matrix`, so the display frames share the same math
    as the exporters and remain invertible.
    """
    return world_to_frame_matrix(frame, affine, cras_offset) @ scene_to_world_matrix(
        affine, mm_scene
    )


def frame_to_scene_matrix(
    frame: str, affine: np.ndarray | None, cras_offset: np.ndarray | None, mm_scene: bool
) -> np.ndarray:
    """Return the inverse of :func:`scene_to_frame_matrix`.

    Maps points typed in a user-selected frame (scanner RAS, voxel or cRAS)
    back into the scene's natural frame so live-edit rows and localization
    inputs land on the same mesh that the viewer renders.  Computed as the
    exact inverse so ``frame_to_scene @ scene_to_frame == I`` holds.
    """
    _validate_frame(frame)
    return np.linalg.inv(scene_to_frame_matrix(frame, affine, cras_offset, mm_scene))


def frame_to_scene_points(
    points: np.ndarray,
    frame: str,
    affine: np.ndarray | None,
    cras_offset: np.ndarray | None,
    mm_scene: bool,
) -> np.ndarray:
    """Map (N, 3) *points* expressed in *frame* into the scene's natural frame."""
    return transform_points(points, frame_to_scene_matrix(frame, affine, cras_offset, mm_scene))


def world_to_frame_points(
    frame: str, points: np.ndarray, affine: np.ndarray | None, cras_offset: np.ndarray | None
) -> np.ndarray:
    """Map scanner-RAS world-mm *points* (N, 3) into *frame*."""
    return transform_points(points, world_to_frame_matrix(frame, affine, cras_offset))


# ---- export serialization -------------------------------------------------


def triangle_faces(cells: np.ndarray) -> np.ndarray:
    """Convert PyVista's raveled ``[3 i j k, ...]`` cell array into an (M, 3) face array."""
    cells = np.asarray(cells, dtype=np.int64)
    if cells.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if cells.ndim != 1 or cells.size % 4 != 0:
        raise ValueError(f"expected raveled triangular cells, got shape {cells.shape}")
    faces = cells.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("expected only triangular mesh cells")
    return faces[:, 1:]


def write_mesh_obj(path: str | Path, points: np.ndarray, faces: np.ndarray, frame: str) -> Path:
    """Write *points* (N, 3) and triangular *faces* (M, 3) as a Wavefront OBJ file."""
    path = Path(path)
    lines = ["# VIRDA scalp mesh export", f"# coordinate frame: {frame}"]
    lines.extend(
        f"v {x:.6g} {y:.6g} {z:.6g}"
        for x, y, z in np.asarray(points, dtype=np.float64)
    )
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"expected triangular (M, 3) faces, got shape {faces.shape}")
    lines.extend(f"f {i0 + 1} {i1 + 1} {i2 + 1}" for i0, i1, i2 in faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_points_tsv(path: str | Path, names: list[str], points: np.ndarray) -> Path:
    """Write one ``name/x/y/z`` row per point, the viewer's tabular convention."""
    path = Path(path)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got shape {pts.shape}")
    if len(names) != pts.shape[0]:
        raise ValueError(f"got {len(names)} names for {pts.shape[0]} points")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "x", "y", "z"])
        writer.writerows(
            [name, float(p[0]), float(p[1]), float(p[2])]
            for name, p in zip(names, pts, strict=True)
        )
    return path


def collect_mesh_export(scene: SceneData, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the scalp mesh ``(points, faces)`` transformed into a chosen frame."""
    if scene.scene_mesh is None:
        raise ValueError("no scalp mesh is loaded to export")
    points = transform_points(np.asarray(scene.scene_mesh.points, dtype=np.float64), matrix)
    faces = triangle_faces(np.asarray(scene.scene_mesh.faces))
    return points, faces


def collect_electrodes_export(scene: SceneData, matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return ``(names, points)`` for every electrode in a chosen frame."""
    names: list[str] = []
    blocks: list[np.ndarray] = []
    for group in scene.electrode_groups:
        pts = group["points"]
        if pts is None or len(pts) == 0:
            continue
        names.extend(group["names"])
        blocks.append(transform_points(np.asarray(pts, dtype=np.float64), matrix))
    if not blocks:
        return [], np.empty((0, 3))
    return names, np.vstack(blocks)


def collect_fiducials_export(scene: SceneData, matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Return ``(labels, points)`` for the fiducials in a chosen frame."""
    if scene.fiducial_points is None or len(scene.fiducial_points) == 0:
        return [], np.empty((0, 3))
    points = transform_points(np.asarray(scene.fiducial_points, dtype=np.float64), matrix)
    return list(scene.fiducial_labels), points
