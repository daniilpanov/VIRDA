"""Data loading and scene preparation for the 3D viewer (no Qt).

Everything that reads a file or resolves coordinates into the scene frame
lives here so it can be tested without a display: mesh loading, MRI volume
loading and downsampling, fiducial points, normal glyphs, electrode loading
(Stage 3 JSON or tabular cRAS files) with automatic frame detection, and the
QC report.  :class:`SceneData` is the plain-data result consumed by
:class:`virda_gui.viewer.ViewerWidget` on the GUI thread.
"""

import colorsys
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from nibabel import aff2axcodes
from scipy.spatial import cKDTree

from virda_gui.scene import (
    compute_normal_lines,
    downsample,
    load_fiducial_points,
    load_normals,
    percentile_clim,
    sample_normals,
    scene_placement,
    transform_points,
)

_BOUNDS = tuple[float, float, float, float, float, float]
_ELECTRODE_COLORS = ["yellow", "lime", "magenta", "cyan", "orange", "white"]


def _build_scene(
    data: np.ndarray | None, affine: np.ndarray | None, mesh_poly: pv.PolyData | None
) -> tuple[pv.ImageData | None, pv.PolyData | None, bool]:
    """Place the volume and mesh into one coordinate frame.

    Returns the volume, the mesh moved into the scene frame and a flag that is
    True when the scene is already in world millimeters.
    """
    volume = None
    if data is not None:
        volume = pv.ImageData(dimensions=data.shape)
        volume.point_data["intensity"] = data.ravel(order="F")

    scene_mesh = mesh_poly.copy() if mesh_poly is not None else None

    spacing, origin, transform, mm_scene = scene_placement(affine)
    if volume is not None and mm_scene and affine is not None:
        volume.spacing = tuple(spacing)
        volume.origin = tuple(origin)
    if scene_mesh is not None and not mm_scene:
        scene_mesh.transform(transform, inplace=True)

    return volume, scene_mesh, mm_scene


def _load_mesh_poly(mesh_path: str) -> pv.PolyData:
    loaded = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    face_array = np.empty((faces.shape[0], 4), dtype=np.int64)
    face_array[:, 0] = 3
    face_array[:, 1:] = faces
    return pv.PolyData(vertices, face_array.ravel())


def _cras_to_scanner_ras_offset(affine: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """FreeSurfer cRAS -> scanner RAS offset for a NIfTI volume.

    FreeSurfer cRAS coordinates are centered at the volume midpoint while
    scanner RAS is the affine world frame, so the conversion offset is the
    affine applied to the center voxel.
    """
    center_voxel = (np.asarray(shape[:3], dtype=np.float64) - 1.0) / 2.0
    return np.array((affine @ np.append(center_voxel, 1.0))[:3], dtype=np.float64)


_CRAS_DETECT_RATIO = 0.8


def _detect_cras_conversion(
    points: np.ndarray,
    mesh_points: np.ndarray,
    affine: np.ndarray | None,
    mm_scene: bool,
    cras_offset: np.ndarray,
) -> tuple[bool, float, float]:
    """Decide whether tabular electrode points are FreeSurfer cRAS coordinates.

    Electrodes sit on the scalp, so the correct frame puts them close to the
    scalp mesh while the wrong frame displaces the whole group by ``|c_ras|``.
    Both hypotheses -- raw scanner RAS and cRAS shifted by ``cras_offset`` --
    are moved into the scene frame (voxel indices when ``mm_scene`` is False)
    and scored by the median distance from each electrode to its nearest mesh
    vertex.  Conversion wins only when it improves the fit by
    ``_CRAS_DETECT_RATIO``; near-ties keep the points untouched.

    Returns ``(needs_conversion, raw_distance_mm, shifted_distance_mm)``.
    """
    scene_transform = None if (affine is None or mm_scene) else np.linalg.inv(affine)

    def _median_distance(pts: np.ndarray) -> float:
        scene_pts = transform_points(pts, scene_transform) if scene_transform is not None else pts
        distances, _ = cKDTree(mesh_points).query(scene_pts)
        return float(np.median(distances))

    d_raw = _median_distance(points)
    d_shift = _median_distance(points + cras_offset)
    return d_shift < _CRAS_DETECT_RATIO * d_raw, d_raw, d_shift


def _cras_decision_message(label: str, converted: bool, d_raw: float, d_shift: float) -> str:
    """One-line QC report of the automatic cRAS frame detection."""
    verdict = "cRAS detected" if converted else "scanner RAS assumed"
    action = "converted" if converted else "unchanged"
    return f"{label}: {verdict} (median dist {d_raw:.1f} -> {d_shift:.1f} mm), {action}"


def parse_electrode_specs(specs: list[list[str]]) -> list[tuple[str, str | None]]:
    """Split ``--electrodes FILE [COLOR]`` occurrences into (path, color) pairs."""
    pairs: list[tuple[str, str | None]] = []
    for spec in specs:
        if len(spec) == 1:
            pairs.append((spec[0], None))
        elif len(spec) == 2:
            pairs.append((spec[0], spec[1]))
        else:
            raise ValueError(
                f"--electrodes expects FILE [COLOR], got {len(spec)} values: {' '.join(spec)}"
            )
    return pairs


def _resolve_group_color(color: str | None, index: int) -> str:
    """Return the explicit group color or the default palette entry.

    Raises ``ValueError`` when an explicit color is not understood by VTK.
    """
    if color is not None:
        try:
            pv.Color(color)
        except ValueError:
            raise ValueError(f"invalid electrode color {color!r}") from None
        return color
    return _ELECTRODE_COLORS[index % len(_ELECTRODE_COLORS)]


def intensify_color(color: str) -> str:
    """Return a deeper, more saturated shade of *color* for flagged electrodes."""
    r, g, b = pv.Color(color).float_rgb
    h, lightness, sat = colorsys.rgb_to_hls(r, g, b)
    darker = colorsys.hls_to_rgb(h, min(0.45, lightness * 0.7), min(1.0, sat * 1.6))
    return pv.Color(darker).hex_rgb


def _create_normal_glyphs(
    points: np.ndarray,
    normals: np.ndarray,
    scale: float,
    density: int,
) -> pv.PolyData:
    """Build line-segment glyphs pointing along *normals* at sampled vertices.

    Returns a ``PolyData`` mesh of line segments that can be added to a
    PyVista plotter.
    """
    idx, sampled = sample_normals(normals, density)
    origins, tips = compute_normal_lines(points[idx], sampled, scale)

    n = len(idx)
    lines = np.empty((n * 2, 3), dtype=np.float64)
    lines[0::2] = origins
    lines[1::2] = tips

    line_cells = np.empty((n, 3), dtype=np.int64)
    line_cells[:, 0] = 2
    line_cells[:, 1] = np.arange(0, n * 2, 2)
    line_cells[:, 2] = np.arange(1, n * 2, 2)

    return pv.PolyData(lines, lines=line_cells.ravel())


def _load_electrodes(
    path: str,
    cras_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], list[str]]:
    """Load electrodes from a Stage 3 JSON or a tabular CSV/TSV file.

    Dispatches to ``_load_electrodes_from_json`` for ``.json`` files and to
    ``_load_electrodes_from_csv`` for everything else (``.tsv``, ``.csv``,
    ``.txt``).  ``cras_offset`` applies only to tabular files: Stage 3 JSON
    output is already in scanner RAS, while TSV/CSV electrode tables may
    store FreeSurfer cRAS coordinates that need converting first.  Returns
    per-electrode display names (the ``electrode_id``/``name`` field, or a
    generated ``E001``-style fallback).
    """
    if path.endswith(".json"):
        return _load_electrodes_from_json(path)
    return _load_electrodes_from_csv(path, cras_offset=cras_offset)


def _load_electrodes_from_json(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], list[str]]:
    """Read ``electrodes.json`` from the Stage 3 output.

    Returns scalp points, residuals, flags, measured distances and display
    names of the localized electrodes (used to draw fiducial links and ID
    labels). Non-localized electrodes are skipped.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    points: list[np.ndarray] = []
    residuals: list[float] = []
    flags: list[bool] = []
    measured: list[dict[str, float]] = []
    names: list[str] = []
    for index, item in enumerate(data):
        coords = item.get("coords") or item.get("scalp_coords")
        if coords is None:
            continue
        points.append(np.asarray(coords, dtype=np.float64))
        residuals.append(float(item.get("residual_error") or 0.0))
        flags.append(bool(item.get("flagged", False)))
        measured.append(
            {
                str(fiducial_id): float(distance)
                for fiducial_id, distance in item.get("measured_distances", {}).items()
            }
        )
        names.append(str(item.get("electrode_id") or f"E{index + 1:03d}"))
    if not points:
        return np.empty((0, 3)), np.empty(0), np.empty(0), [], []
    return (
        np.asarray(points),
        np.asarray(residuals),
        np.asarray(flags, dtype=bool),
        measured,
        names,
    )


def _load_electrodes_from_csv(
    path: str,
    cras_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], list[str]]:
    """Read electrodes from a TSV/CSV with columns: name, x, y, z.

    Returns positions with zero residuals, no flags and no fiducial links.
    Column names are matched case-insensitively.  The delimiter is detected
    automatically by :class:`csv.Sniffer`.  When ``cras_offset`` is given the
    positions are treated as FreeSurfer cRAS and shifted into scanner RAS.
    """
    import csv

    with open(path, encoding="utf-8") as fh:
        sample = fh.read(2048)
        dialect = csv.Sniffer().sniff(sample)
        fh.seek(0)
        reader = csv.DictReader(fh, dialect=dialect)

        if not reader.fieldnames:
            raise ValueError(f"Electrodes file is empty or has no header: {path}")

        col_map = {col.lower().strip(): col for col in reader.fieldnames}

        for required in ("x", "y", "z"):
            if required not in col_map:
                raise ValueError(
                    f"Electrodes file missing required column '{required}', "
                    f"found: {list(reader.fieldnames)}"
                )

        name_col = col_map.get("name")
        points: list[np.ndarray] = []
        names: list[str] = []
        for index, row in enumerate(reader):
            x = float(row[col_map["x"]])
            y = float(row[col_map["y"]])
            z = float(row[col_map["z"]])
            points.append(np.array([x, y, z]))
            raw_name = row.get(name_col) if name_col is not None else None
            names.append(str(raw_name).strip() if raw_name else f"E{index + 1:03d}")

    if not points:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=bool), [], []
    positions = np.asarray(points)
    if cras_offset is not None:
        positions = positions + cras_offset
    return (
        positions,
        np.zeros(len(points)),
        np.zeros(len(points), dtype=bool),
        [{} for _ in points],
        names,
    )


def build_electrode_links(
    points: np.ndarray,
    measured: list[dict[str, float]],
    fiducial_id_to_point: dict[str, np.ndarray],
) -> np.ndarray:
    """Return (N, 2, 3) line segments from each electrode to its measured fiducials."""
    pairs: list[np.ndarray] = []
    for point, distances in zip(points, measured, strict=True):
        for fiducial_id in distances:
            if fiducial_id in fiducial_id_to_point:
                pairs.append(np.vstack([point, fiducial_id_to_point[fiducial_id]]))
    if not pairs:
        return np.empty((0, 2, 3))
    return np.asarray(pairs)


def _scene_bounds_to_world(bounds: _BOUNDS, transform: np.ndarray) -> _BOUNDS:
    xs = (bounds[0], bounds[1])
    ys = (bounds[2], bounds[3])
    zs = (bounds[4], bounds[5])
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    world = corners @ transform[:3, :3].T + transform[:3, 3]
    return (
        float(world[:, 0].min()),
        float(world[:, 0].max()),
        float(world[:, 1].min()),
        float(world[:, 1].max()),
        float(world[:, 2].min()),
        float(world[:, 2].max()),
    )


def _points_bounds_to_world(points: np.ndarray, transform: np.ndarray) -> _BOUNDS:
    world = points @ transform[:3, :3].T + transform[:3, 3]
    return (
        float(world[:, 0].min()),
        float(world[:, 0].max()),
        float(world[:, 1].min()),
        float(world[:, 1].max()),
        float(world[:, 2].min()),
        float(world[:, 2].max()),
    )


def _mesh_within_volume(volume_bounds: _BOUNDS, mesh_bounds: _BOUNDS, margin: float = 5.0) -> bool:
    for axis in range(3):
        if mesh_bounds[2 * axis] < volume_bounds[2 * axis] - margin:
            return False
        if mesh_bounds[2 * axis + 1] > volume_bounds[2 * axis + 1] + margin:
            return False
    return True


def _voxel_samples(volume: pv.ImageData) -> np.ndarray:
    dims = np.asarray(volume.dimensions, dtype=np.float64)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [dims[0] - 1, 0.0, 0.0],
            [0.0, dims[1] - 1, 0.0],
            [0.0, 0.0, dims[2] - 1],
            dims / 2.0,
        ]
    )


def _round_trip_error(affine: np.ndarray, voxel_samples: np.ndarray) -> float:
    world = voxel_samples @ affine[:3, :3].T + affine[:3, 3]
    inverse = np.linalg.inv(affine)
    back = world @ inverse[:3, :3].T + inverse[:3, 3]
    return float(np.abs(voxel_samples - back).max())


def _fmt_bounds(bounds: _BOUNDS) -> str:
    return (
        f"x=[{bounds[0]:.2f}, {bounds[1]:.2f}] "
        f"y=[{bounds[2]:.2f}, {bounds[3]:.2f}] "
        f"z=[{bounds[4]:.2f}, {bounds[5]:.2f}]"
    )


def _report_qc(
    spacing: tuple[float, float, float] | None,
    orientation: tuple[str, str, str] | None,
    volume: pv.ImageData | None,
    mesh: pv.PolyData | None,
    affine: np.ndarray | None,
    mm_scene: bool,
    log: Callable[[str], None],
) -> None:
    scene_transform = affine if (affine is not None and not mm_scene) else np.eye(4)
    scene_space = "world (mm)" if mm_scene else "voxel indices (affine has rotation/flip)"

    log("=" * 62)
    if volume is not None:
        volume_world_bounds = _scene_bounds_to_world(tuple(volume.bounds), scene_transform)
        log("MRI volume")
        log(f"  spacing (mm)                  : {spacing}")
        log(f"  orientation                   : {orientation}")
        log(f"  world bounds (mm)             : {_fmt_bounds(volume_world_bounds)}")
    if mesh is not None:
        mesh_world_bounds = _points_bounds_to_world(np.asarray(mesh.points), scene_transform)
        log("Scalp mesh")
        log(f"  vertices                      : {mesh.n_points}")
        log(f"  world bounds (mm)             : {_fmt_bounds(mesh_world_bounds)}")
    if volume is not None and mesh is not None:
        overlap = _mesh_within_volume(volume_world_bounds, mesh_world_bounds)
        log(f"  within volume bounds (+5 mm)  : {overlap}")
    log(f"  scene coordinates             : {scene_space}")
    if affine is not None:
        log(
            "  voxel->world round-trip error :"
            f" {_round_trip_error(affine, _voxel_samples(volume)):.3e} mm"
        )
    log("=" * 62)


@dataclass
class SceneData:
    """Fully resolved scene data, ready to be rendered.

    Produced off the GUI thread by :func:`collect_scene_data` and consumed on
    the GUI thread by :meth:`virda_gui.viewer.ViewerWidget.set_scene`.
    Contains only data objects (NumPy arrays and PyVista meshes), never actors
    or windows, so the collection step is testable without a display.
    """

    volume: pv.ImageData | None = None
    scene_mesh: pv.PolyData | None = None
    mm_scene: bool = True
    mesh_opacity: float = 0.6
    hi_clim: tuple[float, float] | None = None
    fiducial_points: np.ndarray | None = None
    fiducial_labels: list[str] = field(default_factory=list)
    fiducial_id_to_point: dict[str, np.ndarray] = field(default_factory=dict)
    normals_poly: pv.PolyData | None = None
    electrode_groups: list[dict[str, Any]] = field(default_factory=list)


def collect_scene_data(
    nifti_path: str | None = None,
    mesh_path: str | None = None,
    fiducials_path: str | None = None,
    normals_path: str | None = None,
    downsample_stride: int = 1,
    mesh_opacity: float = 0.6,
    normals_scale: float = 3.0,
    normals_density: int = 500,
    electrode_specs: list[tuple[str, str | None]] | None = None,
    electrodes_cras: bool = False,
    log: Callable[[str], None] = print,
) -> SceneData:
    """Load a full viewer scene without touching any display.

    Runs on a worker thread: nibabel reading, volume downsampling, mesh
    loading, scene placement, fiducials, normals glyphs, electrode loading with
    cRAS frame detection, and QC reporting.  All returned points are already in
    the scene frame.
    """
    if not nifti_path and not mesh_path:
        raise ValueError("at least one of nifti_path or mesh_path is required")
    if electrodes_cras and not nifti_path:
        raise ValueError("electrodes_cras requires nifti_path")

    data = None
    affine = None
    spacing = None
    orientation = None
    hi_clim: tuple[float, float] | None = None
    cras_offset: np.ndarray | None = None
    if nifti_path:
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        affine = nifti_img.affine
        orientation = aff2axcodes(affine)
        cras_offset = _cras_to_scanner_ras_offset(affine, tuple(int(d) for d in data.shape))
        if downsample_stride > 1:
            data, affine = downsample(data, affine, downsample_stride)
        spacing = tuple(float(zoom) for zoom in np.linalg.norm(affine[:3, :3], axis=0))
        hi_clim = percentile_clim(data)

    mesh_poly = _load_mesh_poly(mesh_path) if mesh_path else None
    volume, scene_mesh, mm_scene = _build_scene(data, affine, mesh_poly)
    _report_qc(spacing, orientation, volume, scene_mesh, affine, mm_scene, log)

    fiducial_points = None
    fiducial_labels: list[str] = []
    if fiducials_path:
        fiducial_points, fiducial_labels = load_fiducial_points(fiducials_path)

    normals_data = None
    if normals_path:
        normals_data = load_normals(normals_path)
        if scene_mesh is not None and normals_data.shape[0] != scene_mesh.n_points:
            raise ValueError(
                f"Normals count ({normals_data.shape[0]}) does not match "
                f"mesh vertex count ({scene_mesh.n_points})"
            )

    normals_poly = None
    if normals_data is not None and scene_mesh is not None:
        scene_normals = normals_data
        if not mm_scene:
            inv_rot = np.linalg.inv(affine)[:3, :3]
            scene_normals = normals_data @ inv_rot.T
        normals_poly = _create_normal_glyphs(
            np.asarray(scene_mesh.points), scene_normals, normals_scale, normals_density
        )

    scene_fiducials = None
    fiducial_id_to_point: dict[str, np.ndarray] = {}
    if fiducial_points is not None:
        scene_fiducials = fiducial_points
        if not mm_scene:
            scene_fiducials = transform_points(fiducial_points, np.linalg.inv(affine))
        for label, point in zip(fiducial_labels, scene_fiducials, strict=True):
            fiducial_id_to_point[label.split(" (")[0]] = point

    electrode_groups: list[dict[str, Any]] = []
    for gi, (epath, spec_color) in enumerate(electrode_specs or []):
        points, _, flags, measured, names = _load_electrodes(epath)
        label = epath.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        if not epath.endswith(".json") and cras_offset is not None and len(points) > 0:
            if electrodes_cras:
                points = points + cras_offset
                log(f"{label}: cRAS conversion forced by --electrodes-cras")
            elif scene_mesh is not None:
                converted, d_raw, d_shift = _detect_cras_conversion(
                    points, scene_mesh.points, affine, mm_scene, cras_offset
                )
                if converted:
                    points = points + cras_offset
                log(_cras_decision_message(label, converted, d_raw, d_shift))
        if len(points) > 0 and not mm_scene:
            points = transform_points(points, np.linalg.inv(affine))
        electrode_groups.append(
            {
                "points": points,
                "flags": flags,
                "measured": measured,
                "names": names,
                "label": label,
                "color": _resolve_group_color(spec_color, gi),
            }
        )

    return SceneData(
        volume=volume,
        scene_mesh=scene_mesh,
        mm_scene=mm_scene,
        mesh_opacity=mesh_opacity,
        hi_clim=hi_clim,
        fiducial_points=scene_fiducials,
        fiducial_labels=fiducial_labels,
        fiducial_id_to_point=fiducial_id_to_point,
        normals_poly=normals_poly,
        electrode_groups=electrode_groups,
    )
