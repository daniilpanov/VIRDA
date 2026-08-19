"""Interactive 3D viewer: scalp mesh over the MRI volume.

``virda-gui`` is the interactive visualisation tool of the ``virda_gui``
package. It renders the scalp mesh on top of a semi-transparent MRI volume and
places both objects in the same coordinate frame using the NIfTI affine, so the
mesh overlays the actual scalp in all three views. On-screen checkboxes toggle
the visibility of the mesh, the MRI and the fiducial points (with labels), and
a "Boost contrast" checkbox sharpens the MRI (opaque skin surface) and the mesh
(bright solid surface that makes holes visible).

When ``--normals`` is supplied, per-vertex normal vectors are drawn as cyan
line segments originating at each sampled vertex and pointing outward.

Usage
-----
    virda-gui --nifti <scan.nii.gz>
    virda-gui --mesh <final_mesh.ply>
    virda-gui --nifti <scan.nii.gz> --mesh <final_mesh.ply>
    virda-gui --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --fiducials <fiducials/fiducials.json>
    virda-gui --mesh <ese_mesh.ply> --normals <normals.npy>
    virda-gui --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --ese-mesh <stage2/ese_mesh.ply> --normals <stage2/>
    virda-gui --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --ese-mesh <stage2/ese_mesh.ply> --fiducials <fiducials.json> \\
        --electrodes <localization/electrodes.json>
    virda-gui --nifti <scan.nii.gz> --mesh <final_mesh.ply> \\
        --electrodes <electrodes.tsv>

At least one of ``--nifti`` or ``--mesh`` is required. ``--normals`` points to
the ``stage2/`` output directory (``ese_vertices.npy`` + ``normals.npy``);
``--normals-scale`` sets the arrow length in scene units (default 5) and
``--normals-step`` draws only every N-th normal. ``--electrodes`` accepts either
the Stage 3 ``electrodes.json`` (localized electrodes colored by residual error,
with fiducial links) or a tabular file with ``name``, ``x``, ``y``, ``z``
columns (shown as yellow spheres without links).

If the affine is axis-aligned with positive spacing the scene is shown in world
millimeters; otherwise the mesh is transformed into voxel index space so that
the overlay stays correct for rotated or flipped affines as well. A mesh shown
on its own stays in its native (world) coordinates. Fiducial points are stored
in world coordinates and are transformed into the scene frame accordingly.
"""

import argparse
import json

import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from nibabel import aff2axcodes

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Load electrodes from a Stage 3 JSON or a tabular CSV/TSV file.

    Dispatches to ``_load_electrodes_from_json`` for ``.json`` files and to
    ``_load_electrodes_from_csv`` for everything else (``.tsv``, ``.csv``,
    ``.txt``).
    """
    if path.endswith(".json"):
        return _load_electrodes_from_json(path)
    return _load_electrodes_from_csv(path)


def _load_electrodes_from_json(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Read ``electrodes.json`` from the Stage 3 output.

    Returns scalp points, residuals and flags of the localized electrodes, plus
    the measured distances per electrode (used to draw fiducial links). Non-
    localized electrodes are skipped.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    points: list[np.ndarray] = []
    residuals: list[float] = []
    flags: list[bool] = []
    measured: list[dict[str, float]] = []
    for item in data:
        coords = item.get("scalp_coords")
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
    if not points:
        return np.empty((0, 3)), np.empty(0), np.empty(0), []
    return np.asarray(points), np.asarray(residuals), np.asarray(flags, dtype=bool), measured


def _load_electrodes_from_csv(
    path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Read electrodes from a TSV/CSV with columns: name, x, y, z.

    Returns positions with zero residuals, no flags and no fiducial links.
    Column names are matched case-insensitively.  The delimiter is detected
    automatically by :class:`csv.Sniffer`.
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

        points: list[np.ndarray] = []
        for row in reader:
            x = float(row[col_map["x"]])
            y = float(row[col_map["y"]])
            z = float(row[col_map["z"]])
            points.append(np.array([x, y, z]))

    if not points:
        return np.empty((0, 3)), np.empty(0), np.empty(0, dtype=bool), []
    return (
        np.asarray(points),
        np.zeros(len(points)),
        np.zeros(len(points), dtype=bool),
        [{} for _ in points],
    )


def _build_electrode_links(
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


def _print_qc(
    spacing: tuple[float, float, float] | None,
    orientation: tuple[str, str, str] | None,
    volume: pv.ImageData | None,
    mesh: pv.PolyData | None,
    affine: np.ndarray | None,
    mm_scene: bool,
) -> None:
    scene_transform = affine if (affine is not None and not mm_scene) else np.eye(4)
    scene_space = "world (mm)" if mm_scene else "voxel indices (affine has rotation/flip)"

    print("=" * 62)
    if volume is not None:
        volume_world_bounds = _scene_bounds_to_world(tuple(volume.bounds), scene_transform)
        print("MRI volume")
        print(f"  spacing (mm)                  : {spacing}")
        print(f"  orientation                   : {orientation}")
        print(f"  world bounds (mm)             : {_fmt_bounds(volume_world_bounds)}")
    if mesh is not None:
        mesh_world_bounds = _points_bounds_to_world(np.asarray(mesh.points), scene_transform)
        print("Scalp mesh")
        print(f"  vertices                      : {mesh.n_points}")
        print(f"  world bounds (mm)             : {_fmt_bounds(mesh_world_bounds)}")
    if volume is not None and mesh is not None:
        overlap = _mesh_within_volume(volume_world_bounds, mesh_world_bounds)
        print(f"  within volume bounds (+5 mm)  : {overlap}")
    print(f"  scene coordinates             : {scene_space}")
    if affine is not None:
        print(
            "  voxel->world round-trip error :"
            f" {_round_trip_error(affine, _voxel_samples(volume)):.3e} mm"
        )
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="virda-gui",
        description="Interactive 3D viewer: scalp mesh and/or MRI volume.",
    )
    parser.add_argument("--nifti", help="Path to the T1-weighted NIfTI scan.")
    parser.add_argument("--mesh", help="Path to the scalp mesh (PLY).")
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Voxel stride for volume downsampling (1 = full resolution).",
    )
    parser.add_argument(
        "--mesh-opacity", type=float, default=0.6, help="Scalp mesh opacity (0..1)."
    )
    parser.add_argument(
        "--fiducials",
        help="Path to fiducials JSON (fiducials/fiducials.json).",
    )
    parser.add_argument(
        "--normals",
        help="Path to normals file (normals.npy) for visualisation.",
    )
    parser.add_argument(
        "--normals-scale",
        type=float,
        default=3.0,
        help="Visual length of normal arrows in scene units (default: 3.0).",
    )
    parser.add_argument(
        "--normals-density",
        type=int,
        default=500,
        help="Show one normal per N vertices (default: 500).",
    )
    parser.add_argument(
        "--electrodes",
        help=(
            "Path to electrodes file: Stage 3 JSON (.json) "
            "or tabular with name/x/y/z columns (.tsv/.csv)."
        ),
    )
    args = parser.parse_args()

    if not args.nifti and not args.mesh:
        parser.error("at least one of --nifti or --mesh is required")

    data = None
    affine = None
    spacing = None
    orientation = None
    hi_clim = None
    if args.nifti:
        nifti_img = nib.load(args.nifti)
        data = nifti_img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        affine = nifti_img.affine
        orientation = aff2axcodes(affine)
        if args.downsample > 1:
            data, affine = downsample(data, affine, args.downsample)
        spacing = tuple(float(zoom) for zoom in np.linalg.norm(affine[:3, :3], axis=0))
        hi_clim = percentile_clim(data)

    mesh_poly = _load_mesh_poly(args.mesh) if args.mesh else None
    volume, scene_mesh, mm_scene = _build_scene(data, affine, mesh_poly)
    _print_qc(spacing, orientation, volume, scene_mesh, affine, mm_scene)

    fiducial_points = None
    fiducial_labels: list[str] = []
    if args.fiducials:
        fiducial_points, fiducial_labels = load_fiducial_points(args.fiducials)

    normals_data = None
    if args.normals:
        normals_data = load_normals(args.normals)
        if scene_mesh is not None and normals_data.shape[0] != scene_mesh.n_points:
            raise ValueError(
                f"Normals count ({normals_data.shape[0]}) does not match "
                f"mesh vertex count ({scene_mesh.n_points})"
            )

    plotter = pv.Plotter(title="VIRDA — scalp mesh and/or MRI volume")
    mri_actor = None
    mesh_actor = None
    if volume is not None:
        mri_actor = plotter.add_volume(volume, cmap="bone", opacity="sigmoid", mapper="smart")
    if scene_mesh is not None:
        mesh_actor = plotter.add_mesh(scene_mesh, color="salmon", opacity=args.mesh_opacity)

    fiducial_actor = None
    fiducial_label_actor = None
    if fiducial_points is not None and len(fiducial_points) > 0:
        scene_points = fiducial_points
        if not mm_scene:
            scene_points = transform_points(fiducial_points, np.linalg.inv(affine))
        fiducial_actor = plotter.add_points(
            scene_points, color="red", point_size=10, render_points_as_spheres=True
        )
        fiducial_label_actor = plotter.add_point_labels(
            scene_points,
            fiducial_labels,
            font_size=12,
            text_color="white",
            background_color="black",
            show_points=False,
            shape=None,
        )

    normals_actor = None
    if normals_data is not None and scene_mesh is not None:
        scene_normals = normals_data
        if not mm_scene:
            inv_rot = np.linalg.inv(affine)[:3, :3]
            scene_normals = normals_data @ inv_rot.T
        normals_poly = _create_normal_glyphs(
            np.asarray(scene_mesh.points), scene_normals, args.normals_scale, args.normals_density
        )
        normals_actor = plotter.add_mesh(normals_poly, color="cyan", opacity=0.8, line_width=2)

    fiducial_id_to_point: dict[str, np.ndarray] = {}
    if fiducial_points is not None:
        for label, point in zip(fiducial_labels, fiducial_points, strict=True):
            fiducial_id_to_point[label.split(" (")[0]] = point

    electrode_points = None
    electrode_residuals = None
    electrode_flags = None
    electrode_measured: list[dict[str, float]] = []
    if args.electrodes:
        (
            electrode_points,
            electrode_residuals,
            electrode_flags,
            electrode_measured,
        ) = _load_electrodes(args.electrodes)
    if electrode_points is not None and len(electrode_points) > 0 and not mm_scene:
        electrode_points = transform_points(electrode_points, np.linalg.inv(affine))

    electrode_actor = None
    link_actor = None
    flagged_actor = None
    if electrode_points is not None and len(electrode_points) > 0:
        is_tabular = bool(np.all(electrode_residuals == 0))
        if is_tabular:
            electrode_actor = plotter.add_points(
                electrode_points,
                color="yellow",
                point_size=12,
                render_points_as_spheres=True,
            )
        else:
            healthy = ~electrode_flags
            if healthy.any():
                electrode_actor = plotter.add_points(
                    electrode_points[healthy],
                    scalars=electrode_residuals[healthy],
                    cmap="jet",
                    point_size=12,
                    render_points_as_spheres=True,
                )
            if (~healthy).any():
                flagged_actor = plotter.add_points(
                    electrode_points[~healthy],
                    color="red",
                    point_size=16,
                    render_points_as_spheres=True,
                )
            if electrode_actor is None:
                electrode_actor = flagged_actor
        links = _build_electrode_links(electrode_points, electrode_measured, fiducial_id_to_point)
        if len(links) > 0:
            flat = links.reshape(-1, 3)
            link_actor = plotter.add_lines(flat, color="cyan", width=1)

    y = 10
    mri_visible = {"on": True}

    def set_mesh_visibility(flag: bool) -> None:
        mesh_actor.SetVisibility(flag)

    def set_mri_visibility(flag: bool) -> None:
        mri_visible["on"] = bool(flag)
        mri_actor.SetVisibility(flag)

    def set_contrast(flag: bool) -> None:
        nonlocal mri_actor
        if mesh_actor is not None:
            if flag:
                mesh_actor.prop.opacity = 0.95
                mesh_actor.prop.diffuse = 1.0
                mesh_actor.prop.specular = 0.6
                mesh_actor.prop.specular_power = 40.0
                mesh_actor.prop.ambient = 0.2
            else:
                mesh_actor.prop.opacity = args.mesh_opacity
                mesh_actor.prop.diffuse = 1.0
                mesh_actor.prop.specular = 0.0
                mesh_actor.prop.specular_power = 100.0
                mesh_actor.prop.ambient = 0.0
        if volume is not None:
            if "intensity" in plotter.scalar_bars:
                plotter.remove_scalar_bar("intensity")
            plotter.remove_actor(mri_actor)
            if flag:
                mri_actor = plotter.add_volume(
                    volume,
                    cmap="bone",
                    opacity="sigmoid_10",
                    mapper="smart",
                    clim=hi_clim,
                    opacity_unit_distance=0.5,
                )
            else:
                mri_actor = plotter.add_volume(
                    volume,
                    cmap="bone",
                    opacity="sigmoid",
                    mapper="smart",
                )
            mri_actor.SetVisibility(mri_visible["on"])
        plotter.render()

    if mesh_actor is not None:
        plotter.add_checkbox_button_widget(set_mesh_visibility, value=True, position=(10, y))
        plotter.add_text("Show mesh", position=(75, y + 10), font_size=18, name="mesh_label")
        y += 50
    if mri_actor is not None:
        plotter.add_checkbox_button_widget(set_mri_visibility, value=True, position=(10, y))
        plotter.add_text("Show MRI", position=(75, y + 10), font_size=18, name="mri_label")
        y += 50
    if fiducial_actor is not None:

        def set_fiducials_visibility(flag: bool) -> None:
            fiducial_actor.SetVisibility(flag)
            if fiducial_label_actor is not None:
                fiducial_label_actor.SetVisibility(flag)

        plotter.add_checkbox_button_widget(set_fiducials_visibility, value=True, position=(10, y))
        plotter.add_text(
            "Show fiducials", position=(75, y + 10), font_size=18, name="fiducials_label"
        )
        y += 50
    if normals_actor is not None:

        def set_normals_visibility(flag: bool) -> None:
            normals_actor.SetVisibility(flag)

        plotter.add_checkbox_button_widget(set_normals_visibility, value=True, position=(10, y))
        plotter.add_text("Show normals", position=(75, y + 10), font_size=18, name="normals_label")
        y += 50
    if electrode_actor is not None:

        def set_electrodes_visibility(flag: bool) -> None:
            electrode_actor.SetVisibility(flag)
            if flagged_actor is not None:
                flagged_actor.SetVisibility(flag)

        plotter.add_checkbox_button_widget(set_electrodes_visibility, value=True, position=(10, y))
        plotter.add_text(
            "Show electrodes", position=(75, y + 10), font_size=18, name="electrodes_label"
        )
        y += 50
    if link_actor is not None:

        def set_links_visibility(flag: bool) -> None:
            link_actor.SetVisibility(flag)

        plotter.add_checkbox_button_widget(set_links_visibility, value=True, position=(10, y))
        plotter.add_text("Show links", position=(75, y + 10), font_size=18, name="links_label")
        y += 50
    plotter.add_checkbox_button_widget(set_contrast, value=False, position=(10, y))
    plotter.add_text("Boost contrast", position=(75, y + 10), font_size=18, name="contrast_label")
    plotter.add_axes(interactive=False)
    plotter.show()


if __name__ == "__main__":
    main()
