"""Interactive 3D viewer: scalp mesh over the MRI volume.

Standalone research script for manual quality control of the Stage 1 output.
It renders the scalp mesh on top of a semi-transparent MRI volume and places
both objects in the same coordinate frame using the NIfTI affine, so the mesh
overlays the actual scalp in all three views. On-screen checkboxes toggle the
visibility of the mesh and the MRI, and a "Boost contrast" checkbox sharpens
the MRI (opaque skin surface) and the mesh (bright solid surface that makes
holes visible).

Usage
-----
    python research/scripts/mesh_over_mri_viewer.py --nifti <scan.nii.gz>
    python research/scripts/mesh_over_mri_viewer.py --mesh <final_mesh.ply>
    python research/scripts/mesh_over_mri_viewer.py --nifti <scan.nii.gz> --mesh <final_mesh.ply>

At least one of ``--nifti`` or ``--mesh`` is required.

If the affine is axis-aligned with positive spacing the scene is shown in world
millimeters; otherwise the mesh is transformed into voxel index space so that
the overlay stays correct for rotated or flipped affines as well. A mesh shown
on its own stays in its native (world) coordinates.
"""

import argparse

import nibabel as nib
import numpy as np
import pyvista as pv
import trimesh
from nibabel import aff2axcodes

_BOUNDS = tuple[float, float, float, float, float, float]


def _downsample(data: np.ndarray, affine: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError(f"downsample must be >= 1, got {stride}")
    if stride == 1:
        return data, affine
    sampled = data[::stride, ::stride, ::stride]
    scaled = affine @ np.diag([stride, stride, stride, 1.0])
    return sampled, scaled


def _is_axis_aligned(affine: np.ndarray) -> bool:
    linear = affine[:3, :3]
    if not np.allclose(linear, np.diag(np.diag(linear))):
        return False
    return bool(np.all(np.diag(linear) > 0))


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
    mm_scene = True

    if affine is not None:
        if _is_axis_aligned(affine):
            if volume is not None:
                volume.spacing = (float(affine[0, 0]), float(affine[1, 1]), float(affine[2, 2]))
                volume.origin = (float(affine[0, 3]), float(affine[1, 3]), float(affine[2, 3]))
        else:
            if scene_mesh is not None:
                scene_mesh.transform(np.linalg.inv(affine), inplace=True)
            mm_scene = False

    return volume, scene_mesh, mm_scene


def _load_mesh_poly(mesh_path: str) -> pv.PolyData:
    loaded = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    face_array = np.empty((faces.shape[0], 4), dtype=np.int64)
    face_array[:, 0] = 3
    face_array[:, 1:] = faces
    return pv.PolyData(vertices, face_array.ravel())


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
        prog="mri_viewer",
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
            data, affine = _downsample(data, affine, args.downsample)
        spacing = tuple(float(zoom) for zoom in np.linalg.norm(affine[:3, :3], axis=0))
        hi_clim = tuple(float(v) for v in np.percentile(data, (3, 99.9)))

    mesh_poly = _load_mesh_poly(args.mesh) if args.mesh else None
    volume, scene_mesh, mm_scene = _build_scene(data, affine, mesh_poly)
    _print_qc(spacing, orientation, volume, scene_mesh, affine, mm_scene)

    plotter = pv.Plotter(title="VIRDA — scalp mesh and/or MRI volume")
    mri_actor = None
    mesh_actor = None
    if volume is not None:
        mri_actor = plotter.add_volume(volume, cmap="bone", opacity="sigmoid", mapper="smart")
    if scene_mesh is not None:
        mesh_actor = plotter.add_mesh(scene_mesh, color="salmon", opacity=args.mesh_opacity)

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
            plotter.remove_scalar_bar()
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
    plotter.add_checkbox_button_widget(set_contrast, value=False, position=(10, y))
    plotter.add_text("Boost contrast", position=(75, y + 10), font_size=18, name="contrast_label")
    plotter.add_axes(interactive=False)
    plotter.show()


if __name__ == "__main__":
    main()
