"""QC 3D renders from fixed viewpoints (requires pyvista)."""

from pathlib import Path

import numpy as np

from virda.io.qc.geometry import fiducials_world_coordinates
from virda.models.stage1_result import Stage1Result

_VIEWS = [
    ("front", (0, 1.4, 0.3)),
    ("side_left", (-1.4, 0, 0.2)),
    ("side_right", (1.4, 0, 0.2)),
    ("back", (0, -1.4, 0.3)),
    ("top", (0.2, 0.2, 1.4)),
    ("three_quarter", (0.8, 0.8, 0.7)),
]


def render_3d(result: Stage1Result, output_dir: str | Path) -> Path:
    """Shaded 3D renders of the scalp mesh with fiducials (needs mesh.ply next to output)."""
    out = Path(output_dir)
    try:
        import pyvista as pv
    except ImportError:
        return out
    pv.OFF_SCREEN = True
    if hasattr(pv, "start_xvfb"):
        pv.start_xvfb()
    mesh = pv.read(str(out / "mesh.ply"))
    center = np.asarray(mesh.center, dtype=float)
    diag = float(mesh.length)
    for name, pos in _VIEWS:
        plotter = pv.Plotter(off_screen=True, window_size=[900, 900])
        plotter.add_mesh(mesh, color="lightsteelblue", specular=0.3, smooth_shading=True)
        if result.fiducials:
            points = fiducials_world_coordinates(result.fiducials, result.mri_volume.affine)
            plotter.add_points(
                points,
                color="red",
                point_size=16,
                render_points_as_spheres=True,
                name="fiducials",
            )
            for fiducial in result.fiducials:
                world = fiducials_world_coordinates([fiducial], result.mri_volume.affine)[0]
                plotter.add_point_labels(
                    [list(world)],
                    [fiducial.fiducial_id],
                    text_color="yellow",
                    font_size=14,
                    shape_opacity=0.5,
                    always_visible=True,
                )
        plotter.camera_position = [
            tuple(center + diag * 0.9 * np.array(pos)),
            tuple(center),
            (0, 0, 1),
        ]
        plotter.enable_anti_aliasing()
        plotter.show(screenshot=str(out / f"qc_3d_{name}.png"))
        plotter.close()
    return out
