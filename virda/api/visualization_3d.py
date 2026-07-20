"""3D visualization module — interactive viewers for MRI, mesh, electrodes.

This module lives in the API layer because it depends on pyvista and
matplotlib. Core modules never import from this module.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from core.surface_extractor import MeshData
from core.fiducial_manager import FiducialManager
from core.ese_generator import ESEResult
from core.electrode_localizer import LocalizationResult

logger = logging.getLogger(__name__)


def plot_scalp_mesh(
    mesh: MeshData,
    title: str = "Scalp Mesh",
    show_normals: bool = False,
    normals: Optional[np.ndarray] = None,
    quality: Optional[np.ndarray] = None,
):
    """Plot scalp mesh with optional normals or quality overlay."""
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista not available, falling back to matplotlib")
        _plot_mesh_matplotlib(mesh, title)
        return

    plotter = pv.Plotter()
    plotter.set_background("white")

    surface = pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]))

    if quality is not None:
        plotter.add_mesh(surface, scalars=quality, cmap="viridis", scalar_bar_args={"title": "PCA Quality"})
    else:
        plotter.add_mesh(surface, color="peachpuff", opacity=0.9)

    if show_normals and normals is not None:
        arrows = pv.PolyData(mesh.vertices)
        arrows["normals"] = normals
        glyphs = arrows.glyph(orient="normals", scale=False, factor=2.0)
        plotter.add_mesh(glyphs, color="blue", opacity=0.5)

    plotter.add_axes()
    plotter.add_title(title)
    plotter.show()


def plot_ese(
    ese: ESEResult,
    title: str = "ESE Surface",
    show_scalp: bool = True,
    show_ese: bool = True,
    show_normals: bool = False,
):
    """Plot ESE surface with scalp reference."""
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista not available")
        return

    plotter = pv.Plotter()
    plotter.set_background("white")

    if show_scalp:
        scalp_cloud = pv.PolyData(ese.scalp_vertices)
        plotter.add_mesh(scalp_cloud, color="peachpuff", opacity=0.6, point_size=2, render_points_as_spheres=True)

    if show_ese:
        ese_cloud = pv.PolyData(ese.ese_vertices)
        plotter.add_mesh(ese_cloud, color="cyan", opacity=0.6, point_size=2, render_points_as_spheres=True)

    if show_normals:
        arrows = pv.PolyData(ese.scalp_vertices)
        arrows["normals"] = ese.normals
        glyphs = arrows.glyph(orient="normals", scale=False, factor=2.0)
        plotter.add_mesh(glyphs, color="blue", opacity=0.4)

    plotter.add_axes()
    plotter.add_title(title)
    plotter.show()


def plot_localization(
    ese: ESEResult,
    localization: LocalizationResult,
    fiducial_mgr: FiducialManager,
    title: str = "Electrode Localization",
):
    """Plot localization results with mesh, ESE, fiducials, and electrodes."""
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista not available")
        return

    plotter = pv.Plotter()
    plotter.set_background("white")

    scalp_cloud = pv.PolyData(ese.scalp_vertices)
    plotter.add_mesh(scalp_cloud, color="peachpuff", opacity=0.3, point_size=1, render_points_as_spheres=True)

    ese_cloud = pv.PolyData(ese.ese_vertices)
    plotter.add_mesh(ese_cloud, color="lightblue", opacity=0.3, point_size=1, render_points_as_spheres=True)

    if localization.num_electrodes > 0:
        _, coords = localization.get_electrode_coords()
        electrode_cloud = pv.PolyData(coords)
        plotter.add_mesh(
            electrode_cloud,
            color="red",
            point_size=10,
            render_points_as_spheres=True,
        )

    fid_coords = fiducial_mgr.get_coordinates_matrix()
    if len(fid_coords) > 0:
        fid_cloud = pv.PolyData(fid_coords)
        plotter.add_mesh(fid_cloud, color="green", point_size=15, render_points_as_spheres=True)

    for fid_id, fid in fiducial_mgr.get_all_fiducials().items():
        plotter.add_point_labels(
            [fid.coordinates],
            [fid_id],
            font_size=12,
            text_color="green",
            point_color="green",
            point_size=20,
            render_points_as_spheres=True,
        )

    for loc_e in localization.electrodes:
        plotter.add_point_labels(
            [loc_e.ese_coords],
            [loc_e.electrode_id],
            font_size=10,
            text_color="red",
            point_color="red",
            point_size=8,
            render_points_as_spheres=True,
        )

    plotter.add_axes()
    plotter.add_title(title)
    plotter.show()


def plot_mri_slices(
    mri_data: np.ndarray,
    mesh: Optional[MeshData] = None,
    fiducial_mgr: Optional[FiducialManager] = None,
    title: str = "MRI with Mesh Overlay",
):
    """Plot MRI slices (axial, coronal, sagittal).

    Parameters
    ----------
    mri_data : np.ndarray
        3D MRI voxel data.
    """
    try:
        import pyvista as pv
    except ImportError:
        logger.warning("pyvista not available")
        return

    plotter = pv.Plotter(shape=(1, 3))

    mid_z = mri_data.shape[2] // 2
    mid_y = mri_data.shape[1] // 2
    mid_x = mri_data.shape[0] // 2

    for i, (sl, dim_name) in enumerate(
        [(mid_z, "Axial"), (mid_y, "Coronal"), (mid_x, "Sagittal")]
    ):
        plotter.subplot(0, i)
        if dim_name == "Axial":
            img = mri_data[:, :, sl].T
        elif dim_name == "Coronal":
            img = mri_data[:, sl, :].T
        else:
            img = mri_data[sl, :, :].T

        img = img.astype(np.float64)
        if img.max() > 0:
            img = img / img.max() * 255

        plotter.add_image(img, cmap="gray")
        plotter.add_title(f"{dim_name} (slice {sl})")

    plotter.add_axes()
    plotter.show()


def _plot_mesh_matplotlib(mesh: MeshData, title: str):
    """Fallback matplotlib 3D scatter plot."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    verts = mesh.vertices
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.5, c="peachpuff", alpha=0.6)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
