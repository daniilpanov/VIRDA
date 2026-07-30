"""3D visualization using PyVista."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from ..core.fiducial_manager import FiducialManager
from ..core.types import ESEResult, LocalizationResult, MeshData, NormalResult

logger = logging.getLogger(__name__)


def plot_mesh(
    mesh: MeshData,
    title: str = "Scalp Mesh",
    show_edges: bool = False,
    color: str = "peachpuff",
    opacity: float = 0.8,
) -> pv.Plotter:
    """Display a 3D mesh.

    Parameters
    ----------
    mesh : MeshData
        Mesh to display.
    title : str
        Window title.
    show_edges : bool
        Show mesh edges.
    color : str
        Mesh color.
    opacity : float
        Mesh opacity.

    Returns
    -------
    pv.Plotter
        PyVista plotter with the mesh added.
    """
    plotter = pv.Plotter()
    pv_mesh = pv.PolyData(mesh.vertices, np.column_stack([
        np.full(len(mesh.faces), 3), mesh.faces
    ]))
    plotter.add_mesh(pv_mesh, color=color, opacity=opacity, show_edges=show_edges)
    plotter.add_title(title)
    plotter.add_axes()
    return plotter


def plot_normals(
    mesh: MeshData,
    normals: np.ndarray,
    step: int = 100,
    arrow_length: float = 3.0,
) -> pv.Plotter:
    """Display mesh with PCA normal arrows.

    Parameters
    ----------
    mesh : MeshData
        Mesh vertices.
    normals : np.ndarray
        Normal vectors (N,3).
    step : int
        Show every step-th normal.
    arrow_length : float
        Length of normal arrows.

    Returns
    -------
    pv.Plotter
        PyVista plotter.
    """
    plotter = pv.Plotter()
    pv_mesh = pv.PolyData(mesh.vertices, np.column_stack([
        np.full(len(mesh.faces), 3), mesh.faces
    ]))
    plotter.add_mesh(pv_mesh, color="peachpuff", opacity=0.5)

    indices = np.arange(0, len(mesh.vertices), step)
    centers = mesh.vertices[indices]
    directions = normals[indices] * arrow_length

    plotter.add_arrows(centers, directions, mag=1.0, color="blue")
    plotter.add_title("Scalp Mesh + PCA Normals")
    plotter.add_axes()
    return plotter


def plot_ese_comparison(ese: ESEResult, step: int = 100) -> pv.Plotter:
    """Display scalp vs ESE surface.

    Parameters
    ----------
    ese : ESEResult
        ESE result with scalp and ESE vertices.
    step : int
        Show every step-th connecting line.

    Returns
    -------
    pv.Plotter
        PyVista plotter.
    """
    plotter = pv.Plotter()

    scalp_cloud = pv.PolyData(ese.scalp_vertices)
    ese_cloud = pv.PolyData(ese.ese_vertices)

    plotter.add_mesh(scalp_cloud, color="peachpuff", opacity=0.5, point_size=2, render_points_as_spheres=True)
    plotter.add_mesh(ese_cloud, color="cyan", opacity=0.5, point_size=2, render_points_as_spheres=True)

    indices = np.arange(0, ese.num_points, step)
    for i in indices:
        line = pv.Line(ese.scalp_vertices[i], ese.ese_vertices[i])
        plotter.add_mesh(line, color="gray", opacity=0.2, line_width=1)

    plotter.add_title("Scalp vs ESE Surface")
    plotter.add_axes()
    return plotter


def plot_quality_map(mesh: MeshData, quality: np.ndarray) -> pv.Plotter:
    """Display PCA quality as a color map on the mesh.

    Parameters
    ----------
    mesh : MeshData
        Mesh to display.
    quality : np.ndarray
        Quality values per vertex.

    Returns
    -------
    pv.Plotter
        PyVista plotter.
    """
    plotter = pv.Plotter()
    pv_mesh = pv.PolyData(mesh.vertices, np.column_stack([
        np.full(len(mesh.faces), 3), mesh.faces
    ]))
    pv_mesh["quality"] = quality
    plotter.add_mesh(pv_mesh, scalars="quality", cmap="viridis", opacity=0.8)
    plotter.add_scalar_bar(title="PCA Quality (lower=better)")
    plotter.add_title("PCA Normal Quality")
    plotter.add_axes()
    return plotter


def plot_localization(
    ese: ESEResult,
    result: LocalizationResult,
    fiducial_mgr: FiducialManager,
) -> pv.Plotter:
    """Display electrode localization results.

    Parameters
    ----------
    ese : ESEResult
        ESE surface.
    result : LocalizationResult
        Localization results.
    fiducial_mgr : FiducialManager
        Fiducial manager with coordinates.

    Returns
    -------
    pv.Plotter
        PyVista plotter.
    """
    plotter = pv.Plotter()

    ese_cloud = pv.PolyData(ese.ese_vertices)
    plotter.add_mesh(ese_cloud, color="lightblue", opacity=0.3, point_size=1, render_points_as_spheres=True)

    if result.num_electrodes > 0:
        _, coords = result.get_electrode_coords()
        electrode_cloud = pv.PolyData(coords)
        plotter.add_mesh(
            electrode_cloud, color="red", point_size=12,
            render_points_as_spheres=True, emissive=True,
        )
        for loc in result.electrodes:
            plotter.add_point_labels(
                [loc.ese_coords], [loc.electrode_id],
                font_size=12, text_color="red",
                point_color="red", point_size=1,
            )

    fid_coords = fiducial_mgr.get_coordinates_matrix()
    if len(fid_coords) > 0:
        fid_cloud = pv.PolyData(fid_coords)
        plotter.add_mesh(
            fid_cloud, color="green", point_size=16,
            render_points_as_spheres=True, emissive=True,
        )
        all_fids = fiducial_mgr.get_all_fiducials()
        labels = list(all_fids.keys())
        plotter.add_point_labels(
            fid_coords, labels,
            font_size=14, text_color="green",
            point_color="green", point_size=1,
        )

    plotter.add_title("Electrode Localization")
    plotter.add_axes()
    return plotter


def save_screenshot(plotter: pv.Plotter, path: Path, window_size: tuple[int, int] = (1920, 1080)) -> None:
    """Save a screenshot from a PyVista plotter.

    Parameters
    ----------
    plotter : pv.Plotter
        Active plotter.
    path : Path
        Output image path.
    window_size : tuple[int, int]
        Image dimensions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.window_size = window_size
    plotter.screenshot(str(path))
    logger.info("Screenshot saved to %s", path)
