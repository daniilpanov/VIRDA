"""Export pipeline results to standard file formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh

from ..core.fiducial_manager import FiducialManager
from ..core.types import ESEResult, LocalizationResult, MeshData, NormalResult

logger = logging.getLogger(__name__)


def export_mesh(
    mesh: MeshData,
    path: Path,
    file_format: str | None = None,
) -> None:
    """Export mesh to PLY, STL, OBJ, or VTK format.

    Parameters
    ----------
    mesh : MeshData
        Mesh to export.
    path : Path
        Output file path.
    file_format : str, optional
        Format string. Inferred from extension if None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(path, file_type=file_format)
    logger.info("Exported mesh to %s (%d vertices, %d faces)", path, mesh.num_vertices, mesh.num_faces)


def export_vertices_csv(mesh: MeshData, path: Path) -> None:
    """Export vertex coordinates to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])
    df.index.name = "vertex_id"
    df.to_csv(path)
    logger.info("Exported %d vertices to %s", mesh.num_vertices, path)


def export_faces_csv(mesh: MeshData, path: Path) -> None:
    """Export face indices to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(mesh.faces, columns=["v0", "v1", "v2"])
    df.index.name = "face_id"
    df.to_csv(path)
    logger.info("Exported %d faces to %s", mesh.num_faces, path)


def export_normals_csv(normal_result: NormalResult, path: Path) -> None:
    """Export normals and quality to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([
        normal_result.normals,
        normal_result.quality,
        normal_result.eigenvalues,
    ])
    columns = ["nx", "ny", "nz", "quality", "eigenvalue_1", "eigenvalue_2", "eigenvalue_3"]
    df = pd.DataFrame(data, columns=columns)
    df.index.name = "vertex_id"
    df.to_csv(path)
    logger.info("Exported normals to %s", path)


def export_ese_pairs_csv(ese: ESEResult, path: Path) -> None:
    """Export scalp-to-ESE point pairs to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([ese.scalp_vertices, ese.ese_vertices])
    columns = [
        "scalp_x", "scalp_y", "scalp_z",
        "ese_x", "ese_y", "ese_z",
    ]
    df = pd.DataFrame(data, columns=columns)
    df.index.name = "point_id"
    df.to_csv(path)
    logger.info("Exported %d ESE pairs to %s", ese.num_points, path)


def export_fiducials_json(fiducial_mgr: FiducialManager, path: Path) -> None:
    """Export fiducials to JSON."""
    fiducial_mgr.save(path)
    logger.info("Exported fiducials to %s", path)


def export_electrodes_csv(result: LocalizationResult, path: Path) -> None:
    """Export electrode localization results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for loc in result.electrodes:
        row = {
            "electrode_id": loc.electrode_id,
            "ese_x": loc.ese_coords[0],
            "ese_y": loc.ese_coords[1],
            "ese_z": loc.ese_coords[2],
            "scalp_x": loc.scalp_coords[0],
            "scalp_y": loc.scalp_coords[1],
            "scalp_z": loc.scalp_coords[2],
            "residual_error": loc.residual_error,
            "confidence": loc.confidence,
        }
        for fid, dist in loc.measured_distances.items():
            row[f"measured_{fid}"] = dist
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Exported %d electrodes to %s", result.num_electrodes, path)


def export_electrodes_json(result: LocalizationResult, path: Path) -> None:
    """Export electrode localization results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for loc in result.electrodes:
        data[loc.electrode_id] = {
            "ese_coords": loc.ese_coords.tolist(),
            "scalp_coords": loc.scalp_coords.tolist(),
            "measured_distances": loc.measured_distances,
            "residual_error": loc.residual_error,
            "confidence": loc.confidence,
        }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Exported %d electrodes to %s", result.num_electrodes, path)


def export_config_json(config, path: Path) -> None:
    """Export ESE configuration to JSON."""
    config.save(path)


def create_project_folder(base: Path) -> dict[str, Path]:
    """Create standard project folder structure.

    Parameters
    ----------
    base : Path
        Base project directory.

    Returns
    -------
    dict[str, Path]
        Mapping of folder names to created paths.
    """
    base = Path(base)
    folders = {
        "input_mri": base / "input_mri",
        "segmentation": base / "segmentation",
        "mesh": base / "mesh",
        "fiducials": base / "fiducials",
        "config": base / "config",
        "results": base / "results",
        "quality_control": base / "quality_control",
        "logs": base / "logs",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders
