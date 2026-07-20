"""Export module — save mesh, points, results in standard formats.

This module lives in the API layer because it depends on trimesh and
nibabel for file I/O. Core modules never import from this module.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from core.surface_extractor import MeshData
from core.fiducial_manager import FiducialManager
from core.ese_config import ESEConfig
from core.ese_generator import ESEResult
from core.electrode_localizer import LocalizationResult
from core.pca_normal_estimator import NormalResult

logger = logging.getLogger(__name__)


def export_mesh_ply(mesh: MeshData, path: str | Path) -> None:
    """Export mesh to PLY format."""
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(str(path))
    logger.info("Exported mesh to %s (%d vertices, %d faces)", path, mesh.num_vertices, mesh.num_faces)


def export_mesh_stl(mesh: MeshData, path: str | Path) -> None:
    """Export mesh to STL format."""
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(str(path))
    logger.info("Exported mesh to %s", path)


def export_mesh_obj(mesh: MeshData, path: str | Path) -> None:
    """Export mesh to OBJ format."""
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(str(path))
    logger.info("Exported mesh to %s", path)


def export_vertices_csv(mesh: MeshData, path: str | Path) -> None:
    """Export vertex coordinates to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex_id", "x", "y", "z"])
        for i, v in enumerate(mesh.vertices):
            writer.writerow([i, f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}"])

    logger.info("Exported %d vertices to %s", mesh.num_vertices, path)


def export_faces_csv(mesh: MeshData, path: str | Path) -> None:
    """Export face indices to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["face_id", "v0", "v1", "v2"])
        for i, face in enumerate(mesh.faces):
            writer.writerow([i, face[0], face[1], face[2]])

    logger.info("Exported %d faces to %s", mesh.num_faces, path)


def export_ese_pairs(ese: ESEResult, path: str | Path) -> None:
    """Export scalp-to-ESE point pairs to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "point_id",
            "scalp_x", "scalp_y", "scalp_z",
            "ese_x", "ese_y", "ese_z",
            "normal_x", "normal_y", "normal_z",
            "pca_quality",
        ])
        for ep in ese.ese_points:
            writer.writerow([
                ep.point_id,
                f"{ep.scalp_coords[0]:.6f}", f"{ep.scalp_coords[1]:.6f}", f"{ep.scalp_coords[2]:.6f}",
                f"{ep.ese_coords[0]:.6f}", f"{ep.ese_coords[1]:.6f}", f"{ep.ese_coords[2]:.6f}",
                f"{ep.normal_vector[0]:.6f}", f"{ep.normal_vector[1]:.6f}", f"{ep.normal_vector[2]:.6f}",
                f"{ep.pca_quality:.6f}",
            ])

    logger.info("Exported %d ESE point pairs to %s", ese.num_points, path)


def export_normals_csv(normals: NormalResult, mesh: MeshData, path: str | Path) -> None:
    """Export normal vectors to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex_id", "nx", "ny", "nz", "quality"])
        for i in range(len(normals.normals)):
            n = normals.normals[i]
            writer.writerow([
                i,
                f"{n[0]:.6f}", f"{n[1]:.6f}", f"{n[2]:.6f}",
                f"{normals.quality[i]:.6f}",
            ])

    logger.info("Exported normals to %s", path)


def export_localization_csv(result: LocalizationResult, path: str | Path) -> None:
    """Export localization results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "electrode_id",
            "ese_x", "ese_y", "ese_z",
            "scalp_x", "scalp_y", "scalp_z",
            "residual_error",
            "confidence",
        ])
        for loc in result.electrodes:
            writer.writerow([
                loc.electrode_id,
                f"{loc.ese_coords[0]:.6f}", f"{loc.ese_coords[1]:.6f}", f"{loc.ese_coords[2]:.6f}",
                f"{loc.scalp_coords[0]:.6f}", f"{loc.scalp_coords[1]:.6f}", f"{loc.scalp_coords[2]:.6f}",
                f"{loc.residual_error:.6f}",
                f"{loc.confidence:.6f}",
            ])

    logger.info("Exported %d electrode positions to %s", result.num_electrodes, path)


def export_localization_json(result: LocalizationResult, path: str | Path) -> None:
    """Export localization results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "summary": {
            "num_electrodes": result.num_electrodes,
            "mean_residual_mm": result.mean_residual,
            "max_residual_mm": result.max_residual,
            "flagged_electrodes": result.flagged_electrodes,
        },
        "electrodes": [],
    }

    for loc in result.electrodes:
        data["electrodes"].append({
            "electrode_id": loc.electrode_id,
            "ese_coords": loc.ese_coords.tolist(),
            "scalp_coords": loc.scalp_coords.tolist(),
            "measured_distances": loc.measured_distances,
            "predicted_distances": loc.predicted_distances,
            "residual_error": loc.residual_error,
            "confidence": loc.confidence,
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Exported localization to %s", path)


def export_segmentation_nifti(
    mask: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
) -> None:
    """Export segmentation mask as NIfTI file."""
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not available, skipping NIfTI export")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.Nifti1Image(mask.astype(np.int32), affine)
    nib.save(img, str(path))
    logger.info("Exported segmentation mask to %s", path)
