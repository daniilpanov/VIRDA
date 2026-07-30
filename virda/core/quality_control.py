"""Quality control checks for pipeline stages."""

from __future__ import annotations

import logging

import numpy as np

from .types import (
    ESEResult,
    LocalizationResult,
    MeshData,
    MRIData,
)

logger = logging.getLogger(__name__)


def validate_stage1(
    mri: MRIData,
    mesh: MeshData,
    fiducial_coords: np.ndarray | None = None,
    ese_offset_mm: float | None = None,
) -> list[str]:
    """Validate Stage 1 outputs.

    Parameters
    ----------
    mri : MRIData
        Loaded MRI data.
    mesh : MeshData
        Generated scalp mesh.
    fiducial_coords : np.ndarray, optional
        Fiducial coordinates (K,3).
    ese_offset_mm : float, optional
        ESE offset distance.

    Returns
    -------
    list[str]
        Warning/error messages. Empty if all checks pass.
    """
    messages: list[str] = []

    if mri.affine.shape != (4, 4):
        messages.append("ERROR: MRI affine is not 4x4")

    det = np.linalg.det(mri.affine[:3, :3])
    if abs(det) < 1e-6:
        messages.append(f"ERROR: MRI affine is near-singular (det={det:.6e})")

    if mesh.num_vertices == 0:
        messages.append("ERROR: Mesh has no vertices")
    if mesh.num_faces == 0:
        messages.append("ERROR: Mesh has no faces")

    if mesh.num_vertices > 0:
        coords = mesh.vertices
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        if coord_range.max() > 500:
            messages.append(
                f"WARNING: Mesh coordinate range is {coord_range.max():.1f}mm, "
                "expected <500mm for a head"
            )
        if coord_range.min() < 1:
            messages.append("WARNING: Mesh is extremely flat in one dimension")

    if fiducial_coords is not None and len(fiducial_coords) > 0:
        if mesh.num_vertices > 0:
            for i, fc in enumerate(fiducial_coords):
                dists = np.linalg.norm(mesh.vertices - fc, axis=1)
                min_dist = float(dists.min())
                if min_dist > 30.0:
                    messages.append(
                        f"WARNING: Fiducial {i} is {min_dist:.1f}mm "
                        "from nearest mesh vertex"
                    )

    if ese_offset_mm is not None and ese_offset_mm <= 0:
        messages.append(f"ERROR: ESE offset must be positive, got {ese_offset_mm}")

    return messages


def validate_stage2(ese: ESEResult) -> list[str]:
    """Validate Stage 2 outputs.

    Parameters
    ----------
    ese : ESEResult
        ESE generation result.

    Returns
    -------
    list[str]
        Warning/error messages.
    """
    messages: list[str] = []

    if ese.num_points == 0:
        messages.append("ERROR: ESE has no points")
        return messages

    norms = np.linalg.norm(ese.normals, axis=1)
    non_unit = np.abs(norms - 1.0) > 0.01
    if non_unit.any():
        messages.append(
            f"WARNING: {non_unit.sum()} normals are not unit vectors"
        )

    radial = ese.scalp_vertices - ese.head_centroid
    radial_norms = np.linalg.norm(radial, axis=1)
    valid = radial_norms > 1e-10
    if valid.any():
        dots = np.sum(ese.normals[valid] * radial[valid], axis=1) / radial_norms[valid]
        inward = dots < -0.5
        if inward.any():
            messages.append(
                f"WARNING: {inward.sum()} normals point inward toward head centroid"
            )

    if np.any(ese.quality < 0) or np.any(ese.quality > 1):
        messages.append("WARNING: Quality values outside [0, 1] range")

    if ese.num_points != len(ese.scalp_vertices):
        messages.append(
            f"ERROR: ESE points ({ese.num_points}) != "
            f"scalp vertices ({len(ese.scalp_vertices)})"
        )

    return messages


def validate_stage3(
    result: LocalizationResult,
    max_residual_threshold: float = 5.0,
) -> list[str]:
    """Validate Stage 3 outputs.

    Parameters
    ----------
    result : LocalizationResult
        Electrode localization result.
    max_residual_threshold : float
        Maximum acceptable residual error.

    Returns
    -------
    list[str]
        Warning/error messages.
    """
    messages: list[str] = []

    if result.num_electrodes == 0:
        messages.append("ERROR: No electrodes localized")

    for loc in result.electrodes:
        if loc.residual_error > max_residual_threshold:
            messages.append(
                f"WARNING: Electrode {loc.electrode_id} residual "
                f"{loc.residual_error:.2f}mm exceeds threshold "
                f"{max_residual_threshold:.1f}mm"
            )

        if np.any(np.isnan(loc.ese_coords)):
            messages.append(
                f"ERROR: Electrode {loc.electrode_id} has NaN coordinates"
            )

    return messages
